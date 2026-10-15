import { decode, explainDecode, lib, runDecoder, utils } from "./decoders";
import type { Calibration, MetricStats, ModelConfig, PipelineDefinition, RunSummary, PipelineInstance, Result, Tensor, Transform } from "./types";

/** Builds a calibration from a measured line, or null if either length is missing or not positive. */
export function makeCalibration(pixels: number, amount: number, unit: string): Calibration | null {
  if (!(pixels > 0) || !(amount > 0)) return null;
  return { pixels, amount, unit, unitsPerPixel: amount / pixels };
}

/** Checks a loaded module's default export and returns it as a pipeline. Throws with a readable message. */
export function definePipeline(value: unknown): PipelineDefinition {
  const p = value as Partial<PipelineDefinition> | null;
  if (!p || typeof p !== "object" || typeof p.name !== "string" || !p.name) {
    throw new Error("Pipeline must export { name: string, decode?(outputs, ctx), create?(ctx) }");
  }
  if (typeof p.decode !== "function" && typeof p.create !== "function") {
    throw new Error(`Pipeline "${p.name}" needs decode(outputs, ctx), create(ctx), or both`);
  }
  return p as PipelineDefinition;
}

/**
 * Runs one pipeline: decodes with the pipeline's decoder (or the config's when it has none),
 * then passes the result through the current instance. `reset()` drops the instance; the next
 * frame creates a fresh one. Time going backwards (a looped or rewound video) also resets,
 * so the same footage is never counted twice.
 */
export class PipelineRunner {
  private instance: PipelineInstance | null = null;
  private frames = 0;
  private dropped = 0;
  private lastTimestamp = -Infinity;
  private firstTimestamp: number | null = null;
  /** Distinct frames seen (re-runs of a paused frame excluded). */
  private analyzed = 0;
  private stats = new Map<string, { sum: number; min: number; max: number; last: number; n: number }>();
  readonly options: Record<string, unknown>;

  constructor(
    readonly pipeline: PipelineDefinition,
    options: Record<string, unknown> = {},
    readonly calibration: Calibration | null = null,
  ) {
    this.options = { ...pipeline.defaults, ...options };
  }

  reset() {
    this.instance = null;
    this.frames = 0;
    this.dropped = 0;
    this.lastTimestamp = -Infinity;
    this.firstTimestamp = null;
    this.analyzed = 0;
    this.stats.clear();
  }

  /** What has been measured since the last reset. Numeric metrics are summarized automatically. */
  summary(): RunSummary {
    const metrics: Record<string, MetricStats> = {};
    for (const [k, s] of this.stats) metrics[k] = { mean: s.sum / s.n, min: s.min, max: s.max, last: s.last, frames: s.n };
    const own = this.instance?.summary ? blame(this.pipeline.name, "summary", () => this.instance!.summary!()) : undefined;
    return {
      from: this.firstTimestamp,
      to: this.firstTimestamp === null ? null : this.lastTimestamp,
      framesAnalyzed: this.analyzed,
      framesDropped: this.dropped,
      metrics,
      ...(own && { pipelineResults: own }),
    };
  }

  /** Records a frame that was skipped because inference was busy. */
  markDropped() {
    this.dropped++;
  }

  run(outputs: Record<string, Tensor>, config: ModelConfig, transform: Transform, timestamp: number): Result {
    if (timestamp < this.lastTimestamp) this.reset();
    // A paused video re-runs the same frame; it must not count twice in the statistics.
    const repeat = timestamp === this.lastTimestamp;
    this.lastTimestamp = timestamp;
    this.firstTimestamp ??= timestamp;
    const result = this.process(outputs, config, transform, timestamp);
    if (!repeat) {
      this.analyzed++;
      this.record(result.metrics);
    }
    return result;
  }

  private record(metrics: Result["metrics"]) {
    for (const [k, v] of Object.entries(metrics ?? {})) {
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      const s = this.stats.get(k);
      if (!s) this.stats.set(k, { sum: v, min: v, max: v, last: v, n: 1 });
      else Object.assign(s, { sum: s.sum + v, min: Math.min(s.min, v), max: Math.max(s.max, v), last: v, n: s.n + 1 });
    }
  }

  private process(outputs: Record<string, Tensor>, config: ModelConfig, transform: Transform, timestamp: number): Result {
    const p = this.pipeline;
    const pd = p.decode;
    const result = pd
      ? explainDecode(`Pipeline "${p.name}"'s decode`, outputs, () => runDecoder({ decode: pd }, outputs, config, transform, this.options))
      : decode(outputs, config, transform);
    if (!p.create) return result;
    if (!this.instance) {
      const instance = blame(p.name, "create", () =>
        p.create!({ config, options: this.options, labels: config.labels ?? [], utils, lib, calibration: this.calibration }),
      );
      if (!instance || typeof instance.step !== "function") {
        throw new Error(`Pipeline "${p.name}": create(ctx) must return { step(result, frame) }`);
      }
      this.instance = instance;
    }
    const instance = this.instance;
    const frame = { timestamp, index: this.frames++, dropped: this.dropped, width: transform.frameWidth, height: transform.frameHeight };
    return blame(p.name, "step", () => instance.step(result, frame));
  }
}

/** Prefixes errors from user code with the pipeline and function they came from. */
function blame<T>(pipeline: string, fn: string, run: () => T): T {
  try {
    return run();
  } catch (e) {
    throw new Error(`Pipeline "${pipeline}" failed in ${fn}: ${e instanceof Error ? e.message : String(e)}`);
  }
}
