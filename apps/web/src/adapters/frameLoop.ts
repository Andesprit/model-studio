import { decode, toTensor, type ModelConfig, type PipelineRunner, type Result } from "@detect/core";
import { FrameGrabber } from "./frameGrabber";
import { Renderer } from "./renderer";
import type { ModelRuntime } from "./runtime";

export interface Stats {
  fps: number;
  preMs: number;
  inferMs: number;
  postMs: number;
  detections: number;
  /** Values published by the pipeline. */
  metrics?: Record<string, number | string>;
}

export interface FrameLoopCallbacks {
  onStats(stats: Stats): void;
  onError(error: unknown): void;
}

type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

/**
 * Runs model inference on a video element and draws results on an overlay canvas.
 * At most one inference is in flight; frames that arrive meanwhile are dropped, so latency stays bounded.
 */
export class FrameLoop {
  private grabber = new FrameGrabber();
  private renderer: Renderer;
  private runtime: ModelRuntime | null = null;
  private config: ModelConfig | null = null;
  /** User pipeline; null means decode with the config's decoder and draw as-is. */
  private pipeline: PipelineRunner | null = null;
  private busy = false;
  private inflight: Promise<void> = Promise.resolve();
  /** Set after an error; cleared when the model or config changes. */
  private halted = false;
  /** A refresh was requested while busy; run once more when the current inference finishes. */
  private pending = false;
  private running = false;
  private handle = 0;
  private frameTimes: number[] = [];

  constructor(
    private video: VideoWithRVFC,
    overlay: HTMLCanvasElement,
    private cb: FrameLoopCallbacks,
  ) {
    this.renderer = new Renderer(overlay);
  }

  setModel(runtime: ModelRuntime | null, config: ModelConfig | null) {
    this.runtime = runtime;
    this.config = config;
    this.pipeline?.reset();
    this.halted = false;
    this.renderer.clear();
    this.refresh();
  }

  /** Stops using the current model and resolves once no inference is running on it, so it can be released. */
  detach(): Promise<void> {
    this.setModel(null, null);
    return this.inflight;
  }

  /** Swap config (thresholds etc.) without reloading the model; re-runs on the current frame if paused. */
  setConfig(config: ModelConfig) {
    this.config = config;
    this.halted = false;
    this.refresh();
  }

  /** Swap the user pipeline (or remove it with null) without reloading the model. */
  setPipeline(pipeline: PipelineRunner | null) {
    this.pipeline = pipeline;
    this.halted = false;
    this.refresh();
  }

  /** Clears pipeline state, e.g. when the source changes. */
  resetPipeline() {
    this.pipeline?.reset();
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.schedule();
  }

  stop() {
    this.running = false;
    if (this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this.handle);
    else cancelAnimationFrame(this.handle);
  }

  /** Process the current frame once, e.g. when the video is paused and settings change. */
  refresh() {
    if (this.busy) this.pending = true;
    else if (this.video.readyState >= 2) void this.run();
  }

  private schedule() {
    if (!this.running) return;
    const next = () => {
      if (this.busy) this.pipeline?.markDropped();
      void this.run();
      this.schedule();
    };
    this.handle = this.video.requestVideoFrameCallback
      ? this.video.requestVideoFrameCallback(next)
      : requestAnimationFrame(next);
  }

  private run() {
    if (this.busy || this.halted || !this.runtime || !this.config || this.video.readyState < 2) return;
    this.inflight = this.tick(this.runtime, this.config, this.pipeline);
  }

  private async tick(runtime: ModelRuntime, config: ModelConfig, pipeline: PipelineRunner | null) {
    this.busy = true;
    try {
      const t0 = performance.now();
      const { frame, transform } = this.grabber.grab(this.video, config.input);
      const tensor = toTensor(frame, config.input);
      const t1 = performance.now();
      const outputs = await runtime.run(tensor);
      const t2 = performance.now();
      if (runtime !== this.runtime) return; // model swapped mid-frame; drop the stale result
      if (pipeline !== this.pipeline) return; // pipeline swapped mid-frame; its state must not see this frame
      const result = pipeline
        ? pipeline.run(outputs, config, transform, this.video.currentTime)
        : decode(outputs, config, transform);
      this.renderer.draw(result, transform.frameWidth, transform.frameHeight);
      const t3 = performance.now();
      this.report(t0, t1, t2, t3, result);
    } catch (e) {
      // A failing decoder would fail on every frame; halt and surface it once.
      // Ignore failures from a model or config that was replaced while this frame was in flight.
      if (runtime === this.runtime && config === this.config && pipeline === this.pipeline) {
        this.halted = true;
        this.video.pause();
        this.cb.onError(e);
      }
    } finally {
      this.busy = false;
      if (this.pending) {
        this.pending = false;
        this.refresh();
      }
    }
  }

  private report(t0: number, t1: number, t2: number, t3: number, r: Result) {
    this.frameTimes.push(t3);
    while (this.frameTimes.length && t3 - this.frameTimes[0] > 1000) this.frameTimes.shift();
    this.cb.onStats({
      fps: this.frameTimes.length,
      preMs: t1 - t0,
      inferMs: t2 - t1,
      postMs: t3 - t2,
      detections: (r.boxes?.length ?? 0) + (r.poses?.length ?? 0) + (r.classes?.length ?? 0),
      metrics: r.metrics,
    });
  }
}
