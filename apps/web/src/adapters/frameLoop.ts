import { decodeAll, mergeResults, toTensor, type ModelConfig, type ModelInput, type PipelineRunner, type Result, type SegmentationMap } from "@detect/core";
import { FrameGrabber } from "./frameGrabber";
import { Renderer } from "./renderer";
import type { ModelRuntime } from "./runtime";

export interface Stats {
  fps: number;
  preMs: number;
  /** Total inference time of all models. */
  inferMs: number;
  /** Inference time per loaded model, in slot order (A, B…). */
  inferByModel: number[];
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

/** One frame's analysis: the result and where to draw it, plus timings. */
interface Inferred {
  result: Result;
  frameWidth: number;
  frameHeight: number;
  /** Clockwise turn of the picture the models saw (0 for most models). */
  turn: number;
  preMs: number;
  inferByModel: number[];
  /** When post-processing started (performance.now()). */
  t2: number;
}

/** A frame analyzed in advance (analyze()), kept for playback. The segmentation is stored run-length encoded. */
interface AnalyzedFrame {
  time: number;
  result: Result;
  runs: Uint32Array | null;
  frameWidth: number;
  frameHeight: number;
  turn: number;
  preMs: number;
  inferByModel: number[];
  postMs: number;
}

/** Frames per second of video that analyze() can analyze; fewer is faster, playback shows the nearest one. */
export const ANALYZE_RATES = [10, 15, 30] as const;

/** What analyze() did: the rate, the frames, the wall time, and the mean time per frame of each step. */
export interface AnalysisReport {
  fps: number;
  frames: number;
  seconds: number;
  /** Milliseconds per frame: seeking to the frame, preparing the models' inputs, the models, the pipelines. */
  perFrameMs: { seek: number; prepare: number; model: number; measure: number };
}

/** A loaded model. Replaced (never mutated) on change, so an in-flight frame can tell it went stale. */
interface Slot {
  runtime: ModelRuntime;
  config: ModelConfig;
  grabber: FrameGrabber;
}

/**
 * Runs model inference on a video element and draws results on an overlay canvas.
 * Any number of models (slots A, B, C…) run on the same frame, one after the other, and their results are drawn together.
 * At most one frame is in flight; frames that arrive meanwhile are dropped, so latency stays bounded.
 */
export class FrameLoop {
  private renderer: Renderer;
  private slots: (Slot | null)[] = [];
  /** Pipeline loaded by the user in Advanced; runs on every model's results together. */
  private pipeline: PipelineRunner | null = null;
  /** Each slot's own pipeline (e.g. embedded in its model); runs on that model's output only. Kept apart from
   * the slot so turning a model off and on keeps its pipeline. */
  private modelPipelines: (PipelineRunner | null)[] = [];
  private busy = false;
  private inflight: Promise<void> = Promise.resolve();
  private modelChanges: Promise<void> = Promise.resolve();
  private changingModels = 0;
  private rejectFrame: ((error: unknown) => void) | null = null;
  private onRuntimeError = (event: ErrorEvent | PromiseRejectionEvent) => {
    const error = "reason" in event ? event.reason : event.error;
    // ORT can trap outside session.run's promise, leaving it pending forever.
    if (error instanceof WebAssembly.RuntimeError) this.rejectFrame?.(error);
  };
  /** Set after an error; cleared when the model or config changes. */
  private halted = false;
  /** A refresh was requested while busy; run once more when the current inference finishes. */
  private pending = false;
  /** Only remember that a newer live frame exists, never queue its pixels. */
  private liveFramePending = false;
  private running = false;
  private finished = false;
  private onResize = () => this.refresh();
  private handle = 0;
  private frameTimes: number[] = [];
  /** Frames analyzed in advance; while set, playback draws these instead of running the models. */
  private replay: AnalyzedFrame[] | null = null;
  private replayFps = 30;
  private analyzing = false;

  constructor(
    private video: VideoWithRVFC,
    overlay: HTMLCanvasElement,
    private cb: FrameLoopCallbacks,
  ) {
    this.renderer = new Renderer(overlay);
  }

  /** Puts a model in a slot (0 = A, 1 = B…), or empties the slot with null. */
  setModel(slot: number, runtime: ModelRuntime | null, config: ModelConfig | null) {
    while (this.slots.length <= slot) this.slots.push(null);
    const prev = this.slots[slot];
    this.slots[slot] = runtime && config ? { runtime, config, grabber: prev?.grabber ?? new FrameGrabber() } : null;
    this.replay = null;
    this.resetPipeline();
    this.halted = false;
    this.renderer.clear();
    this.refresh();
  }

  /** Stops using a slot's model and resolves once no inference is running on it, so it can be released. */
  detach(slot: number): Promise<void> {
    this.setModel(slot, null, null);
    return this.inflight;
  }

  /** Session setup shares ORT state with inference. Drain the frame and serialize model changes. */
  async changeModels<T>(change: () => Promise<T>): Promise<T> {
    this.changingModels++;
    const previous = this.modelChanges;
    let release!: () => void;
    this.modelChanges = new Promise<void>((resolve) => { release = resolve; });
    try {
      await previous;
      await this.inflight;
      return await change();
    } finally {
      this.changingModels--;
      release();
      if (!this.changingModels) this.refresh();
    }
  }

  /** Swap a slot's config without reloading its model; re-runs on the current frame if paused. */
  setConfig(slot: number, config: ModelConfig) {
    const s = this.slots[slot];
    if (!s) return;
    this.slots[slot] = { ...s, config };
    this.replay = null;
    this.halted = false;
    this.refresh();
  }

  /**
   * Clockwise turn (degrees) of the picture for a model: a landscape picture is turned onto a portrait canvas so the
   * subject keeps its size (the pipeline then decides which way the rope runs).
   */
  private turnFor(config: ModelConfig): number {
    if (!portraitCanvas(config)) return 0;
    return this.video.videoWidth > this.video.videoHeight ? 90 : 0;
  }

  /** Swap the user pipeline (or remove it with null) without reloading the model. */
  setPipeline(pipeline: PipelineRunner | null) {
    this.pipeline = pipeline;
    this.replay = null;
    this.halted = false;
    this.refresh();
  }

  /** Sets (or clears with null) the pipeline that runs on one slot's model only. */
  setModelPipeline(slot: number, pipeline: PipelineRunner | null) {
    while (this.modelPipelines.length <= slot) this.modelPipelines.push(null);
    this.modelPipelines[slot] = pipeline;
    this.replay = null;
    this.halted = false;
    this.refresh();
  }

  /** Clears every pipeline's state, e.g. when the source changes. */
  resetPipeline() {
    this.pipeline?.reset();
    for (const p of this.modelPipelines) p?.reset();
  }

  start() {
    this.finished = false;
    if (this.running) return;
    this.running = true;
    window.addEventListener("error", this.onRuntimeError);
    window.addEventListener("unhandledrejection", this.onRuntimeError);
    window.addEventListener("resize", this.onResize);
    this.schedule();
  }

  stop() {
    this.running = false;
    window.removeEventListener("error", this.onRuntimeError);
    window.removeEventListener("unhandledrejection", this.onRuntimeError);
    window.removeEventListener("resize", this.onResize);
    if (this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this.handle);
    else cancelAnimationFrame(this.handle);
  }

  /** Freeze this source after its last inference, keeping the pipelines available for export. */
  async finish(): Promise<void> {
    this.finished = true;
    this.running = false;
    if (this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this.handle);
    else cancelAnimationFrame(this.handle);
    // Keep the fatal-error listeners until the pending inference has settled.
    await this.inflight;
    if (this.finished) this.stop();
  }

  /**
   * Analyzes the whole video in advance: seeks to every 1/fps s from the start, paused, and runs the models and
   * pipelines on each frame in order (the pipelines measure the whole video, as in live playback). Playback then
   * draws the stored results instead of running the models, so it never lags behind the video. Resolves with what it
   * did when done; null when aborted or a model, config or pipeline changed meanwhile. Errors are thrown.
   */
  async analyze(fps: number, onProgress: (done: number, total: number) => void, signal: AbortSignal): Promise<AnalysisReport | null> {
    this.stop();
    this.finished = false;
    await this.inflight;
    this.replay = null;
    this.analyzing = true;
    this.resetPipeline();
    this.renderer.clear();
    this.video.pause();
    const slots = [...this.slots];
    const pipeline = this.pipeline;
    const modelPipelines = [...this.modelPipelines];
    const total = Math.max(1, Math.floor(this.video.duration * fps) + 1);
    const frames: AnalyzedFrame[] = [];
    const ms = { seek: 0, prepare: 0, model: 0, measure: 0 };
    const start = performance.now();
    const failed = new Promise<never>((_, reject) => { this.rejectFrame = reject; });
    failed.catch(() => {});
    window.addEventListener("error", this.onRuntimeError);
    window.addEventListener("unhandledrejection", this.onRuntimeError);
    try {
      onProgress(0, total);
      for (let i = 0; i < total; i++) {
        if (signal.aborted || !slots.some(Boolean)) return null;
        const t0 = performance.now();
        await seek(this.video, Math.min(i / fps, this.video.duration));
        ms.seek += performance.now() - t0;
        if (signal.aborted) return null;
        const done = await this.infer(slots, pipeline, modelPipelines, this.video.currentTime, failed);
        if (!done) return null;
        const { result, frameWidth, frameHeight, turn, preMs, inferByModel, t2 } = done;
        const postMs = performance.now() - t2;
        frames.push({ time: this.video.currentTime, ...packSegmentation(result), frameWidth, frameHeight, turn, preMs,
          inferByModel, postMs });
        ms.prepare += preMs;
        ms.model += inferByModel.reduce((a, b) => a + b, 0);
        ms.measure += postMs;
        onProgress(i + 1, total);
      }
      this.replay = frames;
      this.replayFps = fps;
      const per = (v: number) => Math.round(v / total);
      return { fps, frames: total, seconds: Math.round((performance.now() - start) / 100) / 10,
        perFrameMs: { seek: per(ms.seek), prepare: per(ms.prepare), model: per(ms.model), measure: per(ms.measure) } };
    } finally {
      this.analyzing = false;
      this.rejectFrame = null;
      window.removeEventListener("error", this.onRuntimeError);
      window.removeEventListener("unhandledrejection", this.onRuntimeError);
    }
  }

  /** True while playback draws frames analyzed in advance. */
  get analyzed() {
    return this.replay !== null;
  }

  /** Forgets the frames analyzed in advance, e.g. when the source changes. */
  clearReplay() {
    this.replay = null;
  }

  /** Draws the analyzed frame shown at the video's current time. */
  private showReplay(frames: AnalyzedFrame[]) {
    const t = this.video.currentTime + 0.5 / this.replayFps;
    let lo = 0;
    let hi = frames.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (frames[mid].time <= t) lo = mid;
      else hi = mid - 1;
    }
    const f = frames[lo];
    if (!f) return;
    const result = unpackSegmentation(f.result, f.runs);
    this.renderer.draw(result, f.frameWidth, f.frameHeight,
      f.turn ? { turn: f.turn, width: this.video.videoWidth, height: this.video.videoHeight } : undefined);
    this.report(f.preMs, f.inferByModel, f.postMs, performance.now(), result);
  }

  /** Process the current frame once, e.g. when the video is paused and settings change. */
  refresh() {
    if (this.finished || this.analyzing) return;
    if (this.busy) this.pending = true;
    else if (this.video.readyState >= 2) void this.run();
  }

  private schedule() {
    if (!this.running) return;
    const next = () => {
      if (!this.running) return;
      if (this.busy) {
        this.liveFramePending = true;
        for (const p of [this.pipeline, ...this.modelPipelines]) p?.markDropped();
      }
      void this.run();
      this.schedule();
    };
    this.handle = this.video.requestVideoFrameCallback
      ? this.video.requestVideoFrameCallback(next)
      : requestAnimationFrame(next);
  }

  private run() {
    if (this.analyzing) return;
    if (this.replay) {
      if (this.video.readyState >= 2) this.showReplay(this.replay);
      return;
    }
    if (this.busy || this.changingModels || this.halted || !this.slots.some(Boolean) || this.video.readyState < 2) return;
    this.inflight = this.tick([...this.slots], this.pipeline, [...this.modelPipelines]);
  }

  /** True while the slots and pipelines a frame started with are still the current ones. */
  private current(slots: (Slot | null)[], pipeline: PipelineRunner | null, modelPipelines: (PipelineRunner | null)[]) {
    return (
      pipeline === this.pipeline &&
      slots.every((s, i) => s === this.slots[i]) &&
      modelPipelines.every((p, i) => p === this.modelPipelines[i])
    );
  }

  private async tick(slots: (Slot | null)[], pipeline: PipelineRunner | null, modelPipelines: (PipelineRunner | null)[]) {
    this.busy = true;
    this.liveFramePending = false;
    const timestamp = this.video.currentTime;
    const failed = new Promise<never>((_, reject) => { this.rejectFrame = reject; });
    try {
      const done = await this.infer(slots, pipeline, modelPipelines, timestamp, failed);
      if (!done) return;
      const { result, frameWidth, frameHeight, turn, preMs, inferByModel, t2 } = done;
      this.renderer.draw(result, frameWidth, frameHeight,
        turn ? { turn, width: this.video.videoWidth, height: this.video.videoHeight } : undefined);
      const t3 = performance.now();
      this.report(preMs, inferByModel, t3 - t2, t3, result);
    } catch (e) {
      // A failing decoder would fail on every frame; halt and surface it once.
      // Ignore ordinary stale failures; a WASM trap also affects the replacement model's shared runtime.
      if (e instanceof WebAssembly.RuntimeError || this.current(slots, pipeline, modelPipelines)) {
        this.halted = true;
        this.video.pause();
        this.cb.onError(e);
      }
    } finally {
      this.rejectFrame = null;
      this.busy = false;
      if (this.pending) {
        this.pending = false;
        this.refresh();
      } else if (this.running && !this.video.paused && this.liveFramePending && this.video.currentTime !== timestamp
        && slots.some((slot) => slot?.runtime.backend === "wasm")) {
        // Worker replies arrive between video callbacks: use the latest frame instead of waiting again.
        this.refresh();
      }
    }
  }

  /** Runs every model and pipeline on the video's current frame; null when a model or pipeline changed meanwhile. */
  private async infer(slots: (Slot | null)[], pipeline: PipelineRunner | null, modelPipelines: (PipelineRunner | null)[],
    timestamp: number, failed: Promise<never>): Promise<Inferred | null> {
    const slotIndex = slots.flatMap((s, i) => (s ? [i] : []));
    const loaded = slotIndex.map((i) => slots[i]!);
    // Grab every model's input before running any of them, so all see the same video frame
    // (the video keeps playing while the first model runs). Each uses its own input config.
    const t0 = performance.now();
    const prepared = loaded.map(({ config, grabber }) => {
      const turn = this.turnFor(config);
      const { frame, transform } = grabber.grab(this.video, config.input, turn);
      return { tensor: toTensor(frame, config.input), transform, turn };
    });
    const preMs = performance.now() - t0;
    const inputs: ModelInput[] = [];
    const inferByModel: number[] = [];
    for (const [i, { runtime, config }] of loaded.entries()) {
      const t1 = performance.now();
      // The model's own pipeline can set its other inputs (e.g. which branch of the graph runs).
      const outputs = await Promise.race([runtime.run(prepared[i].tensor, modelPipelines[slotIndex[i]]?.feeds(timestamp)), failed]);
      inferByModel.push(performance.now() - t1);
      inputs.push({ outputs, config, transform: prepared[i].transform });
    }
    // A model, config or pipeline swapped mid-frame: drop the stale result (and keep it out of the pipeline's state).
    if (!this.current(slots, pipeline, modelPipelines)) return null;
    const t2 = performance.now();
    const own = slotIndex.map((i) => modelPipelines[i]);
    let result: Result;
    if (own.some(Boolean)) {
      // Each model through its own pipeline (or just its decoder), then the user's pipeline on top, if any.
      const perModel = inputs.map((input, k) => own[k]?.run([input], timestamp) ?? decodeAll([input]));
      result = pipeline ? pipeline.runDecoded(perModel, inputs, timestamp) : mergeResults(perModel);
    } else {
      result = pipeline ? pipeline.run(inputs, timestamp) : decodeAll(inputs);
    }
    const { frameWidth, frameHeight } = inputs[0].transform;
    const turn = prepared[0].turn;
    if (turn) result = { ...result, metrics: { ...result.metrics, "frame turn deg": turn } };
    return { result, frameWidth, frameHeight, turn, preMs, inferByModel, t2 };
  }

  private report(preMs: number, inferByModel: number[], postMs: number, now: number, r: Result) {
    this.frameTimes.push(now);
    while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift();
    this.cb.onStats({
      fps: this.frameTimes.length,
      preMs,
      inferMs: inferByModel.reduce((a, b) => a + b, 0),
      inferByModel,
      postMs,
      detections: (r.boxes?.length ?? 0) + (r.poses?.length ?? 0) + (r.classes?.length ?? 0),
      metrics: r.metrics,
    });
  }
}

/** Turning applies only to models whose input canvas is taller than wide (the vortex lamp). */
const portraitCanvas = (config: ModelConfig) => (config.input?.height ?? 0) > (config.input?.width ?? 0);

/** Resolves once the video shows the frame at `time` (paused). */
function seek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      clearTimeout(timer);
      resolve();
    };
    // Some browsers fire no seeked event when the time doesn't change.
    const timer = setTimeout(done, 3000);
    video.addEventListener("seeked", done);
    video.currentTime = time;
  });
}

/** The result without its segmentation classes, and the classes as (value, length) runs: a mask is mostly background. */
function packSegmentation(result: Result): { result: Result; runs: Uint32Array | null } {
  const s = result.segmentation;
  if (!s) return { result, runs: null };
  const runs: number[] = [];
  const c = s.classes;
  for (let i = 0; i < c.length;) {
    let j = i + 1;
    while (j < c.length && c[j] === c[i]) j++;
    runs.push(c[i], j - i);
    i = j;
  }
  return { result: { ...result, segmentation: { ...s, classes: new Uint16Array(0) } }, runs: Uint32Array.from(runs) };
}

function unpackSegmentation(result: Result, runs: Uint32Array | null): Result {
  const s = result.segmentation;
  if (!s || !runs) return result;
  const classes = new Uint16Array(s.width * s.height);
  for (let r = 0, i = 0; r < runs.length; r += 2) {
    classes.fill(runs[r], i, i + runs[r + 1]);
    i += runs[r + 1];
  }
  return { ...result, segmentation: { ...s, classes } as SegmentationMap };
}
