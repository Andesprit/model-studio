import { decodeAll, mergeResults, toTensor, type ModelConfig, type ModelInput, type PipelineRunner, type Result } from "@detect/core";
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
  /** Pipeline loaded by the user (step 3); runs on every model's results together. */
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
  private onResize = () => this.refresh();
  private handle = 0;
  private frameTimes: number[] = [];

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
    this.halted = false;
    this.refresh();
  }

  /** Swap the user pipeline (or remove it with null) without reloading the model. */
  setPipeline(pipeline: PipelineRunner | null) {
    this.pipeline = pipeline;
    this.halted = false;
    this.refresh();
  }

  /** Sets (or clears with null) the pipeline that runs on one slot's model only. */
  setModelPipeline(slot: number, pipeline: PipelineRunner | null) {
    while (this.modelPipelines.length <= slot) this.modelPipelines.push(null);
    this.modelPipelines[slot] = pipeline;
    this.halted = false;
    this.refresh();
  }

  /** Clears every pipeline's state, e.g. when the source changes. */
  resetPipeline() {
    this.pipeline?.reset();
    for (const p of this.modelPipelines) p?.reset();
  }

  start() {
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

  /** Process the current frame once, e.g. when the video is paused and settings change. */
  refresh() {
    if (this.busy) this.pending = true;
    else if (this.video.readyState >= 2) void this.run();
  }

  private schedule() {
    if (!this.running) return;
    const next = () => {
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
      const slotIndex = slots.flatMap((s, i) => (s ? [i] : []));
      const loaded = slotIndex.map((i) => slots[i]!);
      // Grab every model's input before running any of them, so all see the same video frame
      // (the video keeps playing while the first model runs). Each uses its own input config.
      const t0 = performance.now();
      const prepared = loaded.map(({ config, grabber }) => {
        const { frame, transform } = grabber.grab(this.video, config.input);
        return { tensor: toTensor(frame, config.input), transform };
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
      if (!this.current(slots, pipeline, modelPipelines)) return;
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
      this.renderer.draw(result, frameWidth, frameHeight);
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
