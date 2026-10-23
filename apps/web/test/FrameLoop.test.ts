import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelConfig } from "@detect/core";
import { FrameLoop } from "../src/adapters/frameLoop";
import type { ModelRuntime } from "../src/adapters/runtime";

vi.mock("@detect/core", () => ({
  toTensor: () => ({ dims: [1], data: new Float32Array(1) }),
  decodeAll: () => ({}),
  mergeResults: (results: unknown[]) => results[0] ?? {},
}));
vi.mock("../src/adapters/frameGrabber", () => ({ FrameGrabber: class {
  grab() { return { frame: {}, transform: { frameWidth: 1, frameHeight: 1 } }; }
} }));
vi.mock("../src/adapters/renderer", () => ({ Renderer: class {
  clear() {}
  draw() {}
} }));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

const config = {} as ModelConfig;
const runtime = (run = vi.fn(async () => ({}))): ModelRuntime => ({
  backend: "wasm", input: { name: "image", type: "float32", shape: [1] }, outputNames: [], run, release: vi.fn(),
});
const loops: FrameLoop[] = [];
function setup() {
  const video = { readyState: 2, currentTime: 0, pause: vi.fn(), requestVideoFrameCallback: vi.fn(() => 1), cancelVideoFrameCallback: vi.fn() };
  const cb = { onStats: vi.fn(), onError: vi.fn() };
  const loop = new FrameLoop(video as unknown as HTMLVideoElement, {} as HTMLCanvasElement, cb);
  loops.push(loop);
  return { loop, video, cb };
}

beforeEach(() => vi.stubGlobal("window", new EventTarget()));
afterEach(() => { loops.splice(0).forEach((loop) => loop.stop()); vi.unstubAllGlobals(); });

describe("model changes during inference", () => {
  it("finishes the in-flight frame without resetting its pipeline or accepting more frames", async () => {
    const { loop, video, cb } = setup();
    const frame = deferred<Record<string, never>>();
    const model = runtime(vi.fn(() => frame.promise));
    loop.start();
    loop.setModel(0, model, config);
    const reset = vi.spyOn(loop, "resetPipeline");
    const callback = video.requestVideoFrameCallback.mock.calls[0][0] as unknown as () => void;
    loop.refresh(); // A pending settings refresh must not start another frame after Finish.
    let finished = false;
    const finish = loop.finish().then(() => { finished = true; });
    callback(); // A browser callback already dispatched before cancellation is harmless.
    await Promise.resolve();
    expect(finished).toBe(false);
    expect(video.cancelVideoFrameCallback).toHaveBeenCalled();
    frame.resolve({});
    await finish;
    expect(cb.onStats).toHaveBeenCalledOnce();
    expect(reset).not.toHaveBeenCalled();
    loop.refresh();
    callback();
    window.dispatchEvent(new Event("resize"));
    await Promise.resolve();
    expect(model.run).toHaveBeenCalledOnce();
    expect(cb.onStats).toHaveBeenCalledOnce();
    expect(video.requestVideoFrameCallback).toHaveBeenCalledOnce();
  });

  it("resumes analysis on start after a finished source", async () => {
    const { loop, video, cb } = setup();
    const model = runtime();
    loop.start();
    loop.setModel(0, model, config);
    await loop.finish();
    loop.resetPipeline();
    video.currentTime = 0;
    loop.start();
    const callback = video.requestVideoFrameCallback.mock.calls.at(-1)![0] as unknown as () => void;
    callback();
    await vi.waitFor(() => expect(cb.onStats).toHaveBeenCalledTimes(2));
    expect(model.run).toHaveBeenCalledTimes(2);
  });

  it("keeps fatal-error handling until Finish has drained a pending inference", async () => {
    const { loop, cb } = setup();
    loop.start();
    loop.setModel(0, runtime(vi.fn(() => new Promise(() => {}))), config);
    const finish = loop.finish();
    const error = new WebAssembly.RuntimeError("memory access out of bounds");
    window.dispatchEvent(Object.assign(new Event("error"), { error }));
    await finish;
    expect(cb.onError).toHaveBeenCalledWith(error);
    expect(cb.onStats).not.toHaveBeenCalled();
  });

  it("analyzes the latest CPU frame immediately after a busy callback, without queuing old frames", async () => {
    const { loop, video } = setup();
    const first = deferred<Record<string, never>>();
    const run = vi.fn().mockImplementationOnce(() => first.promise).mockResolvedValue({});
    loop.start();
    loop.setModel(0, runtime(run), config);
    const callback = video.requestVideoFrameCallback.mock.calls[0][0] as unknown as () => void;
    video.currentTime = 1;
    callback();
    video.currentTime = 2;
    callback();
    expect(run).toHaveBeenCalledOnce();
    first.resolve({});
    await vi.waitFor(() => expect(run).toHaveBeenCalledTimes(2));
    await Promise.resolve();
    expect(run).toHaveBeenCalledTimes(2);
  });

  it.each(["paused", "stopped", "gpu"])("does not catch up a %s source", async (kind) => {
    const { loop, video } = setup();
    const frame = deferred<Record<string, never>>();
    const run = vi.fn(() => frame.promise);
    const model = runtime(run);
    if (kind === "gpu") model.backend = "webgpu";
    loop.start();
    loop.setModel(0, model, config);
    video.currentTime = 1;
    (video.requestVideoFrameCallback.mock.calls[0][0] as unknown as () => void)();
    if (kind === "paused") Object.assign(video, { paused: true });
    if (kind === "stopped") loop.stop();
    frame.resolve({});
    await vi.waitFor(() => expect(run).toHaveResolved());
    await Promise.resolve();
    expect(run).toHaveBeenCalledOnce();
  });

  it("refreshes a paused frame after a window resize and removes the listener on stop", async () => {
    const { loop } = setup();
    const model = runtime();
    loop.start();
    loop.setModel(0, model, config);
    await vi.waitFor(() => expect(model.run).toHaveBeenCalledTimes(1));
    window.dispatchEvent(new Event("resize"));
    await vi.waitFor(() => expect(model.run).toHaveBeenCalledTimes(2));
    loop.stop();
    window.dispatchEvent(new Event("resize"));
    await Promise.resolve();
    expect(model.run).toHaveBeenCalledTimes(2);
  });

  it("drains the detached frame before creating a session and blocks refresh until setup finishes", async () => {
    const { loop } = setup();
    const frame = deferred<Record<string, never>>();
    const setupDone = deferred<void>();
    const next = runtime();
    loop.setModel(0, runtime(vi.fn(() => frame.promise)), config);
    loop.setModel(0, null, null);
    const create = vi.fn(async () => { loop.setModel(1, next, config); await setupDone.promise; });
    const change = loop.changeModels(create);
    loop.refresh();
    await Promise.resolve();
    expect(create).not.toHaveBeenCalled();
    frame.resolve({});
    await vi.waitFor(() => expect(create).toHaveBeenCalledOnce());
    loop.refresh();
    expect(next.run).not.toHaveBeenCalled();
    setupDone.resolve();
    await change;
    await vi.waitFor(() => expect(next.run).toHaveBeenCalledOnce());
  });

  it("serializes simultaneous model loads and resumes only after both finish", async () => {
    const { loop } = setup();
    const firstDone = deferred<void>();
    const secondDone = deferred<void>();
    const first = runtime(), second = runtime();
    const a = loop.changeModels(async () => { loop.setModel(0, first, config); await firstDone.promise; });
    const createSecond = vi.fn(async () => { loop.setModel(1, second, config); await secondDone.promise; });
    const b = loop.changeModels(createSecond);
    await Promise.resolve();
    expect(createSecond).not.toHaveBeenCalled();
    firstDone.resolve();
    await a;
    await vi.waitFor(() => expect(createSecond).toHaveBeenCalledOnce());
    expect(first.run).not.toHaveBeenCalled();
    expect(second.run).not.toHaveBeenCalled();
    secondDone.resolve();
    await b;
    await vi.waitFor(() => expect(second.run).toHaveBeenCalledOnce());
    expect(first.run).toHaveBeenCalledOnce();
  });

  it("releases the pause and queue after a failed load", async () => {
    const { loop, cb } = setup();
    const model = runtime();
    loop.setModel(0, model, config);
    await vi.waitFor(() => expect(cb.onStats).toHaveBeenCalledOnce());
    await expect(loop.changeModels(async () => { throw new Error("load failed"); })).rejects.toThrow("load failed");
    await vi.waitFor(() => expect(model.run).toHaveBeenCalledTimes(2));
    await expect(loop.changeModels(async () => "next load")).resolves.toBe("next load");
  });

  it("reports a fatal runtime rejection even after its model was detached", async () => {
    const { loop, cb, video } = setup();
    const frame = deferred<Record<string, never>>();
    loop.setModel(0, runtime(vi.fn(() => frame.promise)), config);
    loop.setModel(0, null, null);
    const error = new WebAssembly.RuntimeError("memory access out of bounds");
    frame.reject(error);
    await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledWith(error));
    expect(video.pause).toHaveBeenCalledOnce();
  });

  it.each(["error", "unhandledrejection"])("settles a pending frame and reports an uncaught WASM %s", async (type) => {
    const { loop, cb } = setup();
    loop.start();
    loop.setModel(0, runtime(vi.fn(() => new Promise(() => {}))), config);
    const error = new WebAssembly.RuntimeError("memory access out of bounds");
    window.dispatchEvent(Object.assign(new Event(type), type === "error" ? { error } : { reason: error }));
    await vi.waitFor(() => expect(cb.onError).toHaveBeenCalledWith(error));
    loop.setModel(0, runtime(), config);
    await vi.waitFor(() => expect(cb.onStats).toHaveBeenCalledOnce());
  });

  it("does not treat unrelated global errors as inference failures", async () => {
    const { loop, cb } = setup();
    const frame = deferred<Record<string, never>>();
    loop.start();
    loop.setModel(0, runtime(vi.fn(() => frame.promise)), config);
    window.dispatchEvent(Object.assign(new Event("error"), { error: new Error("unrelated script") }));
    frame.resolve({});
    await vi.waitFor(() => expect(cb.onStats).toHaveBeenCalledOnce());
    expect(cb.onError).not.toHaveBeenCalled();
  });
});

describe("analyze first", () => {
  /** A paused video of `duration` s whose seeks finish on the next tick, as a browser's do. */
  function seekableVideo(duration: number) {
    const target = new EventTarget();
    let time = 0;
    const video = Object.assign(target, {
      readyState: 2, duration, paused: true, videoWidth: 4, videoHeight: 2, pause: vi.fn(),
      requestVideoFrameCallback: vi.fn(() => 1), cancelVideoFrameCallback: vi.fn(), currentTime: 0,
    });
    Object.defineProperty(video, "currentTime", {
      get: () => time,
      set: (t: number) => { time = t; setTimeout(() => target.dispatchEvent(new Event("seeked")), 0); },
    });
    return video;
  }
  const classes = Uint16Array.from([0, 0, 0, 1, 1, 0, 0, 2]);
  const ropePipeline = () => ({
    run: vi.fn((_: unknown, t: number) => ({ metrics: { t }, segmentation: { width: 4, height: 2, classes, rect: { x: 0, y: 0, width: 4, height: 2 }, background: 0 } })),
    feeds: () => undefined, reset: vi.fn(), markDropped: vi.fn(), summary: () => ({}),
  });

  async function analyzeSetup(duration: number) {
    const video = seekableVideo(duration);
    const cb = { onStats: vi.fn(), onError: vi.fn() };
    const loop = new FrameLoop(video as unknown as HTMLVideoElement, {} as HTMLCanvasElement, cb);
    loops.push(loop);
    const model = runtime();
    loop.setModel(0, model, config);
    const pipeline = ropePipeline();
    loop.setModelPipeline(0, pipeline as never);
    const draw = vi.fn();
    (loop as unknown as { renderer: { draw: typeof draw } }).renderer.draw = draw;
    // setting the model and pipeline refreshes the paused frame once each; analysis is counted from here
    await new Promise((resolve) => setTimeout(resolve, 0));
    model.run.mockClear();
    pipeline.run.mockClear();
    draw.mockClear();
    cb.onStats.mockClear();
    return { loop, video, cb, model, pipeline, draw };
  }

  it("runs every frame in order once, then plays the stored results without running the model", async () => {
    const { loop, video, cb, model, pipeline, draw } = await analyzeSetup(0.1);
    const progress = vi.fn();
    const report = await loop.analyze(30, progress, new AbortController().signal, "seek");
    expect(report).toMatchObject({ fps: 30, frames: 4 });
    expect(Object.keys(report!.perFrameMs)).toEqual(["nextFrame", "prepare", "model", "measure"]);
    expect(model.run).toHaveBeenCalledTimes(4); // 0, 1/30, 2/30, 3/30 s
    expect(pipeline.run.mock.calls.map((c) => c[1])).toEqual([0, 1 / 30, 2 / 30, 3 / 30]);
    expect(progress).toHaveBeenLastCalledWith(4, 4);
    expect(loop.analyzed).toBe(true);
    expect(draw).not.toHaveBeenCalled();

    loop.start();
    video.currentTime = 0.07; // between the frames at 2/30 and 3/30 s: the nearest one is shown
    (video.requestVideoFrameCallback.mock.calls.at(-1)![0] as unknown as () => void)();
    expect(model.run).toHaveBeenCalledTimes(4);
    expect(draw).toHaveBeenCalledOnce();
    const shown = draw.mock.calls[0][0];
    expect(shown.metrics).toEqual({ t: 2 / 30 });
    expect([...shown.segmentation.classes]).toEqual([...classes]);
    expect(cb.onStats).toHaveBeenCalledOnce();
  });

  it("analyzes fewer frames at a lower rate and shows the nearest one", async () => {
    const { loop, video, model, draw } = await analyzeSetup(0.5);
    const report = await loop.analyze(10, () => {}, new AbortController().signal, "seek");
    expect(report).toMatchObject({ fps: 10, frames: 6 }); // 0, 0.1 ... 0.5 s
    expect(model.run).toHaveBeenCalledTimes(6);
    loop.start();
    video.currentTime = 0.26; // nearest analyzed frame: 0.3 s
    (video.requestVideoFrameCallback.mock.calls.at(-1)![0] as unknown as () => void)();
    expect(draw.mock.calls[0][0].metrics.t).toBeCloseTo(0.3);
  });

  it("plays to each wanted time instead of seeking", async () => {
    const { loop, video, model, pipeline } = await analyzeSetup(0.3);
    // A 30 fps video playing at PLAY_RATE: each callback shows a frame 4/30 s later (frames dropped at speed).
    let playing = false;
    const seeks: number[] = [];
    const callbacks: ((now: number, meta: { mediaTime: number }) => void)[] = [];
    let time = 0;
    Object.defineProperty(video, "currentTime", { get: () => time, set: (t: number) => { seeks.push(t); time = t; setTimeout(() => video.dispatchEvent(new Event("seeked")), 0); } });
    video.requestVideoFrameCallback = vi.fn((cb) => { callbacks.push(cb); return callbacks.length; }) as never;
    Object.assign(video, {
      play: vi.fn(async () => {
        playing = true;
        while (playing) {
          await new Promise((resolve) => setTimeout(resolve, 0));
          if (!playing) break;
          time = Math.round((time + 4 / 30) * 30) / 30;
          if (time > 0.3) { time = 0.3; playing = false; video.dispatchEvent(new Event("ended")); break; }
          callbacks.splice(0).forEach((cb) => cb(0, { mediaTime: time }));
        }
      }),
      pause: vi.fn(() => { playing = false; }),
    });
    const report = await loop.analyze(10, () => {}, new AbortController().signal, "play");
    expect(seeks).toEqual([0]); // only the first frame is a seek
    // wanted 0, 0.1, 0.2, 0.3 s; shown frames are 4/30 s apart, so each is the first one at or after the wanted time:
    // 0, 4/30, 8/30, and the last frame (0.3 s) when the video ends
    expect(pipeline.run.mock.calls.map((c) => Math.round((c[1] as number) * 30))).toEqual([0, 4, 8, 9]);
    expect(report).toMatchObject({ stepping: "play" });
    expect(model.run).toHaveBeenCalledTimes(report!.frames);
  });

  it("seeks instead when playing shows no frame (a hidden page)", async () => {
    const { loop, video, model } = await analyzeSetup(0.2);
    Object.assign(video, { play: vi.fn(async () => {}), pause: vi.fn() }); // frame callbacks never fire
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      let report: Awaited<ReturnType<typeof loop.analyze>> | undefined;
      const result = loop.analyze(10, () => {}, new AbortController().signal, "play").then((r) => { report = r; });
      while (report === undefined) await vi.advanceTimersByTimeAsync(100);
      await result;
      expect(report).toMatchObject({ stepping: "seek", frames: 3 });
      expect(model.run).toHaveBeenCalledTimes(3);
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops when aborted and goes back to live analysis", async () => {
    const { loop, model } = await analyzeSetup(10);
    const abort = new AbortController();
    const result = loop.analyze(30, (done) => { if (done === 2) abort.abort(); }, abort.signal, "seek");
    expect(await result).toBeNull();
    expect(model.run).toHaveBeenCalledTimes(2);
    expect(loop.analyzed).toBe(false);
  });

  it("drops the stored results when a model setting changes", async () => {
    const { loop } = await analyzeSetup(0.05);
    await loop.analyze(30, () => {}, new AbortController().signal, "seek");
    expect(loop.analyzed).toBe(true);
    loop.setConfig(0, { ...config });
    expect(loop.analyzed).toBe(false);
  });
});
