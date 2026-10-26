import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelConfig } from "@detect/core";
import { FrameLoop } from "../src/adapters/frameLoop";
import type { ModelRuntime } from "../src/adapters/runtime";

/** The time the mock video shows: its input tensor carries it, so a test can tell frames apart. */
const shown = vi.hoisted(() => ({ time: 0 }));
vi.mock("@detect/core", () => ({
  toTensor: () => ({ dims: [1], data: Float32Array.of(shown.time) }),
  decodeAll: () => ({}),
  mergeResults: (results: unknown[]) => results[0] ?? {},
}));
/** What openDecodedFrames gives analyze(): null (can't decode: seek), or frames from a test. */
const decoding = vi.hoisted(() => ({ open: (() => Promise.resolve(null)) as (...args: unknown[]) => Promise<unknown> }));
vi.mock("../src/adapters/decodedFrames", () => ({ openDecodedFrames: (...args: unknown[]) => decoding.open(...args) }));
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
      set: (t: number) => { time = t; shown.time = t; setTimeout(() => target.dispatchEvent(new Event("seeked")), 0); },
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
    const report = await loop.analyze(30, progress, new AbortController().signal);
    expect(report).toMatchObject({ fps: 30, frames: 4 });
    expect(Object.keys(report!.perFrameMs)).toEqual(["frame", "prepare", "model", "measure"]);
    expect(report!.reading).toBe("seek"); // no file given: seeks the video
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
    const report = await loop.analyze(10, () => {}, new AbortController().signal);
    expect(report).toMatchObject({ fps: 10, frames: 6 }); // 0, 0.1 ... 0.5 s
    expect(model.run).toHaveBeenCalledTimes(6);
    loop.start();
    video.currentTime = 0.26; // nearest analyzed frame: 0.3 s
    (video.requestVideoFrameCallback.mock.calls.at(-1)![0] as unknown as () => void)();
    expect(draw.mock.calls[0][0].metrics.t).toBeCloseTo(0.3);
  });

  it("gets the next frame ready while the model runs, with the same results as one frame at a time", async () => {
    const run = async (overlap: boolean) => {
      const { loop, video, model, pipeline } = await analyzeSetup(0.1); // frames at 0, 1/30, 2/30, 3/30 s
      const seenDuringRun: number[] = [];
      model.run.mockImplementation(async () => {
        await new Promise((resolve) => setTimeout(resolve, 5)); // the model takes a while; a seek takes one tick
        seenDuringRun.push(video.currentTime);
        return {};
      });
      const report = await loop.analyze(30, () => {}, new AbortController().signal, overlap);
      return { report, seenDuringRun, times: pipeline.run.mock.calls.map((c) => c[1]) };
    };
    const one = await run(false);
    const both = await run(true);
    expect(both.times).toEqual(one.times);
    expect(both.report).toMatchObject({ frames: one.report!.frames });
    // one at a time, the video still shows the frame being analyzed; overlapped, it has moved on to the next one
    expect(one.seenDuringRun).toEqual(one.times);
    expect(both.seenDuringRun.slice(0, -1)).toEqual(one.times.slice(1));
  });

  it("runs the model on several frames per call on the GPU, with each frame's own feeds and output", async () => {
    const run = async (batchSize: number) => {
      const video = seekableVideo(0.25); // frames at 0 ... 7/30 s
      const loop = new FrameLoop(video as unknown as HTMLVideoElement, {} as HTMLCanvasElement, { onStats: vi.fn(), onError: vi.fn() });
      loops.push(loop);
      const calls: { n: number; mode: number }[] = [];
      const model: ModelRuntime = {
        backend: "webgpu", input: { name: "image", type: "float32", shape: [null] }, outputNames: ["y"], release: vi.fn(),
        // the output is the input, so each frame's output carries that frame's time
        run: vi.fn(async (t, extra) => {
          calls.push({ n: t.dims[0], mode: extra!.mode.data[0] });
          return { y: { dims: [t.dims[0]], data: Float32Array.from(t.data) } };
        }),
      };
      loop.setModel(0, model, config);
      // like the lamp: both branches (mode 0) until two frames are seen, then one branch (mode 1)
      let seen = 0;
      const got: { t: number; output: number; mode: number }[] = [];
      const pipeline = {
        feeds: () => ({ mode: { dims: [1], data: Float32Array.of(seen >= 2 ? 1 : 0) } }),
        run: vi.fn((inputs: { outputs: { y: { data: Float32Array } } }[], t: number) => {
          got.push({ t, output: inputs[0].outputs.y.data[0], mode: seen >= 2 ? 1 : 0 });
          seen++;
          return {};
        }),
        reset: vi.fn(), markDropped: vi.fn(), summary: () => ({}),
      };
      loop.setModelPipeline(0, pipeline as never);
      await new Promise((resolve) => setTimeout(resolve, 0));
      calls.length = 0;
      got.length = 0;
      seen = 0;
      const report = await loop.analyze(30, () => {}, new AbortController().signal, true, batchSize);
      return { report, calls, got };
    };
    const one = await run(1);
    const four = await run(4);
    expect(four.report).toMatchObject({ batch: 4, frames: 8 });
    expect(one.report).toMatchObject({ batch: 1, frames: 8 });
    // every frame's pipeline got that frame's output, as one at a time
    expect(four.got).toEqual(one.got);
    for (const g of four.got) expect(g.output).toBeCloseTo(g.t);
    // frames 0-3 in one call with mode 0; frames 2 and 3 then want mode 1, so they run alone; frames 4-7 in one call
    expect(four.calls).toEqual([{ n: 4, mode: 0 }, { n: 1, mode: 1 }, { n: 1, mode: 1 }, { n: 4, mode: 1 }]);
  });

  describe("frames decoded in a worker", () => {
    afterEach(() => { decoding.open = () => Promise.resolve(null); });
    /** Decoded frames as the worker sends them: each carries its own time in its tensor. */
    const decodedFrames = (times: number[], failAt = -1) => {
      let i = 0;
      return {
        next: vi.fn(async () => {
          if (i === failAt) throw new Error("decoder broke");
          const t = times[i];
          return { index: i++, decodeMs: 1, preMs: 2, inputs: [{ tensor: { dims: [1], data: Float32Array.of(t) }, transform: { frameWidth: 1, frameHeight: 1 }, turn: 0 }] };
        }),
        close: vi.fn(),
      };
    };

    it("uses the decoded frames instead of seeking, at the same times", async () => {
      const { loop, video, model, pipeline } = await analyzeSetup(0.3);
      let frames!: ReturnType<typeof decodedFrames>;
      let asked: number[] = [];
      decoding.open = async (_file, times) => { asked = times as number[]; frames = decodedFrames(asked); return frames; };
      const seeks: number[] = [];
      Object.defineProperty(video, "currentTime", { get: () => 0, set: (t: number) => { seeks.push(t); setTimeout(() => video.dispatchEvent(new Event("seeked")), 0); } });
      const report = await loop.analyze(10, () => {}, new AbortController().signal, true, 1, new Blob());
      expect(asked).toEqual([0, 0.1, 0.2, 0.3]);
      expect(report).toMatchObject({ reading: "decoded", frames: 4 });
      expect(seeks).toEqual([]);
      expect(pipeline.run.mock.calls.map((c) => c[1])).toEqual(asked);
      // each frame's own picture went to the model (float32 in the tensor)
      expect(model.run.mock.calls.map((c) => Math.round((c[0] as { data: Float32Array }).data[0] * 1000) / 1000)).toEqual(asked);
      expect(frames.close).toHaveBeenCalled();
    });

    it("seeks when the frames can't be decoded, and starts over seeking when decoding fails midway", async () => {
      const { loop, pipeline } = await analyzeSetup(0.3);
      expect(await loop.analyze(10, () => {}, new AbortController().signal, true, 1, new Blob())).toMatchObject({ reading: "seek", frames: 4 });
      pipeline.run.mockClear();
      let frames!: ReturnType<typeof decodedFrames>;
      decoding.open = async (_file, times) => (frames = decodedFrames(times as number[], 2));
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const report = await loop.analyze(10, () => {}, new AbortController().signal, true, 1, new Blob());
      expect(report).toMatchObject({ reading: "seek", frames: 4 });
      expect(warn).toHaveBeenCalledOnce();
      expect(frames.close).toHaveBeenCalled();
      // two frames from the decoder, then the whole video again from the start, seeking
      expect(pipeline.run.mock.calls.map((c) => c[1])).toEqual([0, 0.1, 0, 0.1, 0.2, 0.3]);
      warn.mockRestore();
    });
  });

  it("starts without a live frame's model call running alongside it", async () => {
    const video = seekableVideo(0.1);
    const loop = new FrameLoop(video as unknown as HTMLVideoElement, {} as HTMLCanvasElement, { onStats: vi.fn(), onError: vi.fn() });
    loops.push(loop);
    let busy = false;
    let overlapped = false;
    const model = runtime(vi.fn(async () => {
      if (busy) overlapped = true; // a session can't run two calls at once
      busy = true;
      await new Promise((resolve) => setTimeout(resolve, 5));
      busy = false;
      return {};
    }));
    loop.setModel(0, model, config); // a live frame starts on the paused video
    loop.setModelPipeline(0, ropePipeline() as never); // and queues one more refresh
    const report = await loop.analyze(30, () => {}, new AbortController().signal, true, 1);
    expect(report).toMatchObject({ frames: 4 });
    expect(overlapped).toBe(false);
  });

  it("stops when aborted and goes back to live analysis", async () => {
    const { loop, model } = await analyzeSetup(10);
    const abort = new AbortController();
    const result = loop.analyze(30, (done) => { if (done === 2) abort.abort(); }, abort.signal);
    expect(await result).toBeNull();
    expect(model.run).toHaveBeenCalledTimes(2);
    expect(loop.analyzed).toBe(false);
  });

  it("drops the stored results when a model setting changes", async () => {
    const { loop } = await analyzeSetup(0.05);
    await loop.analyze(30, () => {}, new AbortController().signal);
    expect(loop.analyzed).toBe(true);
    loop.setConfig(0, { ...config });
    expect(loop.analyzed).toBe(false);
  });
});
