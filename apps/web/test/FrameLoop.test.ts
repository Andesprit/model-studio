import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ModelConfig } from "@detect/core";
import { FrameLoop } from "../src/adapters/frameLoop";
import type { ModelRuntime } from "../src/adapters/runtime";

vi.mock("@detect/core", () => ({
  toTensor: () => ({ dims: [1], data: new Float32Array(1) }),
  decodeAll: () => ({}),
  mergeResults: () => ({}),
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
