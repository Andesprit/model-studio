import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createSessionRuntime } from "../src/adapters/ortSession";
import type { ModelRuntime } from "../src/adapters/runtime";
import type { WorkerRequest } from "../src/adapters/workerRuntime";

vi.mock("../src/adapters/ortSession", () => ({ createSessionRuntime: vi.fn() }));
class Scope extends EventTarget {
  onmessage: ((event: { data: WorkerRequest }) => Promise<void>) | null = null;
  postMessage = vi.fn();
  close = vi.fn();
}
let scope: Scope;
let runtime: ModelRuntime;
const bytes = new Uint8Array([1]);
beforeEach(async () => {
  vi.resetModules();
  scope = new Scope();
  vi.stubGlobal("self", scope);
  runtime = { backend: "wasm", input: { name: "image", shape: [1], type: "float32" }, outputNames: ["mask"],
    run: vi.fn(), release: vi.fn(async () => {}) };
  vi.mocked(createSessionRuntime).mockResolvedValue(runtime);
  await import("../src/adapters/ortWorker");
});
afterEach(() => { vi.unstubAllGlobals(); vi.clearAllMocks(); });
const send = (data: WorkerRequest) => scope.onmessage!({ data });

it("creates a CPU session, transfers output buffers once, and disposes on release", async () => {
  await send({ id: 0, type: "create", model: bytes });
  expect(createSessionRuntime).toHaveBeenCalledWith(bytes, "wasm");
  expect(scope.postMessage).toHaveBeenLastCalledWith({ id: 0, value: { input: runtime.input, outputNames: ["mask"] } });
  const tensor = { dims: [1], data: new Float32Array([3]) };
  const outputs = { mask: tensor, alias: tensor };
  vi.mocked(runtime.run).mockResolvedValue(outputs);
  await send({ id: 1, type: "run", input: tensor });
  expect(scope.postMessage).toHaveBeenLastCalledWith({ id: 1, value: outputs }, { transfer: [tensor.data.buffer] });
  await send({ id: 2, type: "release" });
  expect(runtime.release).toHaveBeenCalledOnce();
  expect(scope.postMessage).toHaveBeenLastCalledWith({ id: 2 });
});

it("reports ordinary run errors and treats synchronous WASM traps as fatal", async () => {
  await send({ id: 0, type: "create", model: bytes });
  vi.mocked(runtime.run).mockRejectedValueOnce(new Error("bad dimensions"));
  const request: WorkerRequest = { id: 1, type: "run", input: { dims: [1], data: new Float32Array(1) } };
  await send(request);
  expect(scope.postMessage).toHaveBeenLastCalledWith({ id: 1, error: "bad dimensions" });
  expect(scope.close).not.toHaveBeenCalled();
  vi.mocked(runtime.run).mockRejectedValueOnce(new WebAssembly.RuntimeError("memory access out of bounds"));
  await send(request);
  expect(scope.postMessage).toHaveBeenLastCalledWith({ fatal: "memory access out of bounds" });
  expect(scope.close).toHaveBeenCalledOnce();
});

it.each(["error", "unhandledrejection"])("reports an asynchronous %s even while run stays pending", async (type) => {
  await send({ id: 0, type: "create", model: bytes });
  vi.mocked(runtime.run).mockImplementation(() => new Promise(() => {}));
  void send({ id: 1, type: "run", input: { dims: [1], data: new Float32Array(1) } });
  const error = new WebAssembly.RuntimeError("asynchronous trap");
  scope.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), { error, reason: error }));
  expect(scope.postMessage).toHaveBeenLastCalledWith({ fatal: "asynchronous trap" });
  expect(scope.close).toHaveBeenCalledOnce();
});
