import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createWorkerRuntime } from "../src/adapters/workerRuntime";

class FakeWorker {
  static current: FakeWorker;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: { message: string; preventDefault(): void }) => void) | null = null;
  onmessageerror: (() => void) | null = null;
  postMessage = vi.fn();
  terminate = vi.fn();
  constructor() { FakeWorker.current = this; }
  reply(id: number, value: unknown) { this.onmessage?.({ data: { id, value } } as MessageEvent); }
}

beforeEach(() => vi.stubGlobal("Worker", FakeWorker));
afterEach(() => vi.unstubAllGlobals());

async function setup() {
  const bytes = new Uint8Array([1, 2, 3]);
  const pending = createWorkerRuntime(bytes);
  const worker = FakeWorker.current;
  const create = worker.postMessage.mock.calls[0][0];
  expect(create.model).not.toBe(bytes);
  expect(create.model).toEqual(bytes);
  worker.reply(create.id, { input: { name: "image", type: "float32", shape: [1] }, outputNames: ["mask"] });
  return { runtime: await pending, worker };
}

it("transfers the image without copying and preserves pipeline feed ownership", async () => {
  const { runtime, worker } = await setup();
  const input = { dims: [1], data: new Float32Array([2]) };
  const extra = { mode: { dims: [1], data: new Float32Array([1]) } };
  const run = runtime.run(input, extra);
  const [message, transfer] = worker.postMessage.mock.calls[1];
  expect(message.input).toBe(input);
  expect(message.extra).toBe(extra);
  expect(transfer).toEqual([input.data.buffer]);
  const output = { mask: { dims: [1], data: new Float32Array([3]) } };
  worker.reply(message.id, output);
  expect(await run).toBe(output);
  expect(worker.postMessage).toHaveBeenCalledTimes(2);
});

it("releases the session before terminating its worker", async () => {
  const { runtime, worker } = await setup();
  const release = runtime.release();
  const message = worker.postMessage.mock.calls[1][0];
  expect(message.type).toBe("release");
  expect(worker.terminate).not.toHaveBeenCalled();
  worker.reply(message.id, undefined);
  await release;
  expect(worker.terminate).toHaveBeenCalledOnce();
  await expect(runtime.run({ dims: [1], data: new Float32Array(1) })).rejects.toThrow("released");
  await runtime.release();
});

it("rejects every pending call on a worker crash and permits cleanup", async () => {
  const { runtime, worker } = await setup();
  const a = runtime.run({ dims: [1], data: new Float32Array(1) });
  const b = runtime.run({ dims: [1], data: new Float32Array(1) });
  const assertions = [expect(a).rejects.toThrow("memory access"), expect(b).rejects.toThrow("memory access")];
  worker.onerror?.({ message: "memory access out of bounds", preventDefault: vi.fn() });
  await Promise.all(assertions);
  expect(worker.terminate).toHaveBeenCalledOnce();
  await runtime.release();
});

it("rejects pending runs on a fatal asynchronous WASM rejection", async () => {
  const { runtime, worker } = await setup();
  const run = runtime.run({ dims: [1], data: new Float32Array(1) });
  const assertion = expect(run).rejects.toThrow("WASM trap");
  worker.onmessage?.({ data: { fatal: "WASM trap" } } as MessageEvent);
  await assertion;
  await expect(runtime.run({ dims: [1], data: new Float32Array(1) })).rejects.toThrow("WASM trap");
});

it("surfaces operation and structured-clone failures instead of leaving calls pending", async () => {
  const { runtime, worker } = await setup();
  const run = runtime.run({ dims: [1], data: new Float32Array(1) });
  const assertion = expect(run).rejects.toThrow("bad input");
  worker.onmessage?.({ data: { id: 1, error: "bad input" } } as MessageEvent);
  await assertion;
  worker.postMessage.mockImplementationOnce(() => { throw new Error("could not clone"); });
  await expect(runtime.run({ dims: [1], data: new Float32Array(1) })).rejects.toThrow("could not clone");
  const pending = runtime.run({ dims: [1], data: new Float32Array(1) });
  const rejection = expect(pending).rejects.toThrow("message");
  worker.onmessageerror?.();
  await rejection;
});
