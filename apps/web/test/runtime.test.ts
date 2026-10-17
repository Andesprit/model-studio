import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createRuntime } from "../src/adapters/runtime";
import { createWorkerRuntime } from "../src/adapters/workerRuntime";
import { createSessionRuntime } from "../src/adapters/ortSession";

vi.mock("../src/adapters/workerRuntime", () => ({ createWorkerRuntime: vi.fn() }));
vi.mock("../src/adapters/ortSession", () => ({ createSessionRuntime: vi.fn() }));

const bytes = new Uint8Array([1]);
beforeEach(() => {
  vi.clearAllMocks();
  vi.stubGlobal("Worker", class {});
  vi.stubGlobal("navigator", { gpu: {} });
  vi.mocked(createWorkerRuntime).mockResolvedValue({ backend: "wasm" } as never);
  vi.mocked(createSessionRuntime).mockResolvedValue({ backend: "webgpu" } as never);
  vi.spyOn(console, "warn").mockImplementation(() => {});
});
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); });

it("uses a worker for explicit CPU and keeps WebGPU on the page", async () => {
  expect((await createRuntime(bytes, "wasm")).backend).toBe("wasm");
  expect(createWorkerRuntime).toHaveBeenCalledWith(bytes);
  expect(createSessionRuntime).not.toHaveBeenCalled();
  await createRuntime(bytes, "auto");
  expect(createSessionRuntime).toHaveBeenCalledWith(bytes, "webgpu");
  expect(createWorkerRuntime).toHaveBeenCalledOnce();
});

it("falls back to a page session when workers are absent or fail to load", async () => {
  vi.stubGlobal("Worker", undefined);
  await createRuntime(bytes, "wasm");
  expect(createWorkerRuntime).not.toHaveBeenCalled();
  expect(createSessionRuntime).toHaveBeenCalledWith(bytes, "wasm");
  vi.stubGlobal("Worker", class {});
  vi.mocked(createWorkerRuntime).mockRejectedValueOnce(new Error("module workers unavailable"));
  await createRuntime(bytes, "wasm");
  expect(createSessionRuntime).toHaveBeenCalledTimes(2);
});

it("uses the CPU worker if Auto cannot create a GPU session", async () => {
  vi.mocked(createSessionRuntime).mockRejectedValueOnce(new Error("no adapter"));
  expect((await createRuntime(bytes, "auto")).backend).toBe("wasm");
  expect(createWorkerRuntime).toHaveBeenCalledOnce();
});
