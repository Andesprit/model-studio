import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_INPUT, type Tensor } from "@detect/core";
import { clearGpuVerdicts, createCheckedRuntime, gpuOutputsAgree, readGpuVerdict, syntheticInput, writeGpuVerdict } from "../src/adapters/gpuCheck";
import { createRuntime, type Backend, type ModelRuntime } from "../src/adapters/runtime";

vi.mock("../src/adapters/runtime", () => ({ createRuntime: vi.fn(), errorMessage: (e: Error) => e.message }));
vi.mock("../src/loaders", () => ({ inferConfig: () => ({ config: { input: { ...DEFAULT_INPUT, width: 2, height: 2 } } }) }));
const tensor = (data: number[], dims = [data.length]): Tensor => ({ dims, data: new Float32Array(data) });
const output = (data = [1, -3, 0, 5]) => ({ logits: tensor(data) });
const bytes = new Uint8Array(20);
const identity = "https://example.test/models/lamp.onnx";
let storage: Map<string, string>;

function runtime(backend: Backend, result = output()): ModelRuntime {
  return { backend, input: { name: "image", type: "float32", shape: [1, 3, 2, 2] }, outputNames: ["logits"],
    run: vi.fn(async () => result), release: vi.fn(async () => {}) };
}

beforeEach(() => {
  vi.clearAllMocks();
  storage = new Map();
  vi.stubGlobal("localStorage", { getItem: (k: string) => storage.get(k) ?? null, setItem: (k: string, v: string) => storage.set(k, v),
    removeItem: (k: string) => storage.delete(k), key: (i: number) => [...storage.keys()][i], get length() { return storage.size; } });
  vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => ({ info: { vendor: "intel", architecture: "gen-12lp" } }) } });
});
afterEach(() => vi.unstubAllGlobals());

describe("GPU output agreement", () => {
  it("accepts measured healthy lamp differences and tiny sign flips", () => {
    // First real frame, logits, from gpu-check/lamp.json (probe at f3726fc).
    const gpu = [-6.82421875, -6.1875, -10.328125, -9.6953125, -10.2265625, -10.875, -10.0078125, -10.671875];
    const cpu = [-6.8318572, -6.1442695, -10.3364038, -9.7060833, -10.1990423, -10.8623180, -10.0013180, -10.6798267];
    expect(gpuOutputsAgree(output(gpu), output(cpu))).toBe(true);
    expect(gpuOutputsAgree(output([1e-8, -1e-8]), output([0, 0]))).toBe(true);
  });

  it("rejects offsets even when correlation and sign agree, and checks every output", () => {
    expect(gpuOutputsAgree(output([6, 7, 8]), output([1, 2, 3]))).toBe(false);
    expect(gpuOutputsAgree({ ...output(), other: tensor([0]) }, { ...output(), other: tensor([1]) })).toBe(false);
    expect(gpuOutputsAgree(output([0, 0, 0, 0]), output())).toBe(false);
  });

  it("rejects missing, extra, reshaped, truncated and nonfinite outputs", () => {
    expect(gpuOutputsAgree({}, {})).toBe(false);
    expect(gpuOutputsAgree(output(), {})).toBe(false);
    expect(gpuOutputsAgree({ ...output(), extra: tensor([1]) }, output())).toBe(false);
    expect(gpuOutputsAgree({ logits: tensor([1, 2], [1, 2]) }, { logits: tensor([1, 2], [2, 1]) })).toBe(false);
    expect(gpuOutputsAgree({ logits: tensor([1], [2]) }, { logits: tensor([1, 2], [2]) })).toBe(false);
    for (const n of [NaN, Infinity, -Infinity]) {
      expect(gpuOutputsAgree(output([n]), output([1]))).toBe(false);
      expect(gpuOutputsAgree(output([1]), output([n]))).toBe(false);
    }
  });

  it("generates the same synthetic image with the actual input normalization and layout", () => {
    const cfg = { ...DEFAULT_INPUT, width: 2, height: 2, channels: 1 as const };
    const a = syntheticInput(cfg);
    expect(a).toEqual(syntheticInput(cfg));
    expect(a.dims).toEqual([1, 1, 2, 2]);
    expect(a.data[0]).toBe(0);
    expect(a.data[1]).toBeCloseTo((.299 * 3 + .114) / 255);
  });
});

describe("verdict storage", () => {
  it("persists only recognized verdicts and clears only this feature's entries", () => {
    expect(readGpuVerdict("key")).toBe(null);
    writeGpuVerdict("key", "ok");
    expect(readGpuVerdict("key")).toBe("ok");
    writeGpuVerdict("key", "cpu");
    expect(readGpuVerdict("key")).toBe("cpu");
    storage.set("model-studio.gpu-check.v1:other", "invalid");
    expect(readGpuVerdict("other")).toBe(null);
    storage.set("unrelated", "keep");
    clearGpuVerdicts();
    expect([...storage]).toEqual([["unrelated", "keep"]]);
  });

  it("survives blocked storage access", () => {
    vi.stubGlobal("localStorage", { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); }, get length() { throw new Error("blocked"); } });
    expect(readGpuVerdict("key")).toBe(null);
    expect(() => writeGpuVerdict("key", "ok")).not.toThrow();
    expect(() => clearGpuVerdicts()).not.toThrow();
  });
});

describe("checked runtime lifecycle (called inside changeModels)", () => {
  it("keeps a healthy GPU, releases WASM and skips validation on the next load", async () => {
    const gpu = runtime("webgpu"), cpu = runtime("wasm");
    vi.mocked(createRuntime).mockResolvedValueOnce(gpu).mockResolvedValueOnce(cpu).mockResolvedValueOnce(gpu);
    const first = await createCheckedRuntime(bytes, "auto", identity);
    expect(first).toMatchObject({ runtime: gpu, verification: { status: "verified" } });
    expect(cpu.run).toHaveBeenCalledWith(vi.mocked(gpu.run).mock.calls[0][0]);
    expect(cpu.release).toHaveBeenCalledOnce();
    expect(gpu.release).not.toHaveBeenCalled();
    expect(await createCheckedRuntime(bytes, "auto", identity)).toMatchObject({ verification: { status: "verified" } });
    expect(createRuntime).toHaveBeenCalledTimes(3);
    expect(gpu.run).toHaveBeenCalledOnce();
  });

  it("falls back on disagreement and loads WASM directly after reload", async () => {
    const gpu = runtime("webgpu", output([6, 7, 8, 9])), cpu = runtime("wasm");
    vi.mocked(createRuntime).mockResolvedValueOnce(gpu).mockResolvedValueOnce(cpu).mockResolvedValueOnce(cpu);
    expect(await createCheckedRuntime(bytes, "auto", identity)).toMatchObject({ runtime: cpu, verification: { status: "fallback", notice: expect.stringContaining("CPU") } });
    expect(gpu.release).toHaveBeenCalledOnce();
    expect(cpu.release).not.toHaveBeenCalled();
    await createCheckedRuntime(bytes, "auto", identity);
    expect(createRuntime).toHaveBeenLastCalledWith(bytes, "wasm");
    expect(createRuntime).toHaveBeenCalledTimes(3);
  });

  it("honors forced WebGPU with a warning, and forced WASM without checking", async () => {
    const gpu = runtime("webgpu", output([6, 7, 8, 9])), cpu = runtime("wasm");
    vi.mocked(createRuntime).mockResolvedValueOnce(gpu).mockResolvedValueOnce(cpu).mockResolvedValueOnce(cpu);
    expect(await createCheckedRuntime(bytes, "webgpu", identity)).toMatchObject({ runtime: gpu, verification: { status: "mismatch", notice: expect.stringContaining("WebGPU") } });
    expect(cpu.release).toHaveBeenCalledOnce();
    expect(gpu.release).not.toHaveBeenCalled();
    expect(await createCheckedRuntime(bytes, "wasm", identity)).toMatchObject({ runtime: cpu, verification: { status: "not-checked" } });
    expect(createRuntime).toHaveBeenCalledTimes(3);
  });

  it("does not check large models or an automatic WASM backend", async () => {
    vi.mocked(createRuntime).mockResolvedValueOnce(runtime("webgpu")).mockResolvedValueOnce(runtime("wasm"));
    expect(await createCheckedRuntime(new Uint8Array(40_000_000), "auto", identity)).toMatchObject({ verification: { status: "not-checked" } });
    expect(await createCheckedRuntime(bytes, "auto", identity)).toMatchObject({ verification: { status: "not-checked" } });
    expect(createRuntime).toHaveBeenCalledTimes(2);
  });

  it("separates adapter/model identities and clears verdicts for a recheck", async () => {
    vi.mocked(createRuntime).mockImplementation(async (_b, pref) => runtime(pref === "wasm" ? "wasm" : "webgpu"));
    await createCheckedRuntime(bytes, "auto", identity);
    await createCheckedRuntime(bytes, "auto", identity + "-other");
    vi.stubGlobal("navigator", { gpu: { requestAdapter: async () => ({ info: { vendor: "other", architecture: "gpu" } }) } });
    await createCheckedRuntime(bytes, "auto", identity);
    clearGpuVerdicts();
    await createCheckedRuntime(bytes, "auto", identity);
    expect(createRuntime).toHaveBeenCalledTimes(8);
  });

  it("releases the test session and reports unverified results when the check cannot run", async () => {
    const gpu = runtime("webgpu"), cpu = runtime("wasm");
    vi.mocked(cpu.run).mockRejectedValue(new Error("unsupported test"));
    vi.mocked(createRuntime).mockResolvedValueOnce(gpu).mockResolvedValueOnce(cpu);
    expect(await createCheckedRuntime(bytes, "auto", identity)).toMatchObject({ runtime: gpu, verification: { status: "not-checked", notice: expect.stringContaining("unverified") } });
    expect(cpu.release).toHaveBeenCalledOnce();
    expect(storage.size).toBe(0);
  });
});
