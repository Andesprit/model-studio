import { toTensor, type InputConfig, type Tensor } from "@detect/core";
import { inferConfig } from "../loaders";
import { createRuntime, errorMessage, type BackendPreference, type ModelRuntime } from "./runtime";
import { createWorkerRuntime } from "./workerRuntime";
import { createSessionRuntime } from "./ortSession";

// Local output comparisons: healthy scaled RMS <=0.0129, simulated faults >=0.2225.
// Calibrated on one Intel GPU; this is a gross-error check, not proof of GPU correctness.
const GPU_ERROR_THRESHOLD = 0.05;
// Outputs are clamped to +-OUTPUT_CLAMP (beyond it a logit or score is certain either way), and the RMS leaves out the
// largest TRIM_FRACTION of differences. A segmentation's border cells can round to different sides on a healthy GPU;
// with a sentinel value (the lamp model marks cells outside the water with -10000) each such cell is a huge error,
// and a handful of them judged a healthy GPU faulty. Real faults corrupt most of the output, so they still fail.
// Outputs under 1 / TRIM_FRACTION values are compared whole.
const OUTPUT_CLAMP = 30;
const TRIM_FRACTION = 0.01;
const MAX_CHECK_BYTES = 3_000_000;
const STORAGE_PREFIX = "model-studio.gpu-check.v1:";
type Verdict = "ok" | "cpu";

export interface GpuVerification {
  status: "verified" | "fallback" | "mismatch" | "not-checked";
  label: string;
  notice?: string;
}

export function gpuOutputsAgree(actual: Record<string, Tensor>, reference: Record<string, Tensor>): boolean {
  const names = Object.keys(reference);
  if (!names.length || Object.keys(actual).length !== names.length) return false;
  return names.every((name) => {
    const a = actual[name], b = reference[name];
    if (!a || a.dims.length !== b.dims.length || a.dims.some((d, i) => d !== b.dims[i])) return false;
    if (!b.data.length || a.data.length !== b.data.length || b.dims.reduce((n, d) => n * d, 1) !== b.data.length) return false;
    const squared = new Float64Array(b.data.length);
    for (let i = 0; i < b.data.length; i++) {
      if (!Number.isFinite(a.data[i]) || !Number.isFinite(b.data[i])) return false;
      const ai = Math.max(-OUTPUT_CLAMP, Math.min(OUTPUT_CLAMP, a.data[i]));
      const bi = Math.max(-OUTPUT_CLAMP, Math.min(OUTPUT_CLAMP, b.data[i]));
      const delta = (ai - bi) / Math.max(1, Math.abs(bi));
      squared[i] = delta * delta;
    }
    squared.sort();
    const kept = squared.length - Math.floor(squared.length * TRIM_FRACTION);
    let sum = 0;
    for (let i = 0; i < kept; i++) sum += squared[i];
    return Math.sqrt(sum / kept) <= GPU_ERROR_THRESHOLD;
  });
}

/** The probe's repeatable image, using each model's real normalization and layout. */
export function syntheticInput(input: InputConfig): Tensor {
  const { width, height } = input;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    data[i] = (x * 3 + y * 5) % 256;
    data[i + 1] = (Math.floor(x / 17) * 29 + y) % 256;
    data[i + 2] = (Math.floor(y / 13) * 31 + x) % 256;
    data[i + 3] = 255;
  }
  return toTensor({ width, height, data }, input);
}

export function readGpuVerdict(key: string | null): Verdict | null {
  try {
    const value = key === null ? null : localStorage.getItem(STORAGE_PREFIX + key);
    return value === "ok" || value === "cpu" ? value : null;
  } catch {
    return null; // Private browsing or site settings can block storage.
  }
}

export function writeGpuVerdict(key: string | null, verdict: Verdict) {
  try {
    if (key !== null) localStorage.setItem(STORAGE_PREFIX + key, verdict);
  } catch { /* The current session still uses the verdict when storage is blocked. */ }
}

export function clearGpuVerdicts() {
  try {
    const keys = Array.from({ length: localStorage.length }, (_, i) => localStorage.key(i));
    keys.forEach((key) => { if (key?.startsWith(STORAGE_PREFIX)) localStorage.removeItem(key); });
  } catch { /* Rechecking still works for the current load without storage. */ }
}

async function verdictKey(identity: string): Promise<string | null> {
  try {
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ info: { vendor: string; architecture: string } } | null> } }).gpu;
    const adapter = await gpu?.requestAdapter();
    return adapter ? JSON.stringify([adapter.info.vendor, adapter.info.architecture, identity]) : null;
  } catch {
    return null;
  }
}

function verification(status: GpuVerification["status"]): GpuVerification {
  switch (status) {
    case "verified": return { status, label: "GPU verified" };
    case "fallback": return { status, label: "GPU mismatch; using CPU", notice: "The GPU gave different results. Using the CPU for accurate analysis." };
    case "mismatch": return { status, label: "GPU mismatch; WebGPU forced", notice: "The GPU gave different results. WebGPU is selected; choose Auto or WASM in Advanced to use the CPU." };
    case "not-checked": return { status, label: "GPU not checked" };
  }
}

/** Call inside FrameLoop.changeModels: session creation, checks and releases share ORT state. */
export async function createCheckedRuntime(bytes: Uint8Array, preference: BackendPreference, identity: string): Promise<{ runtime: ModelRuntime; verification: GpuVerification }> {
  const eligible = bytes.byteLength <= MAX_CHECK_BYTES && preference !== "wasm";
  const key = eligible ? await verdictKey(identity) : null;
  const saved = readGpuVerdict(key);
  const rt = await createRuntime(bytes, preference === "auto" && saved === "cpu" ? "wasm" : preference);
  if (rt.backend === "wasm") {
    return { runtime: rt, verification: saved === "cpu" && preference === "auto" ? verification("fallback")
      : { status: "not-checked", label: "CPU; GPU not checked" } };
  }
  if (!eligible) return { runtime: rt, verification: { status: "not-checked", label: "GPU not checked (large model)" } };
  if (saved) return { runtime: rt, verification: verification(saved === "ok" ? "verified" : "mismatch") };

  let cpu: ModelRuntime | undefined;
  let agrees: boolean;
  try {
    const input = syntheticInput(inferConfig(bytes, rt.input).config.input);
    const workerInput = { dims: [...input.dims], data: input.data.slice() };
    // runtime.run zero-fills every other input, including mode=0 (both lamp branches).
    const actual = rt.run(input);
    const reference = (async () => {
      try {
        cpu = await createWorkerRuntime(bytes);
        return await cpu.run(workerInput);
      } catch {
        await cpu?.release();
        cpu = undefined;
        // A worker has its own ORT instance. Page-side fallback must wait for the GPU run.
        await actual;
        cpu = await createSessionRuntime(bytes, "wasm");
        return await cpu.run(input);
      }
    })();
    // Drain both branches on errors before releasing sessions or leaving changeModels.
    const [gpuResult, cpuResult] = await Promise.allSettled([actual, reference]);
    if (gpuResult.status === "rejected") throw gpuResult.reason;
    if (cpuResult.status === "rejected") throw cpuResult.reason;
    agrees = gpuOutputsAgree(gpuResult.value, cpuResult.value);
  } catch (e) {
    await cpu?.release();
    return { runtime: rt, verification: { status: "not-checked", label: `GPU not checked: ${errorMessage(e)}`,
      notice: "The GPU check could not finish; results are unverified. Try WASM in Advanced if the analysis looks wrong." } };
  }
  writeGpuVerdict(key, agrees ? "ok" : "cpu");
  if (!agrees && preference === "auto") {
    await rt.release();
    return { runtime: cpu!, verification: verification("fallback") };
  }
  await cpu!.release();
  return { runtime: rt, verification: verification(agrees ? "verified" : "mismatch") };
}
