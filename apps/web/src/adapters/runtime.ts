import * as ort from "onnxruntime-web/webgpu";
import type { Tensor } from "@detect/core";

export type BackendPreference = "auto" | "webgpu" | "wasm";
export type Backend = "webgpu" | "wasm";

export interface InputInfo {
  name: string;
  /** Static dims if known, symbolic/dynamic dims as null. */
  shape: (number | null)[] | null;
  type: string;
}

/** Thin wrapper over an onnxruntime-web session that speaks core's runtime-agnostic Tensor. */
export interface ModelRuntime {
  backend: Backend;
  input: InputInfo;
  outputNames: readonly string[];
  /**
   * Runs the model on the image tensor. `extra` holds values for the model's other inputs, by name; any other
   * input missing from it gets zeros of its declared shape (unknown dims taken as 1).
   */
  run(input: Tensor, extra?: Record<string, Tensor>): Promise<Record<string, Tensor>>;
  release(): Promise<void>;
}

export const webgpuSupported = () => typeof navigator !== "undefined" && "gpu" in navigator;

export async function createRuntime(model: Uint8Array, preference: BackendPreference): Promise<ModelRuntime> {
  const order: Backend[] =
    preference === "wasm" ? ["wasm"] : preference === "webgpu" ? ["webgpu"] : webgpuSupported() ? ["webgpu", "wasm"] : ["wasm"];

  let lastError: unknown;
  for (const backend of order) {
    try {
      // Creating one EP at a time (instead of passing a fallback list) tells us which one actually loaded.
      const session = await ort.InferenceSession.create(model, {
        executionProviders: [backend],
        graphOptimizationLevel: "all",
      });
      return wrap(session, backend);
    } catch (e) {
      lastError = e;
      console.warn(`[runtime] ${backend} failed, trying next backend`, e);
    }
  }
  throw new Error(`Could not load model: ${errorMessage(lastError)}`);
}

function wrap(session: ort.InferenceSession, backend: Backend): ModelRuntime {
  const metas = (session as unknown as { inputMetadata?: { name: string; type?: string; shape?: (number | string)[] }[] })
    .inputMetadata;
  const info = (name: string, i: number): InputInfo => {
    const meta = metas?.[i];
    return {
      name,
      shape: meta?.shape ? meta.shape.map((d) => (typeof d === "number" && d > 0 ? d : null)) : null,
      type: meta?.type ?? "float32",
    };
  };
  // The first input is the image; any others are set by the model's pipeline (zeros when it doesn't).
  const [input, ...others] = session.inputNames.map(info);
  const toOrt = (to: InputInfo, t: Tensor) =>
    to.type === "float16" ? new ort.Tensor("float16", toHalf(t.data), t.dims) : new ort.Tensor("float32", t.data, t.dims);

  return {
    backend,
    input,
    outputNames: session.outputNames,
    async run(t, extra = {}) {
      const feeds: Record<string, ort.Tensor> = { [input.name]: toOrt(input, t) };
      for (const o of others) {
        const dims = o.shape?.map((d) => d ?? 1) ?? [1];
        feeds[o.name] = toOrt(o, extra[o.name] ?? { dims, data: new Float32Array(dims.reduce((a, b) => a * b, 1)) });
      }
      const results = await session.run(feeds);
      for (const f of Object.values(feeds)) f.dispose();
      const out: Record<string, Tensor> = {};
      for (const name of session.outputNames) {
        const r = results[name];
        out[name] = { dims: [...r.dims], data: toFloat32(r) };
        r.dispose();
      }
      return out;
    },
    release: () => session.release(),
  };
}

/** Normalizes every numeric output type to Float32Array so decoders only handle one type. */
function toFloat32(t: ort.Tensor): Float32Array {
  const data = t.data;
  if (data instanceof Float32Array) return data;
  if (t.type === "float16") {
    if (data instanceof Uint16Array) return fromHalf(data);
    return Float32Array.from(data as ArrayLike<number>);
  }
  if (data instanceof BigInt64Array || data instanceof BigUint64Array) {
    return Float32Array.from(data as ArrayLike<bigint>, Number);
  }
  if (Array.isArray(data)) throw new Error(`Unsupported output type "${t.type}"`);
  return Float32Array.from(data as ArrayLike<number>);
}

function fromHalf(h: Uint16Array): Float32Array {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const v = h[i];
    const sign = v & 0x8000 ? -1 : 1;
    const exp = (v >> 10) & 0x1f;
    const frac = v & 0x3ff;
    out[i] =
      exp === 0 ? sign * 2 ** -14 * (frac / 1024)
      : exp === 31 ? (frac ? NaN : sign * Infinity)
      : sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return out;
}

function toHalf(f: Float32Array): Uint16Array {
  const out = new Uint16Array(f.length);
  const buf = new Float32Array(1);
  const bits = new Uint32Array(buf.buffer);
  for (let i = 0; i < f.length; i++) {
    buf[0] = f[i];
    const x = bits[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = ((x >>> 23) & 0xff) - 127 + 15;
    const mant = x & 0x7fffff;
    if (exp <= 0) out[i] = sign; // flush tiny values to zero; fine for normalized pixels
    else if (exp >= 31) out[i] = sign | 0x7c00;
    else out[i] = sign | (exp << 10) | (mant >> 13);
  }
  return out;
}

export const errorMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));
