import { createSessionRuntime } from "./ortSession";
import { createWorkerRuntime } from "./workerRuntime";
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
   * The CPU worker takes ownership of the image buffer. Extra pipeline feeds remain reusable.
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
      if (backend === "wasm" && typeof Worker !== "undefined") {
        try {
          return await createWorkerRuntime(model);
        } catch (e) {
          console.warn("[runtime] CPU worker unavailable, using the main thread", e);
        }
      }
      return await createSessionRuntime(model, backend);
    } catch (e) {
      lastError = e;
      console.warn(`[runtime] ${backend} failed, trying next backend`, e);
    }
  }
  throw new Error(`Could not load model: ${errorMessage(lastError)}`);
}

export const errorMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));
