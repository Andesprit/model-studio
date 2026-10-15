import {
  configFromMetadata,
  DEFAULT_INPUT,
  definePipeline,
  MANIFEST_METADATA_KEY,
  parseConfig,
  readOnnxMetadata,
  type ModelConfig,
  type PipelineDefinition,
} from "@detect/core";
import type { InputInfo } from "./adapters/runtime";

/** Loads a user pipeline (an ES module whose default export is a PipelineDefinition; old decoder plugins qualify). */
export async function loadPipeline(file: File): Promise<PipelineDefinition> {
  return loadPipelineSource(await file.text());
}

/** Loads a pipeline from its source code, e.g. one embedded in a model's metadata. */
export async function loadPipelineSource(source: string): Promise<PipelineDefinition> {
  const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  try {
    const mod = await import(/* @vite-ignore */ url);
    return definePipeline(mod.default);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function loadConfigFile(file: File): Promise<ModelConfig> {
  let json: unknown;
  try {
    json = JSON.parse(await file.text());
  } catch (e) {
    throw new Error(`Config is not valid JSON: ${(e as Error).message}`);
  }
  return parseConfig(json);
}

/**
 * Best-effort config for a model with no sidecar JSON:
 * Ultralytics metadata if present, then input shape from the ONNX graph, then defaults.
 * `guessed` is true when nothing in the model says how to decode it, so the decoder is only a guess.
 */
export function inferConfig(bytes: Uint8Array, input: InputInfo): { config: ModelConfig; source: string; guessed: boolean } {
  let meta: Record<string, string> = {};
  try {
    meta = readOnnxMetadata(bytes);
  } catch (e) {
    console.warn("Could not read ONNX metadata", e);
  }
  let fromMeta: ModelConfig | null = null;
  try {
    fromMeta = configFromMetadata(meta);
  } catch (e) {
    console.warn("Embedded config is unusable, falling back to input shape", e);
  }
  if (fromMeta && meta[MANIFEST_METADATA_KEY]) return { config: fromMeta, source: "manifest embedded in the model", guessed: false };
  if (fromMeta) {
    // The graph's input shape is the ground truth for channel count; older metadata doesn't record it.
    const s = input.shape;
    const c = s?.length === 4 ? (fromMeta.input.layout === "NCHW" ? s[1] : s[3]) : null;
    if (c === 1) fromMeta.input.channels = 1;
    return { config: fromMeta, source: `Ultralytics metadata (task: ${meta.task})`, guessed: false };
  }

  const config: ModelConfig = { decoder: "yolo-detect", input: { ...DEFAULT_INPUT } };
  const s = input.shape;
  if (s?.length === 4) {
    const isChannels = (d: number | null) => d === 1 || d === 3;
    const nhwc = isChannels(s[3]) && !isChannels(s[1]);
    const [h, w] = nhwc ? [s[1], s[2]] : [s[2], s[3]];
    const c = nhwc ? s[3] : s[1];
    if (h && w) {
      config.input = { ...config.input, width: w, height: h, layout: nhwc ? "NHWC" : "NCHW", channels: c === 1 ? 1 : 3 };
      return { config, source: "model input shape (pick the decoder yourself)", guessed: true };
    }
  }
  return { config, source: "defaults (pick decoder and input size yourself)", guessed: true };
}

/** Hands `data` to the user as a pretty-printed .json download. */
export function downloadJson(name: string, data: unknown) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  URL.revokeObjectURL(url);
}

/** Checks a config against the model's fixed input shape, if it has one. */
export function checkInputShape(config: ModelConfig, input: InputInfo): string | null {
  const s = input.shape;
  if (!s || s.length !== 4) return null;
  const { width, height, layout } = config.input;
  const c = config.input.channels ?? 3;
  const expected = layout === "NCHW" ? [null, c, height, width] : [null, height, width, c];
  for (let i = 1; i < 4; i++) {
    if (s[i] !== null && s[i] !== expected[i]) {
      return `Model input is [${s.map((d) => d ?? "?").join(", ")}] but config produces ${layout} ${width}x${height}, ${c} channel${c === 1 ? "" : "s"}`;
    }
  }
  return null;
}
