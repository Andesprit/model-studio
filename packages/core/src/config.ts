import { configFromManifest, isManifest, MANIFEST_METADATA_KEY } from "./manifest";
import type { InputConfig, ModelConfig } from "./types";

export const DEFAULT_INPUT: InputConfig = {
  width: 640,
  height: 640,
  layout: "NCHW",
  resize: "letterbox",
  scale: 1 / 255,
  mean: [0, 0, 0],
  std: [1, 1, 1],
};

/** ImageNet normalization, for models that expect it (most torchvision/timm models). */
export const IMAGENET_INPUT: Partial<InputConfig> = {
  scale: 1 / 255,
  mean: [0.485, 0.456, 0.406],
  std: [0.229, 0.224, 0.225],
};

/**
 * Validates a (possibly partial) JSON config and fills defaults. Also accepts detector-app manifests.
 * Throws with a readable message.
 */
export function parseConfig(json: unknown): ModelConfig {
  if (!json || typeof json !== "object") throw new Error("Config must be a JSON object");
  const c = (isManifest(json) ? configFromManifest(json) : json) as Record<string, unknown>;
  if (typeof c.decoder !== "string" || !c.decoder) {
    throw new Error('Not a usable config: needs a "decoder" string, or a detector-app manifest with "postprocess"');
  }
  const input = { ...DEFAULT_INPUT, ...(c.input as object | undefined) } as InputConfig;

  const positiveInt = (v: unknown) => Number.isInteger(v) && (v as number) > 0;
  if (!positiveInt(input.width) || !positiveInt(input.height)) throw new Error("input.width/height must be positive integers");
  if (input.layout !== "NCHW" && input.layout !== "NHWC") throw new Error('input.layout must be "NCHW" or "NHWC"');
  if (!["letterbox", "stretch", "scale", "shortest_edge"].includes(input.resize)) {
    throw new Error('input.resize must be "letterbox", "stretch", "scale" or "shortest_edge"');
  }
  if (input.resize === "scale" && !((input.resizeFactor ?? 0) > 0)) throw new Error("input.resizeFactor must be > 0 for scale resize");
  if (input.resize === "shortest_edge" && !positiveInt(input.resizeSize)) {
    throw new Error("input.resizeSize must be a positive integer for shortest_edge resize");
  }
  if (input.align !== undefined && input.align !== "center" && input.align !== "top-left") {
    throw new Error('input.align must be "center" or "top-left"');
  }
  if (input.padValue !== undefined && !(input.padValue >= 0 && input.padValue <= 255)) {
    throw new Error("input.padValue must be between 0 and 255");
  }
  for (const key of ["mean", "std"] as const) {
    const v = input[key];
    if (!Array.isArray(v) || v.length !== 3 || v.some((n) => typeof n !== "number")) {
      throw new Error(`input.${key} must be an array of 3 numbers`);
    }
  }
  if (input.std.some((n) => n === 0)) throw new Error("input.std values must be non-zero");
  if (typeof input.scale !== "number") throw new Error("input.scale must be a number");
  if (input.channels !== undefined && input.channels !== 1 && input.channels !== 3) {
    throw new Error("input.channels must be 1 or 3");
  }

  if (c.labels !== undefined && (!Array.isArray(c.labels) || c.labels.some((l) => typeof l !== "string"))) {
    throw new Error("labels must be an array of strings");
  }
  if (c.options !== undefined && (typeof c.options !== "object" || c.options === null)) {
    throw new Error("options must be an object");
  }
  return {
    decoder: c.decoder,
    input,
    labels: c.labels as string[] | undefined,
    options: c.options as Record<string, unknown> | undefined,
  };
}

const TASK_TO_DECODER: Record<string, string> = {
  detect: "yolo-detect",
  segment: "yolo-seg",
  pose: "yolo-pose",
  classify: "classify",
};

/**
 * Builds a config from ONNX metadata: an embedded detector-app manifest, or
 * Ultralytics keys (task, names, imgsz, kpt_shape). Returns null if neither is present.
 */
export function configFromMetadata(meta: Record<string, string>): ModelConfig | null {
  if (meta[MANIFEST_METADATA_KEY]) return parseConfig(JSON.parse(meta[MANIFEST_METADATA_KEY]));
  const decoder = TASK_TO_DECODER[meta.task];
  if (!decoder) return null;

  const labels: string[] = [];
  for (const m of (meta.names ?? "").matchAll(/(\d+)\s*:\s*(['"])(.*?)\2/g)) labels[Number(m[1])] = m[3];

  const imgsz = (meta.imgsz ?? "").match(/\d+/g)?.map(Number);
  const [height, width] = imgsz?.length === 2 ? imgsz : imgsz?.length === 1 ? [imgsz[0], imgsz[0]] : [640, 640];

  const channels = meta.channels === "1" ? 1 : undefined;

  const options: Record<string, unknown> = {};
  const kpt = (meta.kpt_shape ?? "").match(/\d+/g)?.map(Number);
  if (decoder === "yolo-pose" && kpt?.length === 2) options.kptShape = kpt;

  return {
    decoder,
    input: { ...DEFAULT_INPUT, width, height, channels, resize: decoder === "classify" ? "stretch" : "letterbox" },
    labels: labels.length ? labels : undefined,
    options,
  };
}
