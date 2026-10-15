import type { InputConfig, ModelConfig, ResizeMode } from "./types";

/** ONNX metadata key under which detector-app's embed_manifest.py stores a manifest. */
export const MANIFEST_METADATA_KEY = "detector.manifest";

type Json = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

/** True if a JSON object looks like a detector-app manifest rather than a native config. */
export const isManifest = (json: unknown): json is Json =>
  !!json && typeof json === "object" && "postprocess" in json && !("decoder" in json);

const RESIZE: Record<string, ResizeMode> = {
  none: "scale",
  scale: "scale",
  fit: "letterbox",
  stretch: "stretch",
  shortest_edge: "shortest_edge",
};

const POSTPROCESS_TYPES = ["binary_mask", "boxes", "classification"];

/**
 * Converts a detector-app manifest (see detector-app/js/manifest.js) into a ModelConfig.
 * The result still goes through parseConfig validation.
 */
export function configFromManifest(m: Json): Partial<ModelConfig> & { decoder: string } {
  if (!POSTPROCESS_TYPES.includes(m.postprocess?.type)) {
    throw new Error(`Manifest postprocess.type "${m.postprocess?.type}" is not supported (${POSTPROCESS_TYPES.join(", ")})`);
  }
  const inp: Json = m.input ?? {};
  const resize: Json = inp.resize ?? { mode: "none" };
  const pad: Json = inp.pad ?? {};
  const norm: Json = inp.normalize ?? {};
  const gray = inp.colorspace === "gray";
  const divisor = Number(norm.divisor ?? 255);

  const mode = RESIZE[resize.mode];
  if (!mode) throw new Error(`Manifest input.resize.mode "${resize.mode}" is not supported`);
  const hw: unknown = inp.canvas_hw ?? inp.center_crop ?? resize.hw;
  if (!Array.isArray(hw) || hw.length !== 2) {
    throw new Error("Manifest needs input.canvas_hw (or center_crop / resize.hw); dynamic-size inputs aren't supported");
  }
  const triple = (v: unknown, fallback: number): [number, number, number] =>
    Array.isArray(v) ? (v.length === 1 ? [v[0], v[0], v[0]] : (v as [number, number, number])) : [fallback, fallback, fallback];

  const input: Partial<InputConfig> = {
    height: hw[0],
    width: hw[1],
    layout: inp.layout ?? "NCHW",
    resize: mode,
    resizeFactor: resize.mode === "none" ? 1 : resize.factor,
    resizeSize: resize.size,
    // A center crop is centered by definition; otherwise detector-app defaults to top-left.
    align: inp.center_crop ? "center" : (pad.position ?? "top-left"),
    // Manifest pad values are in [0, 1] space (after dividing by the divisor).
    padValue: Math.min(255, Math.max(0, Math.round(Number(pad.value ?? 0) * divisor))),
    scale: 1 / divisor,
    mean: triple(norm.mean, 0),
    std: triple(norm.std, 1),
    channels: gray ? 1 : 3,
    bgr: inp.channel_order === "bgr",
  };

  const pp: Json = m.postprocess;
  const outputName = m.output?.name ?? "";
  const labels: string[] | undefined = pp.labels;
  switch (pp.type) {
    case "binary_mask":
      return { decoder: "semantic-seg", input: input as InputConfig, options: { outputName, mode: "binary", threshold: pp.threshold ?? 0 } };
    case "boxes":
      if (pp.encoding && pp.encoding !== "xyxysc") throw new Error(`Manifest boxes encoding "${pp.encoding}" is not supported`);
      return {
        decoder: "yolo-detect",
        input: input as InputConfig,
        labels,
        options: { outputName, format: "end2end", scoreThreshold: pp.conf_threshold ?? 0.25, maxDetections: pp.max_detections ?? 300 },
      };
    case "classification":
      return {
        decoder: "classify",
        input: input as InputConfig,
        labels,
        options: { outputName, activation: pp.activation ?? "softmax", topK: pp.top_k ?? 5, scoreThreshold: pp.conf_threshold ?? 0 },
      };
    default:
      throw new Error("unreachable");
  }
}
