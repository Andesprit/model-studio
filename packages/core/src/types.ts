import type { Tracker, TrackerOptions } from "./tracker";

/** RGBA pixels, row-major, 4 bytes per pixel. */
export interface Frame {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** Runtime-agnostic tensor. Platform adapters convert to/from their ONNX runtime's tensor type. */
export interface Tensor {
  dims: number[];
  data: Float32Array;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Box extends Rect {
  classId: number;
  score: number;
  label?: string;
  /** Stable id across frames, set by a tracking pipeline. */
  trackId?: number;
}

/** Per-instance mask. `data` is width*height alpha (0/255), stretched over `rect` (frame coords). */
export interface Mask {
  width: number;
  height: number;
  data: Uint8Array;
  rect: Rect;
  classId: number;
}

/** Dense per-pixel class map, stretched over `rect` (frame coords). */
export interface SegmentationMap {
  width: number;
  height: number;
  classes: Uint16Array;
  rect: Rect;
  /** Class index left transparent when drawing (-1 for none). */
  background: number;
}

export interface Keypoint {
  x: number;
  y: number;
  score: number;
}

export interface Pose {
  box: Box;
  keypoints: Keypoint[];
}

export interface ClassScore {
  classId: number;
  score: number;
  label?: string;
}

/** Everything a decoder can produce. All coordinates are in source-frame pixels. */
export interface Result {
  boxes?: Box[];
  masks?: Mask[];
  segmentation?: SegmentationMap;
  poses?: Pose[];
  /** Pairs of keypoint indices to connect when drawing poses. */
  skeleton?: [number, number][];
  classes?: ClassScore[];
  /** Values a pipeline wants shown next to the stats, e.g. { unique: 132 }. */
  metrics?: Record<string, number | string>;
  /** Extra shapes a pipeline wants drawn, in frame pixels. */
  overlays?: Overlay[];
}

export type Overlay =
  | { type: "line"; from: Point; to: Point; color?: string }
  | { type: "polyline"; points: Point[]; color?: string }
  | { type: "point"; at: Point; color?: string }
  | { type: "text"; at: Point; text: string; color?: string };

export interface Point {
  x: number;
  y: number;
}

export type Layout = "NCHW" | "NHWC";
/**
 * - letterbox:     fit inside width x height keeping aspect, pad the rest
 * - stretch:       fill width x height exactly, aspect not kept
 * - scale:         fixed `resizeFactor`; content beyond the canvas is cropped
 * - shortest_edge: scale the short side to `resizeSize`, then crop/pad to the canvas
 */
export type ResizeMode = "letterbox" | "stretch" | "scale" | "shortest_edge";
export type Align = "center" | "top-left";

export interface InputConfig {
  width: number;
  height: number;
  layout: Layout;
  resize: ResizeMode;
  /** "scale" mode factor. */
  resizeFactor?: number;
  /** "shortest_edge" mode target length. */
  resizeSize?: number;
  /** Where the resized frame sits on the canvas (default "center"). */
  align?: Align;
  /** Padding pixel value 0-255, before normalization (default 114). */
  padValue?: number;
  /** Pixel value is multiplied by `scale`, then (v - mean) / std per channel (RGB order). */
  scale: number;
  mean: [number, number, number];
  std: [number, number, number];
  /** Use BGR channel order instead of RGB. */
  bgr?: boolean;
  /** 3 = color (default), 1 = grayscale (luma; uses mean[0]/std[0]). */
  channels?: 1 | 3;
}

export interface ModelConfig {
  /** Built-in decoder name ("yolo-detect", ...) or "plugin:<name>". */
  decoder: string;
  input: InputConfig;
  labels?: string[];
  /** Decoder-specific options, e.g. scoreThreshold, iouThreshold. */
  options?: Record<string, unknown>;
}

/** Maps model-input pixel coords back to source-frame pixel coords. */
export interface Transform {
  scaleX: number;
  scaleY: number;
  padX: number;
  padY: number;
  frameWidth: number;
  frameHeight: number;
}

export interface DecodeContext {
  config: ModelConfig;
  transform: Transform;
  /** Options merged with decoder defaults. */
  options: Record<string, unknown>;
  labels: string[];
  /** Helpers exposed so plugins don't need to bundle their own. */
  utils: DecoderUtils;
  /** Built-in pieces for pipelines: `utils`, the tracker, and built-in decoders by name. */
  lib: DecodeLib;
}

/** Built-ins available to every pipeline. Pipelines are loaded from a file and can't import the app. */
export interface PipelineLib extends DecoderUtils {
  createTracker(options?: TrackerOptions): Tracker;
}

export interface DecodeLib extends PipelineLib {
  /** Runs a built-in decoder (e.g. "yolo-detect") on these outputs with the current config and transform. */
  decode(name: string, outputs: Record<string, Tensor>, options?: Record<string, unknown>): Result;
}

export interface DecoderUtils {
  nms<T extends Box>(boxes: T[], iouThreshold: number, classAgnostic?: boolean): T[];
  sigmoid(x: number): number;
  softmax(values: Float32Array): Float32Array;
  toFrameRect(x1: number, y1: number, x2: number, y2: number, t: Transform): Rect;
  toFramePoint(x: number, y: number, t: Transform): { x: number; y: number };
}

export interface Decoder {
  name: string;
  /** Default options shown in the UI and merged under user options. */
  defaults?: Record<string, unknown>;
  /** Slider bounds for numeric options, for UIs (thresholds default to [0, 1]). */
  ranges?: Record<string, [number, number]>;
  decode(outputs: Record<string, Tensor>, ctx: DecodeContext): Result;
}

/** Real-world scale, from a line of known length drawn on the video. */
export interface Calibration {
  /** Length of the drawn line in frame pixels. */
  pixels: number;
  /** Real length of that line, in `unit`. */
  amount: number;
  unit: string;
  /** `amount / pixels`: multiply a pixel distance by this to get `unit`. */
  unitsPerPixel: number;
}

/** What a pipeline instance is created with. */
export interface PipelineContext {
  config: ModelConfig;
  /** Pipeline options merged with the pipeline's defaults. */
  options: Record<string, unknown>;
  labels: string[];
  utils: DecoderUtils;
  lib: PipelineLib;
  /** Null until the user calibrates. */
  calibration: Calibration | null;
}

/** Per-frame info passed to `step`. */
export interface FrameInfo {
  /** Media time in seconds. */
  timestamp: number;
  /** Count of frames this instance has processed (dropped frames are not counted). */
  index: number;
  /** Frames skipped since this instance was created because inference was still busy. */
  dropped: number;
  width: number;
  height: number;
}

export interface PipelineInstance {
  step(result: Result, frame: FrameInfo): Result;
  /** Optional totals for the results file (per-particle speeds, counts…). Must be JSON-serializable. */
  summary?(): Record<string, unknown>;
}

/** Statistics of one numeric metric over the analyzed frames. */
export interface MetricStats {
  mean: number;
  min: number;
  max: number;
  last: number;
  frames: number;
}

/** Everything measured since the pipeline last started fresh. */
export interface RunSummary {
  /** Media time (s) of the first and last analyzed frame; null before any frame. */
  from: number | null;
  to: number | null;
  framesAnalyzed: number;
  framesDropped: number;
  /** Every numeric metric the pipeline published, summarized over the analyzed frames. */
  metrics: Record<string, MetricStats>;
  /** Whatever the pipeline's own `summary()` returned. */
  pipelineResults?: Record<string, unknown>;
}

/**
 * A user pipeline: the default export of a loaded JS module.
 * `decode` replaces the config's decoder; `create` builds a stateful post-processor and is
 * called again on every reset, so instances never need to clear their own state.
 * A plain decoder plugin ({ name, decode }) is a valid pipeline.
 */
export interface PipelineDefinition {
  name: string;
  defaults?: Record<string, unknown>;
  ranges?: Record<string, [number, number]>;
  decode?(outputs: Record<string, Tensor>, ctx: DecodeContext): Result;
  create?(ctx: PipelineContext): PipelineInstance;
}
