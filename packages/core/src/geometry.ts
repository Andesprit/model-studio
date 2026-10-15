import type { Box, InputConfig, Rect, Transform } from "./types";

/** How a frame of srcW x srcH is placed onto the model's input canvas. */
export function computeTransform(srcW: number, srcH: number, input: InputConfig): Transform {
  const { width: dstW, height: dstH } = input;
  const base = { frameWidth: srcW, frameHeight: srcH };
  if (input.resize === "stretch") {
    return { ...base, scaleX: dstW / srcW, scaleY: dstH / srcH, padX: 0, padY: 0 };
  }
  const scale =
    input.resize === "scale" ? (input.resizeFactor ?? 1)
    : input.resize === "shortest_edge" ? (input.resizeSize ?? Math.min(dstW, dstH)) / Math.min(srcW, srcH)
    : Math.min(dstW / srcW, dstH / srcH);
  // Negative padding means the resized frame is larger than the canvas and gets cropped.
  const center = (input.align ?? "center") === "center";
  const padX = center ? Math.floor((dstW - Math.round(srcW * scale)) / 2) : 0;
  const padY = center ? Math.floor((dstH - Math.round(srcH * scale)) / 2) : 0;
  return { ...base, scaleX: scale, scaleY: scale, padX, padY };
}

const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);

export function toFramePoint(x: number, y: number, t: Transform) {
  return { x: (x - t.padX) / t.scaleX, y: (y - t.padY) / t.scaleY };
}

/** Converts a model-input-space box (x1,y1,x2,y2) to a clamped frame-space rect. */
export function toFrameRect(x1: number, y1: number, x2: number, y2: number, t: Transform): Rect {
  const a = toFramePoint(x1, y1, t);
  const b = toFramePoint(x2, y2, t);
  const left = clamp(a.x, 0, t.frameWidth);
  const top = clamp(a.y, 0, t.frameHeight);
  const right = clamp(b.x, 0, t.frameWidth);
  const bottom = clamp(b.y, 0, t.frameHeight);
  return { x: left, y: top, width: right - left, height: bottom - top };
}

export function iou(a: Rect, b: Rect): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/** Greedy non-maximum suppression. Returns kept boxes sorted by score (desc). */
export function nms<T extends Box>(boxes: T[], iouThreshold: number, classAgnostic = false): T[] {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: T[] = [];
  for (const box of sorted) {
    const suppressed = kept.some(
      (k) => (classAgnostic || k.classId === box.classId) && iou(k, box) > iouThreshold,
    );
    if (!suppressed) kept.push(box);
  }
  return kept;
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export function softmax(values: Float32Array): Float32Array {
  let max = -Infinity;
  for (const v of values) max = Math.max(max, v);
  const out = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    out[i] = Math.exp(values[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}
