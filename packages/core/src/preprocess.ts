import { computeTransform } from "./geometry";
import type { Frame, InputConfig, Tensor, Transform } from "./types";

/** Default pad color (Ultralytics letterbox gray). */
export const PAD_VALUE = 114;

/**
 * Converts an RGBA frame that is already at model input size into a float tensor.
 * Platforms with a fast native resize (canvas, Skia) should resize first and call this directly.
 */
export function toTensor(frame: Frame, input: InputConfig): Tensor {
  const { width, height, data } = frame;
  if (width !== input.width || height !== input.height) {
    throw new Error(`toTensor: frame is ${width}x${height}, expected ${input.width}x${input.height}`);
  }
  const plane = width * height;
  const channels = input.channels ?? 3;
  const out = new Float32Array(plane * channels);
  const { scale, mean, std, layout } = input;
  if (channels === 1) {
    // BT.601 luma; with one channel NCHW and NHWC share the same memory layout.
    for (let i = 0; i < plane; i++) {
      const luma = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
      out[i] = (luma * scale - mean[0]) / std[0];
    }
  } else {
    const order = input.bgr ? [2, 1, 0] : [0, 1, 2];
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) {
        const v = (data[i * 4 + order[c]] * scale - mean[c]) / std[c];
        if (layout === "NCHW") out[c * plane + i] = v;
        else out[i * 3 + c] = v;
      }
    }
  }
  const dims = layout === "NCHW" ? [1, channels, height, width] : [1, height, width, channels];
  return { dims, data: out };
}

/**
 * Pure-JS bilinear resize into the model input size (letterboxed or stretched).
 * Used where no native resize is available, and in tests.
 */
export function resizeFrame(frame: Frame, input: InputConfig): { frame: Frame; transform: Transform } {
  const t = computeTransform(frame.width, frame.height, input);
  const out = new Uint8ClampedArray(input.width * input.height * 4).fill(input.padValue ?? PAD_VALUE);
  const src = frame.data;
  const sw = frame.width;
  const sh = frame.height;
  for (let y = 0; y < input.height; y++) {
    const fy = (y + 0.5 - t.padY) / t.scaleY - 0.5;
    if (fy < -0.5 || fy > sh - 0.5) continue;
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const wy = Math.min(Math.max(fy - y0, 0), 1);
    for (let x = 0; x < input.width; x++) {
      const fx = (x + 0.5 - t.padX) / t.scaleX - 0.5;
      if (fx < -0.5 || fx > sw - 0.5) continue;
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const wx = Math.min(Math.max(fx - x0, 0), 1);
      const o = (y * input.width + x) * 4;
      for (let c = 0; c < 4; c++) {
        const a = src[(y0 * sw + x0) * 4 + c];
        const b = src[(y0 * sw + x1) * 4 + c];
        const d = src[(y1 * sw + x0) * 4 + c];
        const e = src[(y1 * sw + x1) * 4 + c];
        out[o + c] = (a * (1 - wx) + b * wx) * (1 - wy) + (d * (1 - wx) + e * wx) * wy;
      }
    }
  }
  return { frame: { width: input.width, height: input.height, data: out }, transform: t };
}
