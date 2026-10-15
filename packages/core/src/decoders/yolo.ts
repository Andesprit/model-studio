import type { Box, DecodeContext, Decoder, Keypoint, Mask, Pose, Tensor } from "../types";

/**
 * Supported YOLO output layouts:
 * - "v8":       [1, 4+nc+extra, N]  cx,cy,w,h, class scores, extra (v8/v9/v11/v12, channels-first)
 * - "v8t":      [1, N, 4+nc+extra]  same, transposed
 * - "v5":       [1, N, 5+nc+extra]  cx,cy,w,h, objectness, class scores, extra
 * - "end2end":  [1, N, 6+extra]     x1,y1,x2,y2, score, classId (NMS-free: v10, YOLO26)
 */
export type YoloFormat = "auto" | "v8" | "v8t" | "v5" | "end2end";

export interface YoloCandidate extends Box {
  /** Model-input-space corners, needed for mask cropping. */
  ix1: number;
  iy1: number;
  ix2: number;
  iy2: number;
  /** Offset of the extra features (mask coefs / keypoints) and accessor. */
  extra: (k: number) => number;
}

export const YOLO_DEFAULTS = {
  scoreThreshold: 0.25,
  iouThreshold: 0.45,
  maxDetections: 300,
  classAgnostic: false,
  format: "auto" as YoloFormat,
  outputName: "",
};

export function resolveFormat(dims: number[], extra: number, numLabels: number, format: YoloFormat): YoloFormat {
  if (format !== "auto") return format;
  if (dims.length !== 3) throw new Error(`YOLO output must be 3D, got [${dims.join(", ")}]`);
  const [, a, b] = dims;
  if (a < b) return "v8";
  const features = b - extra;
  if (features === 6 && a <= 1000) return "end2end";
  if (numLabels > 0 && features === 4 + numLabels) return "v8t";
  return "v5";
}

/** Picks the named output, else the first 3D output. */
export function pickOutput(outputs: Record<string, Tensor>, name: string, rank = 3): Tensor {
  if (name) {
    const t = outputs[name];
    if (!t) throw new Error(`Output "${name}" not found. Outputs: ${Object.keys(outputs).join(", ")}`);
    return t;
  }
  const t = Object.values(outputs).find((o) => o.dims.length === rank);
  if (!t) throw new Error(`No ${rank}D output found. Outputs: ${describeOutputs(outputs)}`);
  return t;
}

export function describeOutputs(outputs: Record<string, Tensor>) {
  return Object.entries(outputs)
    .map(([k, t]) => `${k} [${t.dims.join(", ")}]`)
    .join(", ");
}

/** Parses raw YOLO predictions into thresholded, NMS'd candidates in frame coords. */
export function parseYolo(pred: Tensor, extra: number, ctx: DecodeContext): YoloCandidate[] {
  const o = ctx.options;
  const format = resolveFormat(pred.dims, extra, ctx.labels.length, o.format as YoloFormat);
  const channelsFirst = format === "v8";
  const N = channelsFirst ? pred.dims[2] : pred.dims[1];
  const C = channelsFirst ? pred.dims[1] : pred.dims[2];
  const d = pred.data;
  const get = channelsFirst ? (i: number, c: number) => d[c * N + i] : (i: number, c: number) => d[i * C + c];

  const headSize = format === "v5" ? 5 : 4;
  const nc = format === "end2end" ? 0 : C - headSize - extra;
  if (format !== "end2end" && nc < 1) {
    throw new Error(`YOLO output [${pred.dims.join(", ")}] has no room for class scores (format ${format}, extra ${extra})`);
  }
  const extraStart = format === "end2end" ? 6 : headSize + nc;
  const scoreThreshold = Number(o.scoreThreshold);
  const t = ctx.transform;

  const candidates: YoloCandidate[] = [];
  for (let i = 0; i < N; i++) {
    let score: number;
    let classId: number;
    if (format === "end2end") {
      score = get(i, 4);
      classId = Math.round(get(i, 5));
    } else {
      const obj = format === "v5" ? get(i, 4) : 1;
      if (obj < scoreThreshold) continue;
      score = -Infinity;
      classId = 0;
      for (let c = 0; c < nc; c++) {
        const s = get(i, headSize + c);
        if (s > score) {
          score = s;
          classId = c;
        }
      }
      score *= obj;
    }
    if (score < scoreThreshold) continue;

    let x1: number, y1: number, x2: number, y2: number;
    if (format === "end2end") {
      [x1, y1, x2, y2] = [get(i, 0), get(i, 1), get(i, 2), get(i, 3)];
    } else {
      const cx = get(i, 0), cy = get(i, 1), w = get(i, 2), h = get(i, 3);
      [x1, y1, x2, y2] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    }
    const rect = ctx.utils.toFrameRect(x1, y1, x2, y2, t);
    const row = i;
    candidates.push({
      ...rect,
      classId,
      score,
      label: ctx.labels[classId] ?? String(classId),
      ix1: x1,
      iy1: y1,
      ix2: x2,
      iy2: y2,
      extra: (k) => get(row, extraStart + k),
    });
  }

  const kept = format === "end2end"
    ? candidates.sort((a, b) => b.score - a.score)
    : ctx.utils.nms(candidates, Number(o.iouThreshold), Boolean(o.classAgnostic));
  return kept.slice(0, Number(o.maxDetections));
}

const stripInternal = (c: YoloCandidate): Box => ({
  x: c.x, y: c.y, width: c.width, height: c.height, classId: c.classId, score: c.score, label: c.label,
});

export const yoloDetect: Decoder = {
  name: "yolo-detect",
  defaults: YOLO_DEFAULTS,
  decode(outputs, ctx) {
    const pred = pickOutput(outputs, String(ctx.options.outputName));
    return { boxes: parseYolo(pred, 0, ctx).map(stripInternal) };
  },
};

export const yoloSeg: Decoder = {
  name: "yolo-seg",
  defaults: { ...YOLO_DEFAULTS, maskThreshold: 0.5 },
  decode(outputs, ctx) {
    const protos = pickOutput(outputs, "", 4);
    const pred = pickOutput(outputs, String(ctx.options.outputName));
    const [, nm, mh, mw] = protos.dims;
    const kept = parseYolo(pred, nm, ctx);
    const { width: iw, height: ih } = ctx.config.input;
    const sx = mw / iw;
    const sy = mh / ih;
    const p = protos.data;
    const plane = mh * mw;
    const threshold = Number(ctx.options.maskThreshold);
    // sigmoid(x) > threshold  <=>  x > logit(threshold)
    const logit = Math.log(threshold / (1 - threshold));
    const coefs = new Float32Array(nm);

    const masks: Mask[] = kept.map((c) => {
      const px0 = Math.max(0, Math.floor(c.ix1 * sx));
      const py0 = Math.max(0, Math.floor(c.iy1 * sy));
      const px1 = Math.min(mw, Math.ceil(c.ix2 * sx));
      const py1 = Math.min(mh, Math.ceil(c.iy2 * sy));
      const w = Math.max(0, px1 - px0);
      const h = Math.max(0, py1 - py0);
      for (let k = 0; k < nm; k++) coefs[k] = c.extra(k);
      const data = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (py0 + y) * mw + (px0 + x);
          let v = 0;
          for (let k = 0; k < nm; k++) v += coefs[k] * p[k * plane + idx];
          data[y * w + x] = v > logit ? 255 : 0;
        }
      }
      const tl = ctx.utils.toFramePoint(px0 / sx, py0 / sy, ctx.transform);
      const br = ctx.utils.toFramePoint(px1 / sx, py1 / sy, ctx.transform);
      return { width: w, height: h, data, rect: { x: tl.x, y: tl.y, width: br.x - tl.x, height: br.y - tl.y }, classId: c.classId };
    });
    return { boxes: kept.map(stripInternal), masks };
  },
};

/** COCO-17 keypoint skeleton (0-indexed). */
export const COCO_SKELETON: [number, number][] = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
  [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
];

export const yoloPose: Decoder = {
  name: "yolo-pose",
  defaults: { ...YOLO_DEFAULTS, kptShape: [17, 3] },
  decode(outputs, ctx) {
    const [numKpt, kptDim] = ctx.options.kptShape as [number, number];
    const pred = pickOutput(outputs, String(ctx.options.outputName));
    const kept = parseYolo(pred, numKpt * kptDim, ctx);
    const poses: Pose[] = kept.map((c) => {
      const keypoints: Keypoint[] = [];
      for (let k = 0; k < numKpt; k++) {
        const pt = ctx.utils.toFramePoint(c.extra(k * kptDim), c.extra(k * kptDim + 1), ctx.transform);
        keypoints.push({ ...pt, score: kptDim === 3 ? c.extra(k * kptDim + 2) : 1 });
      }
      return { box: stripInternal(c), keypoints };
    });
    const skeleton = numKpt === 17 ? COCO_SKELETON : undefined;
    return { poses, skeleton };
  },
};
