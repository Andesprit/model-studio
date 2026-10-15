import { describe, expect, it } from "vitest";
import {
  computeTransform,
  configFromMetadata,
  decode,
  DEFAULT_INPUT,
  nms,
  parseConfig,
  readOnnxMetadata,
  registerPlugin,
  resizeFrame,
  toFrameRect,
  toTensor,
  type ModelConfig,
  type Tensor,
} from "../src";
import { resolveFormat } from "../src/decoders/yolo";

declare class TextEncoder {
  encode(input: string): Uint8Array;
}

const cfg = (decoder: string, extra: Partial<ModelConfig> = {}): ModelConfig => ({
  decoder,
  input: { ...DEFAULT_INPUT },
  ...extra,
});

describe("geometry", () => {
  it("letterbox round-trips a box to frame coords", () => {
    // 1280x720 into 640x640: scale 0.5, content 640x360, padY 140
    const t = computeTransform(1280, 720, { ...DEFAULT_INPUT, width: 640, height: 640, resize: "letterbox" });
    expect(t).toMatchObject({ scaleX: 0.5, scaleY: 0.5, padX: 0, padY: 140 });
    const r = toFrameRect(100, 190, 200, 290, t);
    expect(r).toEqual({ x: 200, y: 100, width: 200, height: 200 });
  });

  it("stretch uses independent scales and clamps to the frame", () => {
    const t = computeTransform(320, 160, { ...DEFAULT_INPUT, width: 640, height: 640, resize: "stretch" });
    expect(toFrameRect(-10, 0, 640, 640, t)).toEqual({ x: 0, y: 0, width: 320, height: 160 });
  });

  it("nms suppresses overlapping same-class boxes only", () => {
    const b = (x: number, classId: number, score: number) => ({ x, y: 0, width: 10, height: 10, classId, score });
    const kept = nms([b(0, 0, 0.9), b(1, 0, 0.8), b(1, 1, 0.7), b(50, 0, 0.6)], 0.5);
    expect(kept.map((k) => k.score)).toEqual([0.9, 0.7, 0.6]);
    expect(nms([b(0, 0, 0.9), b(1, 1, 0.7)], 0.5, true)).toHaveLength(1);
  });
});

describe("preprocess", () => {
  it("toTensor normalizes into NCHW and NHWC", () => {
    const frame = { width: 1, height: 2, data: new Uint8ClampedArray([255, 0, 51, 255, 0, 255, 102, 255]) };
    const input = { ...DEFAULT_INPUT, width: 1, height: 2 };
    const nchw = toTensor(frame, input);
    expect(nchw.dims).toEqual([1, 3, 2, 1]);
    Array.from(nchw.data).forEach((v, i) => expect(v).toBeCloseTo([1, 0, 0, 1, 0.2, 0.4][i]));
    const nhwc = toTensor(frame, { ...input, layout: "NHWC", bgr: true });
    expect(nhwc.dims).toEqual([1, 2, 1, 3]);
    Array.from(nhwc.data).forEach((v, i) => expect(v).toBeCloseTo([0.2, 0, 1, 0.4, 1, 0][i]));
  });

  it("toTensor produces a 1-channel grayscale tensor", () => {
    const frame = { width: 2, height: 1, data: new Uint8ClampedArray([255, 255, 255, 255, 255, 0, 0, 255]) };
    const input = { ...DEFAULT_INPUT, width: 2, height: 1, channels: 1 as const };
    const t = toTensor(frame, input);
    expect(t.dims).toEqual([1, 1, 1, 2]);
    expect(t.data[0]).toBeCloseTo(1);
    expect(t.data[1]).toBeCloseTo(0.299);
    expect(toTensor(frame, { ...input, layout: "NHWC" }).dims).toEqual([1, 1, 2, 1]);
  });

  it("resizeFrame letterboxes with pad color", () => {
    const src = { width: 2, height: 1, data: new Uint8ClampedArray(8).fill(200) };
    const { frame, transform } = resizeFrame(src, { ...DEFAULT_INPUT, width: 4, height: 4 });
    expect(transform.padY).toBe(1);
    expect(frame.data[0]).toBe(114); // row 0 is padding
    expect(frame.data[4 * 4 + 0]).toBe(200); // row 1 is content
  });
});

/** Builds a v8 channels-first tensor [1, C, N] from per-anchor rows. */
function v8Tensor(rows: number[][]): Tensor {
  const C = rows[0].length;
  const N = rows.length;
  const data = new Float32Array(C * N);
  rows.forEach((row, i) => row.forEach((v, c) => (data[c * N + i] = v)));
  return { dims: [1, C, N], data };
}

describe("yolo decoders", () => {
  const identity = computeTransform(640, 640, { ...DEFAULT_INPUT, width: 640, height: 640, resize: "letterbox" });

  it("yolo-detect decodes v8 layout with threshold and NMS", () => {
    const pred = v8Tensor([
      [100, 100, 20, 20, 0.9, 0.1],
      [101, 100, 20, 20, 0.8, 0.1], // overlaps first, suppressed
      [300, 300, 40, 40, 0.1, 0.7],
      [500, 500, 10, 10, 0.1, 0.1], // below threshold
      [400, 400, 10, 10, 0.2, 0.2], // below threshold
    ]);
    const r = decode({ out: pred }, cfg("yolo-detect", { labels: ["a", "b"], options: { format: "v8" } }), identity);
    expect(r.boxes).toHaveLength(2);
    expect(r.boxes![0]).toMatchObject({ x: 90, y: 90, width: 20, height: 20, classId: 0, label: "a" });
    expect(r.boxes![1]).toMatchObject({ classId: 1, label: "b", score: expect.closeTo(0.7) });
  });

  it("yolo-detect decodes v5 layout (objectness * class score)", () => {
    const data = new Float32Array([100, 100, 20, 20, 0.5, 0.9, 0.1, 50, 50, 10, 10, 0.1, 0.9, 0.9]);
    const r = decode({ out: { dims: [1, 2, 7], data } }, cfg("yolo-detect", { options: { format: "v5" } }), identity);
    expect(r.boxes).toHaveLength(1);
    expect(r.boxes![0].score).toBeCloseTo(0.45);
  });

  it("yolo-detect decodes end2end layout without NMS", () => {
    const data = new Float32Array([10, 10, 50, 50, 0.9, 3, 12, 12, 52, 52, 0.8, 3, 0, 0, 1, 1, 0.01, 0]);
    const r = decode({ out: { dims: [1, 3, 6], data } }, cfg("yolo-detect", { options: { format: "end2end" } }), identity);
    expect(r.boxes!.map((b) => b.classId)).toEqual([3, 3]);
    expect(r.boxes![0]).toMatchObject({ x: 10, y: 10, width: 40, height: 40 });
  });

  it("yolo-seg builds a cropped mask from protos", () => {
    // one class, 1 mask coef, protos 1x4x4 over a 640 input (each proto px = 160 input px)
    const pred = v8Tensor([[160, 160, 320, 320, 0.9, 1]]);
    const protos: Tensor = { dims: [1, 1, 4, 4], data: new Float32Array(16).fill(5) };
    const r = decode({ pred, protos }, cfg("yolo-seg", { options: { format: "v8" } }), identity);
    const m = r.masks![0];
    expect([m.width, m.height]).toEqual([2, 2]);
    expect(Array.from(m.data)).toEqual([255, 255, 255, 255]);
    expect(m.rect).toEqual({ x: 0, y: 0, width: 320, height: 320 });
  });

  it("yolo-pose maps keypoints to frame coords", () => {
    const pred = v8Tensor([[100, 100, 50, 50, 0.9, 110, 120, 0.8]]);
    const r = decode({ pred }, cfg("yolo-pose", { options: { kptShape: [1, 3], format: "v8" } }), identity);
    expect(r.poses![0].keypoints[0]).toEqual({ x: 110, y: 120, score: expect.closeTo(0.8) });
  });
});

describe("yolo format auto-detection on real export shapes", () => {
  it.each([
    [[1, 84, 8400], 0, 80, "v8"],
    [[1, 116, 8400], 32, 80, "v8"],
    [[1, 56, 8400], 51, 0, "v8"],
    [[1, 25200, 85], 0, 80, "v5"],
    [[1, 25200, 85], 0, 0, "v5"],
    [[1, 8400, 84], 0, 80, "v8t"],
    [[1, 300, 6], 0, 80, "end2end"],
    [[1, 300, 38], 32, 80, "end2end"],
  ])("%j extra=%i labels=%i -> %s", (dims, extra, nl, expected) => {
    expect(resolveFormat(dims as number[], extra as number, nl as number, "auto")).toBe(expected);
  });
});

describe("other decoders", () => {
  const t = computeTransform(640, 640, { ...DEFAULT_INPUT, width: 640, height: 640, resize: "stretch" });

  it("semantic-seg argmaxes NCHW logits", () => {
    const data = new Float32Array([1, 0, 0, 1, /* class 1 */ 0, 1, 1, 0]);
    const r = decode({ out: { dims: [1, 2, 2, 2], data } }, cfg("semantic-seg"), t);
    expect(Array.from(r.segmentation!.classes)).toEqual([0, 1, 1, 0]);
    expect(r.segmentation!.rect).toEqual({ x: 0, y: 0, width: 640, height: 640 });
  });

  it("classify applies softmax to logits and returns topK", () => {
    const r = decode({ out: { dims: [1, 3], data: new Float32Array([0, 5, 1]) } }, cfg("classify", { options: { topK: 2 } }), t);
    expect(r.classes!.map((c) => c.classId)).toEqual([1, 2]);
    expect(r.classes![0].score).toBeGreaterThan(0.9);
  });

  it("plugins register and decode", () => {
    const key = registerPlugin({ name: "count", decode: (o) => ({ classes: [{ classId: 0, score: Object.keys(o).length }] }) });
    expect(decode({ a: { dims: [1], data: new Float32Array(1) } }, cfg(key), t).classes![0].score).toBe(1);
  });
});

describe("config", () => {
  it("parseConfig fills defaults and rejects bad input", () => {
    const c = parseConfig({ decoder: "yolo-detect", input: { width: 320, height: 320 } });
    expect(c.input).toMatchObject({ width: 320, layout: "NCHW", scale: 1 / 255 });
    expect(() => parseConfig({})).toThrow(/decoder/);
    expect(parseConfig({ decoder: "x", input: { channels: 1 } }).input.channels).toBe(1);
    expect(() => parseConfig({ decoder: "x", input: { channels: 2 } })).toThrow(/channels/);
    expect(() => parseConfig({ decoder: "x", input: { mean: [1, 2] } })).toThrow(/mean/);
  });

  it("reads Ultralytics metadata from raw onnx bytes", () => {
    const enc = new TextEncoder();
    const field = (num: number, payload: Uint8Array) => [num * 8 + 2, payload.length, ...payload];
    const entry = (k: string, v: string) => new Uint8Array([...field(1, enc.encode(k)), ...field(2, enc.encode(v))]);
    const bytes = new Uint8Array([
      0x08, 0x08, // ir_version = 8 (varint field)
      ...field(7, new Uint8Array([1, 2, 3])), // fake graph, skipped
      ...field(14, entry("task", "pose")),
      ...field(14, entry("names", "{0: 'person', 1: \"it's\"}")),
      ...field(14, entry("imgsz", "[480, 640]")),
      ...field(14, entry("kpt_shape", "[17, 3]")),
    ]);
    const meta = readOnnxMetadata(bytes);
    const c = configFromMetadata(meta)!;
    expect(c.decoder).toBe("yolo-pose");
    expect(c.labels).toEqual(["person", "it's"]);
    expect(c.input).toMatchObject({ width: 640, height: 480 });
    expect(c.options).toEqual({ kptShape: [17, 3] });
    expect(configFromMetadata({})).toBeNull();
  });
});
