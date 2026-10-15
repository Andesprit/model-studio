import { describe, expect, it } from "vitest";
import {
  computeTransform,
  configFromMetadata,
  decode,
  DEFAULT_INPUT,
  parseConfig,
  resizeFrame,
  type InputConfig,
} from "../src";

// The vortex detector manifest from detector-app (tiny U-Net, binary segmentation).
const vortexManifest = {
  label: "Vortex detector (fp32)",
  model_url: "vortex_detector_fp32.onnx",
  input: {
    name: "image",
    colorspace: "gray",
    layout: "NCHW",
    resize: { mode: "scale", factor: 0.5 },
    canvas_hw: [160, 416],
    pad: { position: "top-left", value: 0.0 },
    normalize: { divisor: 255.0 },
  },
  output: { name: "logits" },
  postprocess: { type: "binary_mask", threshold: 0.0, crop_to_frame: true, upsample: true },
};

describe("resize modes and alignment", () => {
  const input = (patch: Partial<InputConfig>): InputConfig => ({ ...DEFAULT_INPUT, ...patch });

  it("scale mode uses a fixed factor and top-left alignment", () => {
    const t = computeTransform(794, 270, input({ width: 416, height: 160, resize: "scale", resizeFactor: 0.5, align: "top-left" }));
    expect(t).toMatchObject({ scaleX: 0.5, scaleY: 0.5, padX: 0, padY: 0 });
  });

  it("shortest_edge scales the short side then center-crops (negative pad)", () => {
    const t = computeTransform(400, 300, input({ width: 256, height: 256, resize: "shortest_edge", resizeSize: 288 }));
    expect(t.scaleX).toBeCloseTo(0.96);
    expect(t.padX).toBe(-64); // 384 wide resized frame, 256 canvas
    expect(t.padY).toBe(-16);
  });

  it("resizeFrame pads with the configured value at top-left", () => {
    const src = { width: 2, height: 2, data: new Uint8ClampedArray(16).fill(200) };
    const { frame } = resizeFrame(src, input({ width: 4, height: 4, resize: "scale", resizeFactor: 1, align: "top-left", padValue: 0 }));
    expect(frame.data[0]).toBe(200); // content at top-left
    expect(frame.data[(3 * 4 + 3) * 4]).toBe(0); // bottom-right is padding
  });
});

describe("detector-app manifests", () => {
  it("maps the vortex binary_mask manifest", () => {
    const c = parseConfig(vortexManifest);
    expect(c.decoder).toBe("semantic-seg");
    expect(c.input).toMatchObject({
      width: 416, height: 160, channels: 1, layout: "NCHW",
      resize: "scale", resizeFactor: 0.5, align: "top-left", padValue: 0,
    });
    expect(c.input.scale).toBeCloseTo(1 / 255);
    expect(c.options).toMatchObject({ mode: "binary", threshold: 0, outputName: "logits" });
  });

  it("maps boxes and classification manifests", () => {
    const boxes = parseConfig({
      input: { resize: { mode: "fit", hw: [640, 640] }, canvas_hw: [640, 640], pad: { position: "center", value: 0.447 } },
      output: { name: "output0" },
      postprocess: { type: "boxes", encoding: "xyxysc", conf_threshold: 0.3, max_detections: 100, labels: ["a"] },
    });
    expect(boxes).toMatchObject({ decoder: "yolo-detect", labels: ["a"] });
    expect(boxes.input).toMatchObject({ resize: "letterbox", align: "center", padValue: 114 });
    expect(boxes.options).toMatchObject({ format: "end2end", scoreThreshold: 0.3, maxDetections: 100 });

    const cls = parseConfig({
      input: { channel_order: "bgr", resize: { mode: "shortest_edge", size: 288 }, center_crop: [256, 256] },
      postprocess: { type: "classification", activation: "softmax", top_k: 5, labels: ["x"] },
    });
    expect(cls).toMatchObject({ decoder: "classify" });
    expect(cls.input).toMatchObject({ width: 256, height: 256, resize: "shortest_edge", resizeSize: 288, align: "center", bgr: true });
    expect(cls.options).toMatchObject({ activation: "softmax", topK: 5 });
  });

  it("explains unusable configs", () => {
    expect(() => parseConfig({ input: "grayscale frame, resized by scale", scale: 0.5 })).toThrow(/manifest/);
    expect(() => parseConfig({ postprocess: { type: "keypoints" } })).toThrow(/keypoints/);
  });

  it("reads a manifest embedded in ONNX metadata", () => {
    const c = configFromMetadata({ "detector.manifest": JSON.stringify(vortexManifest) });
    expect(c?.decoder).toBe("semantic-seg");
  });
});

describe("binary masks", () => {
  it("thresholds 1-channel logits and maps the canvas back onto the frame", () => {
    const c = parseConfig(vortexManifest);
    const input = { ...c.input, width: 4, height: 2 };
    const t = computeTransform(4, 2, { ...input, resizeFactor: 0.5 });
    const logits = { dims: [1, 1, 2, 4], data: new Float32Array([-3, 0.5, 2, -1, 0.1, -0.1, 0, 4]) };
    const r = decode({ logits }, { ...c, input }, t);
    expect(Array.from(r.segmentation!.classes)).toEqual([0, 1, 1, 0, 1, 0, 0, 1]);
    // the 4x2 canvas at scale 0.5 covers 8x4 frame pixels from the top-left; the canvas clips the rest
    expect(r.segmentation!.rect).toEqual({ x: 0, y: 0, width: 8, height: 4 });
    const higher = decode({ logits }, { ...c, input, options: { ...c.options, threshold: 1 } }, t);
    expect(Array.from(higher.segmentation!.classes)).toEqual([0, 0, 1, 0, 0, 0, 0, 1]);
  });

  it("auto mode still treats integer [1,1,H,W] output as class indices", () => {
    const out = { dims: [1, 1, 1, 3], data: new Float32Array([0, 2, 1]) };
    const t = computeTransform(3, 1, { ...DEFAULT_INPUT, width: 3, height: 1, resize: "stretch" });
    const r = decode({ out }, { decoder: "semantic-seg", input: { ...DEFAULT_INPUT, width: 3, height: 1 } }, t);
    expect(Array.from(r.segmentation!.classes)).toEqual([0, 2, 1]);
  });
});

describe("classify activation", () => {
  it("applies sigmoid and a score threshold", () => {
    const t = computeTransform(1, 1, { ...DEFAULT_INPUT, width: 1, height: 1 });
    const r = decode(
      { out: { dims: [1, 3], data: new Float32Array([0, 3, -3]) } },
      { decoder: "classify", input: DEFAULT_INPUT, options: { activation: "sigmoid", scoreThreshold: 0.4 } },
      t,
    );
    expect(r.classes!.map((c) => c.classId)).toEqual([1, 0]);
    expect(r.classes![1].score).toBeCloseTo(0.5);
  });
});
