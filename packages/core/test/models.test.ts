/**
 * End-to-end check against real Ultralytics ONNX exports (run `models/` export first; skipped otherwise).
 * Expected values come from Ultralytics' own predictions on bus.jpg (810x1080).
 */
import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import * as ort from "onnxruntime-web";
import { configFromMetadata, decode, readOnnxMetadata, resizeFrame, toTensor, type Tensor } from "../src";

const dir = `${process.cwd()}/../../models/`;
const available = existsSync(`${dir}yolov8n.onnx`) && existsSync(`${dir}bus.rgba`);

async function run(model: string) {
  const bytes = new Uint8Array(readFileSync(dir + model));
  const config = configFromMetadata(readOnnxMetadata(bytes))!;
  const frame = { width: 810, height: 1080, data: new Uint8ClampedArray(readFileSync(`${dir}bus.rgba`)) };
  const { frame: resized, transform } = resizeFrame(frame, config.input);
  const input = toTensor(resized, config.input);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input.data, input.dims) });
  const outputs: Record<string, Tensor> = {};
  for (const n of session.outputNames) outputs[n] = { dims: [...res[n].dims], data: res[n].data as Float32Array };
  return decode(outputs, config, transform);
}

const near = (box: { x: number; y: number; width: number; height: number }, ref: number[]) =>
  Math.abs(box.x - ref[0]) < 15 && Math.abs(box.y - ref[1]) < 15 &&
  Math.abs(box.x + box.width - ref[2]) < 15 && Math.abs(box.y + box.height - ref[3]) < 15;

describe.skipIf(!available)("real models", () => {
  it("yolov8n detect matches Ultralytics", async () => {
    const r = await run("yolov8n.onnx");
    const top = r.boxes!.filter((b) => b.score > 0.5);
    expect(top.map((b) => b.label).sort()).toEqual(["bus", "person", "person", "person"]);
    expect(top.some((b) => b.label === "bus" && near(b, [31, 230, 801, 775]))).toBe(true);
    expect(top.some((b) => near(b, [670, 380, 809, 879]))).toBe(true);
  }, 60000);

  it("yolov8n-seg produces masks of plausible area", async () => {
    const r = await run("yolov8n-seg.onnx");
    const i = r.boxes!.findIndex((b) => b.label === "bus");
    const m = r.masks![i];
    const frac = m.data.reduce((s, v) => s + (v ? 1 : 0), 0) / m.data.length;
    // Ultralytics: bus mask is ~89.8k px at 640 input scale, where the box is ~461x300 px (~65%)
    expect(frac).toBeGreaterThan(0.55);
    expect(frac).toBeLessThan(0.75);
  }, 60000);

  it("yolov8n-pose finds 3 people with keypoints", async () => {
    const r = await run("yolov8n-pose.onnx");
    const people = r.poses!.filter((p) => p.box.score > 0.5);
    expect(people).toHaveLength(3);
    expect(people.every((p) => p.keypoints.length === 17)).toBe(true);
    const nose = people.find((p) => near(p.box, [48, 396, 244, 905]))!.keypoints[0];
    expect(nose.x).toBeGreaterThan(48);
    expect(nose.x).toBeLessThan(244);
  }, 60000);

  it("yolov8n-cls ranks a bus-like class first", async () => {
    const r = await run("yolov8n-cls.onnx");
    expect(["minibus", "police_van", "trolleybus", "school_bus"]).toContain(r.classes![0].label);
  }, 60000);
});
