/** Drop into model-studio/packages/core/test; exercise the delivered two-stage lamp contract. */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";
import {
  computeTransform, configFromMetadata, definePipeline, PIPELINE_METADATA_KEY,
  PipelineRunner, readOnnxMetadata, type Result, type Tensor,
} from "../src";

const root = `${process.cwd()}/../..`;
const pipelineFile = `${root}/pipelines/lamp.pipeline.js`;
const modelBytes = (precision: string) =>
  new Uint8Array(readFileSync(`${root}/apps/web/public/models/vortex_lamp_auto_${precision}.onnx`));
const config = configFromMetadata(readOnnxMetadata(modelBytes("fp16")))!;
const lamp = definePipeline((await import(pathToFileURL(pipelineFile).href)).default);
const H = 416, W = 232, OUTSIDE = -10000;
const transform = computeTransform(720, 1280, config.input);
const tensor = (data: Float32Array): Tensor => ({ dims: [1, 1, H, W], data });
const blank = () => tensor(new Float32Array(H * W).fill(OUTSIDE));
const placeholder = (): Tensor => ({ dims: [1, 1, 1, 1], data: new Float32Array(1) });

/** Tube and rope share an axis; negative rope logits inside water must still count as water. */
function tube(deg: number, ropeHalf = 8): Tensor {
  const t = blank();
  const a = deg * Math.PI / 180;
  const dx = Math.sin(a), dy = -Math.cos(a);
  const length = Math.abs(deg) === 90 ? 190 : 320;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const along = (x - 116) * dx + (y - 208) * dy;
    const across = Math.abs(-(x - 116) * dy + (y - 208) * dx);
    if (Math.abs(along) <= length / 2 && across <= 35)
      t.data[y * W + x] = across <= ropeHalf ? 4 : -4;
  }
  return t;
}

const pair = (logits: Tensor, logits_asis = logits) => ({ logits, logits_asis });

/** Emulate runtime zero-fill and If placeholders, including PipelineRunner's bootstrap and rewind. */
function play(frames: ReturnType<typeof pair>[], times = frames.map((_, i) => i / 30)) {
  const runner = new PipelineRunner(lamp, {}, null);
  const modes: number[] = [];
  const results: Result[] = frames.map((full, i) => {
    const feed = runner.feeds(times[i]).mode;
    if (feed) {
      expect(feed.dims).toEqual([1]);
      expect(feed.data).toBeInstanceOf(Float32Array);
    }
    const mode = feed?.data[0] ?? 0;
    modes.push(mode);
    return runner.run([{ outputs: {
      logits: mode === 2 ? placeholder() : full.logits,
      logits_asis: mode === 1 ? placeholder() : full.logits_asis,
    }, config, transform }], times[i]);
  });
  return { results, modes, runner };
}

const repeat = (t: Tensor, n = 12) => Array.from({ length: n }, () => pair(t));

describe.each(["fp16"])("vortex lamp model, %s", (precision) => {
  const bytes = modelBytes(precision);
  const meta = readOnnxMetadata(bytes);
  it("embeds a gray portrait top-left letterbox manifest and the delivered pipeline", () => {
    const c = configFromMetadata(meta)!;
    expect(c.decoder).toBe("semantic-seg");
    expect(c.input).toMatchObject({ width: W, height: H, channels: 1, resize: "letterbox", align: "top-left", padValue: 0 });
    expect(c.options).toMatchObject({ outputName: "logits", mode: "binary", threshold: 0 });
    const lf = (s: string) => s.replace(/\r\n/g, "\n");
    expect(lf(meta[PIPELINE_METADATA_KEY])).toBe(lf(readFileSync(pipelineFile, "utf8")));
  });

  it("has exactly two float32 rope outputs with the exact water marker and unused-branch placeholder", async () => {
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    try {
      expect(session.inputNames).toEqual(["image", "mode"]);
      expect(session.outputNames).toEqual(["logits", "logits_asis"]);
      for (const mode of [0, 1, 2]) {
        const outputs = await session.run({
          image: new ort.Tensor("float32", new Float32Array(H * W), [1, 1, H, W]),
          mode: new ort.Tensor("float32", new Float32Array([mode]), [1]),
        });
        for (const [name, t] of Object.entries(outputs)) {
          expect(t.type).toBe("float32");
          if ((mode === 1 && name === "logits_asis") || (mode === 2 && name === "logits")) {
            expect(t.dims).toEqual([1, 1, 1, 1]);
            expect(Array.from(t.data as Float32Array)).toEqual([0]);
          } else {
            expect(t.dims).toEqual([1, 1, H, W]);
            expect((t.data as Float32Array).includes(OUTSIDE)).toBe(true);
          }
          t.dispose();
        }
      }
    } finally { await session.release(); }
  }, 60000);
});

describe("lamp pipeline", () => {
  it("locks vertical using water even when the only rope pixels run horizontally", () => {
    const t = tube(0, -1); // no positive rope pixels
    for (let y = 200; y < 210; y++) for (let x = 90; x < 140; x++) t.data[y * W + x] = 4;
    const { results, modes } = play(repeat(t));
    expect(modes).toEqual([0, ...Array(11).fill(1)]);
    expect(results[0].metrics!["vortex orientation"]).toBe("vertical (deciding 1/10)");
    expect(results[11].metrics!["vortex orientation"]).toBe("vertical");
    expect(results[11].metrics!["water found"]).toBe("yes");
    expect("lamp" in results[11]).toBe(false);
  });

  it("switches to the horizontal branch from water evidence before locking", () => {
    const wrongRope = tube(90, -1);
    for (let y = 180; y < 235; y++) for (let x = 110; x < 120; x++) wrongRope.data[y * W + x] = 4;
    const frames = Array.from({ length: 12 }, () => pair(wrongRope, tube(90)));
    const { results, modes } = play(frames);
    expect(modes).toEqual([0, ...Array(11).fill(2)]);
    expect(results[0].metrics!["vortex orientation"]).toBe("horizontal (deciding 1/10)");
    expect(results[11].metrics!["vortex orientation"]).toBe("horizontal");
    expect(Math.abs(Number(results[11].metrics!["vortex tilt deg"]))).toBeLessThan(1);
    expect(Number(results[11].metrics!["vortex thickness px"])).toBeGreaterThan(0);
  });

  it("bootstraps both branches again after rewind, then requests one branch", () => {
    const times = [...Array.from({ length: 12 }, (_, i) => i / 30), 0, 1 / 30];
    const { results, modes } = play(repeat(tube(0), 14), times);
    expect(modes).toEqual([0, ...Array(11).fill(1), 0, 1]);
    expect(results[12].metrics!["vortex orientation"]).toBe("vertical (deciding 1/10)");
  });

  it.each(["blank", "tiny"])("ignores %s water then locks after ten distinct sideways water frames", (kind) => {
    const empty = blank();
    if (kind === "tiny") empty.data[80 * W + 80] = -4;
    const frames = [...repeat(empty, 15), ...repeat(tube(90), 10)];
    const { results, modes } = play(frames);
    expect(modes.slice(0, 16)).toEqual([0, ...Array(15).fill(1)]);
    expect(modes.slice(16)).toEqual(Array(9).fill(2));
    expect(results[14].metrics!["vortex orientation"]).toBe("vertical (deciding 0/10)");
    expect(results[23].metrics!["vortex orientation"]).toContain("deciding 9/10");
    expect(results[24].metrics!["vortex orientation"]).toBe("horizontal");
    expect(results[15].segmentation).toBeUndefined(); // wait for the requested branch
    expect(results[24].metrics!["lamp warning"]).toBeUndefined();
  });

  it("warns after five distinct no-water frames and clears on recovery", () => {
    const frames = [...repeat(blank(), 5), ...repeat(tube(0), 1)];
    const times = [0, 1 / 30, 2 / 30, 3 / 30, 4 / 30, 5 / 30];
    const { results } = play(frames, times);
    expect(results.slice(0, 4).every(r => r.metrics!["lamp warning"] === undefined)).toBe(true);
    expect(results[4].metrics!["lamp warning"]).toBe("no lamp in view");
    expect(results[4].overlays!.some(o => o.type === "text")).toBe(true);
    expect(results[5].metrics!["lamp warning"]).toBeUndefined();
    expect(results[5].metrics!["water found"]).toBe("yes");
    const paused = play(repeat(blank(), 8), [0, 1, 2, 3, 3, 3, 3, 4]);
    expect(paused.results[6].metrics!["lamp warning"]).toBeUndefined();
    expect(paused.results[7].metrics!["lamp warning"]).toBe("no lamp in view");
    const repeated = play(repeat(tube(90), 12), Array(12).fill(0));
    expect(repeated.results[11].metrics!["vortex orientation"]).toContain("deciding 1/10");
  });

  it("reads exact water markers including negative in-water logits, and ignores placeholders", () => {
    const t = tube(0, -1);
    t.data[200 * W + 116] = -9999; // a negative rope logit is still water
    const { results } = play(repeat(t, 2));
    for (const r of results) {
      expect(r.metrics!["water found"]).toBe("yes");
      expect(r.overlays!.some(o => o.type === "polyline" && o.color === "#4ade80")).toBe(true);
      expect(r.segmentation!.classes.some(v => v !== 0)).toBe(false);
    }
  });

  it("keeps only the largest rope piece, with all displayed pixels inside water", () => {
    const t = tube(0);
    for (let y = 320; y < 326; y++) for (let x = 85; x < 91; x++) t.data[y * W + x] = 4;
    const seg = play([pair(t)]).results[0].segmentation!;
    for (let y = 320; y < 326; y++) for (let x = 85; x < 91; x++) expect(seg.classes[y * W + x]).toBe(0);
    expect(seg.classes.some(v => v === 1)).toBe(true);
    expect(seg.classes.every((v, i) => !v || t.data[i] !== OUTSIDE)).toBe(true);
  });

  it("measures rope tilt and warns above 15 degrees", () => {
    const tilted = play(repeat(tube(25))).results[11];
    expect(Number(tilted.metrics!["vortex tilt deg"])).toBeCloseTo(25, 0);
    expect(String(tilted.metrics!["vortex tilt warning"])).toContain("hold the phone straighter");
    expect(tilted.overlays!.some(o => o.type === "text")).toBe(true);
    const slight = play(repeat(tube(-10))).results[11];
    expect(Number(slight.metrics!["vortex tilt deg"])).toBeCloseTo(-10, 0);
    expect(slight.metrics!["vortex tilt warning"]).toBeUndefined();
  });
});
