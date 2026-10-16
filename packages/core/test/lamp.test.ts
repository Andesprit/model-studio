/**
 * The vortex lamp models (apps/web/public/models/vortex_lamp_auto_fp16.onnx and _fp32.onnx, the same graph in two
 * precisions) and their pipeline (pipelines/lamp.pipeline.js):
 * the embedded manifest and pipeline, and the pipeline's orientation choice, largest-piece filter and tilt warning on
 * synthetic logits. The model outputs both orientations: `logits` (frame seen turned, right for a vertical rope) and
 * `logits_asis` (frame as it is, right for a horizontal rope).
 */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { describe, expect, it } from "vitest";
import {
  computeTransform,
  configFromMetadata,
  definePipeline,
  PIPELINE_METADATA_KEY,
  PipelineRunner,
  readOnnxMetadata,
  type Result,
  type Tensor,
} from "../src";

const root = `${process.cwd()}/../..`;
const pipelineFile = `${root}/pipelines/lamp.pipeline.js`;
const modelMeta = (precision: string) =>
  readOnnxMetadata(new Uint8Array(readFileSync(`${root}/apps/web/public/models/vortex_lamp_auto_${precision}.onnx`)));
const config = configFromMetadata(modelMeta("fp16"))!;
const lamp = definePipeline((await import(pathToFileURL(pipelineFile).href)).default);

const H = 416;
const W = 232;
/** A phone frame held upright. */
const transform = computeTransform(720, 1280, config.input);

/** Logits of a rope: +4 within `half` cells of the segment through (cx, cy) at `deg` from vertical (clockwise), else -4. */
function rope(deg: number, cx = 116, cy = 200, length = 300, half = 10): Tensor {
  const data = new Float32Array(H * W).fill(-4);
  const a = (deg * Math.PI) / 180;
  const [dx, dy] = [Math.sin(a), -Math.cos(a)]; // along the rope, top end first
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const along = (x - cx) * dx + (y - cy) * dy;
      const across = Math.abs(-(x - cx) * dy + (y - cy) * dx);
      if (Math.abs(along) <= length / 2 && across <= half) data[y * W + x] = 4;
    }
  }
  return { dims: [1, 1, H, W], data };
}

/** A weak, scattered mask: isolated cells, as the wrong orientation gives. */
function scattered(): Tensor {
  const data = new Float32Array(H * W).fill(-4);
  for (let y = 4; y < H; y += 9) for (let x = 4; x < W; x += 9) data[y * W + x] = 0.5;
  return { dims: [1, 1, H, W], data };
}

/** The rope of `t` laid left to right (rows become columns, squeezed to the canvas width). */
const sideways = (t: Tensor): Tensor => {
  const data = new Float32Array(H * W).fill(-4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) if (t.data[y * W + x] > 0) data[(x + 90) * W + Math.min(W - 1, Math.round((y * W) / H))] = 4;
  }
  return { dims: [1, 1, H, W], data };
};

const placeholder = (): Tensor => ({ dims: [1, 1, 1, 1], data: new Float32Array(1) });

/**
 * Runs the pipeline as the app does, with the graph's If emulated: before each frame the pipeline's feeds() give
 * `mode` (none = zeros = 0); mode 1 replaces logits_asis by the placeholder, mode 2 replaces logits.
 * `times` are the frames' timestamps (default 0, 1/30, 2/30...).
 */
function play(frames: { logits: Tensor; logits_asis: Tensor }[], times = frames.map((_, i) => i / 30), runner = new PipelineRunner(lamp, {}, null)) {
  const modes: number[] = [];
  const results: Result[] = frames.map((full, i) => {
    const mode = runner.feeds(times[i]).mode?.data[0] ?? 0;
    modes.push(mode);
    const outputs = { logits: mode === 2 ? placeholder() : full.logits, logits_asis: mode === 1 ? placeholder() : full.logits_asis };
    return runner.run([{ outputs, config, transform }], times[i]);
  });
  return { results, modes, runner };
}

const run = (frames: { logits: Tensor; logits_asis: Tensor }[]): Result[] => play(frames).results;

describe.each(["fp16", "fp32"])("vortex lamp model, %s", (precision) => {
  const meta = modelMeta(precision);

  it("embeds a manifest that reads as a gray, portrait, top-left letterbox binary mask", () => {
    const c = configFromMetadata(meta)!;
    expect(c.decoder).toBe("semantic-seg");
    expect(c.input).toMatchObject({ width: W, height: H, channels: 1, resize: "letterbox", align: "top-left", padValue: 0 });
    expect(c.options).toMatchObject({ outputName: "logits", mode: "binary", threshold: 0 });
  });

  it("embeds pipelines/lamp.pipeline.js unchanged", () => {
    const lf = (s: string) => s.replace(/\r\n/g, "\n"); // a Windows checkout may have turned the file to CRLF
    expect(lf(meta[PIPELINE_METADATA_KEY])).toBe(lf(readFileSync(pipelineFile, "utf8")));
  });
});

describe("lamp pipeline", () => {
  it("decides on the first 10 frames and locks vertical for an upright rope", () => {
    const results = run(Array.from({ length: 12 }, () => ({ logits: rope(0), logits_asis: scattered() })));
    expect(results[0].metrics!["vortex orientation"]).toBe("vertical (deciding 1/10)");
    expect(results[11].metrics!["vortex orientation"]).toBe("vertical");
    expect(Math.abs(Number(results[11].metrics!["vortex tilt deg"]))).toBeLessThan(1);
    expect(results[11].metrics!["vortex tilt warning"]).toBeUndefined();
    expect(Number(results[11].metrics!["vortex thickness px"])).toBeGreaterThan(0);
    expect("lamp" in results[11]).toBe(false);
  });

  it("locks horizontal when the as-is output holds the rope", () => {
    const results = run(Array.from({ length: 12 }, () => ({ logits: scattered(), logits_asis: sideways(rope(0)) })));
    expect(results[11].metrics!["vortex orientation"]).toBe("horizontal");
    expect(Math.abs(Number(results[11].metrics!["vortex tilt deg"]))).toBeLessThan(1);
    expect(Number(results[11].metrics!["vortex present %"])).toBeGreaterThan(50);
  });

  it("asks for both branches until it locks, then only the locked one", () => {
    const upright = play(Array.from({ length: 14 }, () => ({ logits: rope(0), logits_asis: scattered() })));
    expect(upright.modes).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]);
    const last = upright.results[13];
    expect(last.metrics!["vortex orientation"]).toBe("vertical");
    expect(Number(last.metrics!["vortex thickness px"])).toBeGreaterThan(0);
    expect(last.segmentation!.classes.some((c) => c === 1)).toBe(true);
    const flat = play(Array.from({ length: 12 }, () => ({ logits: scattered(), logits_asis: sideways(rope(0)) })));
    expect(flat.modes.slice(10)).toEqual([2, 2]);
    expect(flat.results[11].metrics!["vortex orientation"]).toBe("horizontal");
  });

  it("decides again, with both branches, after a seek back", () => {
    const frames = Array.from({ length: 13 }, () => ({ logits: rope(0), logits_asis: scattered() }));
    const times = [...Array.from({ length: 12 }, (_, i) => i / 30), 0]; // the 13th frame seeks back to the start
    const { results, modes } = play(frames, times);
    expect(modes[11]).toBe(1);
    expect(modes[12]).toBe(0); // time went backwards: the runner starts fresh, so the graph runs both branches
    expect(results[12].metrics!["vortex orientation"]).toBe("vertical (deciding 1/10)");
  });

  it("keeps only the largest piece of the mask", () => {
    const logits = rope(0);
    for (let y = 380; y < 400; y++) for (let x = 10; x < 30; x++) logits.data[y * W + x] = 4; // a false piece near the base
    const seg = run([{ logits, logits_asis: scattered() }])[0].segmentation!;
    let stray = 0;
    for (let y = 380; y < 400; y++) for (let x = 10; x < 30; x++) stray += seg.classes[y * W + x];
    expect(stray).toBe(0);
  });

  it("measures the tilt and warns above 15 degrees", () => {
    const tilted = run(Array.from({ length: 12 }, () => ({ logits: rope(25), logits_asis: scattered() })))[11];
    expect(Number(tilted.metrics!["vortex tilt deg"])).toBeCloseTo(25, 0);
    expect(String(tilted.metrics!["vortex tilt warning"])).toContain("hold the phone straighter");
    expect(tilted.overlays!.some((o) => o.type === "text")).toBe(true);
    const slight = run(Array.from({ length: 12 }, () => ({ logits: rope(-10), logits_asis: scattered() })))[11];
    expect(Number(slight.metrics!["vortex tilt deg"])).toBeCloseTo(-10, 0);
    expect(slight.metrics!["vortex tilt warning"]).toBeUndefined();
  });
});
