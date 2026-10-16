/**
 * The vortex lamp model (apps/web/public/models/vortex_lamp_auto_fp32.onnx) and its pipeline (pipelines/lamp.pipeline.js):
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
const meta = readOnnxMetadata(new Uint8Array(readFileSync(`${root}/apps/web/public/models/vortex_lamp_auto_fp32.onnx`)));
const config = configFromMetadata(meta)!;
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

function run(frames: { logits: Tensor; logits_asis: Tensor }[]): Result[] {
  const runner = new PipelineRunner(lamp, {}, null);
  return frames.map((outputs, i) => runner.run([{ outputs, config, transform }], i / 30));
}

describe("vortex lamp model", () => {
  it("embeds a manifest that reads as a gray, portrait, top-left letterbox binary mask", () => {
    expect(config.decoder).toBe("semantic-seg");
    expect(config.input).toMatchObject({ width: W, height: H, channels: 1, resize: "letterbox", align: "top-left", padValue: 0 });
    expect(config.options).toMatchObject({ outputName: "logits", mode: "binary", threshold: 0 });
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
