import { describe, expect, it } from "vitest";
import {
  computeTransform,
  DecodeError,
  DEFAULT_INPUT,
  definePipeline,
  makeCalibration,
  PipelineRunner,
  type ModelConfig,
  type PipelineDefinition,
  type Tensor,
} from "../src";

const define = (p: PipelineDefinition) => definePipeline(p);
const t = computeTransform(640, 480, { ...DEFAULT_INPUT, width: 640, height: 640, resize: "letterbox" });
const config: ModelConfig = { decoder: "classify", input: { ...DEFAULT_INPUT }, labels: ["a", "b", "c"] };
const outputs: Record<string, Tensor> = { out: { dims: [1, 3], data: new Float32Array([0, 5, 1]) } };

/** Counts frames since creation, so resets are visible. */
const counter = define({
  name: "counter",
  create: () => {
    let n = 0;
    return { step: (r) => ({ ...r, metrics: { n: ++n } }) };
  },
});

describe("definePipeline", () => {
  it("rejects modules without a name or without decode/create", () => {
    expect(() => definePipeline(undefined)).toThrow(/name/);
    expect(() => definePipeline({ decode: () => ({}) })).toThrow(/name/);
    expect(() => definePipeline({ name: "x" })).toThrow(/decode.*create/);
  });

  it("accepts a plain decoder plugin", () => {
    expect(definePipeline({ name: "old", decode: () => ({}) }).name).toBe("old");
  });
});

describe("PipelineRunner", () => {
  it("falls back to the config's decoder when the pipeline has none", () => {
    const r = new PipelineRunner(counter).run(outputs, config, t, 0);
    expect(r.classes![0].label).toBe("b");
    expect(r.metrics).toEqual({ n: 1 });
  });

  it("uses the pipeline's decoder with its defaults under user options", () => {
    const p = define({
      name: "dec",
      defaults: { a: 1, b: 2 },
      decode: (_o, ctx) => ({ metrics: { a: ctx.options.a as number, b: ctx.options.b as number } }),
    });
    expect(new PipelineRunner(p, { b: 3 }).run(outputs, config, t, 0).metrics).toEqual({ a: 1, b: 3 });
  });

  it("keeps state across frames and starts fresh after reset", () => {
    const runner = new PipelineRunner(counter);
    runner.run(outputs, config, t, 0);
    expect(runner.run(outputs, config, t, 0.1).metrics).toEqual({ n: 2 });
    runner.reset();
    expect(runner.run(outputs, config, t, 0).metrics).toEqual({ n: 1 });
  });

  it("passes frame info to step", () => {
    const seen: unknown[] = [];
    const p = define({ name: "f", create: () => ({ step: (r, f) => (seen.push(f), r) }) });
    const runner = new PipelineRunner(p);
    runner.run(outputs, config, t, 1.5);
    runner.markDropped();
    runner.markDropped();
    runner.run(outputs, config, t, 2);
    expect(seen).toEqual([
      { timestamp: 1.5, index: 0, dropped: 0, width: 640, height: 480 },
      { timestamp: 2, index: 1, dropped: 2, width: 640, height: 480 },
    ]);
  });

  it("resets when time goes backwards but not when it stays or moves forward", () => {
    const runner = new PipelineRunner(counter);
    runner.run(outputs, config, t, 5);
    runner.run(outputs, config, t, 5); // paused frame re-run
    expect(runner.run(outputs, config, t, 9).metrics).toEqual({ n: 3 }); // seek forward
    expect(runner.run(outputs, config, t, 0).metrics).toEqual({ n: 1 }); // loop
  });

  it("lets a pipeline decoder reuse a built-in decoder and the tracker through ctx.lib", () => {
    const p = define({
      name: "lib",
      decode: (o, ctx) => ({ ...ctx.lib.decode("classify", o, { topK: 1 }), metrics: { tracker: typeof ctx.lib.createTracker } }),
      create: (ctx) => {
        const tracker = ctx.lib.createTracker();
        return { step: (r) => ({ ...r, boxes: tracker.update([{ x: 0, y: 0, width: 1, height: 1, classId: 0, score: 1 }]) }) };
      },
    });
    const r = new PipelineRunner(p).run(outputs, config, t, 0);
    expect(r.classes).toEqual(new PipelineRunner(counter).run(outputs, { ...config, options: { topK: 1 } }, t, 0).classes);
    expect(r.metrics).toEqual({ tracker: "function" });
    expect(r.boxes![0].trackId).toBe(1);
  });

  it("hands the calibration to create", () => {
    const p = define({ name: "cal", create: (ctx) => ({ step: (r) => ({ ...r, metrics: { upp: ctx.calibration?.unitsPerPixel ?? -1 } }) }) });
    expect(new PipelineRunner(p, {}, makeCalibration(90, 180, "cm")).run(outputs, config, t, 0).metrics).toEqual({ upp: 2 });
    expect(new PipelineRunner(p).run(outputs, config, t, 0).metrics).toEqual({ upp: -1 });
  });

  it("makeCalibration needs both lengths", () => {
    expect(makeCalibration(0, 180, "cm")).toBeNull();
    expect(makeCalibration(90, 0, "cm")).toBeNull();
    expect(makeCalibration(90, 180, "cm")).toEqual({ pixels: 90, amount: 180, unit: "cm", unitsPerPixel: 2 });
  });

  it("explains decode failures with the decoder, the output shapes and what to try", () => {
    const run = (cfg: ModelConfig, p = counter) => () => new PipelineRunner(p).run(outputs, cfg, t, 0);
    // yolo-detect needs a 3D output; this model gives [1, 3].
    expect(run({ ...config, decoder: "yolo-detect" })).toThrow(DecodeError);
    expect(run({ ...config, decoder: "yolo-detect" })).toThrow(/Decoder "yolo-detect" couldn't read.*3D.*out \[1, 3\].*Pick another decoder/);
    expect(run({ ...config, decoder: "yolo-detect" })).not.toThrow(/Model outputs/); // already listed by the decoder
    const own = define({ name: "own", decode: () => { throw new Error("bad tensor"); } });
    expect(run(config, own)).toThrow(/Pipeline "own"'s decode couldn't read this model's output: bad tensor/);
    // A built-in decoder called by the pipeline keeps its own name in the message.
    const viaLib = define({ name: "lib", decode: (o, ctx) => ctx.lib.decode("yolo-detect", o) });
    expect(run(config, viaLib)).toThrow(/^Decoder "yolo-detect" couldn't read/);
  });

  it("names the pipeline and function when user code throws", () => {
    const p = define({ name: "boom", create: () => ({ step: () => { throw new Error("oops"); } }) });
    expect(() => new PipelineRunner(p).run(outputs, config, t, 0)).toThrow('Pipeline "boom" failed in step: oops');
  });

  it("summarizes numeric metrics over distinct frames and adds the pipeline's own summary", () => {
    const p = define({
      name: "sum",
      create: () => {
        let n = 0;
        return {
          step: (r) => ({ ...r, metrics: { count: [2, 4, 6][n++ % 3], label: "text is skipped" } }),
          summary: () => ({ calls: n }),
        };
      },
    });
    const runner = new PipelineRunner(p);
    expect(runner.summary()).toEqual({ from: null, to: null, framesAnalyzed: 0, framesDropped: 0, metrics: {} });
    runner.run(outputs, config, t, 1);
    runner.run(outputs, config, t, 2);
    runner.run(outputs, config, t, 2); // paused re-run: not counted (metric 6 is ignored)
    runner.markDropped();
    expect(runner.summary()).toEqual({
      from: 1,
      to: 2,
      framesAnalyzed: 2,
      framesDropped: 1,
      metrics: { count: { mean: 3, min: 2, max: 4, last: 4, frames: 2 } },
      pipelineResults: { calls: 3 },
    });
    runner.run(outputs, config, t, 0); // loop: starts over
    expect(runner.summary()).toMatchObject({ from: 0, to: 0, framesAnalyzed: 1, metrics: { count: { frames: 1 } } });
  });

  it("reports a create that returns no step", () => {
    const p = define({ name: "bad", create: () => ({}) as never });
    expect(() => new PipelineRunner(p).run(outputs, config, t, 0)).toThrow(/step/);
  });
});
