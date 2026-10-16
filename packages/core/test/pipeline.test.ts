import { describe, expect, it } from "vitest";
import {
  computeTransform,
  decodeAll,
  DecodeError,
  DEFAULT_INPUT,
  definePipeline,
  makeCalibration,
  mergeResults,
  PipelineRunner,
  type ModelConfig,
  type PipelineDefinition,
  type Result,
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
    const r = new PipelineRunner(counter).run([{ outputs, config: config, transform: t }], 0);
    expect(r.classes![0].label).toBe("b");
    expect(r.metrics).toEqual({ n: 1 });
  });

  it("uses the pipeline's decoder with its defaults under user options", () => {
    const p = define({
      name: "dec",
      defaults: { a: 1, b: 2 },
      decode: (_o, ctx) => ({ metrics: { a: ctx.options.a as number, b: ctx.options.b as number } }),
    });
    expect(new PipelineRunner(p, { b: 3 }).run([{ outputs, config: config, transform: t }], 0).metrics).toEqual({ a: 1, b: 3 });
  });

  it("keeps state across frames and starts fresh after reset", () => {
    const runner = new PipelineRunner(counter);
    runner.run([{ outputs, config: config, transform: t }], 0);
    expect(runner.run([{ outputs, config: config, transform: t }], 0.1).metrics).toEqual({ n: 2 });
    runner.reset();
    expect(runner.run([{ outputs, config: config, transform: t }], 0).metrics).toEqual({ n: 1 });
  });

  it("passes frame info to step", () => {
    const seen: unknown[] = [];
    const p = define({ name: "f", create: () => ({ step: (r, f) => (seen.push(f), r) }) });
    const runner = new PipelineRunner(p);
    runner.run([{ outputs, config: config, transform: t }], 1.5);
    runner.markDropped();
    runner.markDropped();
    runner.run([{ outputs, config: config, transform: t }], 2);
    expect(seen).toEqual([
      { timestamp: 1.5, index: 0, dropped: 0, width: 640, height: 480 },
      { timestamp: 2, index: 1, dropped: 2, width: 640, height: 480 },
    ]);
  });

  it("asks the instance for the next frame's extra model inputs, and gives none before it exists or on a rewind", () => {
    const p = define({
      name: "feeder",
      create: () => {
        let n = 0;
        return { step: (r) => (n++, r), feeds: () => ({ mode: { dims: [1], data: new Float32Array([n]) } }) };
      },
    });
    const runner = new PipelineRunner(p);
    expect(runner.feeds(0)).toEqual({});
    runner.run([{ outputs, config: config, transform: t }], 0.2);
    expect(runner.feeds(0.3).mode.data[0]).toBe(1);
    expect(runner.feeds(0.2).mode.data[0]).toBe(1); // a paused frame re-run keeps the instance
    expect(runner.feeds(0.1)).toEqual({}); // time went backwards: run() will start a fresh instance
    expect(new PipelineRunner(counter).feeds(0)).toEqual({}); // a pipeline without feeds()
  });

  it("resets when time goes backwards but not when it stays or moves forward", () => {
    const runner = new PipelineRunner(counter);
    runner.run([{ outputs, config: config, transform: t }], 5);
    runner.run([{ outputs, config: config, transform: t }], 5); // paused frame re-run
    expect(runner.run([{ outputs, config: config, transform: t }], 9).metrics).toEqual({ n: 3 }); // seek forward
    expect(runner.run([{ outputs, config: config, transform: t }], 0).metrics).toEqual({ n: 1 }); // loop
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
    const r = new PipelineRunner(p).run([{ outputs, config: config, transform: t }], 0);
    expect(r.classes).toEqual(new PipelineRunner(counter).run([{ outputs, config: { ...config, options: { topK: 1 } }, transform: t }], 0).classes);
    expect(r.metrics).toEqual({ tracker: "function" });
    expect(r.boxes![0].trackId).toBe(1);
  });

  it("hands the calibration to create", () => {
    const p = define({ name: "cal", create: (ctx) => ({ step: (r) => ({ ...r, metrics: { upp: ctx.calibration?.unitsPerPixel ?? -1 } }) }) });
    expect(new PipelineRunner(p, {}, makeCalibration(90, 180, "cm")).run([{ outputs, config: config, transform: t }], 0).metrics).toEqual({ upp: 2 });
    expect(new PipelineRunner(p).run([{ outputs, config: config, transform: t }], 0).metrics).toEqual({ upp: -1 });
  });

  it("makeCalibration needs both lengths", () => {
    expect(makeCalibration(0, 180, "cm")).toBeNull();
    expect(makeCalibration(90, 0, "cm")).toBeNull();
    expect(makeCalibration(90, 180, "cm")).toEqual({ pixels: 90, amount: 180, unit: "cm", unitsPerPixel: 2 });
  });

  it("explains decode failures with the decoder, the output shapes and what to try", () => {
    const run = (cfg: ModelConfig, p = counter) => () => new PipelineRunner(p).run([{ outputs, config: cfg, transform: t }], 0);
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
    expect(() => new PipelineRunner(p).run([{ outputs, config: config, transform: t }], 0)).toThrow('Pipeline "boom" failed in step: oops');
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
    expect(runner.summary()).toEqual({ from: null, to: null, framesAnalyzed: 0, framesDropped: 0, metrics: {}, definitions: {} });
    runner.run([{ outputs, config: config, transform: t }], 1);
    runner.run([{ outputs, config: config, transform: t }], 2);
    runner.run([{ outputs, config: config, transform: t }], 2); // paused re-run: not counted (metric 6 is ignored)
    runner.markDropped();
    expect(runner.summary()).toEqual({
      from: 1,
      to: 2,
      framesAnalyzed: 2,
      framesDropped: 1,
      metrics: { count: { mean: 3, min: 2, max: 4, last: 4, frames: 2 } },
      definitions: {},
      pipelineResults: { calls: 3 },
    });
    runner.run([{ outputs, config: config, transform: t }], 0); // loop: starts over
    expect(runner.summary()).toMatchObject({ from: 0, to: 0, framesAnalyzed: 1, metrics: { count: { frames: 1 } } });
  });

  it("describes each metric that appeared, matching names by their start", () => {
    const p = define({
      name: "desc",
      describe: { particles: "anything about particles", "particles mean speed": "average speed", unused: "never shown" },
      create: () => ({ step: (r) => ({ ...r, metrics: { "particles mean speed cm/s": 1, "particles in frame": 2, other: 3 } }) }),
    });
    const runner = new PipelineRunner(p);
    runner.run([{ outputs, config, transform: t }], 0);
    expect(runner.summary().definitions).toEqual({
      "particles mean speed cm/s": "average speed",
      "particles in frame": "anything about particles",
    });
  });

  describe("with two models", () => {
    const a = { outputs, config, transform: t };
    const b = { outputs: { out: { dims: [1, 3], data: new Float32Array([9, 0, 0]) } }, config, transform: t };

    it("merges both results and hands each one to step by model key", () => {
      let seen: Record<string, Result> = {};
      const p = define({ name: "two", create: () => ({ step: (r, _f, models) => ((seen = models), r) }) });
      const r = new PipelineRunner(p).run([a, b], 0);
      expect(r.classes!.map((c) => c.label)).toEqual(["b", "c", "a", "a", "b", "c"]); // A's top 3, then B's
      expect(seen.A.classes![0].label).toBe("b");
      expect(seen.B.classes![0].label).toBe("a");
      expect(decodeAll([a, b])).toEqual(r);
    });

    it("applies the pipeline's decode to model A only", () => {
      const p = define({ name: "decA", decode: () => ({ metrics: { mine: 1 } }), create: () => ({ step: (r, _f, models) => ({ ...r, overlays: [], metrics: { ...r.metrics, bHasClasses: models.B.classes ? 1 : 0 } }) }) });
      expect(new PipelineRunner(p).run([a, b], 0).metrics).toEqual({ mine: 1, bHasClasses: 1 });
    });

    it("says which model failed to decode", () => {
      const bad = { ...b, config: { ...config, decoder: "yolo-detect" } };
      expect(() => decodeAll([a, bad])).toThrow(/^Model B: Decoder "yolo-detect" couldn't read/);
    });
  });

  it("mergeResults keeps the first segmentation and skeleton and leaves out empty fields", () => {
    const seg = { width: 1, height: 1, classes: new Uint16Array(1), rect: { x: 0, y: 0, width: 1, height: 1 }, background: 0 };
    const box = { x: 0, y: 0, width: 1, height: 1, classId: 0, score: 1 };
    expect(mergeResults([{ segmentation: seg }, { boxes: [box], segmentation: { ...seg, background: 9 } }])).toEqual({ segmentation: seg, boxes: [box] });
  });

  it("runDecoded works on results decoded elsewhere and skips its own decode", () => {
    const p = define({
      name: "on-top",
      decode: () => { throw new Error("must not decode"); },
      create: () => ({ step: (r, _f, models) => ({ ...r, metrics: { boxes: r.boxes?.length ?? 0, models: Object.keys(models).length } }) }),
    });
    const box = { x: 0, y: 0, width: 1, height: 1, classId: 0, score: 1 };
    const runner = new PipelineRunner(p);
    const r = runner.runDecoded([{ boxes: [box] }, { boxes: [box, box] }], [{ outputs, config, transform: t }, { outputs, config, transform: t }], 0);
    expect(r.metrics).toEqual({ boxes: 3, models: 2 });
    expect(runner.summary().metrics.boxes).toMatchObject({ mean: 3, frames: 1 });
  });

  it("reports a create that returns no step", () => {
    const p = define({ name: "bad", create: () => ({}) as never });
    expect(() => new PipelineRunner(p).run([{ outputs, config: config, transform: t }], 0)).toThrow(/step/);
  });
});
