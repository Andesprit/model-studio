import { nms, sigmoid, softmax, toFramePoint, toFrameRect } from "../geometry";
import { createTracker } from "../tracker";
import type { Decoder, DecoderUtils, ModelConfig, PipelineLib, Result, Tensor, Transform } from "../types";
import { classify } from "./classify";
import { semanticSeg } from "./semantic";
import { describeOutputs, yoloDetect, yoloPose, yoloSeg } from "./yolo";

export const utils: DecoderUtils = { nms, sigmoid, softmax, toFrameRect, toFramePoint };
export const lib: PipelineLib = { ...utils, createTracker };

const registry = new Map<string, Decoder>();
for (const d of [yoloDetect, yoloSeg, yoloPose, semanticSeg, classify]) registry.set(d.name, d);

export const builtinDecoderNames = () => [...registry.keys()].filter((n) => !n.startsWith("plugin:"));

/** Registers a plugin decoder under "plugin:<name>". Returns the registered key. */
export function registerPlugin(decoder: Decoder): string {
  if (!decoder || typeof decoder.decode !== "function" || !decoder.name) {
    throw new Error("Plugin must export { name: string, decode(outputs, ctx) }");
  }
  const key = `plugin:${decoder.name}`;
  registry.set(key, decoder);
  return key;
}

export function getDecoder(name: string): Decoder {
  const d = registry.get(name);
  if (!d) throw new Error(`Unknown decoder "${name}". Available: ${[...registry.keys()].join(", ")}`);
  return d;
}

/** A decoder failed on this model's outputs. The message says which decoder, the output shapes, and what to try. */
export class DecodeError extends Error {
  constructor(decoder: string, outputs: Record<string, Tensor>, cause: unknown) {
    const reason = (cause instanceof Error ? cause.message : String(cause)).replace(/\.$/, "");
    const shapes = describeOutputs(outputs);
    // Some decoder errors already list the outputs; don't repeat them.
    const listed = reason.includes(shapes) ? "" : ` Model outputs: ${shapes}.`;
    super(
      `${decoder} couldn't read this model's output: ${reason}.${listed} ` +
        "Pick another decoder in step 2, or load a Config .json or a Pipeline .js with its own decode.",
    );
    this.name = "DecodeError";
  }
}

/** Runs `fn`, turning any failure into a DecodeError for `decoder` (once; nested calls keep the innermost). */
export function explainDecode<T>(decoder: string, outputs: Record<string, Tensor>, fn: () => T): T {
  try {
    return fn();
  } catch (e) {
    throw e instanceof DecodeError ? e : new DecodeError(decoder, outputs, e);
  }
}

export function decode(outputs: Record<string, Tensor>, config: ModelConfig, transform: Transform): Result {
  const decoder = getDecoder(config.decoder);
  return explainDecode(`Decoder "${config.decoder}"`, outputs, () => runDecoder(decoder, outputs, config, transform, config.options));
}

/** Slot names for loaded models, in load order. */
export const modelKey = (index: number) => String.fromCharCode(65 + index);

/** Combines several models' results into one, so they can be drawn together. */
export function mergeResults(results: Result[]): Result {
  if (results.length === 1) return results[0];
  const all = <K extends "boxes" | "masks" | "poses" | "classes" | "overlays">(k: K) => {
    const items = results.flatMap((r) => (r[k] ?? []) as NonNullable<Result[K]>[number][]);
    return items.length ? (items as NonNullable<Result[K]>) : undefined;
  };
  const merged: Result = {
    boxes: all("boxes"),
    masks: all("masks"),
    poses: all("poses"),
    classes: all("classes"),
    overlays: all("overlays"),
    // Only one dense map can be shown; the first model's wins.
    segmentation: results.find((r) => r.segmentation)?.segmentation,
    skeleton: results.find((r) => r.skeleton)?.skeleton,
  };
  const metrics = Object.assign({}, ...results.map((r) => r.metrics ?? {}));
  if (Object.keys(metrics).length) merged.metrics = metrics;
  for (const k of Object.keys(merged) as (keyof Result)[]) if (merged[k] === undefined) delete merged[k];
  return merged;
}

/** Runs any decode function (built-in or a pipeline's) with `options` merged over its defaults. */
export function runDecoder(
  decoder: Pick<Decoder, "defaults" | "decode">,
  outputs: Record<string, Tensor>,
  config: ModelConfig,
  transform: Transform,
  options: Record<string, unknown> | undefined,
): Result {
  return decoder.decode(outputs, {
    config,
    transform,
    options: { ...decoder.defaults, ...options },
    labels: config.labels ?? [],
    utils,
    lib: {
      ...lib,
      decode: (name, o, opts) => {
        const d = getDecoder(name);
        return explainDecode(`Decoder "${name}"`, o, () => runDecoder(d, o, config, transform, opts));
      },
    },
  });
}
