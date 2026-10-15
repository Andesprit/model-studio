import type { ClassScore, Decoder } from "../types";
import { describeOutputs } from "./yolo";

export const classify: Decoder = {
  name: "classify",
  /**
   * activation: "auto" applies softmax when the output doesn't already look like probabilities;
   * "softmax" / "sigmoid" (multi-label) / "none" force it.
   */
  defaults: { outputName: "", topK: 5, activation: "auto", scoreThreshold: 0 },
  decode(outputs, ctx) {
    const name = String(ctx.options.outputName);
    const out = name ? outputs[name] : Object.values(outputs)[0];
    if (!out) throw new Error(`Output not found. Outputs: ${describeOutputs(outputs)}`);
    let probs = out.data;
    const activation = ctx.options.activation;
    if (activation === "softmax" || (activation === "auto" && !looksLikeProbabilities(probs))) {
      probs = ctx.utils.softmax(probs);
    } else if (activation === "sigmoid") {
      probs = probs.map(ctx.utils.sigmoid);
    }

    const threshold = Number(ctx.options.scoreThreshold);
    const scores: ClassScore[] = [];
    probs.forEach((score, classId) => {
      if (score >= threshold) scores.push({ classId, score, label: ctx.labels[classId] ?? String(classId) });
    });
    scores.sort((a, b) => b.score - a.score);
    return { classes: scores.slice(0, Number(ctx.options.topK)) };
  },
};

function looksLikeProbabilities(values: Float32Array) {
  let sum = 0;
  for (const v of values) {
    if (v < 0 || v > 1) return false;
    sum += v;
  }
  return Math.abs(sum - 1) < 0.01;
}
