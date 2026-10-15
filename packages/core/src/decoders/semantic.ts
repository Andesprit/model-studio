import type { Decoder, Tensor } from "../types";
import { describeOutputs } from "./yolo";

type Mode = "auto" | "argmax" | "indices" | "binary";

/**
 * Dense segmentation. Modes:
 * - argmax:  logits/probabilities [1, C, H, W] ("NCHW") or [1, H, W, C] ("NHWC")
 * - indices: class indices [1, H, W] or [1, 1, H, W]
 * - binary:  single-channel logits [1, 1, H, W]; foreground (class 1) where value > threshold
 * - auto:    3D -> indices; one channel -> indices if all values are integers, else binary; otherwise argmax
 * The map covers the whole model input canvas and is mapped back onto the frame.
 */
export const semanticSeg: Decoder = {
  name: "semantic-seg",
  defaults: { outputName: "", layout: "NCHW", mode: "auto", threshold: 0, backgroundClass: 0 },
  ranges: { threshold: [-5, 5] },
  decode(outputs, ctx) {
    const name = String(ctx.options.outputName);
    const out = name ? outputs[name] : Object.values(outputs)[0];
    if (!out) throw new Error(`Output not found. Outputs: ${describeOutputs(outputs)}`);
    const { dims, data } = out;
    if (dims.length !== 3 && dims.length !== 4) {
      throw new Error(`semantic-seg expects a 3D or 4D output, got [${dims.join(", ")}]`);
    }

    const nhwc = ctx.options.layout === "NHWC";
    const [C, h, w] =
      dims.length === 3 ? [1, dims[1], dims[2]] : nhwc ? [dims[3], dims[1], dims[2]] : [dims[1], dims[2], dims[3]];
    const mode = resolveMode(ctx.options.mode as Mode, out, C);

    const plane = h * w;
    const classes = new Uint16Array(plane);
    if (mode === "indices") {
      for (let i = 0; i < plane; i++) classes[i] = data[i];
    } else if (mode === "binary") {
      const threshold = Number(ctx.options.threshold);
      for (let i = 0; i < plane; i++) classes[i] = data[i] > threshold ? 1 : 0;
    } else {
      for (let i = 0; i < plane; i++) {
        let best = 0;
        let bestV = -Infinity;
        for (let c = 0; c < C; c++) {
          const v = nhwc ? data[i * C + c] : data[c * plane + i];
          if (v > bestV) {
            bestV = v;
            best = c;
          }
        }
        classes[i] = best;
      }
    }

    const { width: iw, height: ih } = ctx.config.input;
    const tl = ctx.utils.toFramePoint(0, 0, ctx.transform);
    const br = ctx.utils.toFramePoint(iw, ih, ctx.transform);
    return {
      segmentation: {
        width: w,
        height: h,
        classes,
        rect: { x: tl.x, y: tl.y, width: br.x - tl.x, height: br.y - tl.y },
        background: Number(ctx.options.backgroundClass),
      },
    };
  },
};

function resolveMode(mode: Mode, out: Tensor, channels: number): Exclude<Mode, "auto"> {
  if (mode !== "auto") return mode;
  if (out.dims.length === 3) return "indices";
  if (channels > 1) return "argmax";
  return out.data.every(Number.isInteger) ? "indices" : "binary";
}
