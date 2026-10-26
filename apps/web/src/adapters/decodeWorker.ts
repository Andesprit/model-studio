import { ALL_FORMATS, BlobSource, Input, VideoSampleSink, type VideoSample } from "mediabunny";
import { computeTransform, PAD_VALUE, toTensor, type InputConfig, type Tensor, type Transform } from "@detect/core";
import { turnedSize } from "./frameGrabber";

/**
 * Analyze first's frames, decoded in order in this worker (WebCodecs, through mediabunny) instead of seeking the page's
 * video: each packet is decoded at most once, and the frame shown at each wanted time is turned into every model's
 * input here, while the page runs the models. The page asks for frames by sending credit, so at most `ahead`
 * prepared frames wait at a time.
 */
export type DecodeRequest =
  | { type: "start"; file: Blob; times: number[]; models: { input: InputConfig; turn: number }[]; ahead: number }
  | { type: "credit"; n: number }
  | { type: "stop" };
export type DecodedInput = { tensor: Tensor; transform: Transform; turn: number };
export type DecodeReply =
  | { type: "ready"; width: number; height: number }
  | { type: "unsupported"; reason: string }
  | { type: "frame"; index: number; inputs: DecodedInput[]; decodeMs: number; preMs: number }
  | { type: "done" }
  | { type: "error"; message: string };

let credit = 0;
let stopped = false;
let wake: (() => void) | null = null;
const post = (reply: DecodeReply, transfer: Transferable[] = []) => self.postMessage(reply, { transfer });

self.onmessage = ({ data }: MessageEvent<DecodeRequest>) => {
  if (data.type === "start") void run(data).catch((e) => post({ type: "error", message: e instanceof Error ? e.message : String(e) }));
  else {
    if (data.type === "credit") credit += data.n;
    else stopped = true;
    wake?.();
  }
};

async function run({ file, times, models, ahead }: Extract<DecodeRequest, { type: "start" }>) {
  credit = ahead;
  const input = new Input({ source: new BlobSource(file), formats: ALL_FORMATS });
  try {
    const track = await input.getPrimaryVideoTrack();
    if (typeof VideoDecoder === "undefined" || typeof OffscreenCanvas === "undefined") {
      return post({ type: "unsupported", reason: "no WebCodecs or OffscreenCanvas" });
    }
    if (!track || !(await track.canDecode())) return post({ type: "unsupported", reason: "this video's codec can't be decoded here" });
    post({ type: "ready", width: track.displayWidth, height: track.displayHeight });
    const canvases = models.map(({ input }) => {
      const canvas = new OffscreenCanvas(input.width, input.height);
      return canvas.getContext("2d", { willReadFrequently: true })!;
    });
    const sink = new VideoSampleSink(track);
    let index = 0;
    let t0 = performance.now();
    for await (const sample of sink.samplesAtTimestamps(times)) {
      if (stopped) break;
      const decodeMs = performance.now() - t0;
      if (!sample) throw new Error(`no frame at ${times[index].toFixed(3)} s`);
      const t1 = performance.now();
      const inputs = models.map(({ input, turn }, k) => grab(canvases[k], sample, input, turn));
      const preMs = performance.now() - t1;
      sample.close();
      post({ type: "frame", index, inputs, decodeMs, preMs }, inputs.map((i) => (i.tensor.data as Float32Array).buffer));
      index++;
      credit--;
      while (credit <= 0 && !stopped) await new Promise<void>((resolve) => { wake = resolve; });
      t0 = performance.now();
    }
    if (!stopped) post({ type: "done" });
  } finally {
    input.dispose();
  }
}

/**
 * The frame as the page's FrameGrabber places it: turned by `turn` degrees clockwise (after the file's own rotation,
 * which sample.draw applies), letterboxed top-left into the model's canvas, as one tensor.
 */
function grab(ctx: OffscreenCanvasRenderingContext2D, sample: VideoSample, input: InputConfig, turn: number): DecodedInput {
  const sw = sample.displayWidth;
  const sh = sample.displayHeight;
  const { fw, fh } = turnedSize(sw, sh, turn);
  const t = computeTransform(fw, fh, input);
  const pad = input.padValue ?? PAD_VALUE;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = `rgb(${pad},${pad},${pad})`;
  ctx.fillRect(0, 0, input.width, input.height);
  ctx.translate(t.padX + (fw * t.scaleX) / 2, t.padY + (fh * t.scaleY) / 2);
  ctx.rotate((turn * Math.PI) / 180);
  sample.draw(ctx, (-sw * t.scaleX) / 2, (-sh * t.scaleY) / 2, sw * t.scaleX, sh * t.scaleY);
  const img = ctx.getImageData(0, 0, input.width, input.height);
  return { tensor: toTensor({ width: input.width, height: input.height, data: img.data }, input), transform: t, turn };
}
