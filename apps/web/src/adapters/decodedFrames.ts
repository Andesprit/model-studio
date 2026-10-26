import type { InputConfig } from "@detect/core";
import type { DecodedInput, DecodeReply, DecodeRequest } from "./decodeWorker";

/** Frames waiting, prepared, ahead of the models: enough to keep the GPU busy, small enough for a phone's memory. */
const AHEAD = 8;

export interface DecodedFrame {
  index: number;
  inputs: DecodedInput[];
  /** Time in the worker to decode up to this frame, and to turn it into the models' inputs. */
  decodeMs: number;
  preMs: number;
}

/** Frames decoded in order by decodeWorker.ts; next() resolves them one by one, in the order of `times`. */
export interface DecodedFrames {
  next(): Promise<DecodedFrame>;
  close(): void;
}

/**
 * Starts decoding `file` at `times` (seconds, increasing) for models with these inputs and turns. Resolves null when
 * this browser or this video can't be decoded that way (the caller seeks instead).
 */
export async function openDecodedFrames(file: Blob, times: number[], models: { input: InputConfig; turn: number }[]): Promise<DecodedFrames | null> {
  if (typeof Worker === "undefined" || typeof VideoDecoder === "undefined") return null;
  const worker = new Worker(new URL("./decodeWorker.ts", import.meta.url), { type: "module" });
  const frames: DecodedFrame[] = [];
  let waiting: { resolve(f: DecodedFrame): void; reject(e: Error): void } | null = null;
  let failure: Error | null = null;
  let done = false;
  let ready!: (ok: boolean) => void;
  const started = new Promise<boolean>((resolve) => { ready = resolve; });
  const fail = (e: Error) => {
    failure = e;
    waiting?.reject(e);
    waiting = null;
    ready(false);
  };
  worker.onmessage = ({ data }: MessageEvent<DecodeReply>) => {
    if (data.type === "ready") ready(true);
    else if (data.type === "unsupported") { ready(false); worker.terminate(); }
    else if (data.type === "error") fail(new Error(`Decoding the video failed: ${data.message}`));
    else if (data.type === "done") {
      done = true;
      if (waiting) fail(new Error("The video ended before every frame was decoded"));
    } else if (waiting) { waiting.resolve(data); waiting = null; }
    else frames.push(data);
  };
  worker.onerror = (event) => { event.preventDefault(); fail(new Error(event.message || "Video decoding worker failed")); };
  worker.postMessage({ type: "start", file, times, models, ahead: AHEAD } satisfies DecodeRequest);
  if (!(await started)) {
    worker.terminate();
    // An error before the first frame (e.g. a container mediabunny can't read): seeking may still work.
    return null;
  }
  return {
    next() {
      if (failure) return Promise.reject(failure);
      const frame = frames.shift();
      worker.postMessage({ type: "credit", n: 1 } satisfies DecodeRequest);
      if (frame) return Promise.resolve(frame);
      if (done) return Promise.reject(new Error("The video ended before every frame was decoded"));
      return new Promise((resolve, reject) => { waiting = { resolve, reject }; });
    },
    close() {
      worker.postMessage({ type: "stop" } satisfies DecodeRequest);
      worker.terminate();
    },
  };
}
