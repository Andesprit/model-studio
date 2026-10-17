import type { Tensor } from "@detect/core";
import type { InputInfo, ModelRuntime } from "./runtime";

export type WorkerRequest = { id: number } & (
  | { type: "create"; model: Uint8Array }
  | { type: "run"; input: Tensor; extra?: Record<string, Tensor> }
  | { type: "release" }
);
type SessionInfo = { input: InputInfo; outputNames: readonly string[] };
type ReplyValue = SessionInfo | Record<string, Tensor> | undefined;
export type WorkerReply = { id: number; value?: ReplyValue; error?: string } | { fatal: string };

/** A session lives in one worker until release; image and output buffers cross without copies. */
export async function createWorkerRuntime(model: Uint8Array): Promise<ModelRuntime> {
  const worker = new Worker(new URL("./ortWorker.ts", import.meta.url), { type: "module" });
  const pending = new Map<number, { resolve(value: ReplyValue): void; reject(error: Error): void }>();
  let nextId = 0;
  let closed: Error | null = null;
  const fail = (error: Error) => {
    if (closed) return;
    closed = error;
    for (const call of pending.values()) call.reject(error);
    pending.clear();
    worker.terminate();
  };
  worker.onmessage = ({ data }: MessageEvent<WorkerReply>) => {
    if ("fatal" in data) { fail(new Error(data.fatal)); return; }
    const call = pending.get(data.id);
    if (!call) return;
    pending.delete(data.id);
    if (data.error) call.reject(new Error(data.error));
    else call.resolve(data.value);
  };
  worker.onerror = (event) => {
    event.preventDefault();
    fail(new Error(event.message || "CPU worker failed"));
  };
  worker.onmessageerror = () => fail(new Error("Could not read CPU worker message"));

  function request(message: WorkerRequest, transfer: Transferable[] = []): Promise<ReplyValue> {
    if (closed) return Promise.reject(closed);
    return new Promise((resolve, reject) => {
      pending.set(message.id, { resolve, reject });
      try { worker.postMessage(message, transfer); }
      catch (error) { pending.delete(message.id); reject(error); }
    });
  }

  try {
    // Keep App's model bytes for metadata, backend changes and disk-model reloads.
    const bytes = model.slice();
    const info = await request({ id: nextId++, type: "create", model: bytes }, [bytes.buffer]) as SessionInfo;
    return {
      backend: "wasm",
      ...info,
      async run(input, extra) {
        const transfer = input.data.buffer instanceof ArrayBuffer ? [input.data.buffer] : [];
        return await request({ id: nextId++, type: "run", input, extra }, transfer) as Record<string, Tensor>;
      },
      async release() {
        if (closed) return;
        try { await request({ id: nextId++, type: "release" }); }
        finally { fail(new Error("CPU runtime released")); }
      },
    };
  } catch (error) {
    fail(error instanceof Error ? error : new Error(String(error)));
    throw error;
  }
}
