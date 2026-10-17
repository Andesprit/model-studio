import { createSessionRuntime } from "./ortSession";
import type { ModelRuntime } from "./runtime";
import type { WorkerReply, WorkerRequest } from "./workerRuntime";

let runtime: ModelRuntime;
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const fatal = (error: unknown) => {
  self.postMessage({ fatal: message(error) } satisfies WorkerReply);
  self.close();
};
// A WASM trap can escape session.run's promise. Tell the page even when that promise never settles.
self.addEventListener("error", (event) => { event.preventDefault(); fatal(event.error ?? event.message); });
self.addEventListener("unhandledrejection", (event) => { event.preventDefault(); fatal(event.reason); });
self.onmessage = async ({ data }: MessageEvent<WorkerRequest>) => {
  try {
    if (data.type === "create") {
      runtime = await createSessionRuntime(data.model, "wasm");
      self.postMessage({ id: data.id, value: { input: runtime.input, outputNames: runtime.outputNames } } satisfies WorkerReply);
    } else if (data.type === "run") {
      const outputs = await runtime.run(data.input, data.extra);
      const buffers = Object.values(outputs).map((tensor) => tensor.data.buffer).filter((buffer) => buffer instanceof ArrayBuffer);
      self.postMessage({ id: data.id, value: outputs } satisfies WorkerReply, { transfer: [...new Set(buffers)] });
    } else {
      await runtime.release();
      self.postMessage({ id: data.id } satisfies WorkerReply);
    }
  } catch (error) {
    if (error instanceof WebAssembly.RuntimeError) fatal(error);
    else self.postMessage({ id: data.id, error: message(error) } satisfies WorkerReply);
  }
};
