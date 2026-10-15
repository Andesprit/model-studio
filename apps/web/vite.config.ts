import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cross-origin isolation enables SharedArrayBuffer, which multi-threaded WASM inference needs.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, strictPort: true, headers: isolation },
  preview: { headers: isolation },
  // onnxruntime-web loads its .wasm/.mjs files at runtime; pre-bundling breaks those paths.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
