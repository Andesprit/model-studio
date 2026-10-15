import { defineConfig } from "vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import react from "@vitejs/plugin-react";

// Cross-origin isolation enables SharedArrayBuffer, which multi-threaded WASM inference needs.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// `--mode phone` (npm run dev:phone) serves HTTPS on the local network: phones only allow the camera on
// secure pages, and localhost isn't the phone. The certificate is self-signed, so the browser warns once.
export default defineConfig(({ mode }) => ({
  plugins: mode === "phone" ? [react(), basicSsl()] : [react()],
  // Keep Vite's cache in the root node_modules instead of creating apps/web/node_modules.
  cacheDir: "../../node_modules/.vite/web",
  server: { port: 5173, strictPort: true, headers: isolation, host: mode === "phone" },
  preview: { headers: isolation },
  // onnxruntime-web loads its .wasm/.mjs files at runtime; pre-bundling breaks those paths.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
}));
