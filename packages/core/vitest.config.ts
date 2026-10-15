import { defineConfig } from "vitest/config";

// Keep Vitest's cache in the root node_modules instead of creating packages/core/node_modules.
export default defineConfig({ cacheDir: "../../node_modules/.vite/core" });
