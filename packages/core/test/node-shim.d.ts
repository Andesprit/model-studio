// Minimal Node typings for the real-model test, so core can keep compiling without @types/node
// (core itself must stay free of Node and DOM APIs).
declare module "node:fs" {
  export function existsSync(path: string): boolean;
  export function readFileSync(path: string): Uint8Array;
}
declare const process: { cwd(): string };
