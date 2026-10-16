// Minimal Node typings for the tests that read files, so core can keep compiling without @types/node
// (core itself must stay free of Node and DOM APIs).
declare module "node:fs" {
  export function existsSync(path: string): boolean;
  export function readFileSync(path: string): Uint8Array;
  export function readFileSync(path: string, encoding: "utf8"): string;
}
declare module "node:url" {
  export function pathToFileURL(path: string): { href: string };
}
declare const process: { cwd(): string };
