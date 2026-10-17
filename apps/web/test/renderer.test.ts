import { afterEach, describe, expect, it, vi } from "vitest";
import { Renderer } from "../src/adapters/renderer";

afterEach(() => vi.unstubAllGlobals());

function setup(width = 190, height = 338, dpr = 3) {
  vi.stubGlobal("devicePixelRatio", dpr);
  vi.stubGlobal("document", { createElement: () => ({ getContext: () => ({}) }) });
  const ctx = {
    setTransform: vi.fn(), clearRect: vi.fn(), beginPath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(),
  };
  const canvas = { width: 300, height: 150, clientWidth: width, clientHeight: height, getContext: () => ctx };
  return { canvas, ctx, renderer: new Renderer(canvas as unknown as HTMLCanvasElement) };
}

describe("display-sized overlay", () => {
  it("caps pixel density and draws in frame coordinates without changing stroke sizes", () => {
    const { canvas, ctx, renderer } = setup();
    renderer.draw({ overlays: [{ type: "line", from: { x: 100, y: 200 }, to: { x: 300, y: 400 } }] }, 2160, 3840);
    expect([canvas.width, canvas.height]).toEqual([380, 676]);
    expect(ctx.setTransform).toHaveBeenLastCalledWith(380 / 2160, 0, 0, 676 / 3840, 0, 0);
    expect(ctx.moveTo).toHaveBeenCalledWith(100, 200);
    expect(ctx.lineTo).toHaveBeenCalledWith(300, 400);
    expect(ctx).toHaveProperty("lineWidth", 10.8);
  });

  it("updates backing dimensions after a resize and clears with an identity transform", () => {
    const { canvas, ctx, renderer } = setup(344, 612, 1);
    renderer.draw({}, 2160, 3840);
    expect([canvas.width, canvas.height]).toEqual([344, 612]);
    canvas.clientWidth = 190;
    canvas.clientHeight = 338;
    renderer.draw({}, 2160, 3840);
    expect([canvas.width, canvas.height]).toEqual([190, 338]);
    renderer.clear();
    expect(ctx.setTransform).toHaveBeenLastCalledWith(1, 0, 0, 1, 0, 0);
    expect(ctx.clearRect).toHaveBeenLastCalledWith(0, 0, 190, 338);
  });

  it("limits large displays to two million pixels and handles a hidden stage", () => {
    const { canvas, renderer } = setup(8000, 4000, 2);
    renderer.draw({}, 8000, 4000);
    expect(canvas.width * canvas.height).toBeLessThanOrEqual(2_000_000);
    expect(canvas.width / canvas.height).toBeCloseTo(2, 2);
    canvas.clientWidth = canvas.clientHeight = 0;
    renderer.draw({}, 2160, 3840);
    expect([canvas.width, canvas.height]).toEqual([2, 2]);
  });
});
