import { describe, expect, it } from "vitest";
import { TrendBuffer, trendPath } from "../src/ui/trend";

describe("measurement trend", () => {
  it("keeps ten seconds of distinct analyzed frames, not paused repeats", () => {
    const buffer = new TrendBuffer();
    for (let t = 0; t <= 12; t++) buffer.add(t, { "vortex thickness cm": t }, 0);
    buffer.add(12, { "vortex thickness cm": 999 }, 0);
    expect(buffer.snapshot()?.points.map(p => p.t)).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    expect(buffer.snapshot()?.points.at(-1)?.value).toBe(12);
    expect(buffer.snapshot()?.unit).toBe("cm");
  });

  it("clears on rewind, unit changes and explicit reset", () => {
    const buffer = new TrendBuffer();
    buffer.add(8, { "vortex thickness px": 10 }, 0);
    buffer.add(1, { "vortex thickness px": 20 }, 0);
    expect(buffer.snapshot()?.points).toEqual([{ t: 1, value: 20 }]);
    buffer.add(2, { "vortex thickness mm": 2 }, 0);
    expect(buffer.snapshot()?.points).toEqual([{ t: 2, value: 2 }]);
    buffer.clear();
    expect(buffer.snapshot()).toBeNull();
  });

  it("ignores placeholders and nonfinite readings, and uses particle detections without a pipeline", () => {
    const buffer = new TrendBuffer();
    for (const value of ["-", NaN, Infinity]) buffer.add(0, { "vortex thickness px": value }, 0);
    expect(buffer.snapshot()).toBeNull();
    buffer.add(1, { "particles in frame": 7 }, 99);
    buffer.add(2, {}, 3);
    expect(buffer.snapshot()?.label).toBe("Particles");
    expect(buffer.snapshot()?.points.map(p => p.value)).toEqual([7, 3]);
  });

  it("bounds high-rate input and draws flat values without invalid SVG coordinates", () => {
    const buffer = new TrendBuffer();
    for (let i = 0; i < 2000; i++) buffer.add(i / 1000, { "vortex thickness px": 4 }, 0);
    const points = buffer.snapshot()!.points;
    expect(points.length).toBeLessThanOrEqual(600);
    expect(trendPath(points)).not.toMatch(/NaN|Infinity/);
    expect(trendPath(points)).toMatch(/^M/);
  });
});
