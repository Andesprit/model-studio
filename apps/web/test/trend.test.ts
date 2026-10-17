import { describe, expect, it } from "vitest";
import { TrendBuffer, trendPath, trendWindow } from "../src/ui/trend";

describe("measurement trend", () => {
  it("fills the chart for a short clip and labels its observed span", () => {
    const points = [{ t: 1.2, value: 2 }, { t: 3.2, value: 4 }, { t: 5.2, value: 3 }];
    expect(trendPath(points)).toBe("M2.0,31.0 L150.0,3.0 L298.0,17.0");
    expect(trendWindow(points)).toEqual({ span: 4, label: "last 4 s" });
  });

  it("labels a rolling camera window as ten seconds and handles a single point", () => {
    expect(trendWindow([{ t: 12.04, value: 1 }, { t: 22, value: 2 }]).label).toBe("last 10 s");
    expect(trendPath([{ t: 1, value: 2 }])).toBe("M298.0,17.0");
    expect(trendPath([])).toBe("");
  });

  it("labels the first point and subsecond spans without rounding to zero seconds", () => {
    expect(trendWindow([])).toEqual({ span: 0, label: "last <1 s" });
    expect(trendWindow([{ t: 1, value: 2 }])).toEqual({ span: 0, label: "last <1 s" });
    for (const span of [0.001, 0.25, 0.999]) {
      expect(trendWindow([{ t: 0, value: 2 }, { t: span, value: 3 }])).toEqual({ span, label: "last <1 s" });
    }
    expect(trendWindow([{ t: 0, value: 2 }, { t: 1, value: 3 }]).label).toBe("last 1 s");
  });

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
