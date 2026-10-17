import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { RunSummary } from "@detect/core";
import { SummaryCard } from "../src/ui/SummaryCard";

const metric = (last: number) => ({ mean: 2.3456, min: 1.2, max: 4.5, last, frames: 20 });
const summary: RunSummary = { from: 0.1, to: 4.2, framesAnalyzed: 20, framesDropped: 3, definitions: {}, metrics: {
  "vortex thickness cm": metric(3), "vortex breathing cm": metric(0.1234), "vortex breathing Hz": metric(1.2345),
} };

describe("video recap", () => {
  it("labels a camera recap without changing the default video title", () => {
    const models = [{ name: "Lamp", summary }];
    const camera = renderToStaticMarkup(<SummaryCard models={models} title="Camera session summary" />);
    expect(camera).toContain('aria-label="Camera session summary"');
    expect(camera).toContain("<h2>Camera session summary</h2>");
    expect(renderToStaticMarkup(<SummaryCard models={models} />)).toContain("<h2>Video summary</h2>");
  });

  it("shows the runner's time, frame counts, thickness statistics and latest breathing", () => {
    const html = renderToStaticMarkup(<SummaryCard models={[{ name: "Lamp", summary }]} />);
    for (const text of ["0.1", "4.2", "20 analyzed", "3 skipped", "2.35 cm", "1.2 cm", "4.5 cm", "0.12 cm", "1.23 Hz"]) expect(html).toContain(text);
  });

  it("shows particle statistics for each model and omits absent placeholder metrics", () => {
    const html = renderToStaticMarkup(<SummaryCard models={[{ name: "Particles", summary: { ...summary, metrics: { "particles in frame": metric(3) } } }]} />);
    expect(html).toContain("Particles mean / min / max");
    expect(html).toContain("2.35");
    expect(html).not.toContain("Breathing");
    expect(html).not.toContain("undefined");
  });

  it("shows a short note when too few frames were analyzed", () => {
    const html = renderToStaticMarkup(<SummaryCard models={[{ name: "Lamp", summary: { ...summary, framesAnalyzed: 2, metrics: {} } }]} />);
    expect(html).toContain("Not enough frames");
    expect(html).not.toContain("NaN");
  });
});
