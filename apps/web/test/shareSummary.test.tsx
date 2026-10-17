import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { RunSummary } from "@detect/core";
import { SummaryCard, summaryText } from "../src/ui/SummaryCard";
import { shareSummary } from "../src/ui/shareSummary";

const metric = (last: number) => ({ mean: 2.3456, min: 1.2, max: 4.5, last, frames: 20 });
const summary: RunSummary = { from: 0.1234, to: 4.2345, framesAnalyzed: 20, framesDropped: 3, definitions: {}, metrics: {
  "vortex thickness cm": metric(3), "vortex breathing cm": metric(0.1234), "vortex breathing Hz": metric(1.2345),
} };
const data = { title: "Camera session summary", text: "Lamp measurements" };

describe("share recap text", () => {
  it("uses exactly the card's labels, rounding and calibrated units for every model", () => {
    const models = [{ name: "Vortex lamp", summary }, { name: "Rig vortex", summary },
      { name: "Particles + tracking", summary: { ...summary, metrics: { "in frame": metric(3) } } }];
    const text = summaryText(models, data.title);
    expect(text).toBe([
      "Camera session summary", "", "Vortex lamp", "0.12–4.23 s · 20 analyzed · 3 skipped",
      "Thickness mean / min / max: 2.35 cm / 1.2 cm / 4.5 cm", "Latest breathing: 0.12 cm · 1.23 Hz", "",
      "Rig vortex", "0.12–4.23 s · 20 analyzed · 3 skipped",
      "Thickness mean / min / max: 2.35 cm / 1.2 cm / 4.5 cm", "Latest breathing: 0.12 cm · 1.23 Hz", "",
      "Particles + tracking", "0.12–4.23 s · 20 analyzed · 3 skipped", "Particles mean / min / max: 2.35 / 1.2 / 4.5",
    ].join("\n"));
    const html = renderToStaticMarkup(<SummaryCard models={models} title={data.title} />);
    for (const line of text.split("\n").filter(Boolean)) {
      const [label, value] = line.split(": ");
      expect(html).toContain(label);
      if (value) expect(html).toContain(value);
    }
    expect(html).toContain(">Share</button>");
  });

  it("shares the short-session note without invented statistics and accepts either particle metric name", () => {
    const text = summaryText([{ name: "Lamp", summary: { ...summary, framesAnalyzed: 2 } }]);
    expect(text).toBe("Video summary\n\nLamp\nNot enough frames for a summary.");
    const particles = summaryText([{ name: "Particles", summary: { ...summary, metrics: { "particles in frame": metric(3) } } }]);
    expect(particles).toContain("Particles mean / min / max: 2.35 / 1.2 / 4.5");
    expect(particles).not.toContain("breathing");
  });
});

describe("share recap action", () => {
  it("calls share synchronously before returning its promise and never copies on success", async () => {
    let resolve!: () => void;
    const share = vi.fn(() => new Promise<void>(done => { resolve = done; }));
    const writeText = vi.fn();
    const result = shareSummary(data, { share, clipboard: { writeText } });
    expect(share).toHaveBeenCalledExactlyOnceWith(data);
    resolve();
    expect(await result).toBe("");
    expect(writeText).not.toHaveBeenCalled();
  });

  it("copies when share is absent and reports Copied only after it succeeds", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    expect(await shareSummary(data, { clipboard: { writeText } })).toBe("Copied");
    expect(writeText).toHaveBeenCalledExactlyOnceWith(data.text);
  });

  it("silently handles cancellation without copying", async () => {
    const share = vi.fn().mockRejectedValue(new DOMException("Cancelled", "AbortError"));
    const writeText = vi.fn();
    expect(await shareSummary(data, { share, clipboard: { writeText } })).toBe("");
    expect(writeText).not.toHaveBeenCalled();
  });

  it("copies after a rejected or synchronously throwing share", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    for (const share of [vi.fn().mockRejectedValue(new Error("Unavailable")), () => { throw new Error("Unavailable"); }]) {
      expect(await shareSummary(data, { share, clipboard: { writeText } })).toBe("Copied");
    }
    expect(writeText).toHaveBeenCalledTimes(2);
  });

  it("reports a useful failure if clipboard access is denied or unavailable", async () => {
    for (const browser of [{}, { clipboard: { writeText: vi.fn().mockRejectedValue(new Error("Denied")) } }]) {
      expect(await shareSummary(data, browser)).toBe("Could not share or copy. Use Download results (.json).");
    }
  });
});
