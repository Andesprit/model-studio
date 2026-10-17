import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { KeyMetrics } from "../src/ui/KeyMetrics";

describe("key measurements", () => {
  it("shows placeholders without units", () => {
    const html = renderToStaticMarkup(<KeyMetrics detections={0} metrics={{
      "vortex thickness cm": "-", "vortex breathing Hz": "-", "particles mean speed cm/s": "-",
    }} />);
    expect(html).toContain("<dd>-</dd>");
    expect(html).not.toContain("- cm");
    expect(html).not.toContain("- Hz");
  });
  it("keeps calibrated units and separates breathing amplitude from frequency", () => {
    const html = renderToStaticMarkup(<KeyMetrics detections={0} metrics={{
      "vortex orientation": "vertical",
      "vortex breathing Hz": 1.23456,
      "vortex thickness cm": 2.3456,
      "vortex breathing cm": 0.12345,
      "vortex waviness cm": 0.34567,
    }} />);
    expect(html).toContain("2.35 cm");
    expect(html).toContain("0.12 cm");
    expect(html).toContain("1.23 Hz");
    expect(html).toContain("0.35 cm");
    expect(html).not.toContain("2.3456");
  });

  it("keeps the orientation decision and tilt warning visible", () => {
    const html = renderToStaticMarkup(<KeyMetrics detections={0} metrics={{
      "vortex orientation": "vertical (deciding 3/10)",
      "vortex tilt warning": "Hold the phone straighter",
    }} />);
    expect(html).toContain("vertical (deciding 3/10)");
    expect(html).toContain("Hold the phone straighter");
  });

  it("shows the rig's measurements without requiring lamp orientation", () => {
    const html = renderToStaticMarkup(<KeyMetrics detections={0} metrics={{
      "vortex thickness px": 25.126,
      "vortex breathing px": 0,
      "vortex breathing Hz": 0,
      "vortex waviness px": 12.37,
      "vortex moving": "rising",
    }} />);
    expect(html).toContain("25.13 px");
    expect(html).toContain("0 px");
    expect(html).toContain("rising");
  });

  it("shows particle counts without a pipeline, including zero detections", () => {
    expect(renderToStaticMarkup(<KeyMetrics detections={0} />)).toContain("<dd>0</dd>");
    expect(renderToStaticMarkup(<KeyMetrics detections={7} />)).toContain("<dd>7</dd>");
  });

  it("shows a detector count beside a vortex pipeline when both models run", () => {
    const html = renderToStaticMarkup(<KeyMetrics detections={4} showParticles metrics={{ "vortex thickness px": 25 }} />);
    expect(html).toContain("Particles in frame");
    expect(html).toContain("<dd>4</dd>");
  });

  it("includes tracking measurements alongside vortex measurements", () => {
    const html = renderToStaticMarkup(<KeyMetrics detections={3} metrics={{
      "vortex thickness mm": 15,
      "particles in frame": 3,
      "particles unique": 8,
      "particles mean speed mm/s": 2.567,
    }} />);
    expect(html).toContain("15 mm");
    expect(html).toContain("2.57 mm/s");
    expect(html).toContain("<dd>8</dd>");
  });
});
