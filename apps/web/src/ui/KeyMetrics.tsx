import type { ReactNode } from "react";

/** A short reading of the current analysis; the complete metrics stay in the diagnostics. */
export function KeyMetrics({ metrics = {}, detections, showParticles, trend }: { metrics?: Record<string, number | string>; detections: number; showParticles?: boolean; trend?: ReactNode }) {
  const readings: { label: string; value: string }[] = [];
  const add = (prefix: string, label: string, exclude?: string) => {
    const entry = Object.entries(metrics).find(([key]) => (key === prefix || key.startsWith(`${prefix} `)) && key !== exclude);
    if (!entry) return;
    const [key, value] = entry;
    const unit = key === "vortex breathing Hz" ? "Hz" : key.slice(prefix.length).trim();
    readings.push({ label, value: typeof value === "number" ? `${Number(value.toFixed(2))}${unit ? ` ${unit}` : ""}` : value });
  };
  if (metrics["vortex orientation"] !== undefined) readings.push({ label: "Orientation", value: String(metrics["vortex orientation"]) });
  add("vortex thickness", "Thickness");
  add("vortex breathing", "Breathing", "vortex breathing Hz");
  add("vortex breathing Hz", "Breathing rate");
  add("vortex waviness", "Waviness");
  if (metrics["vortex orientation"] === undefined) {
    add("vortex moving", "Movement");
    add("vortex size", "Size");
  }
  if (showParticles || !readings.length || Object.keys(metrics).some((key) => /^(particles |in frame$|unique$)/.test(key))) {
    readings.push({ label: "Particles in frame", value: String(metrics["particles in frame"] ?? metrics["in frame"] ?? detections) });
    add("particles unique", "Unique particles");
    add("unique", "Unique particles");
    add("particles mean speed", "Mean speed");
  }
  return (
    <section className="key-metrics" aria-label="Key measurements">
      <h2>Key measurements</h2>
      <dl>
        {readings.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      </dl>
      {metrics["vortex tilt warning"] && <p className="warn">{metrics["vortex tilt warning"]}</p>}
      {trend}
    </section>
  );
}
