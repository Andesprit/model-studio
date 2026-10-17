import { useEffect, useState } from "react";
import type { RunSummary } from "@detect/core";
import { reading } from "./trend";
import { shareSummary } from "./shareSummary";

export interface ModelSummary { name: string; summary: RunSummary }

function summaryDetails(s: RunSummary) {
  const rows: { label: string; value: string }[] = [];
  if (s.framesAnalyzed < 6 || s.from === null || s.to === null) return { time: "Not enough frames for a summary.", rows };
  const thickness = Object.entries(s.metrics).find(([key]) => key.startsWith("vortex thickness "));
  const particles = s.metrics["particles in frame"] ?? s.metrics["in frame"];
  const amplitude = Object.entries(s.metrics).find(([key]) => key.startsWith("vortex breathing ") && key !== "vortex breathing Hz");
  const rate = s.metrics["vortex breathing Hz"];
  const range = (label: string, unit: string, metric: typeof particles) => {
    if (metric) rows.push({ label: `${label} mean / min / max`, value: [metric.mean, metric.min, metric.max].map(v => reading(v, unit)).join(" / ") });
  };
  if (thickness) range("Thickness", thickness[0].slice("vortex thickness ".length), thickness[1]);
  range("Particles", "", particles);
  if (amplitude) rows.push({ label: "Latest breathing", value: reading(amplitude[1].last, amplitude[0].slice("vortex breathing ".length)) + (rate ? ` · ${reading(rate.last, "Hz")}` : "") });
  return { time: `${reading(s.from)}–${reading(s.to)} s · ${s.framesAnalyzed} analyzed · ${s.framesDropped} skipped`, rows };
}

export function summaryText(models: ModelSummary[], title = "Video summary"): string {
  return [title, ...models.map(({ name, summary }) => {
    const { time, rows } = summaryDetails(summary);
    return [name, time, ...rows.map(({ label, value }) => `${label}: ${value}`)].join("\n");
  })].join("\n\n");
}

export function SummaryCard({ models, title = "Video summary" }: { models: ModelSummary[]; title?: string }) {
  const [status, setStatus] = useState("");
  const [sharing, setSharing] = useState(false);
  useEffect(() => {
    if (status !== "Copied") return;
    const timer = setTimeout(() => setStatus(""), 3000);
    return () => clearTimeout(timer);
  }, [status]);
  const share = async () => {
    setStatus("");
    setSharing(true);
    setStatus(await shareSummary({ title, text: summaryText(models, title) }));
    setSharing(false);
  };
  return <section className="run-summary" aria-label={title}>
    <div className="summary-actions">
      <h2>{title}</h2>
      <button onClick={share} disabled={sharing}>Share</button>
    </div>
    <p role="status">{status}</p>
    {models.map(({ name, summary: s }, i) => {
      const { time, rows } = summaryDetails(s);
      return <div className="model-summary" key={i}>
        <b>{name}</b>
        <p>{time}</p>
        {rows.length > 0 && <dl>{rows.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>}
      </div>;
    })}
  </section>;
}
