import type { RunSummary } from "@detect/core";
import { reading } from "./trend";

export interface ModelSummary { name: string; summary: RunSummary }

export function SummaryCard({ models, title = "Video summary" }: { models: ModelSummary[]; title?: string }) {
  return <section className="run-summary" aria-label={title}>
    <h2>{title}</h2>
    {models.map(({ name, summary: s }, i) => {
      const thickness = Object.entries(s.metrics).find(([key]) => key.startsWith("vortex thickness "));
      const particles = s.metrics["particles in frame"] ?? s.metrics["in frame"];
      const amplitude = Object.entries(s.metrics).find(([key]) => key.startsWith("vortex breathing ") && key !== "vortex breathing Hz");
      const rate = s.metrics["vortex breathing Hz"];
      const range = (label: string, key: string, metric: typeof particles) => metric && <div><dt>{label} mean / min / max</dt><dd>{[metric.mean, metric.min, metric.max].map(v => reading(v, key)).join(" / ")}</dd></div>;
      return <div className="model-summary" key={i}>
        <b>{name}</b>
        {s.framesAnalyzed < 6 || s.from === null || s.to === null ? <p>Not enough frames for a summary.</p> : <>
          <p>{reading(s.from)}–{reading(s.to)} s · {s.framesAnalyzed} analyzed · {s.framesDropped} skipped</p>
          <dl>
            {thickness && range("Thickness", thickness[0].slice("vortex thickness ".length), thickness[1])}
            {range("Particles", "", particles)}
            {amplitude && <div><dt>Latest breathing</dt><dd>{reading(amplitude[1].last, amplitude[0].slice("vortex breathing ".length))}{rate && ` · ${reading(rate.last, "Hz")}`}</dd></div>}
          </dl>
        </>}
      </div>;
    })}
  </section>;
}
