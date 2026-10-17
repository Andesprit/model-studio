import { memo, useEffect, useState } from "react";
import { reading, TrendBuffer, trendPath, type Trend } from "./trend";

/** Only this component updates for the chart; per-frame stats never enter its props. */
export const Sparkline = memo(function Sparkline({ buffer }: { buffer: TrendBuffer }) {
  const [trend, setTrend] = useState<Trend | null>(null);
  useEffect(() => {
    let version = -1;
    const timer = window.setInterval(() => {
      if (version === buffer.version) return;
      version = buffer.version;
      setTrend(buffer.snapshot());
    }, 100);
    return () => window.clearInterval(timer);
  }, [buffer]);
  if (!trend) return null;
  const path = trendPath(trend.points);
  return <div className="measurement-trend">
    <div>{trend.label} · last 10 s <b>{reading(trend.points[trend.points.length - 1].value, trend.unit)}</b></div>
    <svg role="img" aria-label={`${trend.label} over the last ten seconds${trend.unit ? ` (${trend.unit})` : ""}`} viewBox="0 0 300 34" preserveAspectRatio="none">
      <path d={path} fill="none" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  </div>;
});
