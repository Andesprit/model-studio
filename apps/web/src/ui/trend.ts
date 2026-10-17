export interface TrendPoint { t: number; value: number }
export interface Trend { label: string; unit: string; points: TrendPoint[] }

/** A bounded window of distinct analyzed frames, in seconds of video. */
export class TrendBuffer {
  private points: TrendPoint[] = [];
  private key = "";
  private lastTime = -Infinity;
  version = 0;

  clear() {
    this.points = [];
    this.key = "";
    this.lastTime = -Infinity;
    this.version++;
  }

  add(t: number, metrics: Record<string, number | string> = {}, detections: number) {
    if (t < this.lastTime) this.clear();
    if (t === this.lastTime) return;
    this.lastTime = t;
    const thickness = Object.entries(metrics).find(([key]) => key.startsWith("vortex thickness "));
    const key = thickness?.[0] ?? "particles";
    const value = thickness?.[1] ?? metrics["particles in frame"] ?? metrics["in frame"] ?? detections;
    if (typeof value !== "number" || !Number.isFinite(value)) return;
    if (key !== this.key) this.points = [];
    this.key = key;
    this.points.push({ t, value });
    while (this.points.length > 600 || this.points[0].t < t - 10) this.points.shift();
    this.version++;
  }

  snapshot(): Trend | null {
    return this.points.length ? {
      label: this.key === "particles" ? "Particles" : "Thickness",
      unit: this.key === "particles" ? "" : this.key.slice("vortex thickness ".length),
      points: [...this.points],
    } : null;
  }
}

export function trendPath(points: TrendPoint[]): string {
  if (!points.length) return "";
  const end = points[points.length - 1].t;
  const values = points.map(p => p.value), low = Math.min(...values), high = Math.max(...values);
  return points.map((p, i) => {
    const x = 2 + (p.t - end + 10) / 10 * 296;
    const y = high === low ? 17 : 31 - (p.value - low) / (high - low) * 28;
    return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export const reading = (value: number, unit = "") => `${Number(value.toFixed(2))}${unit ? ` ${unit}` : ""}`;
