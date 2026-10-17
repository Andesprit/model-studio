import type { Overlay, Rect, Result } from "@detect/core";

const PALETTE = [
  [255, 56, 56], [255, 157, 151], [255, 112, 31], [255, 178, 29], [207, 210, 49], [72, 249, 10],
  [146, 204, 23], [61, 219, 134], [26, 147, 52], [0, 212, 187], [44, 153, 168], [0, 194, 255],
  [52, 69, 147], [100, 115, 255], [0, 24, 236], [132, 56, 255], [82, 0, 133], [203, 56, 255],
  [255, 149, 200], [255, 55, 199],
];
const color = (id: number) => PALETTE[((id % PALETTE.length) + PALETTE.length) % PALETTE.length];
const rgb = (id: number, a = 1) => `rgba(${color(id).join(",")},${a})`;

const KEYPOINT_THRESHOLD = 0.5;

/** Draws in source-frame coordinates onto a display-sized canvas. */
export class Renderer {
  private scratch = document.createElement("canvas");
  private scratchCtx = this.scratch.getContext("2d")!;

  constructor(private canvas: HTMLCanvasElement) {}

  clear() {
    const ctx = this.canvas.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  draw(result: Result, frameWidth: number, frameHeight: number) {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    const density = Math.min(devicePixelRatio, 2, Math.sqrt(2_000_000 / (width * height)));
    const backingWidth = Math.max(1, Math.floor(width * density));
    const backingHeight = Math.max(1, Math.floor(height * density));
    if (this.canvas.width !== backingWidth || this.canvas.height !== backingHeight) {
      this.canvas.width = backingWidth;
      this.canvas.height = backingHeight;
    }
    const ctx = this.canvas.getContext("2d")!;
    this.clear();
    ctx.setTransform(backingWidth / frameWidth, 0, 0, backingHeight / frameHeight, 0, 0);
    // Scale strokes/text with resolution so overlays look the same on 480p and 4K.
    const u = Math.max(1, Math.min(frameWidth, frameHeight) / 400);

    if (result.segmentation) {
      const s = result.segmentation;
      this.paint(ctx, s.width, s.height, s.rect, (i) => (s.classes[i] === s.background ? null : color(s.classes[i])), 0.5);
    }
    for (const m of result.masks ?? []) {
      this.paint(ctx, m.width, m.height, m.rect, (i) => (m.data[i] ? color(m.classId) : null), 0.45);
    }
    for (const b of result.boxes ?? []) {
      // Tracked boxes are colored by track so each object keeps its color across frames.
      const id = b.trackId ?? b.classId;
      const track = b.trackId !== undefined ? `#${b.trackId} ` : "";
      drawBox(ctx, b.x, b.y, b.width, b.height, id, `${track}${b.label ?? b.classId} ${(b.score * 100).toFixed(0)}%`, u);
    }
    for (const p of result.poses ?? []) {
      const b = p.box;
      drawBox(ctx, b.x, b.y, b.width, b.height, b.classId, `${b.label ?? b.classId} ${(b.score * 100).toFixed(0)}%`, u);
      ctx.lineWidth = 2 * u;
      for (const [a, c] of result.skeleton ?? []) {
        const ka = p.keypoints[a];
        const kc = p.keypoints[c];
        if (!ka || !kc || ka.score < KEYPOINT_THRESHOLD || kc.score < KEYPOINT_THRESHOLD) continue;
        ctx.strokeStyle = rgb(a + 3, 0.9);
        ctx.beginPath();
        ctx.moveTo(ka.x, ka.y);
        ctx.lineTo(kc.x, kc.y);
        ctx.stroke();
      }
      p.keypoints.forEach((k, i) => {
        if (k.score < KEYPOINT_THRESHOLD) return;
        ctx.fillStyle = rgb(i + 3);
        ctx.beginPath();
        ctx.arc(k.x, k.y, 3 * u, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    if (result.classes?.length) {
      ctx.font = `${14 * u}px system-ui, sans-serif`;
      result.classes.forEach((c, i) => {
        const text = `${c.label ?? c.classId}  ${(c.score * 100).toFixed(1)}%`;
        const y = (10 + i * 22) * u;
        ctx.fillStyle = "rgba(0,0,0,0.6)";
        ctx.fillRect(8 * u, y, ctx.measureText(text).width + 12 * u, 20 * u);
        ctx.fillStyle = "#fff";
        ctx.fillText(text, 14 * u, y + 15 * u);
      });
    }
    for (const o of result.overlays ?? []) drawOverlay(ctx, o, u);
  }

  /** Colorizes a low-res map via a scratch canvas, then stretches it over `rect`. */
  private paint(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    rect: Rect,
    colorAt: (i: number) => number[] | null,
    alpha: number,
  ) {
    if (!w || !h) return;
    this.scratch.width = w;
    this.scratch.height = h;
    const img = this.scratchCtx.createImageData(w, h);
    const a = Math.round(alpha * 255);
    for (let i = 0; i < w * h; i++) {
      const c = colorAt(i);
      if (!c) continue;
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = a;
    }
    this.scratchCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.scratch, rect.x, rect.y, rect.width, rect.height);
  }
}

const OVERLAY_COLOR = "#ffd400";

function drawOverlay(ctx: CanvasRenderingContext2D, o: Overlay, u: number) {
  ctx.strokeStyle = ctx.fillStyle = o.color ?? OVERLAY_COLOR;
  ctx.lineWidth = 2 * u;
  switch (o.type) {
    case "line":
    case "polyline": {
      const pts = o.type === "line" ? [o.from, o.to] : o.points;
      if (pts.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (const p of pts.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.stroke();
      return;
    }
    case "point":
      ctx.beginPath();
      ctx.arc(o.at.x, o.at.y, 4 * u, 0, Math.PI * 2);
      ctx.fill();
      return;
    case "text":
      ctx.font = `${14 * u}px system-ui, sans-serif`;
      // Dark outline keeps text readable on any background.
      ctx.lineWidth = 3 * u;
      ctx.strokeStyle = "rgba(0,0,0,0.7)";
      ctx.strokeText(o.text, o.at.x, o.at.y);
      ctx.fillText(o.text, o.at.x, o.at.y);
      return;
  }
}

function drawBox(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, id: number, text: string, u: number) {
  ctx.lineWidth = 2 * u;
  ctx.strokeStyle = rgb(id);
  ctx.strokeRect(x, y, w, h);
  ctx.font = `${12 * u}px system-ui, sans-serif`;
  const tw = ctx.measureText(text).width + 8 * u;
  const th = 16 * u;
  const ty = y - th < 0 ? y : y - th;
  ctx.fillStyle = rgb(id);
  ctx.fillRect(x, ty, tw, th);
  ctx.fillStyle = "#fff";
  ctx.fillText(text, x + 4 * u, ty + 12 * u);
}
