import type { Point } from "@detect/core";
import { useRef, useState } from "react";

export interface Line {
  from: Point;
  to: Point;
}

export const lineLength = (l: Line) => Math.hypot(l.to.x - l.from.x, l.to.y - l.from.y);

interface Props {
  /** Video frame size; the SVG uses it as its coordinate system so the line is stored in frame pixels. */
  width: number;
  height: number;
  line: Line | null;
  /** While true, a press-drag-release on the video draws a new line. */
  drawing: boolean;
  onLine(line: Line): void;
}

/** What a drag is changing: a new line, one endpoint, or the whole line. */
type Part = "new" | "from" | "to" | "line";

/**
 * Draws the calibration line over the video. An existing line can be resized by dragging its ends
 * and moved by dragging its body. `meet` matches the video's object-fit: contain.
 */
export function CalibrationLine({ width, height, line, drawing, onLine }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ part: Part; start: Point; base: Line } | null>(null);
  const [draft, setDraft] = useState<Line | null>(null);

  const clampX = (x: number) => Math.min(width, Math.max(0, x));
  const clampY = (y: number) => Math.min(height, Math.max(0, y));
  const toFrame = (e: React.PointerEvent): Point => {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svgRef.current!.getScreenCTM()!.inverse());
    return { x: clampX(p.x), y: clampY(p.y) };
  };

  const dragged = (part: Part, start: Point, base: Line, p: Point): Line => {
    switch (part) {
      case "new":
        return { from: start, to: p };
      case "from":
        return { ...base, from: p };
      case "to":
        return { ...base, to: p };
      case "line": {
        // Shift both ends by the same amount, stopping at the frame edges so the length is kept.
        const xs = [base.from.x, base.to.x];
        const ys = [base.from.y, base.to.y];
        const dx = Math.min(width - Math.max(...xs), Math.max(-Math.min(...xs), p.x - start.x));
        const dy = Math.min(height - Math.max(...ys), Math.max(-Math.min(...ys), p.y - start.y));
        return { from: { x: base.from.x + dx, y: base.from.y + dy }, to: { x: base.to.x + dx, y: base.to.y + dy } };
      }
    }
  };

  const shown = draft ?? line;
  if (!width || !height || (!shown && !drawing)) return null;
  // Handle sizes in frame pixels, so they look the same on any video resolution.
  const r = Math.max(width, height) / 120;
  return (
    <svg
      ref={svgRef}
      className={`calibration ${drawing ? "drawing" : ""}`}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      onPointerDown={(e) => {
        const target = (e.target as Element).getAttribute("data-part") as Part | null;
        const part: Part | null = drawing ? "new" : target;
        const base = line ?? { from: { x: 0, y: 0 }, to: { x: 0, y: 0 } };
        if (!part) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        const p = toFrame(e);
        setDrag({ part, start: p, base });
        setDraft(dragged(part, p, base, p));
      }}
      onPointerMove={(e) => drag && setDraft(dragged(drag.part, drag.start, drag.base, toFrame(e)))}
      onPointerUp={(e) => {
        if (!drag) return;
        const done = dragged(drag.part, drag.start, drag.base, toFrame(e));
        setDrag(null);
        setDraft(null);
        if (lineLength(done) > 0) onLine(done);
      }}
    >
      {shown && (
        <>
          <line className="body" x1={shown.from.x} y1={shown.from.y} x2={shown.to.x} y2={shown.to.y} />
          <line className="hit" data-part="line" x1={shown.from.x} y1={shown.from.y} x2={shown.to.x} y2={shown.to.y} />
          <circle className="end" cx={shown.from.x} cy={shown.from.y} r={r} />
          <circle className="end" cx={shown.to.x} cy={shown.to.y} r={r} />
          <circle className="hit end-hit" data-part="from" cx={shown.from.x} cy={shown.from.y} r={r * 3} />
          <circle className="hit end-hit" data-part="to" cx={shown.to.x} cy={shown.to.y} r={r * 3} />
        </>
      )}
    </svg>
  );
}
