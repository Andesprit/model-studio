// Vortex lamp pipeline. Embedded in vortex_lamp_auto_*.onnx by 17_model_studio.py, so it loads with that model.
//
// The model only knows a rope that runs left to right. The ONNX graph therefore runs it twice on every frame:
//   logits       the frame turned 90 deg counter-clockwise, then the mask turned back (a vertical rope: the lamp
//                filmed upright, cap on top)
//   logits_asis  the frame as it is (a rope that already runs left to right)
// This pipeline scores both masks on the first DECIDE_FRAMES analyzed frames, keeps the better orientation for
// the rest of the video, and measures the rope along its own axis. Before it decides, it shows the frame's better
// mask. A reset (loop, seek back, new video) decides again.
//
// Score of a mask = mean probability of its rope pixels x share of those pixels in the largest piece. The wrong
// orientation gives a weak, scattered mask (on the lamp test frames: IoU 0.18 and ~14 pieces vs 0.92 and 1-2).
// After scoring, only the largest piece of each mask is kept (there is one rope), as run_model.py does.
//
// Tilt: the model was trained on ropes within a few degrees of horizontal, and the two branches cover about +-15 deg
// around vertical and horizontal. A diagonal rope (30-75 deg) gives a bad mask in both, and the score can then pick
// the wrong one. The pipeline measures the angle between the rope's main axis (from the mask) and the chosen
// orientation, takes the median over TILT_WINDOW frames, and shows a warning above TILT_WARN_DEG: hold the phone
// straighter.
//
// Measured from the chosen mask, slice by slice across the rope (columns for a horizontal rope, rows for a
// vertical one), with the same definitions as vortex.pipeline.js:
//   area, area rate, thickness (across the rope), center (position across the rope: from the top for a horizontal
//   rope, from the left for a vertical one), drift speed, waviness, breathing.
// Draw a line of known length in step 5 (Calibrate) for real units. Times are seconds of video.

const DECIDE_FRAMES = 10; // analyzed frames whose scores are summed before the orientation is locked
const TILT_WINDOW = 10; // analyzed frames whose tilt estimates are medianed (one frame is noisy, +-10 deg)
const TILT_WARN_DEG = 15; // above this the mask gets worse (lamp test frames: IoU 0.90 at 10 deg, 0.81 at 20, 0.40 at 30)
const BREATHING_WINDOW = 60;
const TREND_FRAMES = 5;

export default {
  name: "vortex-lamp",

  describe: {
    "vortex orientation": "which way the rope runs: vertical (the model saw the frame turned) or horizontal (as filmed); locked after the first 10 analyzed frames",
    "vortex orientation score": "confidence of each orientation's mask (mean rope-pixel probability x share in the largest piece), summed over the deciding frames",
    "vortex tilt deg": "angle between the rope's main axis and the chosen orientation (vertical or horizontal), median of the last 10 analyzed frames",
    "vortex tilt warning": "shown above 15 deg: the masks get worse with tilt and fail around 30-75 deg; hold the phone straighter",
    "vortex present %": "share of the slices across the rope where the rope is seen",
    "vortex area": "area of the rope in the frame",
    "vortex area rate": "change of the rope's area per second, against 5 analyzed frames before (positive = growing)",
    "vortex size": "growing / shrinking / steady, from the area change (steady = under 1%)",
    "vortex thickness": "mean width of the rope across its axis",
    "vortex center": "position of the rope's centreline across its axis (from the top for a horizontal rope, from the left for a vertical one)",
    "vortex drift speed": "speed of the centreline across the axis per second (positive = toward the top or the left)",
    "vortex waviness": "half the span of the rope's centreline across its axis",
    "vortex breathing": "breathing amplitude: half the p10-p90 range of the thickness over the last 60 analyzed frames",
    "vortex breathing Hz": "breathing frequency: expansions per second, from crossings of the average thickness",
  },

  decode(outputs, ctx) {
    const opts = (name) => ({ outputName: name, mode: "binary", threshold: 0 });
    const turned = ctx.lib.decode("semantic-seg", { logits: outputs.logits }, opts("logits")).segmentation;
    const asis = ctx.lib.decode("semantic-seg", { logits_asis: outputs.logits_asis }, opts("logits_asis")).segmentation;
    // scored on the whole mask; then only the largest piece is kept (there is one rope), as run_model.py does
    const scores = { vertical: maskScore(outputs.logits, turned), horizontal: maskScore(outputs.logits_asis, asis) };
    keepLargest(turned);
    keepLargest(asis);
    return {
      segmentation: scores.vertical >= scores.horizontal ? turned : asis,
      lamp: { masks: { vertical: turned, horizontal: asis }, scores },
    };
  },

  create(ctx) {
    const cal = ctx.calibration;
    const unit = cal ? cal.unit : "px";
    const len = (px) => (cal ? px * cal.unitsPerPixel : px);
    const area = (px2) => (cal ? px2 * cal.unitsPerPixel ** 2 : px2);
    const vortex = vortexAnalyzer({ len, area, unit });
    const sums = { vertical: 0, horizontal: 0 };
    const tilts = [];
    let seen = 0;
    let lastT = null;
    let locked = null;

    return {
      step(result, frame) {
        const out = { ...result, metrics: {}, overlays: [...(result.overlays ?? [])] };
        delete out.lamp;
        const lamp = result.lamp;
        if (!lamp) return out;
        const isNew = lastT === null || frame.timestamp > lastT;
        if (!locked && isNew) {
          sums.vertical += lamp.scores.vertical;
          sums.horizontal += lamp.scores.horizontal;
          if (++seen >= DECIDE_FRAMES) locked = sums.vertical >= sums.horizontal ? "vertical" : "horizontal";
        }
        lastT = frame.timestamp;
        const which = locked ?? (lamp.scores.vertical >= lamp.scores.horizontal ? "vertical" : "horizontal");
        out.segmentation = lamp.masks[which];
        out.metrics["vortex orientation"] = locked ? which : `${which} (deciding ${seen}/${DECIDE_FRAMES})`;
        out.metrics["vortex orientation score vertical"] = round(sums.vertical, 2);
        out.metrics["vortex orientation score horizontal"] = round(sums.horizontal, 2);

        const t = axisTilt(out.segmentation, which === "vertical");
        if (isNew && t !== null) {
          tilts.push(t);
          if (tilts.length > TILT_WINDOW) tilts.shift();
        }
        const tilt = tilts.length ? quantile(tilts, 0.5) : null;
        out.metrics["vortex tilt deg"] = tilt === null ? "-" : round(tilt, 1);
        if (tilt !== null && Math.abs(tilt) > TILT_WARN_DEG) {
          const text = `rope tilted ${Math.round(Math.abs(tilt))} deg: hold the phone straighter`;
          out.metrics["vortex tilt warning"] = text;
          out.overlays.push({ type: "text", at: { x: frame.width * 0.05, y: frame.height * 0.05 }, text, color: "#ff3b3b" });
        }
        if (which === "vertical") {
          // measure on the transposed mask (the rope then runs left to right), draw back in frame coordinates
          const sub = { ...out, metrics: {}, overlays: [] };
          vortex.step(transpose(out.segmentation), { ...frame, width: frame.height, height: frame.width }, frame.timestamp, sub);
          Object.assign(out.metrics, sub.metrics);
          out.overlays.push(...sub.overlays.map(swapOverlay));
        } else {
          vortex.step(out.segmentation, frame, frame.timestamp, out);
        }
        return out;
      },
      summary: () => ({
        orientation: locked,
        orientationScores: { ...sums },
        tiltDeg: tilts.length ? round(quantile(tilts, 0.5), 1) : null,
        ...(vortex.used() ? vortex.summary() : {}),
      }),
    };
  },
};

// ---------------------------------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------------------------------

/** Mean sigmoid of the rope pixels x share of the rope pixels in the largest 4-connected piece. */
function maskScore(tensor, seg) {
  const { classes } = seg;
  let n = 0;
  let prob = 0;
  for (let i = 0; i < classes.length; i++) {
    if (classes[i]) {
      n++;
      prob += 1 / (1 + Math.exp(-tensor.data[i]));
    }
  }
  return n ? (prob / n) * (pieces(seg).largestSize / n) : 0;
}

/** Sets every rope cell outside the largest piece to background, in place. */
function keepLargest(seg) {
  const { label, largestId } = pieces(seg);
  for (let i = 0; i < seg.classes.length; i++) if (seg.classes[i] && label[i] !== largestId) seg.classes[i] = seg.background;
}

/** 4-connected pieces of the rope cells: a label per cell (0 = none), and the largest piece's label and size. */
function pieces(seg) {
  const { width: w, height: h, classes } = seg;
  const label = new Int32Array(classes.length);
  const stack = [];
  let id = 0, largestId = 0, largestSize = 0;
  for (let i = 0; i < classes.length; i++) {
    if (!classes[i] || label[i]) continue;
    let size = 0;
    label[i] = ++id;
    stack.push(i);
    while (stack.length) {
      const p = stack.pop();
      size++;
      const x = p % w;
      const y = (p - x) / w;
      for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
        if (q >= 0 && classes[q] && !label[q]) {
          label[q] = id;
          stack.push(q);
        }
      }
    }
    if (size > largestSize) [largestId, largestSize] = [id, size];
  }
  return { label, largestId, largestSize };
}

/**
 * Angle in degrees between the main axis of the rope cells (principal axis of their positions in frame pixels) and
 * the chosen orientation; positive = turned clockwise on screen. null when there are too few cells.
 */
function axisTilt(seg, vertical) {
  const { width: w, height: h, classes, rect } = seg;
  const cw = rect.width / w;
  const ch = rect.height / h;
  let n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!classes[y * w + x] || classes[y * w + x] === seg.background) continue;
      const px = x * cw;
      const py = y * ch;
      n++;
      sx += px;
      sy += py;
      sxx += px * px;
      syy += py * py;
      sxy += px * py;
    }
  }
  if (n < 10) return null;
  const vx = sxx / n - (sx / n) ** 2;
  const vy = syy / n - (sy / n) ** 2;
  const cxy = sxy / n - (sx / n) * (sy / n);
  let a = (0.5 * Math.atan2(2 * cxy, vx - vy) * 180) / Math.PI; // main axis against +x, in (-90, 90]
  if (vertical) a = a > 0 ? a - 90 : a + 90; // against +y instead
  return a;
}

function transpose(seg) {
  const { width: w, height: h, classes, rect, background } = seg;
  const t = new Uint16Array(classes.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) t[x * h + y] = classes[y * w + x];
  return { width: h, height: w, classes: t, rect: { x: rect.y, y: rect.x, width: rect.height, height: rect.width }, background };
}

const swapPoint = (p) => ({ x: p.y, y: p.x });

function swapOverlay(o) {
  if (o.type === "polyline") return { ...o, points: o.points.map(swapPoint) };
  if (o.type === "point" || o.type === "text") return { ...o, at: swapPoint(o.at) };
  if (o.type === "line") return { ...o, from: swapPoint(o.from), to: swapPoint(o.to) };
  return o;
}

// ---------------------------------------------------------------------------------------------------
// Vortex rope (as vortex.pipeline.js; metric names made axis-neutral)
// ---------------------------------------------------------------------------------------------------

function vortexAnalyzer({ len, area, unit }) {
  const series = [];
  let lastT = null;

  return {
    used: () => series.length > 0,

    step(seg, frame, t, out) {
      const rope = measureRope(seg, frame.width, frame.height);
      const isNew = lastT === null || t > lastT;
      const s = {
        t,
        present: rope.present,
        area: area(rope.areaPx),
        thickness: len(rope.thicknessPx),
        center: len(rope.centerYPx),
        waviness: len(rope.wavinessPx),
      };
      if (isNew) {
        series.push(s);
        lastT = t;
      }

      const ref = series[series.length - 1 - TREND_FRAMES];
      const dt = ref ? s.t - ref.t : 0;
      const areaRate = dt > 0 ? (s.area - ref.area) / dt : null;
      const drift = dt > 0 ? -(s.center - ref.center) / dt : null;
      const b = breathing(series.slice(-BREATHING_WINDOW));

      Object.assign(out.metrics, {
        "vortex present %": round(s.present * 100, 1),
        [`vortex area ${unit}²`]: round(s.area, 2),
        [`vortex area rate ${unit}²/s`]: areaRate === null ? "-" : round(areaRate, 2),
        "vortex size": areaRate === null ? "-" : trend(areaRate * dt, s.area * 0.01, "growing", "shrinking"),
        [`vortex thickness ${unit}`]: round(s.thickness, 2),
        [`vortex center ${unit}`]: round(s.center, 2),
        [`vortex drift speed ${unit}/s`]: drift === null ? "-" : round(drift, 2),
        [`vortex waviness ${unit}`]: round(s.waviness, 2),
        [`vortex breathing ${unit}`]: b ? round(b.amplitude, 2) : "-",
        "vortex breathing Hz": b?.frequency ? round(b.frequency, 2) : "-",
      });

      const step = Math.max(1, Math.floor(rope.columns.length / 120));
      const cols = rope.columns.filter((_, i) => i % step === 0);
      if (cols.length > 1) {
        out.overlays.push(
          { type: "polyline", points: cols.map((c) => ({ x: c.x, y: c.top })), color: "#00e5ff" },
          { type: "polyline", points: cols.map((c) => ({ x: c.x, y: c.bottom })), color: "#00e5ff" },
          { type: "polyline", points: cols.map((c) => ({ x: c.x, y: (c.top + c.bottom) / 2 })), color: "#ffd400" },
        );
      }
      if (rope.present > 0) {
        out.overlays.push({ type: "point", at: { x: rope.centerXPx, y: rope.centerYPx }, color: "#ff3b3b" });
      }
    },

    summary() {
      const valid = series.filter((s) => s.present > 0);
      const b = breathing(valid);
      const span = valid.length > 1 ? valid[valid.length - 1].t - valid[0].t : 0;
      return {
        unit,
        timeNote: "times are seconds of video",
        framesAnalyzed: series.length,
        framesWithRope: valid.length,
        samplingRateHz: span > 0 ? round((valid.length - 1) / span, 3) : null,
        breathing: b && {
          amplitude: round(b.amplitude, 4),
          relativePercent: round(b.relative * 100, 2),
          frequencyHz: b.frequency === null ? null : round(b.frequency, 4),
          note: "amplitude = half the p10-p90 range of the thickness; frequency from crossings of its mean",
        },
        thickness: stats(valid.map((s) => s.thickness)),
        area: stats(valid.map((s) => s.area)),
        center: stats(valid.map((s) => s.center)),
        waviness: stats(valid.map((s) => s.waviness)),
      };
    },
  };
}

/** Per-column top and bottom edge of the rope (the longest run of mask cells in each column), in frame pixels. */
function measureRope(seg, frameWidth, frameHeight) {
  const { width: w, height: h, classes, rect, background } = seg;
  const cellW = rect.width / w;
  const cellH = rect.height / h;
  const columns = [];
  let colsInFrame = 0;
  let areaPx = 0;
  for (let x = 0; x < w; x++) {
    const cx = rect.x + (x + 0.5) * cellW;
    if (cx < 0 || cx >= frameWidth) continue;
    colsInFrame++;
    let best = 0, bestStart = -1, run = 0, start = 0;
    for (let y = 0; y < h; y++) {
      const cy = rect.y + (y + 0.5) * cellH;
      const on = classes[y * w + x] !== background && cy >= 0 && cy < frameHeight;
      if (on) {
        if (run === 0) start = y;
        run++;
        if (run > best) {
          best = run;
          bestStart = start;
        }
      } else run = 0;
    }
    if (best > 0) {
      const top = rect.y + bestStart * cellH;
      const bottom = rect.y + (bestStart + best) * cellH;
      columns.push({ x: cx, top, bottom });
      areaPx += best * cellW * cellH;
    }
  }
  const thick = columns.map((c) => c.bottom - c.top);
  const center = columns.map((c) => (c.top + c.bottom) / 2);
  return {
    columns,
    present: colsInFrame ? columns.length / colsInFrame : 0,
    areaPx,
    thicknessPx: thick.length ? mean(thick) : 0,
    centerYPx: center.length ? mean(center) : 0,
    centerXPx: columns.length ? mean(columns.map((c) => c.x)) : 0,
    wavinessPx: center.length ? (quantile(center, 0.95) - quantile(center, 0.05)) / 2 : 0,
  };
}

function breathing(series) {
  const pts = series.filter((s) => s.present > 0);
  if (pts.length < 6) return null;
  const v = pts.map((s) => s.thickness);
  const m = mean(v);
  const amplitude = (quantile(v, 0.9) - quantile(v, 0.1)) / 2;
  let crossings = 0;
  for (let i = 1; i < v.length; i++) if ((v[i - 1] - m) * (v[i] - m) < 0) crossings++;
  const span = pts[pts.length - 1].t - pts[0].t;
  return { amplitude, relative: m ? amplitude / m : 0, frequency: span > 0 && crossings ? crossings / 2 / span : null };
}

// ---------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------

function trend(change, deadBand, up, down) {
  if (Math.abs(change) <= deadBand) return "steady";
  return change > 0 ? up : down;
}

function stats(xs) {
  if (!xs.length) return null;
  return {
    mean: round(mean(xs), 4),
    min: round(Math.min(...xs), 4),
    max: round(Math.max(...xs), 4),
    p10: round(quantile(xs, 0.1), 4),
    p90: round(quantile(xs, 0.9), 4),
  };
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function quantile(xs, q) {
  const s = [...xs].sort((a, b) => a - b);
  const i = (s.length - 1) * q;
  const lo = Math.floor(i);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo);
}

function round(v, digits) {
  return Number.isFinite(v) ? Math.round(v * 10 ** digits) / 10 ** digits : v;
}
