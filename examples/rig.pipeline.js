// Rig pipeline: vortex rope ("torch") + particles, for the two rig models loaded together.
// Load it with "Pipeline .js". It works with either model alone too: each part runs only when its
// data is there (the vortex model gives a mask, the particle model gives boxes).
//
// Before measuring, draw a line of known length in step 5 (Calibrate) so sizes and speeds are in real
// units (else pixels). Times and rates are in seconds of video.
//
// VORTEX (from the mask; the rope runs left to right, so it is measured column by column)
//   area            rope area in the frame
//   area rate       change of area per second (positive = growing, negative = shrinking)
//   thickness       mean height of the rope across its columns
//   center y        height of the rope's centre in the frame (0 = top)
//   vertical speed  how fast the centre moves (positive = rising, negative = falling)
//   waviness        half the up-and-down span of the centreline (how wavy the rope is)
//   breathing       how much the rope expands and contracts: amplitude and frequency of the thickness
//                   over time (frequency from crossings of its average, over the last BREATHING_WINDOW frames)
//
// PARTICLES (from the boxes)
//   in frame, unique (distinct particles followed) and mean speed, plus per-particle speed, direction and size.
//
// Download results gives one JSON with a "vortex" and a "particles" section, plus every numeric
// metric summarized (mean / min / max) by the app.
//
// Sampling note: the app skips frames when the models are slower than the video, so the time series
// has gaps. The results file says how many frames per second of video were analyzed; a breathing
// frequency above half that rate cannot be measured.

const BREATHING_WINDOW = 60; // frames used for the live breathing frequency and amplitude
const TREND_FRAMES = 5; // compare against this many analyzed frames ago to say rising/falling, growing/shrinking
// Speed is measured over this many analyzed frames, not frame to frame: the box centre jitters by a
// pixel or two, which over one short step reads as a large speed (as SPEED_WINDOW in parti/flake_metrics.py).
const SPEED_WINDOW = 3;

export default {
  name: "rig",

  // What each metric means; the app copies these into the results file next to the measurements.
  // Keys match the start of a metric's name, so units added at the end (px, cm…) don't matter.
  describe: {
    "vortex present %": "share of the frame's columns where the vortex rope is seen",
    "vortex area": "area of the rope in the frame",
    "vortex area rate": "change of the rope's area per second, against 5 analyzed frames before (positive = growing)",
    "vortex size": "growing / shrinking / steady, from the area change (steady = under 1%)",
    "vortex thickness": "mean height of the rope across its columns",
    "vortex center y": "height of the rope's centre in the frame, measured from the top",
    "vortex vertical speed": "speed of the rope's centre up or down per second (positive = rising)",
    "vortex moving": "rising / falling / steady, from the centre's movement (steady = under half a pixel)",
    "vortex waviness": "half the up-and-down span of the rope's centreline (how wavy it is)",
    "vortex breathing": "breathing amplitude: how much the rope expands and contracts (half the p10-p90 range of its thickness over the last 60 analyzed frames)",
    "vortex breathing Hz": "breathing frequency: expansions per second, from crossings of the average thickness over the last 60 analyzed frames",
    "particles in frame": "particles detected in this frame",
    "particles unique": "distinct particles followed since the start",
    "particles mean speed": "average speed of the particles in this frame, each measured over its last 3 analyzed positions",
    "particles dropped frames": "frames skipped so far because the models were still busy",
  },

  defaults: {
    particleMaxDistance: 60, // px per 1/30 s of video a particle may move and keep its id
    particleMaxMissed: 6, // analyzed frames a particle may vanish and keep its id
  },

  create(ctx) {
    const cal = ctx.calibration;
    const unit = cal ? cal.unit : "px";
    const len = (px) => (cal ? px * cal.unitsPerPixel : px); // px -> unit
    const area = (px2) => (cal ? px2 * cal.unitsPerPixel ** 2 : px2); // px² -> unit²
    const vortex = vortexAnalyzer({ len, area, unit });
    const particles = particleAnalyzer(ctx, { len, area, unit });

    return {
      step(result, frame) {
        const t = frame.timestamp; // seconds of video
        const out = { ...result, metrics: {}, overlays: [...(result.overlays ?? [])] };
        if (result.segmentation) vortex.step(result.segmentation, frame, t, out);
        if (result.boxes) particles.step(result.boxes, frame, t, out);
        return out;
      },
      summary() {
        return {
          ...(vortex.used() && { vortex: vortex.summary() }),
          ...(particles.used() && { particles: particles.summary() }),
        };
      },
    };
  },
};

// ---------------------------------------------------------------------------------------------------
// Vortex rope
// ---------------------------------------------------------------------------------------------------

function vortexAnalyzer({ len, area, unit }) {
  const series = []; // one entry per analyzed frame
  let lastT = null;

  return {
    used: () => series.length > 0,

    step(seg, frame, t, out) {
      const rope = measureRope(seg, frame.width, frame.height);
      // A paused video re-runs the same frame: draw it, but don't add it to the time series again.
      const isNew = lastT === null || t > lastT;
      const s = {
        t,
        present: rope.present,
        area: area(rope.areaPx),
        thickness: len(rope.thicknessPx),
        centerY: len(rope.centerYPx),
        waviness: len(rope.wavinessPx),
        wavelength: rope.wavelengthPx ? len(rope.wavelengthPx) : null,
      };
      if (isNew) {
        series.push(s);
        lastT = t;
      }

      // Rates against TREND_FRAMES analyzed frames ago (a single-frame difference is mostly detector noise).
      const ref = series[series.length - 1 - TREND_FRAMES];
      const dt = ref ? s.t - ref.t : 0;
      const areaRate = dt > 0 ? (s.area - ref.area) / dt : null;
      const verticalSpeed = dt > 0 ? -(s.centerY - ref.centerY) / dt : null; // image y grows downward
      const b = breathing(series.slice(-BREATHING_WINDOW));

      Object.assign(out.metrics, {
        "vortex present %": round(s.present * 100, 1),
        [`vortex area ${unit}²`]: round(s.area, 2),
        [`vortex area rate ${unit}²/s`]: areaRate === null ? "-" : round(areaRate, 2),
        "vortex size": areaRate === null ? "-" : trend(areaRate * dt, s.area * 0.01, "growing", "shrinking"),
        [`vortex thickness ${unit}`]: round(s.thickness, 2),
        [`vortex center y ${unit}`]: round(s.centerY, 2),
        [`vortex vertical speed ${unit}/s`]: verticalSpeed === null ? "-" : round(verticalSpeed, 2),
        "vortex moving": verticalSpeed === null ? "-" : trend(verticalSpeed * dt, len(0.5), "rising", "falling"),
        [`vortex waviness ${unit}`]: round(s.waviness, 2),
        [`vortex breathing ${unit}`]: b ? round(b.amplitude, 2) : "-",
        "vortex breathing Hz": b?.frequency ? round(b.frequency, 2) : "-",
      });

      // Draw the rope's edges, centreline and centre.
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
      const thick = valid.map((s) => s.thickness);
      const span = valid.length > 1 ? valid[valid.length - 1].t - valid[0].t : 0;
      const first = valid[0];
      const last = valid[valid.length - 1];
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
          periodS: b.frequency ? round(1 / b.frequency, 5) : null,
          note: "amplitude = half the p10-p90 range of the thickness; frequency from crossings of its mean",
        },
        thickness: stats(thick),
        area: stats(valid.map((s) => s.area)),
        centerY: stats(valid.map((s) => s.centerY)),
        waviness: stats(valid.map((s) => s.waviness)),
        netChange: first && last && span > 0
          ? {
              areaRatePerS: round((last.area - first.area) / span, 4),
              verticalSpeedPerS: round(-(last.centerY - first.centerY) / span, 4),
            }
          : null,
      };
    },
  };
}

/**
 * Per-column top and bottom edge of the rope (the longest run of mask cells in each column), in frame pixels.
 * The mask covers the model's whole canvas, which can reach past the video, so cells outside the frame are skipped.
 */
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
  const centerYPx = center.length ? mean(center) : 0;
  return {
    columns,
    present: colsInFrame ? columns.length / colsInFrame : 0,
    areaPx,
    // Mean over all columns: each column is quantized to the mask's cell height (2 px for the vortex model,
    // which works at half resolution), and averaging hundreds of columns resolves changes well below that.
    thicknessPx: thick.length ? mean(thick) : 0,
    centerYPx,
    centerXPx: columns.length ? mean(columns.map((c) => c.x)) : 0,
    // Robust half-span of the centreline: (p95 - p5) / 2.
    wavinessPx: center.length ? (quantile(center, 0.95) - quantile(center, 0.05)) / 2 : 0,
    // Spatial period of the waves: twice the rope's length divided by how many times the centreline crosses its mean.
    wavelengthPx: wavelength(columns.map((c) => c.x), center),
  };
}

function wavelength(xs, ys) {
  if (ys.length < 8) return null;
  const m = mean(ys);
  let crossings = 0;
  for (let i = 1; i < ys.length; i++) if ((ys[i - 1] - m) * (ys[i] - m) < 0) crossings++;
  return crossings ? (2 * (xs[xs.length - 1] - xs[0])) / crossings : null;
}

/** Expansion/contraction of the thickness over a stretch of frames. */
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
// Particles
// ---------------------------------------------------------------------------------------------------

function particleAnalyzer(ctx, { len, area, unit }) {
  const { particleMaxDistance, particleMaxMissed } = ctx.options;
  // maxGrowth caps how far a particle may jump when frames are skipped (see the tracker's options).
  const tracker = ctx.lib.createTracker({ maxDistance: particleMaxDistance, maxMissed: particleMaxMissed, maxGrowth: 4 });
  const tracks = new Map(); // trackId -> per-particle record
  let frames = 0;
  let firstT = null;
  let lastT = null;

  return {
    used: () => frames > 0,

    step(boxes, frame, t, out) {
      const isNew = lastT === null || t > lastT;
      // Track in video time (the tracker's distance allowance is per 1/30 s of video).
      const tracked = tracker.update(boxes, frame.timestamp);
      if (isNew) {
        frames++;
        firstT ??= t;
        lastT = t;
      }
      const speedsNow = [];
      for (const b of tracked) {
        const x = b.x + b.width / 2;
        const y = b.y + b.height / 2;
        let p = tracks.get(b.trackId);
        if (!p) {
          p = { id: b.trackId, firstSeen: t, lastSeen: t, sightings: 0, path: 0, speedSum: 0, speedN: 0, maxSpeed: 0, sizeSum: 0, trail: [], x, y, t, x0: x, y0: y };
          tracks.set(b.trackId, p);
        }
        if (isNew && t > p.t) {
          // Windowed speed: distance to the position SPEED_WINDOW sightings back, over that time.
          const ref = p.trail[p.trail.length - SPEED_WINDOW];
          if (ref && t > ref.t) {
            const speed = len(Math.hypot(x - ref.x, y - ref.y)) / (t - ref.t);
            p.path += speed * (t - p.t);
            p.speedSum += speed;
            p.speedN++;
            p.maxSpeed = Math.max(p.maxSpeed, speed);
            p.speed = speed;
            speedsNow.push(speed);
          }
        }
        if (p.speed !== undefined) b.label = `${round(p.speed, 1)} ${unit}/s`; // the app adds "#id" itself
        if (isNew) {
          p.sightings++;
          p.sizeSum += area(b.width * b.height);
          Object.assign(p, { x, y, t, lastSeen: t });
          p.trail.push({ x, y, t });
          if (p.trail.length > SPEED_WINDOW + 1) p.trail.shift();
        }
      }

      out.boxes = tracked;
      Object.assign(out.metrics, {
        "particles in frame": tracked.length,
        "particles unique": tracker.total,
        [`particles mean speed ${unit}/s`]: speedsNow.length ? round(mean(speedsNow), 2) : "-",
        "particles dropped frames": frame.dropped,
      });
    },

    summary() {
      const list = [...tracks.values()].map((p) => {
        const dt = p.lastSeen - p.firstSeen;
        return {
          id: p.id,
          firstSeen: round(p.firstSeen, 6),
          lastSeen: round(p.lastSeen, 6),
          sightings: p.sightings,
          meanSpeed: p.speedN ? round(p.speedSum / p.speedN, 4) : null,
          maxSpeed: p.speedN ? round(p.maxSpeed, 4) : null,
          // Net velocity, start to end: negative along = right to left; negative across = upward.
          alongPerS: dt > 0 ? round(len(p.x - p.x0) / dt, 4) : null,
          acrossPerS: dt > 0 ? round(len(p.y - p.y0) / dt, 4) : null,
          pathLength: round(p.path, 4),
          meanSize: round(p.sizeSum / p.sightings, 4),
          equivalentDiameter: round(2 * Math.sqrt(p.sizeSum / p.sightings / Math.PI), 4),
        };
      });
      const moving = list.filter((p) => p.meanSpeed !== null);
      const span = lastT - firstT;
      return {
        unit,
        timeNote: "times are seconds of video",
        framesAnalyzed: frames,
        samplingRateHz: span > 0 ? round((frames - 1) / span, 3) : null,
        particlesTotal: list.length,
        // Particles seen in at least 3 analyzed frames; the rest are often detector noise.
        particlesSeen3Plus: list.filter((p) => p.sightings >= 3).length,
        meanSpeed: moving.length ? round(mean(moving.map((p) => p.meanSpeed)), 4) : null,
        medianSpeed: moving.length ? round(median(moving.map((p) => p.meanSpeed)), 4) : null,
        meanAlongPerS: moving.length ? round(mean(moving.filter((p) => p.alongPerS !== null).map((p) => p.alongPerS)), 4) : null,
        meanEquivalentDiameter: list.length ? round(mean(list.map((p) => p.equivalentDiameter)), 4) : null,
        particles: list,
      };
    },
  };
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
const median = (xs) => quantile(xs, 0.5);

function quantile(xs, q) {
  const s = [...xs].sort((a, b) => a - b);
  const i = (s.length - 1) * q;
  const lo = Math.floor(i);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo);
}

function round(v, digits) {
  return Number.isFinite(v) ? Math.round(v * 10 ** digits) / 10 ** digits : v;
}
