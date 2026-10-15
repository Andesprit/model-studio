// Particle pipeline: tracking, speed and count. Not embedded in any shipped model: load it with
// "Pipeline .js" on any model that gives boxes (e.g. the particle detector).
//
// Draw a line of known length in step 5 (Calibrate) so sizes and speeds are in real units (else pixels).
// Times and rates are in seconds of video.
//
// Measured: particles in frame, unique particles followed, mean speed, plus per particle: speed,
// direction (along / across the frame), path length and size.

// Speed is measured over this many analyzed frames, not frame to frame: the box centre jitters by a
// pixel or two, which over one short step reads as a large speed (as SPEED_WINDOW in parti/flake_metrics.py).
const SPEED_WINDOW = 3;

export default {
  name: "particles",

  // What each metric means; the app copies these into the results file next to the measurements.
  describe: {
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
    const particles = particleAnalyzer(ctx, { len, area, unit });
    return {
      step(result, frame) {
        const out = { ...result, metrics: {}, overlays: [...(result.overlays ?? [])] };
        if (result.boxes) particles.step(result.boxes, frame, frame.timestamp, out);
        return out;
      },
      summary: () => (particles.used() ? particles.summary() : {}),
    };
  },
};

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
