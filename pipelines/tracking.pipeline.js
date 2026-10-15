// Tracking pipeline: gives every detection an id that stays the same while the object is followed, and
// shows its speed next to it. Embedded in particle_detector_with_tracking.onnx. It adds no analysis beyond
// that; the particles pipeline (particles.pipeline.js) is the one with the full measurements.
//
// Draw a line of known length in step 5 (Calibrate) so speeds are in real units (else pixels).

export default {
  name: "tracking",

  describe: {
    "in frame": "objects detected in this frame",
    unique: "distinct objects followed since the start",
    dropped: "frames skipped so far because the models were still busy",
  },

  create(ctx) {
    const tracker = ctx.lib.createTracker({ maxDistance: 60, maxMissed: 6, maxGrowth: 4 });
    const cal = ctx.calibration;
    const unit = cal ? cal.unit : "px";
    const last = new Map(); // trackId -> { x, y, t, speed }

    return {
      step(result, frame) {
        const boxes = tracker.update(result.boxes ?? [], frame.timestamp);
        for (const b of boxes) {
          const x = b.x + b.width / 2;
          const y = b.y + b.height / 2;
          const prev = last.get(b.trackId);
          let speed = prev?.speed;
          if (prev && frame.timestamp > prev.t) {
            speed = Math.hypot(x - prev.x, y - prev.y) / (frame.timestamp - prev.t);
            if (cal) speed *= cal.unitsPerPixel;
          }
          // A paused video re-runs the same frame (no time passed), so keep the last speed.
          if (speed !== undefined) b.label = `${b.label ?? b.classId} ${speed.toFixed(1)} ${unit}/s`;
          last.set(b.trackId, { x, y, t: frame.timestamp, speed });
        }
        return { ...result, boxes, metrics: { "in frame": boxes.length, unique: tracker.total, dropped: frame.dropped } };
      },
      summary: () => ({ total: tracker.total }),
    };
  },
};
