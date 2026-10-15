// Example pipeline. Load yolov8n.onnx, then load this file with "Pipeline .js".
//
// A pipeline is an ES module whose default export has a `name` plus one or both of:
//
//   decode(outputs, ctx)  Turns the model's raw output tensors into boxes. Only needed when no
//                         built-in decoder understands your model; when present it replaces the
//                         decoder picked in step 2.
//   create(ctx)           Runs once (and again after every reset) and returns { step(result, frame) }.
//                         `step` runs on every frame after decoding and can remember things between
//                         frames: tracking, counting, speeds...
//
// Pipelines can't `import` anything. What they need from the app comes in `ctx.lib`.

export default {
  name: "tracking-example",

  // ---------------------------------------------------------------------------------------------
  // decode: a hand-written YOLOv8 decoder. The app already has one ("yolo-detect"), so in a real
  // pipeline for YOLOv8 you'd delete this whole function. It's here to show what one looks like.
  //
  //   outputs: { [outputName]: { dims: number[], data: Float32Array } }
  //   ctx:     { options, labels, transform, lib: { nms, toFrameRect, decode, ... } }
  //   returns: { boxes: [{ x, y, width, height, classId, score, label }] } in video pixels
  // ---------------------------------------------------------------------------------------------
  decode(outputs, ctx) {
    const out = Object.values(outputs)[0]; // YOLOv8: [1, 4 + classes, anchors], e.g. [1, 84, 8400]
    const [, channels, anchors] = out.dims;
    const d = out.data;
    const at = (channel, anchor) => d[channel * anchors + anchor];

    const boxes = [];
    for (let i = 0; i < anchors; i++) {
      // Rows 0-3 are the box (center x, center y, width, height); the rest are one score per class.
      let classId = 0;
      let score = 0;
      for (let c = 4; c < channels; c++) {
        if (at(c, i) > score) {
          score = at(c, i);
          classId = c - 4;
        }
      }
      if (score < 0.25) continue;

      const cx = at(0, i), cy = at(1, i), w = at(2, i), h = at(3, i);
      // The model sees a resized, padded image; toFrameRect maps the box back onto the video.
      const rect = ctx.lib.toFrameRect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, ctx.transform);
      boxes.push({ ...rect, classId, score, label: ctx.labels[classId] });
    }
    // Remove overlapping duplicates of the same object.
    return { boxes: ctx.lib.nms(boxes, 0.45) };

    // The short version, reusing the built-in decoder:
    //   return ctx.lib.decode("yolo-detect", outputs);
  },

  // ---------------------------------------------------------------------------------------------
  // create: tracking plus speed. Everything declared here is the pipeline's memory.
  //
  //   ctx:   { labels, calibration, lib: { createTracker, ... } }
  //          calibration is null, or { pixels, amount, unit, unitsPerPixel } from step 5.
  //   frame: { timestamp (s), index, dropped, width, height }
  //   step returns the result to draw. Extra fields you can add:
  //          metrics:  { name: value }   shown in the bar under the video. Numeric ones also go
  //                                      into "Download results" as mean / min / max / last.
  //          overlays: [{ type: "line", from, to } | { type: "text", at, text } | ...]
  //
  //   summary() is optional: whatever it returns is added to the downloaded results file.
  // ---------------------------------------------------------------------------------------------
  create(ctx) {
    const tracker = ctx.lib.createTracker({ maxDistance: 50, maxMissed: 5 });
    const cal = ctx.calibration;
    const unit = cal ? cal.unit : "px";
    const last = new Map(); // trackId -> { x, y, t, speed }
    const particles = new Map(); // trackId -> per-particle totals for the results file

    return {
      step(result, frame) {
        // Gives every box a trackId that stays the same while the object is followed.
        const boxes = tracker.update(result.boxes ?? [], frame.timestamp);

        for (const b of boxes) {
          const x = b.x + b.width / 2;
          const y = b.y + b.height / 2;
          const prev = last.get(b.trackId);
          let speed = prev?.speed;
          const p = particles.get(b.trackId) ?? { label: b.label ?? b.classId, firstSeen: frame.timestamp, speedSum: 0, speedSamples: 0, maxSpeed: 0 };
          p.lastSeen = frame.timestamp;
          if (prev && frame.timestamp > prev.t) {
            speed = Math.hypot(x - prev.x, y - prev.y) / (frame.timestamp - prev.t); // px/s
            if (cal) speed *= cal.unitsPerPixel; // calibrated: cm/s, m/s...
            p.speedSum += speed;
            p.speedSamples++;
            p.maxSpeed = Math.max(p.maxSpeed, speed);
          }
          particles.set(b.trackId, p);
          // A paused video re-runs the same frame (no time passed), so keep the last speed.
          if (speed !== undefined) b.label = `${b.label ?? b.classId} ${speed.toFixed(1)} ${unit}/s`;
          last.set(b.trackId, { x, y, t: frame.timestamp, speed });
        }

        return {
          ...result,
          boxes,
          // "in frame" averaged over the video = particles per frame on average.
          metrics: { "in frame": boxes.length, unique: tracker.total, dropped: frame.dropped },
        };
      },

      summary() {
        const list = [...particles].map(([id, p]) => ({
          id,
          label: p.label,
          firstSeen: p.firstSeen,
          lastSeen: p.lastSeen,
          meanSpeed: p.speedSamples ? p.speedSum / p.speedSamples : null,
          maxSpeed: p.speedSamples ? p.maxSpeed : null,
        }));
        const withSpeed = list.filter((p) => p.meanSpeed !== null);
        return {
          speedUnit: `${unit}/s`,
          particlesTotal: list.length,
          // Mean of each particle's mean speed, so fast-moving and long-lived particles weigh the same.
          meanSpeed: withSpeed.length ? withSpeed.reduce((s, p) => s + p.meanSpeed, 0) / withSpeed.length : null,
          particles: list,
        };
      },
    };
  },
};
