// Vortex metrics pipeline. Load a vortex_detector_*.onnx (its manifest is embedded, so no config is
// needed), then load this file with "Pipeline .js". Calibrate in step 5 to get areas and speeds in
// real units instead of pixels.
//
// No decode here: the model's manifest already selects the built-in binary-mask decoder, which
// gives `result.segmentation`. This pipeline only reads that mask and adds:
//
//   metrics (bar under the video; "Download results" saves their mean / min / max / last)
//     coverage %   % of the frame covered by the mask
//     detected     1 when coverage >= minCoverage, else 0 (its mean = share of frames with a vortex)
//     vortices     separate blobs in this frame (blobs smaller than minBlobArea are ignored)
//     largest      area of the biggest blob (px² or unit² when calibrated)
//     unique       distinct vortices tracked since the start
//     dropped      frames skipped because inference was slower than the video
//   boxes          one per blob, with a track id, its area and speed
//   summary()      per-vortex area and speed for the results file

export default {
  name: "vortex-metrics",

  defaults: {
    minBlobArea: 50, // frame px²; smaller blobs are treated as noise
    minCoverage: 0.1, // % of the frame needed to say "detected"
  },

  create(ctx) {
    const { minBlobArea, minCoverage } = ctx.options;
    const tracker = ctx.lib.createTracker({ maxDistance: 40, maxMissed: 5 });
    const cal = ctx.calibration;
    const unit = cal ? cal.unit : "px";
    const areaUnit = `${unit}²`;
    const toArea = (px2) => (cal ? px2 * cal.unitsPerPixel ** 2 : px2);
    const last = new Map(); // trackId -> { x, y, t, speed }
    const vortices = new Map(); // trackId -> per-vortex totals for the results file

    return {
      step(result, frame) {
        const seg = result.segmentation;
        if (!seg) return result; // not a mask model

        const blobs = findBlobs(seg, frame.width, frame.height);
        const framePixels = frame.width * frame.height;
        const covered = blobs.reduce((sum, b) => sum + b.area, 0);
        const coverage = (100 * covered) / framePixels;
        const kept = blobs.filter((b) => b.area >= minBlobArea);

        // Each blob becomes a box so the tracker can follow it between frames.
        const boxes = tracker.update(
          kept.map((b) => ({ x: b.x, y: b.y, width: b.width, height: b.height, classId: 0, score: 1, area: b.area })),
          frame.timestamp,
        );

        for (const box of boxes) {
          const cx = box.x + box.width / 2;
          const cy = box.y + box.height / 2;
          const prev = last.get(box.trackId);
          let speed = prev?.speed;
          const isNewFrame = !prev || frame.timestamp > prev.t; // a paused video re-runs the same frame
          const v = vortices.get(box.trackId) ?? { firstSeen: frame.timestamp, areaSum: 0, areaSamples: 0, maxArea: 0, speedSum: 0, speedSamples: 0, maxSpeed: 0 };
          v.lastSeen = frame.timestamp;
          if (isNewFrame) {
            const a = toArea(box.area);
            v.areaSum += a;
            v.areaSamples++;
            v.maxArea = Math.max(v.maxArea, a);
          }
          if (prev && frame.timestamp > prev.t) {
            speed = Math.hypot(cx - prev.x, cy - prev.y) / (frame.timestamp - prev.t); // px/s
            if (cal) speed *= cal.unitsPerPixel;
            v.speedSum += speed;
            v.speedSamples++;
            v.maxSpeed = Math.max(v.maxSpeed, speed);
          }
          vortices.set(box.trackId, v);
          last.set(box.trackId, { x: cx, y: cy, t: frame.timestamp, speed });
          box.label = `vortex ${fmt(toArea(box.area))} ${areaUnit}` + (speed !== undefined ? ` ${speed.toFixed(1)} ${unit}/s` : "");
        }

        const largest = kept.reduce((max, b) => Math.max(max, b.area), 0);
        return {
          ...result,
          boxes,
          // Numbers (not text) so the results file can average them.
          metrics: {
            "coverage %": round(coverage, 2),
            detected: coverage >= minCoverage ? 1 : 0,
            vortices: kept.length,
            [`largest ${areaUnit}`]: round(toArea(largest), 3),
            unique: tracker.total,
            dropped: frame.dropped,
          },
        };
      },

      summary() {
        const list = [...vortices].map(([id, v]) => ({
          id,
          firstSeen: v.firstSeen,
          lastSeen: v.lastSeen,
          meanArea: v.areaSamples ? v.areaSum / v.areaSamples : null,
          maxArea: v.areaSamples ? v.maxArea : null,
          meanSpeed: v.speedSamples ? v.speedSum / v.speedSamples : null,
          maxSpeed: v.speedSamples ? v.maxSpeed : null,
        }));
        const withSpeed = list.filter((v) => v.meanSpeed !== null);
        return {
          areaUnit,
          speedUnit: `${unit}/s`,
          vorticesTotal: list.length,
          meanSpeed: withSpeed.length ? withSpeed.reduce((s, v) => s + v.meanSpeed, 0) / withSpeed.length : null,
          vortices: list,
        };
      },
    };
  },
};

const round = (v, digits) => Math.round(v * 10 ** digits) / 10 ** digits;
/** Short display form of an area. */
const fmt = (a) => (a >= 100 ? String(Math.round(a)) : a.toPrecision(3));

/**
 * Connected blobs of the mask (8-neighbour), measured in frame pixels.
 * The mask covers the whole model canvas, which can reach past the video (it's padded), so cells
 * whose center falls outside the frame are ignored.
 */
function findBlobs(seg, frameWidth, frameHeight) {
  const { width: w, height: h, classes, rect, background } = seg;
  const cellW = rect.width / w; // frame px per mask cell
  const cellH = rect.height / h;
  const inside = (i) => {
    const x = rect.x + ((i % w) + 0.5) * cellW;
    const y = rect.y + (Math.floor(i / w) + 0.5) * cellH;
    return x >= 0 && y >= 0 && x < frameWidth && y < frameHeight;
  };
  const isMask = (i) => classes[i] !== background && inside(i);

  const seen = new Uint8Array(w * h);
  const stack = [];
  const blobs = [];
  for (let start = 0; start < w * h; start++) {
    if (seen[start] || !isMask(start)) continue;
    let cells = 0, minX = w, minY = h, maxX = 0, maxY = 0;
    seen[start] = 1;
    stack.push(start);
    while (stack.length) {
      const i = stack.pop();
      const x = i % w, y = (i - x) / w;
      cells++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const n = ny * w + nx;
          if (!seen[n] && isMask(n)) {
            seen[n] = 1;
            stack.push(n);
          }
        }
      }
    }
    blobs.push({
      area: cells * cellW * cellH,
      x: rect.x + minX * cellW,
      y: rect.y + minY * cellH,
      width: (maxX - minX + 1) * cellW,
      height: (maxY - minY + 1) * cellH,
    });
  }
  return blobs;
}
