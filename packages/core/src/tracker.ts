import type { Box } from "./types";

export interface TrackerOptions {
  /**
   * Max pixels a box center may move per 1/30 s and still keep its id. The allowance grows with
   * the time since the track was last seen, so dropped frames don't break tracks. Default 50.
   */
  maxDistance?: number;
  /** Updates a track may go unmatched before it ends. Default 5. */
  maxMissed?: number;
  /** Only match boxes of the same class. Default true. */
  sameClass?: boolean;
}

export interface Tracker {
  /** Assigns a `trackId` to each box. `timestamp` is in seconds; without it each update counts as 1/30 s. */
  update<T extends Box>(boxes: T[], timestamp?: number): (T & { trackId: number })[];
  /** Number of distinct tracks started so far. */
  readonly total: number;
  /** Tracks currently alive (matched recently, within `maxMissed`). */
  readonly active: number;
}

interface Track {
  id: number;
  x: number;
  y: number;
  classId: number;
  seenAt: number;
  missed: number;
}

const STEP = 1 / 30;

/** Greedy nearest-center tracker: closest box/track pairs are matched first; ids are never reused. */
export function createTracker(options: TrackerOptions = {}): Tracker {
  const maxDistance = options.maxDistance ?? 50;
  const maxMissed = options.maxMissed ?? 5;
  const sameClass = options.sameClass ?? true;
  let tracks: Track[] = [];
  let nextId = 1;
  let clock = 0;

  return {
    get total() {
      return nextId - 1;
    },
    get active() {
      return tracks.length;
    },
    update(boxes, timestamp) {
      clock = timestamp ?? clock + STEP;
      const centers = boxes.map((b) => ({ x: b.x + b.width / 2, y: b.y + b.height / 2 }));

      const pairs: { t: number; b: number; d: number }[] = [];
      tracks.forEach((tr, t) => {
        const gate = maxDistance * Math.max(1, (clock - tr.seenAt) / STEP);
        boxes.forEach((box, b) => {
          if (sameClass && box.classId !== tr.classId) return;
          const d = Math.hypot(centers[b].x - tr.x, centers[b].y - tr.y);
          if (d <= gate) pairs.push({ t, b, d });
        });
      });
      pairs.sort((p, q) => p.d - q.d);

      const trackOf = new Array<number>(boxes.length).fill(-1);
      const usedTrack = new Set<number>();
      for (const { t, b } of pairs) {
        if (usedTrack.has(t) || trackOf[b] !== -1) continue;
        usedTrack.add(t);
        trackOf[b] = t;
      }

      const out = boxes.map((box, b) => {
        let tr = tracks[trackOf[b]];
        if (!tr) {
          tr = { id: nextId++, x: 0, y: 0, classId: box.classId, seenAt: clock, missed: 0 };
          tracks.push(tr);
        }
        Object.assign(tr, { x: centers[b].x, y: centers[b].y, seenAt: clock, missed: 0 });
        return { ...box, trackId: tr.id };
      });

      tracks = tracks.filter((tr) => tr.seenAt === clock || ++tr.missed <= maxMissed);
      return out;
    },
  };
}
