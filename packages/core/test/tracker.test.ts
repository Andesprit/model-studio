import { describe, expect, it } from "vitest";
import { createTracker, type Box } from "../src";

const box = (x: number, y = 0, classId = 0): Box => ({ x, y, width: 10, height: 10, classId, score: 1 });
const ids = (boxes: { trackId: number }[]) => boxes.map((b) => b.trackId);

describe("tracker", () => {
  it("keeps ids for objects moving within maxDistance", () => {
    const t = createTracker({ maxDistance: 20 });
    expect(ids(t.update([box(0), box(100)]))).toEqual([1, 2]);
    // Order of boxes changes; matching is by position, not index.
    expect(ids(t.update([box(110), box(10)]))).toEqual([2, 1]);
    expect(t.total).toBe(2);
  });

  it("starts a new id when a box jumps too far or changes class", () => {
    const t = createTracker({ maxDistance: 20 });
    t.update([box(0)]);
    expect(ids(t.update([box(50)]))).toEqual([2]);
    expect(ids(t.update([box(50, 0, 1)]))).toEqual([3]);
  });

  it("ends a track after maxMissed updates and never reuses ids", () => {
    const t = createTracker({ maxDistance: 20, maxMissed: 2 });
    t.update([box(0)], 0);
    t.update([], 0.1);
    t.update([], 0.2);
    expect(t.active).toBe(1);
    t.update([], 0.3);
    expect(t.active).toBe(0);
    expect(ids(t.update([box(0)], 0.4))).toEqual([2]);
  });

  it("widens the gate with elapsed time, so dropped frames don't break tracks", () => {
    const t = createTracker({ maxDistance: 20 });
    t.update([box(0)], 0);
    // 3 frame-steps later (0.1 s) the allowance is 60 px.
    expect(ids(t.update([box(50)], 0.1))).toEqual([1]);
  });

  it("caps the gate growth, so a long gap can't link far-apart boxes", () => {
    const t = createTracker({ maxDistance: 20, maxGrowth: 4 });
    t.update([box(0)], 0);
    // 1.4 s later the uncapped allowance would be 840 px; the cap keeps it at 80 px.
    expect(ids(t.update([box(70)], 1.4))).toEqual([1]);
    expect(ids(t.update([box(300)], 2.8))).toEqual([2]);
  });

  it("gives the closest pair priority", () => {
    const t = createTracker({ maxDistance: 20 });
    t.update([box(0)]);
    // Both are in range; the closer one keeps id 1.
    expect(ids(t.update([box(15), box(5)]))).toEqual([2, 1]);
  });
});
