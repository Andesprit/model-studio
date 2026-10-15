// Two-model example: which objects are people holding?
//
// Load "YOLOv8n pose" as the first model and "YOLOv8n detect" as the second (step 1), then this
// file (step 3). Point the camera at someone holding a cup, bottle, phone, book, remote...
//
// The pose model finds people and their wrists; the detect model finds objects. An object counts
// as held when a wrist is inside its box. Neither model can tell that alone.
//
// With two models, step gets a third argument:
//   step(result, frame, models)
//     result  both models' output merged (what gets drawn): result.poses + result.boxes
//     models  each model separately: models.A, models.B (in load order)
// This pipeline only needs the merged result, since poses only come from the pose model and boxes
// only from the detector, so it works whichever order the models were loaded in.

const LEFT_WRIST = 9; // COCO keypoint indices
const RIGHT_WRIST = 10;

export default {
  name: "held-objects",

  defaults: {
    margin: 0.15, // grow object boxes by 15% so a hand at the edge still counts
    minKeypointScore: 0.5,
  },

  create(ctx) {
    const { margin, minKeypointScore } = ctx.options;
    const tracker = ctx.lib.createTracker({ maxDistance: 60, maxMissed: 8 });
    const objects = new Map(); // trackId -> { label, seen, held } frame counts for the results file

    return {
      step(result, frame, models) {
        const people = result.poses ?? [];
        const wrists = people
          .flatMap((p) => [p.keypoints[LEFT_WRIST], p.keypoints[RIGHT_WRIST]])
          .filter((k) => k && k.score >= minKeypointScore);

        // People are already covered by the pose model; track everything else.
        const things = tracker.update(
          (result.boxes ?? []).filter((b) => b.label !== "person"),
          frame.timestamp,
        );

        let held = 0;
        for (const b of things) {
          const mx = b.width * margin;
          const my = b.height * margin;
          const isHeld = wrists.some(
            (w) => w.x >= b.x - mx && w.x <= b.x + b.width + mx && w.y >= b.y - my && w.y <= b.y + b.height + my,
          );
          const o = objects.get(b.trackId) ?? { label: b.label ?? b.classId, seen: 0, held: 0 };
          o.seen++;
          if (isHeld) {
            o.held++;
            held++;
            b.label = `${b.label ?? b.classId} HELD`;
          }
          objects.set(b.trackId, o);
        }

        return {
          ...result,
          boxes: things,
          // Mark every wrist the pipeline is using.
          overlays: wrists.map((w) => ({ type: "point", at: { x: w.x, y: w.y } })),
          metrics: {
            people: people.length,
            objects: things.length,
            held,
            models: Object.keys(models).length,
          },
        };
      },

      summary() {
        const list = [...objects].map(([id, o]) => ({ id, label: o.label, framesSeen: o.seen, framesHeld: o.held }));
        return {
          objectsTotal: list.length,
          objectsEverHeld: list.filter((o) => o.framesHeld > 0).length,
          objects: list,
        };
      },
    };
  },
};
