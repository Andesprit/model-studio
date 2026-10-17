# Writing pipelines

A pipeline is an ES module (a `.js` file) whose default export describes what to measure. It can't
`import` anything; what it needs from the app comes in through `ctx`.

```js
export default {
  name: "my-pipeline",

  // Optional: what each metric means. Copied into the results file. Keys match the start of a
  // metric's name, so "area" describes "area px²" and "area cm²".
  describe: { "area": "area of the thing in the frame" },

  // Optional: decode the model's raw output yourself, if no built-in decoder understands it.
  // decode(outputs, ctx) { return { boxes: [...] }; },

  // create runs once, and again on every reset (loop, seek back, new video, new calibration...).
  // Whatever it declares is the pipeline's memory.
  create(ctx) {
    const tracker = ctx.lib.createTracker();
    return {
      // Runs on every analyzed frame with that frame's detections.
      step(result, frame) {
        const boxes = tracker.update(result.boxes ?? [], frame.timestamp);
        return { ...result, boxes, metrics: { "unique": tracker.total } };
      },
      // Optional: anything to add to the results file.
      summary() {
        return { total: tracker.total };
      },
    };
  },
};
```

## What a pipeline gets

**`ctx`** (in `create`):

| Field | What it is |
|---|---|
| `ctx.calibration` | `null`, or `{ pixels, amount, unit, unitsPerPixel }` from **Advanced → Calibration** |
| `ctx.options` | the pipeline's `defaults` |
| `ctx.labels` | the model's class names |
| `ctx.lib.createTracker({ maxDistance, maxMissed, maxGrowth, sameClass })` | the tracker: `update(boxes, time)` gives each box a `trackId`; `total`, `active` |
| `ctx.lib.nms`, `toFrameRect`, `toFramePoint`, `sigmoid`, `softmax`, `decode(name, outputs)` | helpers and the built-in decoders |

**`frame`** (in `step`): `timestamp` (s of video), `index`, `dropped` (frames skipped so far), `width`,
`height`.

**`result`** (in `step`): what the decoder produced, in video pixels: `boxes`, `masks`, `segmentation`,
`poses`, `classes`. A pipeline loaded in **Advanced → Pipeline** also gets a third argument, `models`,
with each model's result separately (`models.A`, `models.B`).

## What it returns

The result to draw, plus optionally:

- `metrics: { name: value }`: shown in the bar under the video. Numbers are also summarized (mean, min, max)
  in the results file.
- `overlays: [...]`: extra drawings in video pixels: `{ type: "line", from, to }`,
  `{ type: "polyline", points }`, `{ type: "point", at }`, `{ type: "text", at, text }`, each with an
  optional `color`.

## Setting the model's other inputs

A model can have inputs besides the image, for example a switch that picks which part of the graph runs.
The model's own pipeline sets them with an optional `feeds()` next to `step`. The app calls it before each
frame and passes the values to the model:

```js
create(ctx) {
  let mode = 0;
  return {
    step(result, frame) { /* ... decide something, set mode ... */ return result; },
    feeds: () => ({ mode: { dims: [1], data: new Float32Array([mode]) } }),
  };
}
```

- Keys are the model's input names, and values are float32 tensors (`dims`, `data`).
- Any input that isn't supplied gets zeros of its declared shape. That includes the first frame, before the
  pipeline has an instance, and the first frame after a rewind.
- Only a model's own (embedded) pipeline is asked; `feeds()` is ignored for a pipeline loaded in
  **Advanced → Pipeline**.

The lamp pipeline uses this to run its model's network once per frame instead of twice
(`pipelines/lamp.pipeline.js`).

## Putting a pipeline inside a model

```bash
python scripts/embed_pipeline.py models/vortex_detector_fp32.onnx pipelines/vortex.pipeline.js
python scripts/embed_pipeline.py --extract models/vortex_detector_fp32_with_pipeline.onnx out.js
python scripts/embed_pipeline.py --remove MODEL_with_pipeline.onnx MODEL.onnx
```

Embedding writes `MODEL_with_pipeline.onnx`, keeping the model's other metadata (manifest, Ultralytics
keys); the graph isn't touched. It needs the `onnx` Python package. **Re-run it after every change** to the
pipeline, and copy the new model to `apps/web/public/models/`.

The sources of the embedded pipelines (vortex, tracking, lamp) are in `pipelines/`. Edit those, not the copy inside the model.

## Examples

In `examples/`, for other kinds of models:

- `tracking.pipeline.js`: commented example with both a hand-written YOLOv8 decoder and tracking with
  speed.
- `held-objects.pipeline.js`: two models (pose + detect): which objects people are holding.
- `people-counter.plugin.js`: a decoder only.
- `rig.pipeline.js`: vortex and particles in one pipeline, from before each model had its own.

## Tips from building these

- **Measure speed over a few frames**, not frame to frame: the box centre jitters a pixel or two, which
  over one short step reads as a large speed. The particle pipeline uses 3 analyzed positions.
- **Skipped frames**: between two analyzed frames a lot can happen. Keep the tracker's `maxGrowth` cap (4 by
  default) so a long gap can't link far-apart particles.
- **A paused video re-runs the same frame**: check `frame.timestamp` against the last one before adding to a
  time series, or the frame counts twice.
- **Mask resolution**: the vortex mask is at half resolution (2 px per cell), so per-column heights move in
  2 px steps. Averaging over all columns resolves much smaller changes.
