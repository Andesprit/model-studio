# How it works

## The big picture

```
video / camera ──► grab the frame ──► model A ──► decoder ──► A's pipeline ──┐
                     (once)       └─► model B ──► decoder ──► B's pipeline ──┼─► drawn on the video
                                                                             └─► metrics ─► results file
```

For each analyzed frame the app:

1. grabs the video frame once and prepares it separately for each model (each has its own input size and
   colours), so both models see exactly the same frame;
2. runs model A, then model B, on the GPU (WebGPU) or the CPU (WASM);
3. **decodes** each model's raw output into something meaningful: a mask, boxes, keypoints or classes;
4. passes each model's result to **its own pipeline**, which measures things over time (area, speed,
   tracking…);
5. draws everything over the video and records the measurements.

The code is split in two:

- **`packages/core`**: all the logic, with no browser code: reading configs, preparing frames, decoders,
  pipelines, the tracker, calibration. It could be reused as-is in a future mobile app.
- **`apps/web`**: the browser part: the UI, the camera and video, running the models with onnxruntime-web,
  drawing.

## Live analysis and skipped frames

The video plays at its normal speed and the app analyzes whatever frame is on screen whenever the models
are free. Frames that pass while the models are busy are **skipped**, never queued, so the overlay never
falls behind. The **dropped** number in the bar counts them.

With both rig models on a desktop GPU, one frame takes about 0.75 s, so roughly 1 in 20 frames of a 30 fps
video gets analyzed. This matters for measuring; see [findings](findings.md#skipped-frames-limit-what-can-be-measured).

Measurements start over (a pipeline is reset) when the video loops or you seek back, when you change the
video or camera, change a model, or change the calibration. A paused video re-analyzes the same frame
when a setting changes, but that frame isn't counted twice.

## Models and how they're configured

The app needs three things to use an ONNX model: how to prepare the image for it, how to read its output,
and (optionally) what to measure. It finds them in this order:

1. **A config file** (step 2, `Config .json`), if you load one. An example is in
   `examples/semantic-seg.example.json`.
2. **Metadata inside the `.onnx`**:
   - a **detector-app manifest** (key `detector.manifest`, written by detector-app's `embed_manifest.py`).
     The vortex model has one: grayscale, scaled to half size onto a 416×160 canvas, mask = logits above 0;
   - **Ultralytics metadata** (task, class names, image size). The particle model has it: YOLO26s-P2,
     1 class "particle", 1280×1280 input.
3. **The model's input shape** alone. The app then guesses the decoder and shows a yellow warning, because
   the guess can be wrong.

If a decoder can't read a model's output, the error says which model, which decoder, the output shapes, and
what to try.

### Decoders

| Decoder | Reads | Used by |
|---|---|---|
| `semantic-seg` | a mask: per-class logits, class indices, or one-channel logits (binary) | vortex model |
| `yolo-detect` | YOLO boxes (v5/v8/v11/v26 layouts, auto-detected) + NMS | particle model |
| `yolo-seg` | YOLO boxes + instance masks | |
| `yolo-pose` | YOLO boxes + keypoints | |
| `classify` | class scores | |

### The model list

`apps/web/public/models/index.json` is the list the app shows under **Change model**, and the first two
entries are loaded on startup. An entry can carry recommended decoder options; the particle model's are
from its training notes (confidence 0.20, overlap 0.35, up to 1000 detections, since the default 300 cuts
off dense frames).

## Pipelines

A pipeline is a small JavaScript file that turns each frame's detections into measurements. Each rig model
can carry its own **inside the `.onnx`** (metadata key `model-studio.pipeline`), so it loads with the model:

| Model file | Pipeline inside | Source |
|---|---|---|
| `vortex_detector_fp32_with_pipeline.onnx` | `vortex` | `pipelines/vortex.pipeline.js` |
| `particle_detector_with_tracking.onnx` | `tracking` | `pipelines/tracking.pipeline.js` |
| `particle_detector.onnx` | none (detection only) | `pipelines/particles.pipeline.js` can be loaded on top in step 3 |
| `vortex_lamp_auto_fp32.onnx` | `vortex-lamp` | `pipelines/lamp.pipeline.js` |

Each model's pipeline runs on that model's output only. A model opened from disk asks before running its
pipeline, since that's code; the app's own models don't ask.

Step 3 can load one more pipeline that runs on **both** models' results together (for analyses that relate
them, like "particles inside the vortex"). Writing and embedding pipelines is in [pipelines.md](pipelines.md).

## Calibration

Drawing a line over something of known length and entering that length gives the scale (units per pixel).
Pipelines then report lengths, areas and speeds in that unit. The line is measured in the video's real
pixels, not the size it's shown at on screen. Without calibration everything is in pixels.

Times and rates are in **seconds of video**. For high-speed footage (recorded at 2000 fps, played at 30
fps), real rates are much higher; see [findings](findings.md#high-speed-footage-and-time).

## What gets measured

### Vortex (`vortex` pipeline)

The rope runs left to right, so it's measured column by column: for each column of the mask, its top and
bottom edge.

| Measurement | Meaning |
|---|---|
| present % | share of the frame's columns where the rope is seen |
| area | rope area in the frame |
| area rate | change of area per second, against 5 analyzed frames before (positive = growing) |
| size | growing / shrinking / steady (steady = change under 1%) |
| thickness | mean height of the rope across its columns |
| center y | height of the rope's centre from the top of the frame (the red dot) |
| vertical speed | how fast the centre moves (positive = rising) |
| moving | rising / falling / steady (steady = under half a pixel) |
| waviness | half the up-and-down span of the centreline: how wavy the rope is |
| breathing | how much the rope expands and contracts: half the p10–p90 range of its thickness over the last 60 analyzed frames |
| breathing Hz | expansions per second, from crossings of the average thickness over the last 60 analyzed frames |

Drawn: the rope's top and bottom edges (blue), its centreline (yellow) and its centre (red dot).

### Vortex lamp (`vortex-lamp` pipeline)

For the vortex lamp filmed with a phone. The model only knows a rope that runs left to right. Its graph runs
the network twice on every frame and returns two masks:
- `logits`: the frame turned 90° and the mask turned back, which is right for a rope running up and down;
- `logits_asis`: the frame as it is.

The pipeline picks between them:

1. It scores both masks on every frame: mean probability of the rope pixels × share of those pixels in the
   largest piece. The wrong orientation gives a weak, scattered mask.
2. It adds up the scores over the first 10 analyzed frames, then locks the better orientation until the next
   reset (loop, seek back, new video).
3. It keeps only the largest piece of the mask, since there is one rope.
4. It measures the rope slice by slice across its own axis: rows for a vertical rope, columns for a
   horizontal one.

| Measurement | Meaning |
|---|---|
| orientation | vertical or horizontal; `(deciding n/10)` until it locks |
| orientation score | each orientation's summed score over the deciding frames |
| tilt deg | angle between the rope's main axis and the chosen orientation, median of the last 10 analyzed frames (positive = turned clockwise) |
| tilt warning | shown above 15°, also drawn on the video: hold the phone straighter |
| present %, area, area rate, size, thickness, waviness, breathing, breathing Hz | as in the `vortex` pipeline, taken across the rope's axis |
| center | position of the centreline across the axis (from the top for a horizontal rope, from the left for a vertical one) |
| drift speed | how fast the centreline moves across the axis (positive = toward the top or the left) |

Drawn: the same edges, centreline and centre as the `vortex` pipeline, along the rope's real direction.

What it needs (framing, tilt, speed): [findings](findings.md#the-vortex-lamp-model).

### Particles (`particles` pipeline)

| Measurement | Meaning |
|---|---|
| in frame | particles detected in this frame |
| unique | distinct particles followed since the start (can run high if the tracker loses one and gives it a new number) |
| mean speed | average speed of the particles in this frame, each measured over its last 3 analyzed positions |
| dropped frames | frames skipped so far because the models were busy |

Each box shows the particle's number and speed. Per particle, the results file adds: first and last time
seen, sightings, mean and max speed, net velocity along and across the frame, path length and size.

## The results file

**Download results** (bottom of the panel) saves one JSON:

```
{
  exportedAt, source, videoDuration, frameSize, calibration,
  models: [
    {
      slot: "A", name, backend, pipeline: "vortex",
      from, to,                     // first and last analyzed time (s of video)
      framesAnalyzed, framesDropped,
      metrics:     { "vortex thickness px": { mean, min, max, last, frames }, ... },   // every numeric metric summarized
      definitions: { "vortex thickness px": "mean height of the rope across its columns", ... },
      pipelineResults: { ... }      // the pipeline's own summary: breathing, statistics, per-particle list...
    },
    { slot: "B", ... }
  ]
}
```

Only models that were switched on appear.
