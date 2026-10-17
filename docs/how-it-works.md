# How it works

## The main path

Choose **Vortex lamp**, **Rig vortex** or **Particles**, then **Camera** or **Video file**. A fresh visit
downloads no model until you choose; only the chosen model loads. You can start the source while it
loads. The live mask or detections appear with **Key measurements** and a trend of the last ten seconds
of thickness or particle count. The lamp card shows orientation, thickness, breathing and waviness,
with a warning when the rope looks tilted.

On the lamp camera, framing lines help fit the lamp to about 70 % of the image width until orientation
locks. **Phone tilt** uses the motion sensor when permission and readings are available; the guide
still works without it. The sensor's alignment with the camera has not been verified on a real phone.
See the [lamp limits](findings.md#the-vortex-lamp-model). Portrait puts the readings below the video;
the compact phone landscape layout puts them beside it, keeping the live view and **Finish** on screen.
Desktop also shows the video and readings side by side.

With an embedded pipeline, **Finish** ends a camera session and releases the camera, leaving its last
frame and **Camera session summary**. A video reaching its end shows **Video summary**. Both recaps
show the analyzed time span and frame counts, measurement statistics and latest breathing readings;
too few frames gives a short note. **Share** sends the recap text, or copies it when native sharing is
unavailable. **Download results (.json)** saves the full measurements in the [results format](#the-results-file).

The subject is remembered on this device. Links with `?target=lamp`, `?target=vortex` or
`?target=particles` override that choice. Lamp and rig models start loading on return; particles waits
for a source or another subject tap, avoiding an automatic large download. A returning lamp user only
needs to tap **Camera**. **Advanced** starts collapsed and keeps model combinations, backend choice,
model and config files, pipeline files and calibration available.

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

CPU (WASM) sessions run in dedicated module workers, including the CPU reference used by the automatic
GPU check. Each session keeps its worker until it is released. Image buffers transfer to the worker and
output buffers transfer back; extra pipeline feeds stay reusable. Frame preparation, decoding, pipelines
and drawing still run on the page. WebGPU sessions also stay on the page. If a worker cannot start, CPU
inference falls back to the page. A worker failure reports an inference error and stops that analysis;
loading a model again creates a fresh worker. Session changes still wait for the current frame to finish.

The code is split in two:

- **`packages/core`**: all the logic, with no browser code: reading configs, preparing frames, decoders,
  pipelines, the tracker, calibration. It could be reused as-is in a future mobile app.
- **`apps/web`**: the browser part: the UI, the camera and video, running the models with onnxruntime-web,
  drawing.

## Live analysis and skipped frames

The video plays at its normal speed and the app analyzes whatever frame is on screen whenever the models
are free. Frames that pass while the models are busy are **skipped**, never queued, so the overlay never
falls behind. The **dropped** number in the bar counts them.

On the CPU path, if another live frame arrived during inference, the loop analyzes the latest available
frame as soon as the worker replies, without waiting for another video callback. It never queues old
frames or repeats a paused frame for this catch-up. WebGPU keeps its existing callback scheduling.

With both rig models on a desktop GPU, one frame takes about 0.75 s, so roughly 1 in 20 frames of a 30 fps
video gets analyzed. This matters for measuring; see [findings](findings.md#skipped-frames-limit-what-can-be-measured).

Measurements start over (a pipeline is reset) when the video loops or you seek back, when you change the
video or camera, change a model, or change the calibration. A paused video re-analyzes the same frame
when a setting changes, but that frame isn't counted twice.

With a pipeline running (embedded or loaded in **Advanced → Pipeline**), a video plays once. When a video
with an embedded pipeline ends, **Video summary** shows the analyzed time span, analyzed and skipped
frames, thickness or particle count statistics, and latest breathing readings. Fewer than six analyzed
frames shows a short note instead.
The summary uses the same pipeline statistics as **Download results** and clears on replay, seeking or
a reset. Detector-only video playback still loops.

For a live camera with an active embedded pipeline, **Finish** stops new analysis and waits for the last
inference before reading the same statistics into **Camera session summary**. It releases the camera and
pauses the last frame with a **Session finished** label. **Download results** saves that finished session;
**Camera** starts a fresh one, clearing the recap and resetting the measurements. Detector-only cameras
have no Finish action.

## Models and how they're configured

The app needs three things to use an ONNX model: how to prepare the image for it, how to read its output,
and (optionally) what to measure. It finds them in this order:

1. **A config file** (**Advanced → Model config → Config .json**), if you load one. An example is in
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

`apps/web/public/models/index.json` is the full list shown in **Advanced → Models and backend**. A fresh visit
loads no model. A remembered subject or a `?target=lamp|vortex|particles` link preselects it; lamp and rig
models load on open, while particles waits for a source or another subject tap. Choosing **Vortex lamp**,
**Rig vortex** or **Particles** downloads only that target's model and switches the others off.
Advanced checkboxes can enable any combination, downloading each
model only the first time it is selected. An entry can carry recommended decoder options; the particle model's are
from its training notes (confidence 0.20, overlap 0.35, up to 1000 detections, since the default 300 cuts
off dense frames).

## Pipelines

A pipeline is a small JavaScript file that turns each frame's detections into measurements. Each rig model
can carry its own **inside the `.onnx`** (metadata key `model-studio.pipeline`), so it loads with the model:

| Model file | Pipeline inside | Source |
|---|---|---|
| `vortex_detector_fp32_with_pipeline.onnx` | `vortex` | `pipelines/vortex.pipeline.js` |
| `particle_detector_with_tracking.onnx` | `tracking` | `pipelines/tracking.pipeline.js` |
| `particle_detector.onnx` | none (detection only) | `pipelines/particles.pipeline.js` can be loaded in **Advanced → Pipeline** |
| `vortex_lamp_auto_fp16.onnx`, `vortex_lamp_auto_fp32.onnx` | `vortex-lamp` | `pipelines/lamp.pipeline.js` |

Each model's pipeline runs on that model's output only. A model opened from disk asks before running its
pipeline, since that's code; the app's own models don't ask.

**Advanced → Pipeline** can load one more pipeline that runs on **both** models' results together (for analyses that relate
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

For the vortex lamp filmed with a phone. The model only knows a rope that runs left to right. Its graph can
run the network two ways and returns two masks:
- `logits`: the frame turned 90° and the mask turned back, which is right for a rope running up and down;
- `logits_asis`: the frame as it is.

A second model input, `mode`, picks what runs: 0 both, 1 only `logits`, 2 only `logits_asis`. The output that
doesn't run is a 1×1 placeholder.

The pipeline picks between the two:

1. While deciding, it asks for both (`mode` 0, through its `feeds()`, see [pipelines](pipelines.md#setting-the-models-other-inputs))
   and scores both masks on every frame: mean probability of the rope pixels × share of those pixels in the
   largest piece. The wrong orientation gives a weak, scattered mask.
2. It adds up the scores over the first 10 analyzed frames and locks the better orientation until the next
   reset (loop, seek back, new video). From then on it asks for that branch only, so the network runs once per
   frame instead of twice.
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

**Download results** (under the analysis) saves one JSON:

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
