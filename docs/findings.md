# Findings

What we learned building and testing the app, with the numbers we measured. Unless said otherwise, the PC
numbers are from an AMD Ryzen 3 3200G (4 cores, Radeon Vega GPU) in Chrome.

## Phones: the GPU can compute wrong results

**Symptom.** On an Oppo A6K (Qualcomm Adreno 6xx GPU) with the GPU backend, every frame gave the maximum
number of detections (300), all labelled "airplane 50%", whatever the camera pointed at. With **Backend →
WASM (CPU)** the same models gave correct results.

**What we checked.** We ran each model once on the GPU and once on the CPU with the exact same input tensor
and compared the raw outputs, before any decoding:

| | Detector, GPU | Detector, CPU |
|---|---|---|
| minimum value | **−2.47** | 0 |
| values exactly 0 | **0** | 98,928 |
| largest difference | 363.7 | |

- The input was identical (same tensor fed to both), so it isn't the preprocessing.
- Most of that output is class probabilities, which can't be negative. A negative minimum means the GPU's
  math is wrong.
- On the CPU, ~99k values are exactly 0 (classes that clearly aren't there); on the GPU none are, so every
  one of the 8,400 candidate positions passed the 25% threshold. All classes tied near 50% (the value a
  score of 0 maps to), noise picked "airplane", and the boxes came out near zero size, so overlap removal
  couldn't merge them: 300 detections.
- The pose model was off the same way (largest difference 308).
- No errors or warnings from onnxruntime: the GPU computes wrong values silently.
- The first GPU run took **77–90 s** per model (compiling its shaders), against ~1 s on the CPU.

**Why.** Most likely a driver issue on that Adreno family with the operations onnxruntime's WebGPU backend
uses. It's outside our code; the same models are correct on that phone's CPU and on the PC's GPU and CPU.
We only tested one phone, so we don't know which other phones are affected.

**Automatic check for small models.** With Backend **Auto**, models up to 3 MB (both lamp entries, rig
vortex, and small models opened from disk) are checked when they first load on WebGPU. The app runs the
same deterministic image through WebGPU and WASM, using the model's own input normalization. Other inputs
are zero-filled, including lamp `mode=0`, which exercises both orientation branches. For each output it
computes `sqrt(mean(((gpu - wasm) / max(1, abs(wasm))) ** 2))`. Any output above **0.05**, a nonfinite
value, or mismatched output names or shapes fails the check. On disagreement, Auto keeps the WASM session
and shows a CPU notice above the video. On agreement, it releases the test session and keeps WebGPU.

The verdict is remembered in localStorage by GPU vendor/architecture and model URL (file name for disk
models). A later visit skips validation, going straight to WASM after a mismatch. Blocked storage does
not prevent analysis, but the check repeats on later loads. **Advanced** shows per-model verification and
offers **Check GPU again**, which forgets the verdicts and reloads the loaded models. Explicit **WebGPU**
keeps that backend and warns after a mismatch; explicit **WASM** does not run a check. All session changes
and test inferences run inside the frame loop's model-change guard.

**Calibration and cost, PC only.** On an Intel gen-12lp GPU in visible Chromium 148, we compared each of
the three target models on a synthetic input and five real video frames, prepared with the app's frame
grabber and tensor conversion. We also checked the lamp's locked modes. Worst healthy scaled RMS was
**0.01290**; the smallest simulated fault was **0.22252**. Faults were a constant offset, scrambled values
plus noise, all zeros, and particle confidences fixed at 0.5. The 0.05 threshold sits 3.88 times above the
worst healthy error and 4.45 times below the smallest tested fault. Sign agreement alone fails on tiny
particle scores (only 80% agreement despite near-identical numbers); correlation misses constant offsets.
Raw measurements and the reproducing scripts are in `.atelier/goal/evidence/gpu-check/separation.json`.

Creating the additional WASM session and running one test took **198–220 ms** for the lamp with four
threads, **303 ms** single-threaded (as deployed), and **68–102 ms** for rig vortex. The 40 MB particle
models are deliberately **not checked**: their test alone took **1.49 s** with four threads and **4.58 s**
with one on this PC. Select **Advanced → Backend → WASM (CPU)** if particle results look wrong. Changing
the backend reloads the models straight away.

In the full app, a fresh phone-sized camera run reached the first lamp frame in **3.45 s**, against
**2.85–3.0 s** before checking, with the same two taps and measurements visible without scrolling.
Reloading in that profile took **1.11 s** and created no WASM test session. Desktop measured **3.20 s**
then **0.97 s**. Under the same simulated offset, the old app's mask had IoU as low as **0.687** against
WASM; automatic fallback matched forced WASM masks and measurements exactly on 12 identical 4K frames.
These are local PC browser timings, not physical-phone or production-network measurements.

This threshold has been calibrated on **one healthy GPU**, and fallback tested with simulated errors.
The affected Adreno phone was unavailable for this check. A single synthetic image can catch gross math
faults like the tested cases; agreement does not prove correct results for all inputs or devices.

**Other possible fixes, not built yet.**
1. Try onnxruntime settings that change which GPU code runs (convolution layout, optimization level), in
   case one avoids the bug.
2. For real phone performance, a native app: it can use the phone's NPU/DSP (e.g. LiteRT with the
   `.tflite` int8 model, or native onnxruntime with NNAPI/QNN), which a browser can't reach. `packages/core`
   has no browser code, so it could be reused there.

## CPU inference runs off the page

WASM sessions now live in dedicated module workers, including the automatic GPU check's CPU reference.
WebGPU stays on the page: its steady-state runs did not produce main-thread long tasks in this test.
The page still prepares frames, decodes outputs, runs pipelines and draws. The worker takes ownership
of each image buffer and transfers output buffers back; extra pipeline feeds remain reusable. A session
keeps its worker until release. If workers are unavailable or cannot load, CPU inference uses the page.
Worker errors, including asynchronous WASM traps, reject pending calls and reach the existing error
panel; loading another model can recover without reloading the page.

**Blocking and throughput, PC only.** We compared the milestone's base (`1d69324`) with the worker build
in visible Chromium at 390x844, using the same 720x1280 fake lamp camera. Each environment has six
ten-second pairs after orientation locks: three main-first and three worker-first, alternating. Production
is served under `/model-studio/` without cross-origin isolation, as deployed (single-threaded WASM); dev
uses its normal isolation headers and four WASM threads. These are medians of three runs per order:

| Environment / order | Infer, page / worker | Analyzed fps, page / worker | Total blocking time, page / worker |
|---|---|---|---|
| Production, page first | 78.40 / 78.50 ms | 11.38 / 11.50 | 4176 / 0 ms |
| Production, worker first | 78.10 / 79.90 ms | 11.32 / 11.30 | 4227 / 0 ms |
| Dev, page first | 27.63 / 28.84 ms | 20.87 / 26.29 | 0 / 0 ms |
| Dev, worker first | 27.51 / 28.66 ms | 21.14 / 26.09 | 2 / 0 ms |

Total blocking time sums the part of each main-thread long task beyond 50 ms. Across the six production
windows it fell from **26,271 ms to zero**; across dev's six windows, **1371 ms to zero**. All twelve worker
windows had zero long tasks. Dev's median is small because most individual CPU calls are below the 50 ms
threshold, even though they interrupt animation. The largest animation-callback gap was 250 ms on the
production page path and 33 ms with workers; dev was 100 ms and 17 ms. No worker window had a gap over
50 ms. These animation callbacks measure page responsiveness, not the camera's capture rate.

The first prototype lost analyzed frames by waiting for another video callback after a reply. The final
CPU path remembers whether a live frame arrived while busy, then analyzes the latest frame immediately
after inference if the source time changed. It queues no old frames and does not catch up paused video.
Along with transferring image buffers without a copy, this kept production throughput essentially flat
and raised dev throughput by about 23-26%. Infer time rose by at most about 4.5% in the per-order medians.
The GPU retains its original scheduling.

A paired ten-second GPU check measured camera inference at 31.3 / 31.6 ms before / after, and 4K video at
29.1 / 28.5 ms. Analyzed rates were 15.5 / 16.4 fps and 17.8 / 19.0 fps respectively; these single pairs
are consistent with the unchanged GPU path. The 5.2-second video was replayed only by the timing
harness to avoid measuring idle time after it ended. Normal playback with a pipeline still does not loop.

On 24 identical seek-stepped 4K frames in each threading setup, the worker's raw CPU outputs, decoded
masks, key measurements and complete embedded-pipeline summaries matched the original exactly. The
tests also verified that image buffers transferred and pipeline feed buffers stayed usable. Worker and
WASM assets load under the production URL prefix. Reproduce with
`node .atelier/goal/evidence/worker/matrix.mjs` and `node .atelier/goal/evidence/worker/equality.mjs`
(also set `SINGLE_THREAD=1` for the latter). Raw runs and screenshots are under
`.atelier/goal/evidence/worker/`; `blocking.json` retains the earlier probe and the final build comparison.
This does not establish performance on a physical phone or remove the remaining preparation and pipeline
work from the page.

## int8 models are slower on the GPU, faster on the CPU

Particle model on `tube_1` (874×212 strip), particle model alone, average of 5 frames:

| | fp32 | int8 |
|---|---|---|
| GPU (WebGPU) | **692 ms** | 2,640 ms |
| CPU (WASM) | 2,990 ms | **1,295 ms** |

The vortex model shows the same pattern on the GPU: ~85 ms fp32 against ~220 ms int8. Both versions found
the same particles (5–6 per frame).

**Why.** The int8 models are "QDQ" quantized: they keep the same 129 float convolutions and wrap each with
Quantize/Dequantize steps:

| | operations | convolutions | Quantize + Dequantize |
|---|---|---|---|
| fp32 | 460 | 129 | 0 |
| int8 | 1,230 | 129 | 770 |

- **On the CPU**, onnxruntime recognizes the Dequantize → Conv → Quantize pattern and replaces it with real
  int8 convolutions (fast SIMD, 4× less data to move).
- **On the GPU**, onnxruntime's WebGPU backend has no int8 convolutions, so it runs all 1,230 operations: all
  of fp32's work plus 770 extra small GPU jobs.

**Rule: fp32 (or fp16) on the GPU, int8 on the CPU.** Quantizing always makes the file smaller (39 → 10 MB
here); it only makes it faster where the hardware computes in int8: CPUs, phone NPUs and DSPs (native apps),
TensorRT. Not a GPU in the browser.

Note: detector-app's README says the vortex int8 model only runs on WASM. In this app (onnxruntime-web
1.30) it runs on WebGPU correctly, just slower than fp32.

## The particle model wastes most of its work on padding

The particle model takes a 1280×1280 square. A tube strip (about 4:1) is scaled to 1280 wide and the rest
filled with grey bars, so **about 75% of the pixels it processes are padding**.

Re-exporting it from `particle_detector.pt` with a **1280×384** input (`imgsz=(384, 1280)` in Ultralytics)
keeps every tube at the same scale and drops most of the padding:

| | 1280×1280 | 1280×384 | |
|---|---|---|---|
| GPU fp32 | 692 ms | **267 ms** | 2.6× faster |
| CPU fp32 | 2,990 ms | **858 ms** | 3.5× faster |

Detections were practically identical: all 21 particles the square model found, in the same place (box
overlap 0.997, confidence differing by 0.004 on average), plus one borderline extra.

The rectangular model is only for strips. On a full 1024×1024 frame (the three tubes) it would have to
shrink the image to 384 tall and the particles would get too small, so the square model stays the default
for unknown inputs. The 1280×384 export and the `.pt` are in `original-mpeg4/models` with the other unused
models. A good next step: pick the particle model automatically from the video's shape.

Why 384 and not 320: at 320, tubes 2 and 3 (less elongated) would have to shrink to fit; at 384 all three
keep the same scale as the square model.

## Timings on the PC (GPU)

Lamp fp16, measured in a visible Chrome window on 2026-10-17. Phone means a **390×844 viewport** on the PC;
desktop is 1440×900, not a different device. The input is 232×416 grayscale. The 4K rows summarize 536
orientation-locked frames in each 600-frame run, replaying the whole clip:

| Lamp source and viewport | Prepare | Network | Decode, pipeline and draw | Analyzed fps |
|---|---|---|---|---|
| 2160×3840 video, phone | 5.0 ms | 30.2 ms | 3.2 ms | 17 |
| 2160×3840 video, desktop | 5.2 ms | 30.4 ms | 3.3 ms | 18 |
| 720×1280 fake camera, phone, short sample | 4.4 ms | 29.0 ms | 3.3 ms | 19 |
| 720×1280 fake camera, desktop, short sample | 4.3 ms | 27.3 ms | 3.3 ms | 17 |

Before GPU frame scaling, preparing the 4K input took 30.2–30.7 ms and the app analyzed about 14 fps.
The first 100 frames with the smaller overlay had median prepare times of 5.2 ms (phone) and 5.3 ms
(desktop); locked-frame p95 was 6.0 and 6.1 ms. A separate same-session comparison of the old and smaller
overlays gave 17 fps with each, prepare 5.67 → 5.05 ms, and postprocessing 3.53 → 3.55 ms: the overlay's
clear win is memory, with essentially unchanged postprocessing cost. The phone's 18 fps target is not
consistently reached.

Camera samples use the earlier three-second protocol after orientation locks. Longer fake-camera runs
slowed to 5.4–6.4 ms prepare and 11–12 fps, with network time rising to about 60 ms. Alternating the old
and smaller renderers in one camera session gave 5.47 / 5.48 ms prepare and 15 fps with both, so that
long-run slowdown is also present with the old renderer. It is not evidence of a physical phone's speed.

Earlier network-only measurements on other inputs:

| Model | Input | Time per frame |
|---|---|---|
| Vortex fp32 | 416×160, grey | 45–100 ms |
| Particles fp32 | 1280×1280 | 650–700 ms |
| Particles 1280×384 | 1280×384 | ~270 ms |
| YOLOv8n detect / pose (tests) | 640×640 | ~80 / ~170 ms |

Both rig models together: about 0.75 s per frame, roughly 1.3 analyzed frames per second.

## Skipped frames limit what can be measured

The app analyzes live and skips frames while the models are busy (about 1 in 20 frames analyzed with both
models). Consequences:

- **Breathing frequency**: you can only measure oscillations slower than half the analysis rate. Faster
  ones are missed or, worse, show up as a false slower frequency. The results file states the analysis rate
  (`samplingRateHz`).
- **Tracking**: particles move a lot between analyzed frames, so the tracker can mix them up and speeds get
  coarser.
- **Fewer skipped frames**: untick the model you don't need, or use the rectangular particle model on strips.

We built a mode that stepped through every frame of a video (300 of 300 frames analyzed, none skipped, in
56 s with the vortex model) and removed it, to keep only live analysis for now.

## Tracking: the distance allowance needs a cap

The tracker lets a particle move further the longer it hasn't been seen, so skipped frames don't break
tracks. Without a limit, a 1.4 s gap made that allowance over 2,500 px, larger than the frame: particles
from opposite ends got linked and their trails crossed the whole image. The allowance is now capped at 4×
the base distance (`maxGrowth`, 240 px with the particle pipeline's settings). Past that, a particle gets a
new number instead of stealing another's. The trail lines were removed.

## Speeds need a window, not one step

Frame-to-frame speed was inflated by detection jitter (the box centre moves a pixel or two between frames):
the mean per-particle speed came out ~3.7× the net speed. Measuring over 3 analyzed positions, as
`SPEED_WINDOW` in `parti/flake_metrics.py` does, brought it down from ~10,000 to ~7,600 px/s. The remaining
gap is real in `turbina-antorcha`: the flakes swirl around the vortex rope.

## The vortex measurements match the earlier research

On `type-1_sup.mp4`, thickness came out 23–26 px and waviness 11–15 px. The vortex research notes report a
median thickness of 25 px for type-1 superior and a centreline amplitude of 10–31 px.

The mask is at half resolution (2 px per cell), so per-column thickness moves in 2 px steps. The pipeline
averages over all columns, which resolves much smaller changes, needed for breathing.

## The two models come from different experiments

- **Vortex**: the vortex-wall videos (`type-1/2/3`, ~800×270).
- **Particles**: flakes in a glass tube filmed by a high-speed camera, 1024×1024 frames holding three mirror
  views of the tube (`parti` project).

On the vortex videos the particle model finds nothing (highest score 0.198, just under its 0.20 threshold),
and on the particle videos the vortex model paints a meaningless mask. Using both together needs footage
where both appear.

On `turbina-antorcha` the particle model also fires on the bubble string of the vortex rope; `parti` filters
those by colour (yellow flakes only), which this app doesn't do yet.

## High-speed footage and time

The recordings are high-speed (2000 fps; bomba at 250 fps) but the video files play at 30 fps. Times,
speeds and frequencies in the app are in **seconds of video**, so for 2000 fps footage real speeds and
frequencies are 2000/30 ≈ 67× higher. Convert when reading the results (we built an fps setting for this and
removed it for now).

## Video formats

Five particle clips (`bomba…`, `turbina…`) were MPEG-4 Part 2, an old codec Chrome and Edge can't play
("no supported source was found"). They were converted to H.264, which plays everywhere:

```bash
ffmpeg -i in.mp4 -an -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p -movflags +faststart out.mp4
```

The app now says which format problem it is and shows that command. `.avi` files don't play in browsers
either.

The three tube views were cut with the crop boxes from `parti` (`00_split_tubes.ipynb`): tube_1 x 90, y 60,
934×252; tube_2 x 120, y 308, 904×248; tube_3 x 0, y 606, 1024×294 (on the 1024×1024 frame). Tube 1 was then
tightened to the inside of the tube only (x 150, y 92, 874×212).

## The vortex model crops large videos

The vortex model's manifest scales the frame to half size onto a fixed 416×160 canvas. Frames up to about
832×320 fit; larger ones get cut off (a 1024×1024 frame becomes 512×512 and only part fits). detector-app
shrinks the scale automatically when a frame doesn't fit; this app doesn't yet.

## The vortex lamp model

`vortex_lamp_auto_fp16.onnx` (and its float32 twin `vortex_lamp_auto_fp32.onnx`) comes from the `vortex-detector` repo: the rig's vortex training plus 200 frames
from five handheld phone videos of a vortex lamp, each under a different light colour. The full write-up is in
that repo, in `docs/lamp_frames.md` and `docs/model_studio.md`. What matters here:

- **Colour.** Any light colour works. The model takes gray, and the app converts frames before the model.
- **Orientation is chosen automatically.** The network only knows a rope that runs left to right. Turning
  the frame before the network turned out better than training it on all orientations:

  | test | turn the frame first | train on all orientations |
  |---|---|---|
  | lamp IoU | 0.92 | 0.86 |
  | rig edge error | 0.81 px | 1.3–1.7 px |

  So the graph runs both ways, and the pipeline locks the better one after 10 frames. That choice was right on
  every frame tested:
  - 30 of 30 held-out lamp frames filmed upright;
  - 30 of 30 of the same frames turned sideways;
  - 156 of 156 frames of a full phone video.
- **Tilt: keep the phone within about 15° of upright.** The training had no tilt.

  | tilt | 0–10° | 20° | 30° | 45° |
  |---|---|---|---|---|
  | IoU | 0.90 | 0.76 | 0.39 | 0.16 |

  On diagonals the orientation score is no longer reliable either. The pipeline estimates the tilt from the
  mask and warns above 15°.
- **Framing: hold the phone upright, with the lamp about 70 % of the frame's width.** The canvas is a fixed
  416×232 (portrait), so how big the tube looks to the model depends only on how much of the frame it fills.
  At about 70 % it matches the training size. The model was trained without zoom, so a much smaller or larger
  lamp gives a worse mask.
- **Live camera guidance.** The lamp camera draws two light lines at 15% and 85% of the video width,
  with a framing hint, until the pipeline locks orientation after ten analyzed frames. The guides stay
  out of video-file and rig/particle workflows. A separate **Phone tilt** reading uses motion-sensor
  gravity, rotated for the screen orientation (including the older `window.orientation` fallback).
  Its rounded reading is green through 10 degrees, amber through 15, and red above 15; this is separate
  from the pipeline's estimate of the rope's tilt. A low-pass filter and at most ten UI updates per second
  steady the level without rendering React on sensor events or touching the frame loop.
  If less than half of gravity lies in the screen plane (pointing the phone steeply up or down), it says
  tilt is unavailable. Without readings, no level appears. On iOS the existing Camera tap requests motion
  permission; denial or failure silently leaves the framing guide, with no level. The listener is removed
  when leaving the lamp camera. Synthetic events verify angles, screen rotations and permission paths;
  **sensor-to-camera alignment is untested on a real phone**. Moving the phone can bias gravity readings,
  and the level cannot correct a tilted lamp or guarantee a good mask.
- **Cost: one network pass per frame once the orientation is locked.** A graph keeps no state between frames,
  so the choice lives in the pipeline. The pipeline tells the graph which branch to run through the `mode`
  input (an ONNX If node) and its `feeds()` hook.

  Measured with the app's runtime and the embedded pipeline on a 4K phone video, on a PC:

  | backend | network, deciding (both) | network, locked (one) | pipeline, deciding → locked |
  |---|---|---|---|
  | WASM | 100 ms | 50 ms | 20 → 12 ms |
  | WebGPU | 40 ms | 30 ms | 18 → 9 ms |

  - **On the GPU the gain is smaller**, because for a model this small, fixed per-run costs dominate.
  - **Each If branch holds its own copy of the weights**, so the float32 file is 1.5 MB, not 0.5. With the
    weights shared from outside the branches, onnxruntime couldn't pre-pack them, and every mode ran about 35 %
    slower.
  - **onnxruntime-web can't take a default value for `mode`**: feeding an input that also has a built-in value
    fails with "invalid input". So `mode` is a plain input, and the app feeds zeros to any input nobody
    supplies.
- **Float16 is the default; float32 is listed for comparison.** The network runs in float16 inside, with
  float32 input and output, so the app and the pipeline see no difference. The masks came out identical to
  float32: 0 of 96,512 cells differ. The file is half the size (0.8 MB).

  | backend (PC, Intel integrated GPU) | fp32, one pass | fp16, one pass | fp32, both | fp16, both |
  |---|---|---|---|---|
  | WebGPU | 28.6 ms | 23.9 ms | 36.4 ms | 29.1 ms |
  | WASM | 23.0 ms | 23.3 ms | 43.7 ms | 46.0 ms |

  GPUs compute float16 natively, so WebGPU gains (1.2× here). WASM doesn't, and stays the same. Phone GPUs may
  gain more or less; compare the two entries on the phone. Once it's clear the fp16 entry is no worse there,
  the fp32 entry can go.
- **GPU scaling makes large sources cheaper to prepare.** The old HTML canvas used
  `willReadFrequently: true`, putting the full-source copy and downscale on the CPU: about 30 ms for a 4K
  phone video, even though the model only needs 232×416 pixels. For sources above one megapixel the app
  now uses an OffscreenCanvas 2D without that flag, so the GPU can shrink the frame before the small
  readback. Sources up to one megapixel, including the 720×1280 camera, retain the original CPU path;
  browsers without OffscreenCanvas 2D also use that fallback. The padding, geometry and tensor conversion
  are unchanged.
  - On 600 consecutive displayed frames, grab plus tensor conversion had median 4.1 ms and p95 5.2 ms,
    with no return to the slow path. In the app, preparing the 4K frame is about 5 ms; see the timing table.
  - On 24 identical seek-stepped frames, old/new mask IoU averaged 0.9985 (minimum 0.9969). The normalized
    input tensor's mean absolute difference was 0.000018; median thickness and area differences were both
    under 0.02%. Default image smoothing was sufficient.
  - The overlay also uses its displayed size times device pixel ratio (capped at 2× and two million
    pixels), with drawing transformed from video coordinates. For the 4K clip its backing store fell
    from 8,294,400 pixels to 256,880 on the phone view and 210,528 on desktop, about 97% fewer pixels.
    Strokes and text retain their source-resolution scaling; calibration remains a separate SVG. Paused
    video refreshes after a window resize. These numbers are from the PC, not a physical phone.
- **Not tested yet:**
  - a phone;
  - a lamp video the model has never seen. All five lamp videos were used in training.

## Deploying

What limits the choice: the particle model is 39 MB and onnxruntime's engine ~27 MB; the CPU needs two
security headers (COOP/COEP) to use several threads; and the models may need to stay private.

| Option | Large files | Headers | Private | Notes |
|---|---|---|---|---|
| Hugging Face Space (static) | yes | yes | yes, org-only Space | app and models together |
| Cloudflare Pages | no, 25 MB per file: model to R2 or Hugging Face, engine from a CDN | yes (`_headers`) | yes (Cloudflare Access) | free with a private repo |
| GitHub Pages | yes (up to 100 MB) | no | Enterprise only | private repo needs a paid plan |

On GitHub Pages, without the headers the app works and the GPU too, but the CPU runs on a single thread
(slower, which matters on phones). A small service worker (`coi-serviceworker`) can add the headers from the
browser, at the cost of one page reload on the first visit; it isn't added.

Other notes:

- Only the two models the app ships with are in git (in the private repo); the others are kept out by
  `.gitignore`. Hosting models on Hugging Face and having the app download them is a common setup, but it
  requires the models to be public: a token to read private ones would be visible in the page.
- Whoever opens the page can download the models.
- The app fetches `/models/...` from the site root; a site served from a subfolder (GitHub Pages) needs that
  path adjusted.
