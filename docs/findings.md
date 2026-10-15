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

**What to do now.** On a phone, if detections look wrong, set **Backend → WASM (CPU)**. Changing the backend
reloads the models straight away.

**Possible fixes, not built yet.**
1. Start every model on the CPU and switch to the GPU only after a test inference matches the CPU's output;
   remember the verdict per device. Every device would end up on whichever backend is correct for it.
2. Try onnxruntime settings that change which GPU code runs (convolution layout, optimization level), in
   case one avoids the bug.
3. For real phone performance, a native app: it can use the phone's NPU/DSP (e.g. LiteRT with the
   `.tflite` int8 model, or native onnxruntime with NNAPI/QNN), which a browser can't reach. `packages/core`
   has no browser code, so it could be reused there.

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
