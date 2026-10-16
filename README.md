# Model Studio

Runs ONNX vision models on a video file or the live camera, in the browser, and measures what they see.
It's set up for the rig's two models:

- **Vortex**: segments the vortex rope ("torch") and measures its area, thickness, movement, waviness and
  breathing (how much it expands and contracts).
- **Particles**: detects the particles (flakes), tracks them, and measures how many there are and how fast
  they move.

There is also a **Vortex lamp** model for a desk vortex lamp filmed with a phone. It works out whether the rope
runs up and down or left to right, and takes the same measurements along the rope (see
[how it works](docs/how-it-works.md#vortex-lamp-vortex-lamp-pipeline)).

Both rig models run at the same time on the same frames. Everything runs on your computer (WebGPU on the GPU, or WASM
on the CPU); nothing is uploaded. At the end, **Download results** saves everything that was measured as a
JSON file.

## Quick start

You need [Node.js](https://nodejs.org) 20 or newer and Chrome or Edge.

```bash
git clone https://github.com/Andesprit/model-studio.git
cd model-studio
npm install
```

The models come with the repo, in `apps/web/public/models/`, and the app downloads them by URL
(listed in `apps/web/public/models/index.json`; a `url` there can be relative or any absolute link):
`vortex_detector_fp32_with_pipeline.onnx` (0.5 MB), `particle_detector.onnx` (39 MB, detection only),
`particle_detector_with_tracking.onnx` (39 MB, the same detector plus a tracking pipeline) and
`vortex_lamp_auto_fp16.onnx` (0.8 MB, the vortex lamp model with its pipeline; `vortex_lamp_auto_fp32.onnx`, 1.5 MB,
is the same in float32, listed for comparison).
Then:

```bash
npm run dev
```

and open http://localhost:5173. Both models load on their own, each with its analysis built in.

## Using it

1. **Model**: one checkbox per model (vortex, particles, particles + tracking, vortex lamp). Vortex and
   particles are loaded on start; the others download when you tick them. For the lamp, untick the other two
   and hold the phone upright with the lamp about 70 % of the frame's width
   ([why](docs/findings.md#the-vortex-lamp-model)). Any ticked models run together, and each frame takes about
   the sum of their times, so fewer models is faster.
2. **Config**: how each model reads its input and output. Filled in from the model itself; you normally don't
   touch it.
3. **Pipeline**: optional. The models already bring their own; load a `.js` here only to run another
   analysis on top of both.
4. **Source**: **Video file** or **Camera**. Videos must be H.264 MP4 (see
   [findings](docs/findings.md#video-formats)). The Source step has three one-click examples (Vortex, Particles, Both; in `apps/web/public/examples/`). More test videos are in `videos/`: `type-*` for the vortex model,
   `bomba*` and `turbina*` (full frame, or one tube per `_tube_N` file) for the particle model.
5. **Calibrate**: click **Calibrate**, drag a line over something of known length in the video, then enter
   its length and unit. Sizes and speeds are then in that unit instead of pixels. The line can be moved,
   resized or removed.
6. **Results → Download results (.json)**: everything measured so far, with one section per model.

What each measurement means is in [docs/how-it-works.md](docs/how-it-works.md#what-gets-measured), and it's
also written into every results file.

### On a phone

```bash
npm run dev:phone
```

prints a `Network: https://<pc-ip>:5173` address. Open it on a phone on the same Wi-Fi and accept the
certificate warning once. On some phones the GPU gives wrong results; if every detection looks wrong, set
**Backend** to **WASM (CPU)** (see [findings](docs/findings.md#phones-the-gpu-can-compute-wrong-results)).

## Commands

```bash
npm run dev          # the app at http://localhost:5173
npm run dev:phone    # the same over HTTPS on the local network, for phones
npm test             # unit tests (packages/core)
npm run typecheck    # TypeScript, both packages
npm run build        # static site in apps/web/dist
python scripts/embed_pipeline.py MODEL.onnx PIPELINE.js   # put a pipeline inside a model
                                                          # (particle_detector_with_tracking.onnx = particle model + pipelines/tracking.pipeline.js)
```

## Project layout

```
apps/web/              the app (React + Vite): UI, camera/video, onnxruntime-web, drawing
  public/models/       the models the app loads, and index.json (the list it shows and preloads)
packages/core/         the logic, no browser code: config, preprocessing, decoders, pipelines, tracker
  test/                unit tests
pipelines/             the analysis built into each model (vortex, particles, lamp): the source to edit
examples/              other pipelines and a config, for other kinds of models
scripts/               embed_pipeline.py
docs/                  how it works, findings, history
videos/                test videos (H.264): vortex (type-*) and particles (bomba*, turbina*)
```

## Docs

- [How it works](docs/how-it-works.md): the flow, models and config, pipelines, the results file.
- [Writing pipelines](docs/pipelines.md): how to write one and put it inside a model.
- [Findings](docs/findings.md): what we learned testing it: phones, int8 vs fp32, speed, video formats,
  deploying.

## Deploying

Every push to `main` builds the app and publishes it to GitHub Pages (`.github/workflows/pages.yml`);
the models are served from the same site under `models/`, so their download URLs are
`https://<owner>.github.io/<repo>/models/<file>.onnx`. Pages can't set the cross-origin-isolation headers,
so the deployed app runs inference single-threaded; locally (`npm run dev`) it is multi-threaded.
