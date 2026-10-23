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

You can run both rig models at the same time on the same frames. Everything runs on your computer (WebGPU on the GPU, or WASM
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
`vortex_lamp_auto_fp16.onnx` (20 MB, the vortex lamp model with its pipeline).
Then:

```bash
npm run dev
```

and open http://localhost:5173. Choose a subject to download its model; a fresh visit downloads no model.

## Using it

See [the main path](docs/how-it-works.md#the-main-path) and the [measured before/after gains](docs/findings.md#before-and-after).

1. Choose **Vortex lamp**, **Rig vortex** or **Particles**. Only that model downloads and runs.
   Your subject is remembered for the next visit. Lamp and rig models start loading on return;
   the 39 MB particle model waits until you start a source or tap **Particles** again.
   Share the current address to preselect the subject for someone else, or add
   `?target=lamp`, `?target=vortex` or `?target=particles` to the page URL.
2. Tap **Camera** or **Video file**, even while the model is loading. The mask or detections and key
   measurements appear together. Videos must be H.264 MP4 (see [findings](docs/findings.md#video-formats)).
   After choosing a video file, pick **Analyze first** or **Live**. **Analyze first** analyzes the video at
   10, 15 or 30 frames per second (10 by default; fewer is faster) with a progress bar and **Cancel**, then plays
   it with the stored results, showing the nearest analyzed frame, so playback never lags behind on a slow
   device. Its summary covers the whole video, and a line under it gives the time per frame spent seeking,
   preparing, in the model and measuring (also in the downloaded results). **Live** plays right
   away and analyzes as it goes, skipping frames when the device is slow. Changing a model or setting after
   the analysis, including drawing a calibration line, goes back to live analysis.
   For the lamp, hold the phone upright with the lamp about 70 % of the frame's width
   ([why](docs/findings.md#the-vortex-lamp-model)). The lamp camera shows framing lines until orientation
   locks. If motion access is available, **Phone tilt** stays visible: green through 10 degrees, amber
   through 15, then red. A phone pointed steeply up or down shows that tilt is unavailable. Motion
   permission uses the same Camera tap; denial leaves the guide working. The level uses the phone sensor,
   not the mask, and its alignment has not been tested on a real phone.
   Rig vortex and Particles also offer **Try an example**;
   the rig's Video 10 and 11 switch on both rig models. More test clips are in `videos/`.
3. **Download results (.json)** saves everything measured, with one section per model.
   The key card plots the last ten seconds of thickness (or particle count). Calibrated thickness
   uses your chosen unit. When a video with an embedded pipeline ends, **Video summary** shows its
   analyzed time span, analyzed and skipped frames, thickness or particle count statistics, and latest
   breathing readings beside the download. Fewer than six analyzed frames shows a short note instead.
   On the camera with an embedded pipeline, tap **Finish** to end the session, release the camera and
   see the same recap as **Camera session summary**. The last frame stays visible with a **Session finished**
   label. Tap **Camera** again to start fresh. The recap clears on replay or a new source.
   Tap **Share** in either recap to send its text and measurements to another app. Browsers without sharing
   copy the same text to the clipboard and show **Copied**. **Download results (.json)** keeps the full data.
   **All measurements and timing** opens the full metric list, backend and per-frame costs.

Open **Advanced** for the power tools:

- **Models and backend**: every model remains a checkbox, including particles + tracking. Any combination can run together. A model downloads the first time it is enabled;
  choosing a main target switches the others off. Fewer models is faster. **Load .onnx** opens a model
  from disk and asks before running an embedded pipeline. **Backend** switches between Auto, WebGPU and WASM.
- **Model config**: the decoder and **Config .json** for each loaded model. Normally these are inferred
  from the model itself.
- **Pipeline**: load or unload a `.js` analysis on top of all enabled models. Models with an embedded
  pipeline already run it on their own.
- **Calibration**: click **Calibrate**, drag a line over something of known length in the video, then
  enter its length and unit. Sizes and speeds use that unit. The line can be moved, resized or removed;
  changing the source clears it.

What each measurement means is in [docs/how-it-works.md](docs/how-it-works.md#what-gets-measured), and it's
also written into every results file.

### On a phone

```bash
npm run dev:phone
```

prints a `Network: https://<pc-ip>:5173` address. Open it on a phone on the same Wi-Fi and accept the
certificate warning once. On some phones the GPU gives wrong results. **Auto** checks small models
(including the lamp and rig vortex) against the CPU on their first load. If they disagree, it uses the CPU,
shows a notice above the video, and remembers the choice for that GPU and model. The large particle models
are not checked because their CPU test is slow; if their detections look wrong, set **Advanced → Backend**
to **WASM (CPU)**. Advanced shows each model's check status and offers **Check GPU again**. Explicit
**WebGPU** keeps the GPU even after a mismatch, with a warning; explicit **WASM** skips the check.
See [findings](docs/findings.md#phones-the-gpu-can-compute-wrong-results) for measurements and limits.

Use this checklist on the phone; these behaviors have been checked in browser emulation, but still need
real-device verification. If a step differs, note the phone, browser, orientation and what appeared.

1. **First visit:** start a fresh private browsing session without a `?target=` parameter. Expect **Vortex lamp**,
   **Rig vortex** and **Particles**. Tap **Vortex lamp**, then **Camera**, allowing camera access.
   After loading, the live mask and key measurements should fit without scrolling. If the camera is
   denied, allow it in the browser's site settings and tap **Camera** again; if the view is clipped,
   capture a screenshot and the browser's zoom setting.
2. **Framing and level:** on iPhone, allow motion access when prompted by the **Camera** tap.
   Expect **Fit the lamp between the lines** until orientation locks (the guide may disappear quickly).
   Hold the lamp at about 70 % of the frame width, then gently tilt the phone: **Phone tilt** should be
   green through 10 degrees, amber through 15, and red above 15. If it says
   **Tilt unavailable: point camera forward**, point it forward. If no level appears, check motion
   permission; the guide still works without sensor access. If the level disagrees with the camera
   image, record the angle and orientation; sensor-to-camera alignment is still unverified.
3. **GPU check:** with **Advanced** open, leave **Backend** on **Auto (WebGPU → WASM)**. On a healthy
   WebGPU phone, expect **GPU verified** and no CPU notice. If the GPU disagrees, expect
   **The GPU gave different results. Using the CPU for accurate analysis.** above the video and
   **GPU mismatch; using CPU** in Advanced. **CPU; GPU not checked** is also normal without WebGPU.
   If the mask looks wrong, choose **WASM (CPU)**; **Check GPU again** retries the comparison in Auto.
   Particle models are not checked automatically. See the [GPU limits](docs/findings.md#phones-the-gpu-can-compute-wrong-results).
4. **Landscape:** close **Advanced** and turn the phone sideways. The live video, key card and **Finish**
   should remain on screen without scrolling. If they do not, record a screenshot in both orientations
   and the browser's zoom setting; return to portrait to continue.
5. **Finish and share:** after several seconds of analysis, tap **Finish**. Expect the last frame,
   **Session finished** and **Camera session summary**, with the camera released. Tap **Share** and
   choose a destination: it should receive the recap's numbers. Without native sharing, expect
   **Copied** and paste to check the text. If sharing or copying fails, note the displayed error and
   use **Download results (.json)**. If the recap says too few frames, tap **Camera**, wait longer and
   finish again; if the camera stays active, close the tab and report it.
6. **Return and links:** reopen the same address in the same tab/profile. **Vortex lamp** should stay
   selected, leaving just **Camera** to tap. Open `?target=vortex` or `?target=particles` to check that
   a link overrides the remembered subject with **Rig vortex** or **Particles** (particles waits for a
   source before downloading). If it does not, check the URL spelling and whether browser storage is
   blocked or the private session ended; selecting the subject manually still works.

See [lamp limits](docs/findings.md#the-vortex-lamp-model) for the untested sensor alignment and phone
performance, and for the model's framing and tilt limits. A green level or a passed GPU check does not
guarantee an accurate mask.

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
  public/models/       models downloaded on demand, and index.json (the full selectable list)
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
