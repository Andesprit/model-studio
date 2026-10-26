# Detectors (Android)

A native Android app for the vortex lamp, the rig vortex and the particle detector, from the camera or a video on the
phone. It was written because model studio (`apps/web`) in a phone browser was too slow. The models are the web app's
(the particle model here is its INT8 LiteRT export).

| detector | model (in `app/src/main/assets`) | steps ported from |
|---|---|---|
| Vortex lamp | `vortex_lamp_rope.onnx` (fp32, 13.2 MB) and `vortex_lamp_rope_int8.onnx` (3.8 MB): the rope network of the "0.4" lamp model (phone-footage augmentations on every training frame at a 0.4 chance; the lamp file on model studio's `feat/lamp-orientation` branch), from `tools/export_lamp_model.py` | model studio's `lamp.pipeline.js` for a model with no gate: landscape pictures turned upright, both branches while deciding and the stronger rope locked after 10 frames, decided again after 15 frames with no rope or a turned phone, "no lamp in view", largest piece, and the rope measurements |
| Rig vortex | `rig_vortex_fp32.onnx` = model studio's `vortex_detector_fp32_with_pipeline.onnx`, and `rig_vortex_int8.onnx` = `vortex_detector/models/vortex_detector_int8.onnx`, the INT8 model that file was made from | model studio's "Rig vortex": the frame halved, top-left on a 160 x 416 canvas, rope = logits > 0; `vortex.pipeline.js`'s measurements |
| Particles | `particle_detector_int8.tflite` (lc_100, INT8, float decode) | `particle_detector/run_model.py`: CLAHE, letterbox 1280 x 384, conf 0.20, NMS 0.35 |

The rope measurements (`RopeAnalyzer.kt`) are the pipelines': thickness, center (center y on the rig), drift (vertical
speed on the rig), waviness, area, area rate, breathing, and at the end a summary (frames, thickness, breathing, center,
waviness, area). They are in pixels of the analysed picture; times are seconds of video (frame index over the mean frame
rate) or since the camera started.

The two vortex models run on a backend picked on screen: CPU fp32, CPU INT8 (the default) or NNAPI (the phone's GPU or
AI chip, half precision allowed), with ONNX Runtime Android 1.20; the status line shows one model run and its median
over the last 30 frames, with the backend that ran it (NNAPI falls back to CPU fp32 if it can't be set up, and says
so). Each lamp backend has one session per canvas shape (the frame as is and turned), with the shape fixed: NNAPI needs
fixed shapes. The particle model runs on LiteRT 1.0.1, CPU 4 threads or the GPU delegate.

Model studio's lamp file wraps the network in If nodes that pick the branch (as is, or the frame turned 90 deg). The app
runs the bare network and turns the frame itself (`VortexDetector.runBranch`), the same arithmetic: one copy of the weights
instead of three, and no If nodes, which NNAPI can't run.

A video runs live or, for the vortex detectors, "Analyze first" (asked after picking it):
- Live: every frame in order, as fast as the model allows, each result shown as it comes.
- Analyze first at 10, 15 or 30 frames per second of video: a progress bar (Stop cancels), then the video plays (looping,
  with its own controls) with the stored result of the last analysed frame up to its current time drawn over it, that
  frame's readings and the summary. Frames are read in order with `MediaMetadataRetriever.getFrameAtIndex`, every one
  of them (reading only every few would decode again from the last keyframe each time, seconds per frame) and only
  the wanted ones analysed; masks are kept run-length encoded.
Frames are turned by the file's rotation tag when the retriever has not done it.

Not ported yet: calibration to real units, particle tracking, speed and spin, Analyze first for particles. The tilt
warning is left out on purpose: the lamp model was trained on lamps tilted up to 45 deg.

## Build and install

Needs JDK 17 and the Android SDK (`local.properties` points at it).

```bash
./gradlew assembleDebug
```

The APK is `app/build/outputs/apk/debug/app-debug.apk`; it holds the arm64 phone and x86_64 emulator libraries.
A phone-only APK, about half the size, signed with the debug key (for sideloading, not for a store):

```bash
./gradlew assembleRelease -Pabi=arm64-v8a
```

Built APKs are attached to GitHub releases, not committed: GitHub refuses files over 100 MB. Copy it to the phone and open it to install (allow installs
from that source), or with the phone on USB debugging:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Smoke test

`tools/export_lamp_model.py` (vortex_detector's venv) writes the lamp network and checks it against model studio's file
on the desktop. `tools/make_reference.py` (particle_detector's venv) copies the particle model and the test images (a
lying and an upright lamp) into the app and writes the desktop outputs; `app/src/androidTest/.../SmokeTest.kt` compares
against them on a phone or emulator, and checks the rope measurements on a band of known thickness:

```bash
./gradlew connectedDebugAndroidTest
```

Results on the x86_64 emulator (2026-10-26), against model studio's files, in the branch the app uses:
- lamp (0.4): rope IoU 0.9994-0.9997 (CPU fp32, NNAPI) and 0.988-0.993 (CPU INT8) on the desktop's own input,
  0.991-0.997 on the app's own resize (Android's bilinear resize differs from cv2's at the mask boundary). The lying
  lamp (a landscape picture, turned upright) and the upright one give the same orientation, rope strength and thickness.
- rig: rope IoU 1.000 (fp32, NNAPI) and 0.994 (INT8) on the desktop's input, 0.994 on the app's resize.
- particles, INT8 tflite vs the fp32 onnx desktop boxes: 38 vs 35 boxes, F1 0.932 at IoU 0.5.

Emulator times say nothing about a phone (one lamp model run: CPU fp32 80-430 ms, CPU INT8 90-95 ms, NNAPI 460 ms on
the emulator's reference NNAPI with no GPU or AI chip; rig 22-30 ms on the CPU; particles ~680 ms). Analyze first on a
4 s 720p lamp video at 10 per second: 41 frames in 17 s with CPU INT8. The test logs the phone's times
under the logcat tag `SmokeTest`, including whether the GPU delegate took the particle model (`backend`).
