# Detectors (Android)

A native Android app for the vortex lamp model and the particle detector, from the camera or a video on the phone.
It was written because model studio (`apps/web`) in a phone browser was too slow. The models are the same
files the web app ships (the particle model here is its INT8 LiteRT export).

| detector | model (in `app/src/main/assets`) | runtime | steps ported from |
|---|---|---|---|
| Vortex | `vortex_lamp_rope.onnx`: the rope network of model studio's live lamp model (steep tilt, no water gate), fp32, from `tools/export_lamp_model.py` | ONNX Runtime Android 1.20, 4 threads | model studio's `lamp.pipeline.js` for a model with no gate: landscape pictures turned upright, both branches while deciding and the stronger rope locked after 10 frames, decided again after 15 frames with no rope or a turned phone, "no lamp in view", largest piece, and the rope measurements (`RopeAnalyzer.kt`: thickness, center, drift, waviness, area, area rate, breathing) |
| Particles | `particle_detector_int8.tflite` (lc_100, INT8, float decode) | LiteRT 1.0.1, CPU 4 threads or GPU delegate | `particle_detector/run_model.py`: CLAHE, letterbox 1280 x 384, conf 0.20, NMS 0.35 |

Model studio's lamp file wraps the network in If nodes that pick the branch (as is, or the frame turned 90 deg). The app
runs the bare network and turns the frame itself (`VortexDetector.runBranch`), the same arithmetic: one copy of the weights
instead of three, and no If nodes, which NNAPI (the phone's AI chip) can't run. Measurements are in pixels of the analysed
picture; times are seconds of video (frame index over the mean frame rate) or since the camera started.

Not ported yet: calibration to real units, the summary at the end, particle tracking, speed and spin. The tilt warning
is left out on purpose: this model was trained on lamps tilted up to 45 deg. Videos are read frame by frame with
`MediaMetadataRetriever.getFrameAtIndex`, as fast as the model allows, not in real time; frames are turned by the file's
rotation tag when the retriever has not done it.

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

Results on the x86_64 emulator (2026-10-26):
- vortex, against model studio's live lamp file: rope IoU 0.9998-1.000 on the desktop's own input in the branch that
  finds the rope, 0.995-0.997 on the app's own resize (Android's bilinear resize differs from cv2's at the mask
  boundary); the nearly empty other branch differs by a few cells. The lying lamp (a landscape picture, turned upright)
  and the upright one give the same orientation, rope strength and thickness.
- particles, INT8 tflite vs the fp32 onnx desktop boxes: 38 vs 35 boxes, F1 0.932 at IoU 0.5.

Emulator times (vortex 310-410 ms with both branches while deciding, particles ~600 ms) say nothing about a phone. The test logs the phone's times
under the logcat tag `SmokeTest`, including whether the GPU delegate took the particle model (`backend`).
