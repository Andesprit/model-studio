# Detectors (Android)

A native Android app for the vortex lamp model and the particle detector, from the camera or a video on the phone.
It was written because model studio (`apps/web`) in a phone browser was too slow. The models are the same
files the web app ships (the particle model here is its INT8 LiteRT export).

| detector | model (in `app/src/main/assets`) | runtime | steps ported from |
|---|---|---|---|
| Vortex | `vortex_lamp.onnx` = `vortex_detector/models/model_studio/vortex_lamp_auto_fp32.onnx` | ONNX Runtime Android 1.20, 4 threads | `vortex_detector/model_studio/lamp.pipeline.js`: water region, orientation decided over 10 frames with water and then locked, one rope branch per frame, largest piece |
| Particles | `particle_detector_int8.tflite` (lc_100, INT8, float decode) | LiteRT 1.0.1, CPU 4 threads or GPU delegate | `particle_detector/run_model.py`: CLAHE, letterbox 1280 x 384, conf 0.20, NMS 0.35 |

Not ported yet: the rope measurements of lamp.pipeline.js (thickness, center, breathing...), the tilt warning,
particle tracking, speed and spin. Videos are read frame by frame with `MediaMetadataRetriever.getFrameAtIndex`, as fast as
the model allows, not in real time.

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

`tools/make_reference.py` copies the models and one test image per detector into the app and writes the desktop
outputs; `app/src/androidTest/.../SmokeTest.kt` compares against them on a phone or emulator:

```bash
./gradlew connectedDebugAndroidTest
```

Results on the x86_64 emulator (2026-10-17):
- vortex, on the desktop's own input tensor: rope and water IoU 1.000 in both modes. On the app's own resize, rope
  IoU 0.961 (mode 1, a 549-cell mask) and 0.996 (mode 2); water IoU 0.997. Android's bilinear resize differs
  from cv2's at the mask boundary.
- particles, INT8 tflite vs the fp32 onnx desktop boxes: 38 vs 35 boxes, F1 0.932 at IoU 0.5.

Emulator times (vortex ~360 ms, particles ~600 ms) say nothing about a phone. The test logs the phone's times
under the logcat tag `SmokeTest`, including whether the GPU delegate took the particle model (`backend`).
