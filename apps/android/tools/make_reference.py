"""Reference outputs for the app's on-device smoke test (app/src/androidTest).

    PIPE=D:/pipe D:/pipe/particle_detector/.venv/Scripts/python.exe tools/make_reference.py

PIPE is the research folder that holds particle_detector/ and vortex_detector/ (default D:/pipe).

Copies the two models and one test image of each detector into the app, and writes what the desktop runtimes give
on those images, for SmokeTest.kt to compare against:
  particles: boxes from particle_detector.onnx (fp32) through run_model.py's detect; the app runs the INT8 .tflite,
             so the test asks for matching boxes, not identical ones
  vortex:    rope masks of model studio's lamp file in mode 1 and 2, on a lying and an upright lamp (the app runs the
             bare network from tools/export_lamp_model.py and turns the frame itself, so the test checks both branches)
"""
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

APP = Path(__file__).resolve().parents[1]
PIPE = Path(os.environ.get("PIPE", "D:/pipe"))
ASSETS = APP / "app/src/main/assets"
TEST = APP / "app/src/androidTest/assets"

PARTICLE_MODEL = PIPE / "particle_detector/models/particle_detector_int8.tflite"
PARTICLE_REF_MODEL = PIPE / "particle_detector/models/particle_detector.onnx"
PARTICLE_IMAGE = PIPE / "particle_detector/data/learning_curve_level/lc_100/images/val/bomba-2_tube_3_frame_005750.jpg"
# model studio's file with the app's checkpoint (tools/export_lamp_model.py)
VORTEX_MODEL = PIPE / ("vortex_detector/models/lamp_r2_lite0_bg_strong_scale_tilt_steep_nogate/"
                       "vortex_lamp_auto_lamp_r2_lite0_bg_strong_scale_tilt_steep_nogate_fp16.onnx")
VORTEX_IMAGE = PIPE / "vortex_detector/lamp_fixed/frames/IMG_2117_f00006.png"
H, W = 416, 232

sys.path.insert(0, str(PIPE / "particle_detector"))
import run_model  # noqa: E402


def lamp_input(bgr):
    """As model studio: BT.601 luma, fit into 416 x 232 at the top-left, zero padding, /255."""
    h, w = bgr.shape[:2]
    s = min(W / w, H / h)
    sw, sh = round(w * s), round(h * s)
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = np.zeros((1, 1, H, W), np.float32)
    x[0, 0, :sh, :sw] = (0.299 * small[..., 2] + 0.587 * small[..., 1] + 0.114 * small[..., 0]) / 255
    return x


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    TEST.mkdir(parents=True, exist_ok=True)
    shutil.copy(PARTICLE_MODEL, ASSETS / "particle_detector_int8.tflite")
    cv2.imwrite(str(TEST / "particles.png"), cv2.imread(str(PARTICLE_IMAGE)))  # png: the same pixels on both sides
    lying = cv2.imread(str(VORTEX_IMAGE))
    cv2.imwrite(str(TEST / "lamp.png"), lying)
    cv2.imwrite(str(TEST / "lamp_upright.png"), cv2.rotate(lying, cv2.ROTATE_90_CLOCKWISE))

    detect = run_model.load(PARTICLE_REF_MODEL)
    boxes = detect(cv2.imread(str(TEST / "particles.png")))

    sess = ort.InferenceSession(str(VORTEX_MODEL), providers=["CPUExecutionProvider"])
    vortex = {}
    for image in ("lamp", "lamp_upright"):
        x = lamp_input(cv2.imread(str(TEST / f"{image}.png")))
        (TEST / f"{image}_input.bin").write_bytes(x.tobytes())  # float32 little-endian, 1 x 1 x 416 x 232
        for mode, name in [(1, "logits"), (2, "logits_asis")]:
            out = sess.run([name], {"image": x, "mode": np.array([mode], np.float32)})[0][0, 0]
            (TEST / f"{image}_rope_mode{mode}.bin").write_bytes((out > 0).astype(np.uint8).tobytes())
            vortex[f"{image} mode {mode}"] = int((out > 0).sum())

    ref = {"particles": boxes.round(2).tolist(), "vortex": vortex}
    (TEST / "reference.json").write_text(json.dumps(ref, indent=1))
    print(f"particles: {len(boxes)} boxes; vortex: {vortex}")


if __name__ == "__main__":
    main()
