"""The lamp rope network for the app (app/src/main/assets/vortex_lamp_rope.onnx), fp32, from its training checkpoint.

    PIPE=D:/pipe D:/pipe/vortex_detector/.venv/Scripts/python.exe tools/export_lamp_model.py

PIPE is the research folder that holds vortex_detector/ (default D:/pipe).

Model studio's file (vortex_lamp_auto_fp16.onnx) wraps the same network in If nodes that pick the branch: as is
("logits_asis", mode 2), or the frame turned 90 deg counter-clockwise and the mask turned back ("logits", mode 1).
The app runs the bare network and turns the frame itself (VortexDetector.runBranch): one copy of the weights instead
of three, and no If nodes, which NNAPI can't run. fp32: ONNX Runtime's CPU kernels have no fp16 convolutions.

Checks on the desktop that the bare network plus the app's turning gives model studio's masks, on the smoke test's
lamp image, in both branches.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

APP = Path(__file__).resolve().parents[1]
PIPE = Path(os.environ.get("PIPE", "D:/pipe"))
CHECKPOINT = PIPE / "vortex_detector/runs/lamp_r2_lite0_bg_strong_scale_tilt_steep_motion_defocus/lite0_lamp_s0.pt"
# model studio's file with this checkpoint (the live site's vortex_lamp_auto_fp16.onnx)
STUDIO_MODEL = PIPE / ("vortex_detector/models/lamp_r2_lite0_bg_strong_scale_tilt_steep_nogate/"
                       "vortex_lamp_auto_lamp_r2_lite0_bg_strong_scale_tilt_steep_nogate_fp16.onnx")
IMAGE = PIPE / "vortex_detector/lamp_fixed/frames/IMG_2117_f00006.png"
OUT = APP / "app/src/main/assets/vortex_lamp_rope.onnx"
H, W = 416, 232


class Normalized(nn.Module):
    """ImageNet gray normalisation inside the model: the input stays gray / 255."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(0.449))
        self.register_buffer("std", torch.tensor(0.226))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class Pad32(nn.Module):
    """Pads the input to multiples of 32 for the U-Net and crops the output back."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        return self.net(F.pad(x, (0, pw, 0, ph)))[:, :, :h, :w]


def lamp_input(bgr):
    """As model studio: BT.601 luma, fit into 416 x 232 at the top-left, zero padding, /255."""
    h, w = bgr.shape[:2]
    s = min(W / w, H / h)
    sw, sh = round(w * s), round(h * s)
    small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = np.zeros((1, 1, H, W), np.float32)
    x[0, 0, :sh, :sw] = (0.299 * small[..., 2] + 0.587 * small[..., 1] + 0.114 * small[..., 0]) / 255
    return x


def branch(sess, x, vertical):
    """As VortexDetector.runBranch: vertical = turned 90 deg counter-clockwise, run, turned back."""
    if not vertical:
        return sess.run(None, {"x": x})[0][0, 0]
    turned = np.ascontiguousarray(x[0, 0].T[::-1])[None, None]       # transpose, then flip the rows
    y = sess.run(None, {"x": turned})[0][0, 0]
    return np.ascontiguousarray(y.T[:, ::-1])                        # transpose back, then flip the columns


def main():
    ck = torch.load(CHECKPOINT, map_location="cpu")
    net = Normalized(smp.Unet("timm-tf_efficientnet_lite0", encoder_weights=None, in_channels=1, classes=1,
                              decoder_channels=(64, 32, 16, 8, 8)))
    net.load_state_dict(ck["model"])
    model = Pad32(net).eval()
    torch.onnx.export(model, torch.zeros(1, 1, H, W), str(OUT), input_names=["x"], output_names=["y"], opset_version=17,
                      dynamo=False, dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}})
    print(f"{OUT.name}: {OUT.stat().st_size / 1e6:.1f} MB")

    app = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    studio = ort.InferenceSession(str(STUDIO_MODEL), providers=["CPUExecutionProvider"])
    lying = cv2.imread(str(IMAGE))
    for label, bgr in [("lying", lying), ("upright", cv2.rotate(lying, cv2.ROTATE_90_CLOCKWISE))]:
        x = lamp_input(bgr)
        for mode, name, vertical in [(1, "logits", True), (2, "logits_asis", False)]:
            want = studio.run([name], {"image": x, "mode": np.array([mode], np.float32)})[0][0, 0] > 0
            got = branch(app, x, vertical) > 0
            union = (want | got).sum()
            print(f"{label} lamp, mode {mode}: rope cells {got.sum()} vs {want.sum()}, "
                  f"IoU {(want & got).sum() / union if union else 1:.4f}")


if __name__ == "__main__":
    main()
