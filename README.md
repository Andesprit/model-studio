# Detect

Run any ONNX vision model (detection, instance/semantic segmentation, pose, classification) in the
browser on a video file or the live camera. Uses WebGPU when available and falls back to WASM.
Everything runs locally; nothing is uploaded.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # core unit tests + real-model checks (if models/ exists)
npm run build      # static site in apps/web/dist
```

Sample models: the buttons under **Model** load whatever is listed in
`apps/web/public/models/index.json`. To (re)create the YOLOv8n samples:

```bash
cd models
python -c "from ultralytics import YOLO; [YOLO(m).export(format='onnx', opset=17) for m in ['yolov8n.pt','yolov8n-seg.pt','yolov8n-pose.pt','yolov8n-cls.pt']]"
cp *.onnx ../apps/web/public/models/
```

## How a model gets configured

In order of priority:

1. **Config file** (`Config .json`), which describes preprocessing and which decoder reads the outputs.
   See `apps/web/public/examples/semantic-seg.example.json`. Any field you leave out gets a default.
   **detector-app manifests** (`*.manifest.json` with `input` / `postprocess`) are accepted as-is:
   `binary_mask`, `boxes` (xyxysc) and `classification`.
2. **Metadata embedded in the `.onnx`**: a detector-app manifest (`detector.manifest`, written by
   `embed_manifest.py`) or Ultralytics metadata (task, class names, image size). These models work
   with zero setup.
3. **The model's input shape**, after which you pick the decoder yourself.

Preprocessing covers grayscale or color input, RGB/BGR, NCHW/NHWC, `letterbox` / `stretch` /
`scale` (fixed factor) / `shortest_edge` resizing, center or top-left alignment, pad value and
mean/std normalization.

The decoder can be switched live in the sidebar.

| Decoder        | Expects |
|----------------|---------|
| `yolo-detect`  | YOLO v5/v8/v9/v11/v12 `[1,4+nc,N]` / `[1,N,5+nc]`, NMS-free v10/YOLO26 `[1,N,6]` (`format` option to force) |
| `yolo-seg`     | as above + prototype masks `[1,32,H,W]` |
| `yolo-pose`    | as above + keypoints (`kptShape`, default `[17,3]`) |
| `semantic-seg` | logits `[1,C,H,W]` / `[1,H,W,C]`, class indices `[1,H,W]`, or binary logits `[1,1,H,W]` (`mode`, `threshold`) |
| `classify`     | `[1,C]` logits or probabilities |

## Pipelines

For anything else, write a pipeline as an ES module and load it with **Pipeline .js**. A pipeline can
bring its own `decode` (replacing the config's decoder) and/or a `create(ctx)` that returns a stateful
`step(result, frame)` run after decoding, e.g. for tracking or counting. See
`apps/web/public/examples/tracking.pipeline.js` for a commented example of both (a hand-written
YOLOv8 decoder, then tracking and speed using the calibration), and
`apps/web/public/examples/people-counter.plugin.js` for a decoder-only one. A decoder gets the raw
output tensors plus helpers (`nms`, `softmax`, coordinate mapping) and returns boxes/masks/keypoints/classes
in frame pixels.

## Layout

```
packages/core   platform-free TS: config, preprocessing, decoders, NMS, ONNX metadata reader
apps/web        React + Vite: onnxruntime-web runtime, video/camera capture, canvas overlay, UI
```

`core` has no DOM or Node dependencies, so a future React Native app can reuse it unchanged and only
add its own adapters (`onnxruntime-react-native`, `react-native-vision-camera`, Skia). Runtime `.js`
pipelines are web-only. On mobile, pipelines have to be bundled into the app.
