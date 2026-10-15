# Generic Vision Webapp — Design

## Goal
Client-side web app that runs any ONNX vision model (detection, segmentation, pose,
classification) on a video file or the live camera, using WebGPU when available
(WASM fallback), drawing results as a live overlay. Built so the logic can later be
reused by a React Native app.

## Decisions
- Models: ONNX only, loaded from a local file. Nothing is uploaded.
- Output: live overlay + stats only (no export, one model at a time).
- Decoding: built-in presets + JS decoder plugins, optionally described by a JSON
  sidecar config. Ultralytics ONNX metadata (task, names, imgsz) auto-configures.
- Stack: npm workspaces. `packages/core` (pure TS, no DOM) + `apps/web` (Vite + React).

## Layout
```
packages/core/   config, preprocess (letterbox math, RGBA→tensor), decoders,
                 nms/geometry, ONNX metadata reader, types. Vitest unit tests.
apps/web/        adapters (frame grabber, ort-web runtime, canvas renderer),
                 plugin loader, React UI.
```

## Contracts
- `Frame {width,height,data: Uint8ClampedArray}` RGBA
- `Tensor {dims:number[], data: Float32Array}` runtime-agnostic
- `ModelConfig {decoder, input{width,height,layout,resize,mean,std,scale}, labels?, options?}`
- `Decoder {name, decode(outputs: Record<string,Tensor>, ctx): Result}`
- `Result {boxes?, masks?, segmentation?, poses?, classes?}` in frame coordinates

## Frame loop
`requestVideoFrameCallback` (rAF fallback). At most one inference in flight; frames
are dropped, never queued. Per frame: draw letterboxed into offscreen canvas →
`toTensor` → `session.run` → `decode` → render. Stats: FPS, pre/infer/post ms, backend.

## Errors
Typed messages in a status bar: model load failure, config/shape mismatch, camera
denied, WebGPU unavailable (warning, falls back to WASM). Decoder exceptions pause
the loop and show the message.

## Testing
Vitest on core (preprocess, NMS, letterbox round-trip, each decoder on synthetic
tensors, metadata reader). Browser smoke test with a real YOLOv8n ONNX.

## Out of scope
Export/recording, multi-model, the RN app, GPU-only preprocessing.
Note: runtime JS plugins are web-only; on mobile, plugins get bundled.
