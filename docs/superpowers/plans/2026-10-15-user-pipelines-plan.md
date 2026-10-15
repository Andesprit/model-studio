# Plan: User Pipelines (model → config → pipeline → source → calibrate)

## Goal
Let anyone attach their own logic to a model without touching the app: tracking,
counting, measuring, or anything else that needs to look at results over time. Today
the app can only do "one frame in, one overlay out", and the only extension point is a
stateless decoder plugin. We replace that plugin with a **pipeline**: a user-supplied
JS module that can (optionally) decode the model's raw output and then post-process
results with memory across frames, publish numbers, draw extra overlays, and ask the
user for calibration input. The app flow becomes five steps:

1. **Load model** — the `.onnx` (unchanged).
2. **Load config** — optional, data only (input prep, labels, built-in decoder name +
   options). Skipped when embedded in the model (manifest / Ultralytics metadata).
3. **Load pipeline** — optional, code. Brings its own decoder if the model needs one,
   plus post-decode logic.
4. **Source** — camera or video file (unchanged).
5. **Calibrate** — only shown if the pipeline asks for something (a scale, a line, a
   region, a number).

With no pipeline loaded the app behaves exactly as it does today.

## Assumptions
- **Config stays pure data, pipeline is all code.** Rule for users: "is it code? → pipeline."
  The config never references a JS file.
- **Decoder resolution:** if the loaded pipeline provides `decode`, it is used and the
  config's `decoder` is ignored (the UI says so). Otherwise the config/metadata decoder is used.
- **The "decoder plugin" concept is retired, but old plugin files keep working.** A
  module whose default export is `{ name, decode }` is a valid pipeline with only a decoder.
  The "Plugin .js" button becomes "Pipeline .js". `"decoder": "plugin:<name>"` in an
  existing config resolves to the loaded pipeline's decoder when the names match.
- **One pipeline at a time**, mirroring "one model at a time" in the current spec.
- **Fresh state = fresh instance.** The pipeline module exports a definition with a
  `create(...)` factory; the app discards the instance and calls `create` again on every
  reset. Pipelines never have to write their own reset logic.
- **Synchronous per frame in v1.** The pipeline's per-frame function returns a `Result`,
  not a promise. Chaining a second model (e.g. classify each crop) is out of scope for v1.
- **Pipelines can't `import` from the app**, because they are loaded from a Blob URL
  (see `apps/web/src/loaders.ts`). Everything reusable is handed to them through the
  context object (`ctx.lib`), extending today's `ctx.utils`.
- **Pipeline logic lives in `packages/core`** (pure TS, no DOM) so the React Native app
  can reuse it; only file loading and drawing live in `apps/web`. Per the existing spec,
  runtime-loaded JS stays web-only; on mobile pipelines get bundled.
- **The built-in tracker uses greedy nearest-centroid matching** (gated by a max distance
  and a max number of missed frames), not Hungarian/Kalman. It's simple, testable, and
  enough for slow-moving objects; a better tracker can replace it behind the same API.
- **The frame-loop class is renamed** from `Pipeline` (`apps/web/src/adapters/pipeline.ts`)
  to `FrameLoop`, so "pipeline" means only the user concept.
- **Calibration values and pipeline options live in app state for the session** and are
  exportable/importable as JSON, like the config. No automatic persistence.

## Approach
Add one new stage to the existing frame loop, between `decode` and `renderer.draw`:

```
grab → toTensor → runtime.run → decode (pipeline's or built-in) → pipeline step → draw
```

**The pipeline definition** (default export of the user's module) has these fields. Only
`name` is required, plus at least one of `decode` / `create`:

| Field | Purpose |
|---|---|
| `name` | Shown in the UI. |
| `decode(outputs, ctx)` | Optional custom decoder, the same signature as today's plugin `decode`. |
| `defaults`, `ranges` | Options and slider bounds, the same convention as decoders. |
| `calibration` | Optional list of inputs the user must provide: `number` (with unit), `point`, `line`, `region` (rectangle). Each has a key and a label. |
| `create(ctx)` | Returns an instance with `step(result, frameCtx) → Result`. Called again on every reset. |

**The context** gives the pipeline: merged `options`, `labels`, `calibration` values
(in frame pixel coordinates), and `lib` = built-in decoders callable by name, the
tracker factory, NMS/geometry/sigmoid/softmax (today's `utils`). The per-frame context
adds the video timestamp, a processed-frame counter, and frame size.

**The `Result` gains three optional fields** that any pipeline can fill:
- `Box.trackId?` — the renderer shows it on the box and colors by it.
- `metrics?: Record<string, number | string>` — shown in the stats bar ("unique: 132").
- `overlays?` — simple drawing primitives in frame coordinates (line, polyline, point,
  text) for counting lines, trails, and zones.

**Reset triggers** (re-call `create`): new model, new pipeline, option or calibration
change, new source, and the video time going backwards (the player has `loop` on and
can seek). Without the last one a looping video would count the same particles again.

**Errors** behave like decoder errors today: an exception in `step` or `decode` halts
the loop, pauses the video, and shows the message.

Minimal version: Tasks 1–3 (contract, loading, drawing). That already lets someone
write a tracker and counter themselves. Tasks 4–8 make it pleasant and add calibration.

**Fragile assumption:** this plan assumes processed frames are close enough in time for
frame-to-frame association to work. The loop deliberately drops frames when inference
is slower than the video, so on a slow device (WASM, large model) objects can jump too
far between processed frames and the tracker swaps or loses IDs, and counts inflate.
Mitigations built into the plan: the per-frame context exposes the real timestamp so
trackers can scale their gating by elapsed time, and the example pipeline reports
"frames processed / dropped" so users can see when results are unreliable. An
"analyze every frame" mode (seek frame by frame instead of real-time playback) is the
real fix and is listed as an open question rather than built now.

## Tasks

### Phase 1: Foundation

## Task 1: Pipeline contract and runner in core

**Description:** Define the pipeline definition/instance/context types and the new
`Result` fields in `packages/core`, plus a small runner that owns the current instance,
resolves the decoder (pipeline → config), runs `step`, and recreates the instance on
`reset()`. A plain `{ name, decode }` object must validate as a pipeline. No app changes.

**Acceptance criteria:**
- [ ] Validation rejects modules without `name`, or without both `decode` and `create`, with a readable message (matching the style of `registerPlugin`'s error).
- [ ] The runner uses the pipeline's `decode` when present, else the config's decoder; `"plugin:<name>"` resolves to a loaded pipeline of that name.
- [ ] `reset()` calls `create` again so state from before the reset is gone (proven by a counting test pipeline).

**Verification:**
- [ ] Tests pass: `npm test`
- [ ] Types: `npm run typecheck`

**Dependencies:** None

**Files likely touched:**
- `packages/core/src/types.ts`
- `packages/core/src/pipeline.ts` (new)
- `packages/core/src/decoders/index.ts`
- `packages/core/src/index.ts`
- `packages/core/test/pipeline.test.ts` (new)

**Scope:** M

## Task 2: Load a pipeline in the web app and run it per frame

**Description:** Rename the frame loop to `FrameLoop`, give it a `setPipeline()` that
plugs the core runner in between decode and draw, and replace the "Plugin .js" button
with a "Pipeline" step (load / unload, shows the name and whether it supplies the
decoder). Reorder the sidebar into the five steps: Model, Config, Pipeline, Source,
Calibrate (Calibrate stays hidden until Task 6).

**Acceptance criteria:**
- [ ] With no pipeline, behavior and the stats bar match today's app.
- [ ] Loading an existing decoder-only plugin file works with no edits to that file.
- [ ] Unloading a pipeline returns to the config's decoder without reloading the model.

**Verification:**
- [ ] Tests pass: `npm test`; types: `npm run typecheck`
- [ ] Manual check (`npm run dev`): sample YOLO model + video, then load and unload a decoder-only plugin.

**Dependencies:** Task 1

**Files likely touched:**
- `apps/web/src/adapters/pipeline.ts` → `apps/web/src/adapters/frameLoop.ts`
- `apps/web/src/loaders.ts`
- `apps/web/src/App.tsx`
- `apps/web/src/ui/ConfigEditor.tsx`

**Scope:** M

### Checkpoint: Foundation — tests and typecheck pass; app works with no pipeline and with an old decoder plugin.

### Phase 2: Core features

## Task 3: Draw pipeline output (track IDs, metrics, overlays)

**Description:** Teach the renderer the new `Result` fields: label and color boxes by
`trackId` when present, draw `overlays`, and show `metrics` in the stats bar next to fps.

**Acceptance criteria:**
- [ ] Boxes with a `trackId` show it in the label and keep a stable color per ID.
- [ ] Line, polyline, point and text overlays render in frame coordinates and scale with resolution like current boxes.
- [ ] Each metric shows as one `key: value` item in the stats bar.

**Verification:**
- [ ] Types: `npm run typecheck`
- [ ] Manual check: a throwaway pipeline that adds a fixed line, a text overlay and a metric renders all three.

**Dependencies:** Task 2

**Files likely touched:**
- `apps/web/src/adapters/renderer.ts`
- `apps/web/src/adapters/frameLoop.ts`
- `apps/web/src/App.tsx`

**Scope:** S

## Task 4: Reset rules

**Description:** Trigger a pipeline reset on new model, new pipeline, option change,
new source, and when the video timestamp goes backwards (loop or seek). Pass the
timestamp and processed/dropped frame counts in the per-frame context.

**Acceptance criteria:**
- [ ] A counting pipeline's total returns to 0 when the video loops or the user seeks back.
- [ ] Switching camera ↔ file or loading another model resets state.
- [ ] Seeking forward or pausing does not reset.

**Verification:**
- [ ] Tests pass: `npm test` (runner reset unit tests)
- [ ] Manual check: a frame-counter pipeline across loop, seek back, seek forward, pause.

**Dependencies:** Task 3

**Files likely touched:**
- `apps/web/src/adapters/frameLoop.ts`
- `apps/web/src/App.tsx`
- `packages/core/src/pipeline.ts`

**Scope:** S

## Task 5: Built-in library for pipelines (`ctx.lib`) and the tracker

**Description:** Expose built-in decoders callable by name, today's `utils`, and a
`createTracker(options)` from core through the pipeline context. The tracker does greedy
nearest-centroid matching with `maxDistance` (scaled by elapsed time) and `maxMissed`,
and returns boxes with `trackId` plus the count of unique IDs so far.

**Acceptance criteria:**
- [ ] Tracker unit tests: stable IDs for objects moving under `maxDistance`; new ID after an object is missing more than `maxMissed` frames; no ID reused.
- [ ] A pipeline can call `ctx.lib.decode("yolo-detect", outputs)` and get the same result as the built-in path.
- [ ] Pipeline `defaults`/`ranges` appear as editable options in the Pipeline step, using the same slider pattern as `ConfigEditor`.

**Verification:**
- [ ] Tests pass: `npm test`
- [ ] Manual check: a ~15-line pipeline using `ctx.lib` shows track IDs and a unique-count metric on a sample video.

**Dependencies:** Task 4

**Files likely touched:**
- `packages/core/src/tracker.ts` (new)
- `packages/core/src/pipeline.ts`
- `packages/core/test/tracker.test.ts` (new)
- `apps/web/src/App.tsx` (pipeline options UI)

**Scope:** M

### Checkpoint: Core — a tracking/counting pipeline written only against `ctx.lib` works end to end on a video, and the count survives pause but resets on loop.

### Phase 3: Calibrate

## Task 6: Calibration step for number inputs

**Description:** Read the pipeline's `calibration` declaration and show the Calibrate step
only when it is non-empty. Start with `number` fields (e.g. "pixels per µm"). Values reach
the pipeline via `ctx.calibration`; changing one resets the pipeline. Calibration and
pipeline options can be exported/imported as one JSON file.

**Acceptance criteria:**
- [ ] No Calibrate step for pipelines that declare nothing.
- [ ] Missing required values block the pipeline with a clear message instead of failing inside user code.
- [ ] Export, then import, restores the same values.

**Verification:**
- [ ] Tests pass: `npm test` (validation of declarations and values)
- [ ] Manual check: a pipeline that displays a µm size metric from a calibrated scale.

**Dependencies:** Task 5

**Files likely touched:**
- `packages/core/src/pipeline.ts`
- `apps/web/src/ui/Calibrate.tsx` (new)
- `apps/web/src/App.tsx`
- `apps/web/src/loaders.ts`

**Scope:** M

## Task 7: Draw calibration geometry on the video

**Description:** Support `point`, `line` and `region` calibration fields by clicking or
dragging on the overlay canvas, converting from displayed CSS pixels to frame pixels.
The current value stays visible as an overlay while the step is open.

**Acceptance criteria:**
- [ ] A line drawn on a scaled-down video arrives in `ctx.calibration` in frame-pixel coordinates (unit test on the conversion).
- [ ] Points and lines can be redrawn; a region can be redrawn by dragging a new one.
- [ ] Drawing works with the video paused and playing.

**Verification:**
- [ ] Tests pass: `npm test`
- [ ] Manual check: draw a counting line on a sample video, then resize the window and confirm it stays on the same spot of the frame.

**Dependencies:** Task 6

**Files likely touched:**
- `apps/web/src/ui/Calibrate.tsx`
- `apps/web/src/adapters/renderer.ts`
- `apps/web/src/App.tsx`
- `apps/web/src/styles.css`

**Scope:** M

## Task 8: Example pipeline, authoring guide, spec update

**Description:** Ship an example "particle count" pipeline (tracker + line crossing +
unique count + processed/dropped frames + optional µm scale), a short guide on writing
pipelines (definition fields, `ctx.lib`, reset rules, the decoder-only case), and update
the design spec's flow, contracts and "out of scope" sections.

**Acceptance criteria:**
- [ ] The example loads from a file and counts line crossings on a sample video.
- [ ] The guide is enough to write a pipeline without reading app source (checked by writing a second tiny example from the guide alone).
- [ ] The spec no longer describes decoder plugins as a separate concept.

**Verification:**
- [ ] Manual check: run the example from a fresh `npm run dev`.

**Dependencies:** Task 7

**Files likely touched:**
- `examples/pipelines/particle-count.js` (new)
- `docs/pipelines.md` (new)
- `docs/superpowers/specs/2026-10-13-generic-vision-webapp-design.md`

**Scope:** S

### Checkpoint: Complete — all five steps work end to end; tests and typecheck pass; a new user can write a pipeline from the guide.

## Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Dropped frames break tracking on slow devices | High | Timestamp-scaled gating; show processed/dropped counts; "analyze every frame" mode as a follow-up (Open Questions). |
| User code runs in the page with full access | Medium | Same exposure as today's plugins: only files the user picks are run. Say so in the UI and the guide. Never auto-load code from model metadata. |
| Slow `step` code lowers fps | Medium | Time it separately in stats ("pipeline ms") so users see the cost. |
| `Result` additions break the RN app's types later | Low | All new fields are optional; nothing existing changes shape. |
| Breaking existing configs that use `plugin:<name>` | Low | Name-based resolution to the loaded pipeline (Task 1). |

## Open Questions
- **Analyze every frame?** Real-time playback drops frames, which hurts counting accuracy.
  An offline mode would step the video frame by frame at whatever speed inference allows.
  *Assumed for now:* real-time only; revisit after Task 8.
- **Async pipelines / chaining a second model** (e.g. classify each detected crop).
  *Assumed:* synchronous `step` in v1; the contract can later accept a promise without
  breaking sync pipelines.
- **Built-in pipelines picked from a list** (e.g. "Track + count") without a file.
  *Assumed:* file-only in v1, with the example shipped as a file to copy.
- **Exporting per-frame results** (CSV of tracks/counts). The current spec lists export
  as out of scope. *Assumed:* still out of scope; metrics are on screen only.
- **Tracker quality.** Greedy matching may swap IDs when particles cross.
  *Assumed:* greedy is enough for v1; Hungarian/Kalman can replace it behind `createTracker`.
