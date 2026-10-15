import { DecodeError, makeCalibration, modelKey, PIPELINE_METADATA_KEY, PipelineRunner, readOnnxMetadata, type ModelConfig, type PipelineDefinition } from "@detect/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { FrameLoop, type Stats } from "./adapters/frameLoop";
import { createRuntime, errorMessage, webgpuSupported, type BackendPreference, type ModelRuntime } from "./adapters/runtime";
import { checkInputShape, downloadJson, inferConfig, loadConfigFile, loadPipeline, loadPipelineSource } from "./loaders";
import { CalibrationLine, lineLength, type Line } from "./ui/CalibrationLine";
import { ConfigEditor, OptionField, SHOW_OPTIONS } from "./ui/ConfigEditor";

interface Sample {
  name: string;
  /** Where to download the model: absolute, or relative to the index.json that lists it. */
  url: string;
  /** Recommended decoder options for this model (e.g. its best confidence threshold), applied over the detected config. */
  options?: Record<string, unknown>;
}

/** Applies a sample's recommended options on top of a detected config. */
const withOptions = (config: ModelConfig, options?: Record<string, unknown>): ModelConfig =>
  options ? { ...config, options: { ...config.options, ...options } } : config;

type Source = "camera" | "file" | null;

/** The list of downloadable models; each entry's `url` is resolved against this file's location. */
const MODELS_INDEX = `${import.meta.env.BASE_URL}models/index.json`;

/** Sample models loaded on startup as model A and model B. */
const PRELOAD = ["vortex_detector_fp32_with_pipeline.onnx", "particle_detector_with_pipeline.onnx"];

interface SecondModel {
  runtime: ModelRuntime;
  name: string;
  config: ModelConfig;
  configSource: string;
  decoderGuessed: boolean;
}

/** Calibration units: [symbol used in results, name shown in the dropdown]. */
const UNITS: [string, string][] = [
  ["mm", "millimeters (mm)"],
  ["cm", "centimeters (cm)"],
  ["m", "meters (m)"],
  ["in", "inches (in)"],
  ["ft", "feet (ft)"],
  ["yd", "yards (yd)"],
];

export function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const loopRef = useRef<FrameLoop | null>(null);
  const runnerRef = useRef<PipelineRunner | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [source, setSource] = useState<Source>(null);
  const [sourceName, setSourceName] = useState("");
  const [aspect, setAspect] = useState(16 / 9);
  const [backendPref, setBackendPref] = useState<BackendPreference>("auto");
  const [runtime, setRuntime] = useState<ModelRuntime | null>(null);
  const [modelName, setModelName] = useState("");
  const [loading, setLoading] = useState(false);
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [explicitConfig, setExplicitConfig] = useState<ModelConfig | null>(null);
  const [configSource, setConfigSource] = useState("");
  /** Nothing told us how to decode this model; the decoder is a default the user should confirm. */
  const [decoderGuessed, setDecoderGuessed] = useState(false);
  /** Text of the last decoder failure; while it's the error shown, it already explains what to do. */
  const [decodeErrorText, setDecodeErrorText] = useState("");
  /** Optional second model (slot B), run on the same frames as model A. Its config is auto-detected. */
  const [modelB, setModelB] = useState<SecondModel | null>(null);
  /** Whether each loaded model runs; unticking keeps it loaded so it can be turned back on instantly. */
  const [useA, setUseA] = useState(true);
  const [useB, setUseB] = useState(true);
  const preloadStarted = useRef(false);
  /** Bytes of the model in each slot, kept so a backend change can reload it. */
  const loadedBytes = useRef<({ bytes: Uint8Array; name: string; options?: Record<string, unknown>; trusted: boolean } | null)[]>([
    null,
    null,
  ]);
  /** Pipelines embedded in each slot's model; each runs on that model's output only. */
  const [modelPipelines, setModelPipelines] = useState<(PipelineDefinition | null)[]>([null, null]);
  const modelRunners = useRef<(PipelineRunner | null)[]>([null, null]);
  const [pipeline, setPipeline] = useState<PipelineDefinition | null>(null);
  const [pipelineOptions, setPipelineOptions] = useState<Record<string, unknown>>({});
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });
  const [calLine, setCalLine] = useState<Line | null>(null);
  const [calAmount, setCalAmount] = useState(0);
  const [calUnit, setCalUnit] = useState("cm");
  const [drawingLine, setDrawingLine] = useState(false);
  const calPixels = calLine ? lineLength(calLine) : 0;
  const calibration = useMemo(() => makeCalibration(calPixels, calAmount, calUnit), [calPixels, calAmount, calUnit]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const p = new FrameLoop(videoRef.current!, overlayRef.current!, {
      onStats: setStats,
      onError: (e) => {
        const text = `Inference error: ${errorMessage(e)}`;
        setError(text);
        if (e instanceof DecodeError) setDecodeErrorText(text);
      },
    });
    loopRef.current = p;
    p.start();
    fetch(MODELS_INDEX)
      .then((r) => (r.ok ? r.json() : []))
      .then((list: Sample[]) => {
        setSamples(list);
        // Start with the vortex (A) + particle (B) models loaded and on. The ref keeps React's dev double-mount from loading twice.
        if (preloadStarted.current) return;
        preloadStarted.current = true;
        void preload(list);
      })
      .catch(() => setSamples([]));
    return () => p.stop();
  }, []);

  // Config edits only swap config; model swaps go through setModel.
  useEffect(() => {
    if (config) loopRef.current?.setConfig(0, config);
  }, [config]);
  const configB = modelB?.config;
  useEffect(() => {
    if (configB) loopRef.current?.setConfig(1, configB);
  }, [configB]);

  // A new runner starts with fresh pipeline state.
  useEffect(() => {
    runnerRef.current = pipeline ? new PipelineRunner(pipeline, pipelineOptions, calibration) : null;
    loopRef.current?.setPipeline(runnerRef.current);
  }, [pipeline, pipelineOptions, calibration]);

  // Each model's own pipeline gets a fresh runner whenever it or the calibration changes.
  useEffect(() => {
    modelPipelines.forEach((def, slot) => {
      const runner = def ? new PipelineRunner(def, {}, calibration) : null;
      modelRunners.current[slot] = runner;
      loopRef.current?.setModelPipeline(slot, runner);
    });
  }, [modelPipelines, calibration]);

  const anyPipeline = !!pipeline || modelPipelines.some(Boolean);

  /** Saves what the pipelines measured, plus what it was measured on, as one .json file with a section per model. */
  function downloadResults() {
    const global = runnerRef.current;
    const video = videoRef.current!;
    const model = (slot: number, name: string, backend: string) => {
      const runner = modelRunners.current[slot];
      return { slot: modelKey(slot), name, backend, pipeline: runner?.pipeline.name ?? null, ...(runner ? runner.summary() : {}) };
    };
    const results = {
      exportedAt: new Date().toISOString(),
      source: source === "camera" ? "camera" : sourceName,
      videoDuration: source === "file" && Number.isFinite(video.duration) ? video.duration : undefined,
      frameSize,
      calibration,
      // Only the models that were switched on, each with what its own pipeline measured.
      models: [
        ...(useA && runtime ? [model(0, modelName, runtime.backend)] : []),
        ...(useB && modelB ? [model(1, modelB.name, modelB.runtime.backend)] : []),
      ],
      // A pipeline loaded in step 3 runs on all models together; its results go here.
      ...(global && { pipeline: global.pipeline.name, ...global.summary() }),
    };
    const names = [global?.pipeline.name, ...modelRunners.current.map((r) => r?.pipeline.name)].filter(Boolean);
    const base = `${names.join("+") || "results"}-${(sourceName || "camera").replace(/\.[^.]+$/, "")}`;
    downloadJson(`${base}-results.json`, results);
  }

  /**
   * The pipeline embedded in a model file, if any. A model from outside the app (picked from disk) runs
   * code when its pipeline loads, so the user is asked first; the app's own sample models are trusted.
   */
  async function embeddedPipeline(bytes: Uint8Array, name: string, slot: number): Promise<PipelineDefinition | null> {
    let source: string | undefined;
    try {
      source = readOnnxMetadata(bytes)[PIPELINE_METADATA_KEY];
    } catch {
      return null; // unreadable metadata: no pipeline
    }
    if (!source) return null;
    const entry = loadedBytes.current[slot];
    if (entry && !entry.trusted) {
      const ok = window.confirm(`"${name}" includes a pipeline (code that runs in this page to measure its results). Run it?`);
      if (!ok) return null;
      entry.trusted = true; // don't ask again when the same model reloads (e.g. backend change)
    }
    try {
      return await loadPipelineSource(source);
    } catch (e) {
      setError(`${name}: its embedded pipeline failed to load: ${errorMessage(e)}`);
      return null;
    }
  }

  function setModelPipeline(slot: number, def: PipelineDefinition | null) {
    setModelPipelines((p) => p.map((d, i) => (i === slot ? def : d)));
  }

  function editConfig(next: ModelConfig) {
    setError(""); // a previous inference error may not apply to the new settings
    if (next.decoder !== config?.decoder) setDecoderGuessed(false); // the user picked one
    setConfig(next);
  }

  function editConfigB(next: ModelConfig) {
    setError("");
    setModelB((m) => m && { ...m, config: next, decoderGuessed: m.decoderGuessed && next.decoder === m.config.decoder });
  }

  const shapeWarning = useA && runtime && config ? checkInputShape(config, runtime.input) : null;
  const shapeWarningB = useB && modelB ? checkInputShape(modelB.config, modelB.runtime.input) : null;
  // A decode error already gives the same advice, with the actual reason.
  const showGuessWarnings = !(error && error === decodeErrorText);
  const decoderWarning =
    useA && runtime && config && decoderGuessed && !pipeline?.decode && showGuessWarnings
      ? `We don't know how to read ${modelB ? "model A's" : "this model's"} output, so we're trying "${config.decoder}". ` +
        "If the results look wrong, pick another decoder in step 2, or load a Config .json or a Pipeline .js with its own decode."
      : null;
  const decoderWarningB =
    useB && modelB?.decoderGuessed && showGuessWarnings
      ? `We don't know how to read model B's output, so we're trying "${modelB.config.decoder}". ` +
        "If the results look wrong, pick another decoder for model B in step 2."
      : null;

  /** Loads a model into slot 0 (A) or 1 (B). `on` says whether it starts switched on. */
  async function loadModel(
    bytes: Uint8Array,
    name: string,
    slot = 0,
    backend = backendPref,
    on = true,
    options?: Record<string, unknown>,
    trusted = false,
  ) {
    const prev = loadedBytes.current[slot];
    // A reload of the same model (backend change) keeps the user's earlier answer about its pipeline.
    loadedBytes.current[slot] = { bytes, name, options, trusted: trusted || (prev?.bytes === bytes && prev.trusted) };
    if (slot === 1) return loadModelB(bytes, name, backend, on, options);
    setLoading(true);
    setError("");
    try {
      if (runtime) {
        await loopRef.current!.detach(0); // let the in-flight frame finish before freeing its GPU buffers
        await runtime.release();
      }
      const rt = await createRuntime(bytes, backend);
      const inferred = inferConfig(bytes, rt.input);
      const cfg = explicitConfig ?? withOptions(inferred.config, options);
      setRuntime(rt);
      setModelName(name);
      setConfig(cfg);
      setConfigSource(explicitConfig ? "config file" : inferred.source);
      setDecoderGuessed(!explicitConfig && inferred.guessed);
      setUseA(on);
      loopRef.current!.setModel(0, on ? rt : null, on ? cfg : null);
      setModelPipeline(0, await embeddedPipeline(bytes, name, 0));
    } catch (e) {
      setRuntime(null);
      setModelName("");
      loopRef.current!.setModel(0, null, null);
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  async function loadModelB(bytes: Uint8Array, name: string, backend: BackendPreference, on: boolean, options?: Record<string, unknown>) {
    setLoading(true);
    setError("");
    try {
      await unloadModelB();
      const rt = await createRuntime(bytes, backend);
      const inferred = inferConfig(bytes, rt.input);
      const cfg = withOptions(inferred.config, options);
      setModelB({ runtime: rt, name, config: cfg, configSource: inferred.source, decoderGuessed: inferred.guessed });
      setUseB(on);
      loopRef.current!.setModel(1, on ? rt : null, on ? cfg : null);
      setModelPipeline(1, await embeddedPipeline(bytes, name, 1));
    } catch (e) {
      setError(`Model B: ${errorMessage(e)}`);
    } finally {
      setLoading(false);
    }
  }

  async function unloadModelB() {
    if (!modelB) return;
    setModelB(null);
    await loopRef.current!.detach(1);
    await modelB.runtime.release();
  }

  async function preload(list: Sample[]) {
    const byFile = (f: string) => list.find((s) => s.url.split("/").pop() === f);
    const first = byFile(PRELOAD[0]);
    const second = byFile(PRELOAD[1]);
    if (first) await onSample(first, 0);
    if (second) await onSample(second, 1);
  }

  /** Switches backend and reloads the loaded models on it, keeping each one on or off as it was. */
  async function changeBackend(backend: BackendPreference) {
    setBackendPref(backend);
    const [a, b] = loadedBytes.current;
    if (a) await loadModel(a.bytes, a.name, 0, backend, useA, a.options, a.trusted);
    if (b) await loadModel(b.bytes, b.name, 1, backend, useB, b.options, b.trusted);
  }

  /** Turns a loaded model on or off without unloading it. */
  function toggleModel(slot: number, on: boolean) {
    setError("");
    if (slot === 0) {
      setUseA(on);
      loopRef.current!.setModel(0, on ? runtime : null, on ? config : null);
    } else {
      setUseB(on);
      loopRef.current!.setModel(1, on && modelB ? modelB.runtime : null, on && modelB ? modelB.config : null);
    }
  }

  async function onModelFile(file: File, slot = 0) {
    await loadModel(new Uint8Array(await file.arrayBuffer()), file.name, slot);
  }

  async function onSample(sample: Sample, slot = 0) {
    setLoading(true);
    try {
      const url = new URL(sample.url, new URL(MODELS_INDEX, location.href)).href;
      const res = await fetch(url);
      // The dev server answers a missing file with the app's HTML page, not a 404.
      if (!res.ok || res.headers.get("content-type")?.includes("text/html")) {
        throw new Error(`Could not download ${sample.name} from ${url} (HTTP ${res.status})`);
      }
      // The models listed in index.json come from the app's own deployment, so their embedded pipelines are trusted like the app's code.
      await loadModel(new Uint8Array(await res.arrayBuffer()), sample.name, slot, backendPref, true, sample.options, true);
    } catch (e) {
      setError(errorMessage(e));
      setLoading(false);
    }
  }

  async function onConfigFile(file: File) {
    try {
      const cfg = await loadConfigFile(file);
      setExplicitConfig(cfg);
      setConfig(cfg);
      setConfigSource(`config file (${file.name})`);
      setDecoderGuessed(false);
      setError("");
    } catch (e) {
      setError(errorMessage(e));
    }
  }

  async function onPipelineFile(file: File) {
    try {
      setPipeline(await loadPipeline(file));
      setPipelineOptions({});
      setError("");
    } catch (e) {
      setError(`Pipeline failed to load: ${errorMessage(e)}`);
    }
  }

  function unloadPipeline() {
    setPipeline(null);
    setPipelineOptions({});
    setError("");
  }

  function stopSource() {
    loopRef.current?.resetPipeline();
    // A scale measured on one view doesn't apply to another.
    setCalLine(null);
    setDrawingLine(false);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const v = videoRef.current!;
    if (v.src) URL.revokeObjectURL(v.src);
    v.removeAttribute("src");
    v.srcObject = null;
  }

  async function startCamera() {
    stopSource();
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      setSource("camera");
      setSourceName("");
    } catch (e) {
      setSource(null);
      const name = e instanceof DOMException ? e.name : "";
      setError(name === "NotAllowedError" ? "Camera permission denied" : `Camera failed: ${errorMessage(e)}`);
    }
  }

  async function startVideoFile(file: File) {
    stopSource();
    setError("");
    const v = videoRef.current!;
    v.src = URL.createObjectURL(file);
    setSource("file");
    setSourceName(file.name);
    try {
      await v.play();
    } catch (e) {
      // The browser can't decode this codec (e.g. MPEG-4 Part 2, H.265, or an .avi container).
      const unsupported = e instanceof DOMException && e.name === "NotSupportedError";
      setError(
        unsupported
          ? `Could not play ${file.name}: the browser doesn't support this video's format. Convert it to H.264 MP4, ` +
              `e.g. ffmpeg -i "${file.name}" -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4`
          : `Could not play video: ${errorMessage(e)}`,
      );
    }
  }

  return (
    <div className="app">
      <aside>
        <h1>Detect</h1>

        <section>
          <h2>1. Model</h2>
          <label className="field">
            <span>Backend</span>
            <select value={backendPref} disabled={loading} onChange={(e) => void changeBackend(e.target.value as BackendPreference)}>
              <option value="auto">Auto (WebGPU → WASM)</option>
              <option value="webgpu" disabled={!webgpuSupported()}>
                WebGPU{webgpuSupported() ? "" : " (unavailable)"}
              </option>
              <option value="wasm">WASM (CPU)</option>
            </select>
          </label>
          <p className="hint">Both models run on the same frames; untick one to run only the other.</p>
          {[0, 1].map((slot) => {
            const rt = slot === 0 ? runtime : modelB?.runtime;
            const name = slot === 0 ? modelName : modelB?.name;
            const on = slot === 0 ? useA : useB;
            return (
              <div key={slot}>
                {rt ? (
                  <label className="check">
                    <input type="checkbox" checked={on} disabled={loading} onChange={(e) => toggleModel(slot, e.target.checked)} />{" "}
                    {modelKey(slot)}: <b>{name}</b> on <b className={rt.backend}>{rt.backend}</b>
                    {modelPipelines[slot] && <> · pipeline <b>{modelPipelines[slot]!.name}</b></>}
                  </label>
                ) : (
                  <p className="hint">
                    {modelKey(slot)}: {loading ? "loading…" : "none"}
                  </p>
                )}
                <details>
                  <summary>Change model {modelKey(slot)}</summary>
                  <FileButton label={loading ? "Loading…" : "Load .onnx"} accept=".onnx" onFile={(f) => onModelFile(f, slot)} disabled={loading} small />
                  {samples.length > 0 && (
                    <div className="samples">
                      {samples.map((s) => (
                        <button key={s.url} disabled={loading} onClick={() => onSample(s, slot)}>
                          {s.name}
                        </button>
                      ))}
                    </div>
                  )}
                </details>
              </div>
            );
          })}
        </section>

        <section>
          <h2>2. Config</h2>
          <FileButton label="Config .json" accept=".json,application/json" onFile={onConfigFile} small />
          {runtime && (
            <p className="hint">
              {modelB && "Model A: "}Config from {configSource}
            </p>
          )}
          {explicitConfig && (
            <button className="link" onClick={() => setExplicitConfig(null)}>
              Forget config file (auto-detect on next model load)
            </button>
          )}
          {config && (
            <ConfigEditor config={config} decoderFromPipeline={pipeline?.decode ? pipeline.name : undefined} onChange={editConfig} />
          )}
          {modelB && (
            <>
              <h3>Model B</h3>
              <p className="hint">Config from {modelB.configSource}</p>
              <ConfigEditor config={modelB.config} onChange={editConfigB} />
            </>
          )}
        </section>

        <section>
          <h2>3. Pipeline</h2>
          <div className="row">
            <FileButton label="Pipeline .js" accept=".js,.mjs,text/javascript" onFile={onPipelineFile} small />
            {pipeline && (
              <button className="small" onClick={unloadPipeline}>
                Unload
              </button>
            )}
          </div>
          {pipeline ? (
            <p className="hint">
              <b>{pipeline.name}</b>: {[pipeline.decode && "decoder", pipeline.create && "post-processing"].filter(Boolean).join(" + ")}
            </p>
          ) : (
            <p className="hint">
              {modelPipelines.some(Boolean)
                ? "Models with an embedded pipeline run it on their own. Load one here to run another on all models together."
                : "Optional. Your own decoder and/or logic across frames (tracking, counting…)."}
            </p>
          )}
          {SHOW_OPTIONS &&
            pipeline &&
            Object.entries({ ...pipeline.defaults, ...pipelineOptions }).map(([key, value]) => (
              <OptionField
                key={`${pipeline.name}.${key}`}
                name={key}
                value={value}
                range={pipeline.ranges?.[key]}
                onChange={(v) => setPipelineOptions((o) => ({ ...o, [key]: v }))}
              />
            ))}
        </section>

        <section>
          <h2>4. Source</h2>
          <div className="row">
            <button onClick={startCamera} className={source === "camera" ? "active" : ""}>
              Camera
            </button>
            <FileButton label="Video file" accept="video/*" onFile={startVideoFile} active={source === "file"} />
          </div>
        </section>

        <section>
          <h2>5. Calibrate</h2>
          <div className="row">
            <label className="field">
              <span>Pixels</span>
              <input value={calPixels ? calPixels.toFixed(1) : ""} readOnly placeholder="draw a line" />
            </label>
            <label className="field">
              <span>Amount</span>
              <input type="number" min={0} step="any" value={calAmount || ""} onChange={(e) => setCalAmount(Number(e.target.value))} />
            </label>
            <label className="field">
              <span>Unit</span>
              <select value={calUnit} onChange={(e) => setCalUnit(e.target.value)}>
                {UNITS.map(([value, name]) => (
                  <option key={value} value={value}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="row">
            <button className={drawingLine ? "active" : ""} disabled={!source} onClick={() => setDrawingLine((d) => !d)}>
              {drawingLine ? "Drag on the video…" : calLine ? "Redraw line" : "Calibrate"}
            </button>
            {calLine && <button onClick={() => setCalLine(null)}>Remove line</button>}
          </div>
          {calLine && !drawingLine && <p className="hint">Drag the ends to resize the line, or the line itself to move it.</p>}
          {calibration && (
            <p className="hint">
              1 px = {calibration.unitsPerPixel.toPrecision(3)} {calibration.unit}
            </p>
          )}
        </section>

        {anyPipeline && (
          <section>
            <h2>Results</h2>
            <button disabled={!source} onClick={downloadResults}>
              Download results (.json)
            </button>
            <p className="hint">Everything the pipelines measured so far, with a section per model.</p>
          </section>
        )}
      </aside>

      <main>
        {(error || shapeWarning || decoderWarning || shapeWarningB || decoderWarningB) && (
          <div className="status">
            {error && <div className="err">{error}</div>}
            {decoderWarning && <div className="warn">{decoderWarning}</div>}
            {shapeWarning && <div className="warn">{modelB && "Model A: "}{shapeWarning}</div>}
            {decoderWarningB && <div className="warn">{decoderWarningB}</div>}
            {shapeWarningB && <div className="warn">Model B: {shapeWarningB}</div>}
          </div>
        )}
        <div className="stage" style={{ aspectRatio: aspect }}>
          <video
            ref={videoRef}
            muted
            playsInline
            // With a pipeline the video plays once, so its results cover the whole video
            // (a loop would restart the measurements).
            loop={!pipeline}
            controls={source === "file"}
            onLoadedMetadata={(e) => {
              const { videoWidth: width, videoHeight: height } = e.currentTarget;
              setAspect(width / height || 16 / 9);
              setFrameSize({ width, height });
            }}
          />
          <canvas ref={overlayRef} />
          <CalibrationLine
            width={frameSize.width}
            height={frameSize.height}
            line={calLine}
            drawing={drawingLine}
            onLine={(l) => {
              setCalLine(l);
              setDrawingLine(false);
            }}
          />
          {!source && <div className="placeholder">Load a model, then pick a camera or video file.</div>}
        </div>
        {stats && runtime && (
          <div className="stats">
            <span>{stats.fps} fps</span>
            <span>pre {stats.preMs.toFixed(1)} ms</span>
            <span>
              infer{" "}
              {stats.inferByModel.length > 1
                ? stats.inferByModel.map((ms, i) => `${modelKey(i)} ${ms.toFixed(1)}`).join(" · ")
                : stats.inferMs.toFixed(1)}{" "}
              ms
            </span>
            <span>post {stats.postMs.toFixed(1)} ms</span>
            <span>{stats.detections} results</span>
            <span>{runtime.backend}</span>
            {Object.entries(stats.metrics ?? {}).map(([k, v]) => (
              <span key={k} className="metric">
                {k}: <b>{v}</b>
              </span>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function FileButton(props: {
  label: string;
  accept: string;
  onFile(f: File): void;
  active?: boolean;
  disabled?: boolean;
  small?: boolean;
}) {
  return (
    <label className={`button ${props.active ? "active" : ""} ${props.small ? "small" : ""} ${props.disabled ? "disabled" : ""}`}>
      {props.label}
      <input
        type="file"
        accept={props.accept}
        hidden
        disabled={props.disabled}
        onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) props.onFile(f);
        }}
      />
    </label>
  );
}
