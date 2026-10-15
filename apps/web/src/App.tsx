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
const VORTEX = "vortex_detector_fp32_with_pipeline.onnx";
const PARTICLES = "particle_detector.onnx";
const PRELOAD = [VORTEX, PARTICLES];

/** Short clips in public/examples, for trying the models without a video of your own. */
const EXAMPLE_VIDEOS = [
  { label: "Vortex", file: "vortex.mp4", models: [VORTEX], hint: "A vortex wall: the vortex model" },
  { label: "Particles", file: "particles.mp4", models: [PARTICLES], hint: "Particles in a pump: the particle model" },
  { label: "Both", file: "both.mp4", models: [VORTEX, PARTICLES], hint: "A turbine tube with a vortex rope and flakes: both models" },
];

/** One entry of the model list: a sample to download or a file the user picked. Loaded the first time it is ticked. */
interface ModelEntry {
  key: string;
  name: string;
  sample?: Sample;
  bytes?: Uint8Array;
  runtime?: ModelRuntime;
  config?: ModelConfig;
  /** A config file the user loaded for this model; it wins over auto-detection, also on reload. */
  explicitConfig?: ModelConfig;
  configSource: string;
  /** Nothing told us how to decode this model; the decoder is a default the user should confirm. */
  decoderGuessed: boolean;
  /** Whether it runs; unticking keeps it loaded so it can be turned back on instantly. */
  on: boolean;
  loading: boolean;
  /** Its embedded pipeline may run without asking: the app's own samples, or after the user said yes. */
  trusted: boolean;
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
  /** Every model in the list; its index is its slot in the frame loop. */
  const [models, setModels] = useState<ModelEntry[]>([]);
  const modelsRef = useRef<ModelEntry[]>([]);
  const loading = models.some((m) => m.loading);
  /** Text of the last decoder failure; while it's the error shown, it already explains what to do. */
  const [decodeErrorText, setDecodeErrorText] = useState("");
  const preloadStarted = useRef(false);
  /** Pipelines embedded in each slot's model; each runs on that model's output only. */
  const [modelPipelines, setModelPipelines] = useState<(PipelineDefinition | null)[]>([]);
  const modelRunners = useRef<(PipelineRunner | null)[]>([]);
  const [pipeline, setPipeline] = useState<PipelineDefinition | null>(null);
  const [pipelineOptions, setPipelineOptions] = useState<Record<string, unknown>>({});
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });
  const [calLine, setCalLine] = useState<Line | null>(null);
  const [calAmount, setCalAmount] = useState(0);
  const [calUnit, setCalUnit] = useState("cm");
  const [drawingLine, setDrawingLine] = useState(false);
  const calPixels = calLine ? lineLength(calLine) : 0;
  const calibration = useMemo(() => makeCalibration(calPixels, calAmount, calUnit), [calPixels, calAmount, calUnit]);
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
      .catch(() => [])
      .then((list: Sample[]) => {
        // The ref keeps React's dev double-mount from loading twice.
        if (preloadStarted.current) return;
        preloadStarted.current = true;
        updateModels(() => list.map((sample) => ({ key: sample.url, name: sample.name, sample, configSource: "", decoderGuessed: false, on: false, loading: false, trusted: true })));
        // Start with the vortex and particle models loaded and on; the rest download when ticked.
        void (async () => {
          for (const file of PRELOAD) {
            const i = list.findIndex((x) => x.url.split("/").pop() === file);
            if (i >= 0) await toggleModel(i, true);
          }
        })();
      });
    return () => p.stop();
  }, []);

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
      models: models.flatMap((m, i) => (m.on && m.runtime ? [model(i, m.name, m.runtime.backend)] : [])),
      // A pipeline loaded in step 3 runs on all models together; its results go here.
      ...(global && { pipeline: global.pipeline.name, ...global.summary() }),
    };
    const names = [global?.pipeline.name, ...modelRunners.current.filter((_, i) => models[i]?.on).map((r) => r?.pipeline.name)].filter(Boolean);
    const base = `${names.join("+") || "results"}-${(sourceName || "camera").replace(/\.[^.]+$/, "")}`;
    downloadJson(`${base}-results.json`, results);
  }

  /** Replaces the model list (keeping the ref, which async code reads, in step with the state). */
  function updateModels(fn: (m: ModelEntry[]) => ModelEntry[]) {
    modelsRef.current = fn(modelsRef.current);
    setModels(modelsRef.current);
  }

  const patchModel = (i: number, patch: Partial<ModelEntry>) => updateModels((ms) => ms.map((m, k) => (k === i ? { ...m, ...patch } : m)));

  /**
   * The pipeline embedded in a model file, if any. A model from outside the app (picked from disk) runs
   * code when its pipeline loads, so the user is asked first; the app's own sample models are trusted.
   */
  async function embeddedPipeline(i: number, bytes: Uint8Array): Promise<PipelineDefinition | null> {
    const entry = modelsRef.current[i];
    let source: string | undefined;
    try {
      source = readOnnxMetadata(bytes)[PIPELINE_METADATA_KEY];
    } catch {
      return null; // unreadable metadata: no pipeline
    }
    if (!source) return null;
    if (!entry.trusted) {
      const ok = window.confirm(`"${entry.name}" includes a pipeline (code that runs in this page to measure its results). Run it?`);
      if (!ok) return null;
      patchModel(i, { trusted: true }); // don't ask again when the same model reloads (e.g. backend change)
    }
    try {
      return await loadPipelineSource(source);
    } catch (e) {
      setError(`${entry.name}: its embedded pipeline failed to load: ${errorMessage(e)}`);
      return null;
    }
  }

  function setModelPipeline(slot: number, def: PipelineDefinition | null) {
    setModelPipelines((p) => {
      const next = [...p];
      while (next.length <= slot) next.push(null);
      next[slot] = def;
      return next;
    });
  }

  function editConfig(i: number, next: ModelConfig) {
    setError(""); // a previous inference error may not apply to the new settings
    const m = modelsRef.current[i];
    patchModel(i, { config: next, decoderGuessed: m.decoderGuessed && next.decoder === m.config?.decoder }); // picking another decoder answers the guess
    loopRef.current!.setConfig(i, next);
  }

  /** Downloads a model from the sample list. */
  async function download(sample: Sample): Promise<Uint8Array> {
    const url = new URL(sample.url, new URL(MODELS_INDEX, location.href)).href;
    const res = await fetch(url);
    // The dev server answers a missing file with the app's HTML page, not a 404.
    if (!res.ok || res.headers.get("content-type")?.includes("text/html")) {
      throw new Error(`Could not download ${sample.name} from ${url} (HTTP ${res.status})`);
    }
    return new Uint8Array(await res.arrayBuffer());
  }

  /** Loads (downloading first if needed) the model in slot `i`. `on` says whether it starts switched on. */
  async function loadModel(i: number, on: boolean, backend = backendPref) {
    const entry = modelsRef.current[i];
    patchModel(i, { loading: true });
    setError("");
    try {
      const bytes = entry.bytes ?? (await download(entry.sample!));
      if (entry.runtime) {
        await loopRef.current!.detach(i); // let the in-flight frame finish before freeing its GPU buffers
        await entry.runtime.release();
      }
      const rt = await createRuntime(bytes, backend);
      const inferred = inferConfig(bytes, rt.input);
      const cfg = entry.explicitConfig ?? withOptions(inferred.config, entry.sample?.options);
      patchModel(i, {
        bytes,
        runtime: rt,
        config: cfg,
        configSource: entry.explicitConfig ? "config file" : inferred.source,
        decoderGuessed: !entry.explicitConfig && inferred.guessed,
        on,
      });
      loopRef.current!.setModel(i, on ? rt : null, on ? cfg : null);
      setModelPipeline(i, await embeddedPipeline(i, bytes));
    } catch (e) {
      patchModel(i, { runtime: undefined, on: false });
      loopRef.current!.setModel(i, null, null);
      setError(`${entry.name}: ${errorMessage(e)}`);
    } finally {
      patchModel(i, { loading: false });
    }
  }

  /** Switches a model on or off; one that isn't loaded yet is loaded (and downloaded) first. Unticking keeps it loaded. */
  async function toggleModel(i: number, on: boolean) {
    setError("");
    const m = modelsRef.current[i];
    if (on && !m.runtime) return loadModel(i, true);
    patchModel(i, { on });
    loopRef.current!.setModel(i, on && m.runtime ? m.runtime : null, on && m.config ? m.config : null);
  }

  /** Switches backend and reloads the loaded models on it, keeping each one on or off as it was. */
  async function changeBackend(backend: BackendPreference) {
    setBackendPref(backend);
    for (const [i, m] of modelsRef.current.entries()) if (m.runtime) await loadModel(i, m.on, backend);
  }

  /** Adds a model file picked from disk to the list and turns it on. */
  async function addModelFile(file: File) {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const i = modelsRef.current.length;
    updateModels((ms) => [...ms, { key: `file:${i}:${file.name}`, name: file.name, bytes, configSource: "", decoderGuessed: false, on: false, loading: false, trusted: false }]);
    await loadModel(i, true);
  }

  async function onConfigFile(i: number, file: File) {
    try {
      const cfg = await loadConfigFile(file);
      patchModel(i, { explicitConfig: cfg, config: cfg, configSource: `config file (${file.name})`, decoderGuessed: false });
      loopRef.current!.setConfig(i, cfg);
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

  const startVideoFile = (file: File) => startVideo(URL.createObjectURL(file), file.name);

  async function startVideo(src: string, name: string) {
    stopSource();
    setError("");
    const v = videoRef.current!;
    v.src = src;
    setSource("file");
    setSourceName(name);
    try {
      await v.play();
    } catch (e) {
      // The browser can't decode this codec (e.g. MPEG-4 Part 2, H.265, or an .avi container).
      const unsupported = e instanceof DOMException && e.name === "NotSupportedError";
      setError(
        unsupported
          ? `Could not play ${name}: the browser doesn't support this video's format. Convert it to H.264 MP4, ` +
              `e.g. ffmpeg -i "${name}" -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4`
          : `Could not play video: ${errorMessage(e)}`,
      );
    }
  }

  const active = models.filter((m) => m.on && m.runtime);
  // A decode error already gives the same advice, with the actual reason.
  const showGuessWarnings = !(error && error === decodeErrorText);
  const warnings = active.flatMap((m) => {
    const prefix = active.length > 1 ? `${m.name}: ` : "";
    const out: string[] = [];
    if (m.decoderGuessed && !pipeline?.decode && showGuessWarnings) {
      out.push(
        `${prefix}We don't know how to read this model's output, so we're trying "${m.config!.decoder}". ` +
          "If the results look wrong, pick another decoder in step 2, or load a Config .json or a Pipeline .js with its own decode.",
      );
    }
    const shape = checkInputShape(m.config!, m.runtime!.input);
    if (shape) out.push(prefix + shape);
    return out;
  });

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
          <p className="hint">Tick the models to run; they all run on the same frames. A model downloads the first time it is ticked.</p>
          {models.map((m, i) => (
            <label className="check" key={m.key}>
              <input type="checkbox" checked={m.on} disabled={loading} onChange={(e) => void toggleModel(i, e.target.checked)} />
              <span>
                <b>{m.name}</b>
                {m.loading ? " · loading…" : m.runtime && <> on <b className={m.runtime.backend}>{m.runtime.backend}</b></>}
                {modelPipelines[i] && <> · pipeline <b>{modelPipelines[i]!.name}</b></>}
              </span>
            </label>
          ))}
          <FileButton label="Load .onnx" accept=".onnx" onFile={addModelFile} disabled={loading} small />
        </section>

        <section>
          <h2>2. Config</h2>
          {models.map(
            (m, i) =>
              m.runtime &&
              m.config && (
                <div key={m.key}>
                  {models.filter((x) => x.runtime).length > 1 && <h3>{m.name}</h3>}
                  <p className="hint">Config from {m.configSource}</p>
                  <FileButton label="Config .json" accept=".json,application/json" onFile={(f) => void onConfigFile(i, f)} small />
                  {m.explicitConfig && (
                    <button className="link" onClick={() => patchModel(i, { explicitConfig: undefined })}>
                      Forget config file (auto-detect on next model load)
                    </button>
                  )}
                  <ConfigEditor config={m.config} decoderFromPipeline={pipeline?.decode ? pipeline.name : undefined} onChange={(c) => editConfig(i, c)} />
                </div>
              ),
          )}
          {!models.some((m) => m.runtime) && <p className="hint">Tick a model to configure it.</p>}
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
          <p className="hint">Or try an example:</p>
          <div className="samples">
            {EXAMPLE_VIDEOS.map((x) => (
              <button key={x.file} title={x.hint} onClick={() => {
                  // Run only the model(s) the clip is made for.
                  models.forEach((m, i) => {
                    const want = !!m.sample && x.models.includes(m.sample.url.split("/").pop()!);
                    if (m.sample && m.on !== want) void toggleModel(i, want);
                  });
                  void startVideo(`${import.meta.env.BASE_URL}examples/${x.file}`, x.file);
                }}>
                {x.label}
              </button>
            ))}
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
        {(error || warnings.length > 0) && (
          <div className="status">
            {error && <div className="err">{error}</div>}
            {warnings.map((w) => (
              <div key={w} className="warn">
                {w}
              </div>
            ))}
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
        {stats && active.length > 0 && (
          <div className="stats">
            <span>{stats.fps} fps</span>
            <span>pre {stats.preMs.toFixed(1)} ms</span>
            <span>
              infer{" "}
              {stats.inferByModel.length > 1
                ? stats.inferByModel.map((ms, i) => `${active[i]?.name ?? modelKey(i)} ${ms.toFixed(1)}`).join(" · ")
                : stats.inferMs.toFixed(1)}{" "}
              ms
            </span>
            <span>post {stats.postMs.toFixed(1)} ms</span>
            <span>{stats.detections} results</span>
            <span>{active[0].runtime!.backend}</span>
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
