import { DecodeError, makeCalibration, PipelineRunner, type ModelConfig, type PipelineDefinition } from "@detect/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { FrameLoop, type Stats } from "./adapters/frameLoop";
import { createRuntime, errorMessage, webgpuSupported, type BackendPreference, type ModelRuntime } from "./adapters/runtime";
import { checkInputShape, downloadJson, inferConfig, loadConfigFile, loadPipeline } from "./loaders";
import { CalibrationLine, lineLength, type Line } from "./ui/CalibrationLine";
import { ConfigEditor, OptionField, SHOW_OPTIONS } from "./ui/ConfigEditor";

interface Sample {
  name: string;
  file: string;
}

type Source = "camera" | "file" | null;

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
    fetch("/models/index.json")
      .then((r) => (r.ok ? r.json() : []))
      .then(setSamples)
      .catch(() => setSamples([]));
    return () => p.stop();
  }, []);

  // Config edits only swap config; model swaps go through setModel.
  useEffect(() => {
    if (config) loopRef.current?.setConfig(config);
  }, [config]);

  // A new runner starts with fresh pipeline state.
  useEffect(() => {
    runnerRef.current = pipeline ? new PipelineRunner(pipeline, pipelineOptions, calibration) : null;
    loopRef.current?.setPipeline(runnerRef.current);
  }, [pipeline, pipelineOptions, calibration]);

  /** Saves what the pipeline measured, plus what it was measured on, as a .json file. */
  function downloadResults() {
    const runner = runnerRef.current;
    if (!runner) return;
    const video = videoRef.current!;
    const results = {
      exportedAt: new Date().toISOString(),
      model: modelName,
      backend: runtime?.backend,
      pipeline: runner.pipeline.name,
      source: source === "camera" ? "camera" : sourceName,
      videoDuration: source === "file" && Number.isFinite(video.duration) ? video.duration : undefined,
      frameSize,
      calibration,
      ...runner.summary(),
    };
    const base = `${runner.pipeline.name}-${(sourceName || "camera").replace(/\.[^.]+$/, "")}`;
    downloadJson(`${base}-results.json`, results);
  }

  function editConfig(next: ModelConfig) {
    setError(""); // a previous inference error may not apply to the new settings
    if (next.decoder !== config?.decoder) setDecoderGuessed(false); // the user picked one
    setConfig(next);
  }

  const shapeWarning = runtime && config ? checkInputShape(config, runtime.input) : null;
  const decoderWarning =
    // A decode error already gives the same advice, with the actual reason.
    runtime && config && decoderGuessed && !pipeline?.decode && !(error && error === decodeErrorText)
      ? `We don't know how to read this model's output, so we're trying "${config.decoder}". ` +
        "If the results look wrong, pick another decoder in step 2, or load a Config .json or a Pipeline .js with its own decode."
      : null;

  async function loadModel(bytes: Uint8Array, name: string) {
    setLoading(true);
    setError("");
    try {
      if (runtime) {
        await loopRef.current!.detach(); // let the in-flight frame finish before freeing its GPU buffers
        await runtime.release();
      }
      const rt = await createRuntime(bytes, backendPref);
      const inferred = inferConfig(bytes, rt.input);
      const cfg = explicitConfig ?? inferred.config;
      setRuntime(rt);
      setModelName(name);
      setConfig(cfg);
      setConfigSource(explicitConfig ? "config file" : inferred.source);
      setDecoderGuessed(!explicitConfig && inferred.guessed);
      loopRef.current!.setModel(rt, cfg);
    } catch (e) {
      setRuntime(null);
      setModelName("");
      loopRef.current!.setModel(null, null);
      setError(errorMessage(e));
    } finally {
      setLoading(false);
    }
  }

  async function onModelFile(file: File) {
    await loadModel(new Uint8Array(await file.arrayBuffer()), file.name);
  }

  async function onSample(sample: Sample) {
    setLoading(true);
    try {
      const res = await fetch(`/models/${sample.file}`);
      if (!res.ok) throw new Error(`Could not fetch ${sample.file} (${res.status})`);
      await loadModel(new Uint8Array(await res.arrayBuffer()), sample.name);
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
      setError(`Could not play video: ${errorMessage(e)}`);
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
            <select value={backendPref} onChange={(e) => setBackendPref(e.target.value as BackendPreference)}>
              <option value="auto">Auto (WebGPU → WASM)</option>
              <option value="webgpu" disabled={!webgpuSupported()}>
                WebGPU{webgpuSupported() ? "" : " (unavailable)"}
              </option>
              <option value="wasm">WASM (CPU)</option>
            </select>
          </label>
          <FileButton label={loading ? "Loading…" : "Load .onnx"} accept=".onnx" onFile={onModelFile} disabled={loading} />
          {samples.length > 0 && (
            <div className="samples">
              {samples.map((s) => (
                <button key={s.file} disabled={loading} onClick={() => onSample(s)}>
                  {s.name}
                </button>
              ))}
            </div>
          )}
          {runtime && (
            <p className="hint">
              <b>{modelName}</b> on <b className={runtime.backend}>{runtime.backend}</b>
            </p>
          )}
        </section>

        <section>
          <h2>2. Config</h2>
          <FileButton label="Config .json" accept=".json,application/json" onFile={onConfigFile} small />
          {runtime && <p className="hint">Config from {configSource}</p>}
          {explicitConfig && (
            <button className="link" onClick={() => setExplicitConfig(null)}>
              Forget config file (auto-detect on next model load)
            </button>
          )}
          {config && (
            <ConfigEditor config={config} decoderFromPipeline={pipeline?.decode ? pipeline.name : undefined} onChange={editConfig} />
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
            <p className="hint">Optional. Your own decoder and/or logic across frames (tracking, counting…).</p>
          )}
          {pipeline && (
            <button className="small" disabled={!source} onClick={downloadResults}>
              Download results (.json)
            </button>
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
      </aside>

      <main>
        {(error || shapeWarning || decoderWarning) && (
          <div className="status">
            {error && <div className="err">{error}</div>}
            {decoderWarning && <div className="warn">{decoderWarning}</div>}
            {shapeWarning && <div className="warn">{shapeWarning}</div>}
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
            <span>infer {stats.inferMs.toFixed(1)} ms</span>
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
