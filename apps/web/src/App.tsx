import { DecodeError, makeCalibration, modelKey, PIPELINE_METADATA_KEY, PipelineRunner, readOnnxMetadata, type ModelConfig, type PipelineDefinition } from "@detect/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { ANALYZE_FPS, FrameLoop, type Stats } from "./adapters/frameLoop";
import { errorMessage, webgpuSupported, type BackendPreference, type ModelRuntime } from "./adapters/runtime";
import { clearGpuVerdicts, createCheckedRuntime, type GpuVerification } from "./adapters/gpuCheck";
import { checkInputShape, downloadJson, inferConfig, loadConfigFile, loadPipeline, loadPipelineSource } from "./loaders";
import { readInitialTarget, rememberTarget, type Target } from "./subjects";
import { CalibrationLine, lineLength, type Line } from "./ui/CalibrationLine";
import { ConfigEditor, OptionField, SHOW_OPTIONS } from "./ui/ConfigEditor";
import { KeyMetrics } from "./ui/KeyMetrics";
import { FramingGuide, requestMotionPermission } from "./ui/FramingGuide";
import { Sparkline } from "./ui/Sparkline";
import { SummaryCard, type ModelSummary } from "./ui/SummaryCard";
import { TrendBuffer } from "./ui/trend";

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

/** Main-path targets; the full downloadable list remains available in Advanced. */
const VORTEX = "vortex_detector_fp32_with_pipeline.onnx";
const PARTICLES = "particle_detector.onnx";
const PARTICLES_TRACKING = "particle_detector_with_tracking.onnx";
const TARGETS = [
  { id: "lamp", name: "Vortex lamp", file: "vortex_lamp_auto_fp16.onnx" },
  { id: "vortex", name: "Rig vortex", file: VORTEX },
  { id: "particles", name: "Particles", file: PARTICLES },
] as const;

type ExampleModel = "vortex" | "particles" | "both";

interface ExampleVideo {
  label: string;
  file: string;
  src: string;
  models: ExampleModel;
}

/** One example for each unique clip. Cropped variants stand in for their full-frame source. */
const EXAMPLE_VIDEOS: ExampleVideo[] = [
  { label: "Video 1", file: "type-1_inf.mp4", src: new URL("../../../videos/type-1_inf.mp4", import.meta.url).href, models: "vortex" },
  { label: "Video 2", file: "type-1_sup.mp4", src: new URL("../../../videos/type-1_sup.mp4", import.meta.url).href, models: "vortex" },
  { label: "Video 3", file: "type-2_inf.mp4", src: new URL("../../../videos/type-2_inf.mp4", import.meta.url).href, models: "vortex" },
  { label: "Video 4", file: "type-2_sup.mp4", src: new URL("../../../videos/type-2_sup.mp4", import.meta.url).href, models: "vortex" },
  { label: "Video 5", file: "type-3_inf.mp4", src: new URL("../../../videos/type-3_inf.mp4", import.meta.url).href, models: "vortex" },
  { label: "Video 6", file: "type-3_sup.mp4", src: new URL("../../../videos/type-3_sup.mp4", import.meta.url).href, models: "vortex" },
  { label: "Video 7", file: "bomba-2_tube_3.mp4", src: new URL("../../../videos/bomba-2_tube_3.mp4", import.meta.url).href, models: "particles" },
  { label: "Video 8", file: "bomba_tube_3.mp4", src: new URL("../../../videos/bomba_tube_3.mp4", import.meta.url).href, models: "particles" },
  { label: "Video 9", file: "turbina-sin-antorcha_tube_3.mp4", src: new URL("../../../videos/turbina-sin-antorcha_tube_3.mp4", import.meta.url).href, models: "particles" },
  { label: "Video 10", file: "turbina-antorcha_tube_3.mp4", src: new URL("../../../videos/turbina-antorcha_tube_3.mp4", import.meta.url).href, models: "both" },
  { label: "Video 11", file: "turbina-antorcha-ultima_tube_3.mp4", src: new URL("../../../videos/turbina-antorcha-ultima_tube_3.mp4", import.meta.url).href, models: "both" },
];

/** One entry of the model list: a sample to download or a file the user picked. Loaded the first time it is ticked. */
interface ModelEntry {
  key: string;
  name: string;
  sample?: Sample;
  bytes?: Uint8Array;
  runtime?: ModelRuntime;
  gpuVerification?: GpuVerification;
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
  const [motionAllowed, setMotionAllowed] = useState(false);
  const [target, setTarget] = useState<Target | null>(readInitialTarget);
  const [sourceName, setSourceName] = useState("");
  const [aspect, setAspect] = useState(16 / 9);
  const [backendPref, setBackendPref] = useState<BackendPreference>("auto");
  /** Every model in the list; its index is its slot in the frame loop. */
  const [models, setModels] = useState<ModelEntry[]>([]);
  const modelsRef = useRef<ModelEntry[]>([]);
  const particleModelPreference = useRef(PARTICLES);
  const loading = models.some((m) => m.loading);
  /** Text of the last decoder failure; while it's the error shown, it already explains what to do. */
  const [decodeErrorText, setDecodeErrorText] = useState("");
  const modelListLoaded = useRef(false);
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
  const trend = useRef(new TrendBuffer());
  const [summaries, setSummaries] = useState<ModelSummary[]>([]);
  const [finishing, setFinishing] = useState(false);
  const [cameraFinished, setCameraFinished] = useState(false);
  /** A chosen video file waiting for the user to pick Analyze first or Live. */
  const [pendingFile, setPendingFile] = useState<File | null>(null);
  /** Progress of the analysis in advance (frames), while it runs. */
  const [analysis, setAnalysis] = useState<{ done: number; total: number } | null>(null);
  /** Frames analyzed in advance for the playing video, or 0 for live analysis. */
  const [analyzedFrames, setAnalyzedFrames] = useState(0);
  const analysisAbort = useRef<AbortController | null>(null);

  function resetReadings() {
    trend.current.clear();
    setSummaries([]);
  }

  function readSummaries(): ModelSummary[] {
    return modelsRef.current.flatMap((m, i) => {
      const runner = modelRunners.current[i];
      return m.on && m.runtime && runner ? [{ name: m.name, summary: runner.summary() }] : [];
    });
  }

  useEffect(() => {
    const p = new FrameLoop(videoRef.current!, overlayRef.current!, {
      onStats: (s) => {
        trend.current.add(videoRef.current!.currentTime, s.metrics, s.detections);
        setStats(s);
        // Include a final inference that completes after the media's ended event.
        if (videoRef.current!.ended) setSummaries(readSummaries());
      },
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
        if (modelListLoaded.current) return;
        modelListLoaded.current = true;
        updateModels(() => list.map((sample) => ({ key: sample.url, name: sample.name, sample, configSource: "", decoderGuessed: false, on: false, loading: false, trusted: true })));
        if (target && target !== "particles") chooseTarget(target);
      });
    return () => p.stop();
  }, []);

  // A new runner starts with fresh pipeline state.
  useEffect(() => {
    resetReadings();
    runnerRef.current = pipeline ? new PipelineRunner(pipeline, pipelineOptions, calibration) : null;
    loopRef.current?.setPipeline(runnerRef.current);
  }, [pipeline, pipelineOptions, calibration]);

  // Each model's own pipeline gets a fresh runner whenever it or the calibration changes.
  useEffect(() => {
    resetReadings();
    modelPipelines.forEach((def, slot) => {
      const runner = def ? new PipelineRunner(def, {}, calibration) : null;
      modelRunners.current[slot] = runner;
      loopRef.current?.setModelPipeline(slot, runner);
    });
  }, [modelPipelines, calibration]);

  const anyPipeline = !!pipeline || modelPipelines.some((def, i) => !!def && models[i]?.on);

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
      // A pipeline loaded in Advanced runs on all models together; its results go here.
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
    resetReadings();
    const entry = modelsRef.current[i];
    patchModel(i, { loading: true });
    setError("");
    try {
      const bytes = entry.bytes ?? (await download(entry.sample!));
      await loopRef.current!.changeModels(async () => {
        if (entry.runtime) {
          await loopRef.current!.detach(i); // let the in-flight frame finish before freeing its GPU buffers
          await entry.runtime.release();
        }
        const identity = entry.sample ? new URL(entry.sample.url, new URL(MODELS_INDEX, location.href)).href : `file:${entry.name}`;
        const checked = await createCheckedRuntime(bytes, backend, identity);
        const rt = checked.runtime;
        const inferred = inferConfig(bytes, rt.input);
        const cfg = entry.explicitConfig ?? withOptions(inferred.config, entry.sample?.options);
        patchModel(i, {
          bytes,
          runtime: rt,
          gpuVerification: checked.verification,
          config: cfg,
          configSource: entry.explicitConfig ? "config file" : inferred.source,
          decoderGuessed: !entry.explicitConfig && inferred.guessed,
          on,
        });
        loopRef.current!.setModel(i, on ? rt : null, on ? cfg : null);
        setModelPipeline(i, await embeddedPipeline(i, bytes));
      });
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
    resetReadings();
    setError("");
    const m = modelsRef.current[i];
    const modelFile = m.sample?.url.split("/").pop();
    if (on && (modelFile === PARTICLES || modelFile === PARTICLES_TRACKING)) {
      particleModelPreference.current = modelFile;
    }
    if (on && !m.runtime) return loadModel(i, true);
    patchModel(i, { on });
    loopRef.current!.setModel(i, on && m.runtime ? m.runtime : null, on && m.config ? m.config : null);
  }

  /** A main-path choice runs only its model; Advanced can still combine any models. */
  function chooseTarget(next: Target) {
    resetReadings();
    const file = TARGETS.find((t) => t.id === next)!.file;
    setTarget(next);
    rememberTarget(next);
    setStats(null);
    loopRef.current?.resetPipeline();
    modelsRef.current.forEach((m, i) => {
      const want = m.sample?.url.split("/").pop() === file;
      if (m.on !== want) void toggleModel(i, want);
    });
    loopRef.current?.refresh();
  }

  /** A preselected large model waits for a source tap instead of using mobile data on open. */
  function startSelectedTarget() {
    if (!target) return;
    rememberTarget(target);
    const file = TARGETS.find((t) => t.id === target)!.file;
    const i = modelsRef.current.findIndex((m) => m.sample?.url.split("/").pop() === file);
    if (i >= 0 && !modelsRef.current[i].on && !modelsRef.current[i].loading) void toggleModel(i, true);
  }

  function chooseExample(example: ExampleVideo) {
    setStats(null);
    const needsVortex = example.models === "vortex" || example.models === "both";
    const needsParticles = example.models === "particles" || example.models === "both";
    modelsRef.current.forEach((m, i) => {
      const file = m.sample?.url.split("/").pop();
      const want = file === VORTEX ? needsVortex
        : file === PARTICLES || file === PARTICLES_TRACKING ? needsParticles && file === particleModelPreference.current : false;
      if (m.sample && m.on !== want) void toggleModel(i, want);
    });
    void startVideo(example.src, example.label);
  }

  /** Switches backend and reloads the loaded models on it, keeping each one on or off as it was. */
  async function changeBackend(backend: BackendPreference) {
    setBackendPref(backend);
    for (const [i, m] of modelsRef.current.entries()) if (m.runtime) await loadModel(i, m.on, backend);
  }

  async function checkGpuAgain() {
    clearGpuVerdicts();
    await changeBackend(backendPref);
  }

  /** Adds a model file picked from disk to the list and turns it on. */
  async function addModelFile(file: File) {
    setTarget(null);
    setStats(null);
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
    analysisAbort.current?.abort();
    analysisAbort.current = null;
    setAnalysis(null);
    setAnalyzedFrames(0);
    loopRef.current?.clearReplay();
    setCameraFinished(false);
    resetReadings();
    loopRef.current?.resetPipeline();
    setStats(null);
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

  async function finishCamera() {
    const stream = streamRef.current;
    if (!stream || finishing) return;
    setFinishing(true);
    await loopRef.current!.finish();
    // A new source may have been chosen while the last inference was draining.
    if (streamRef.current === stream) {
      setSummaries(readSummaries());
      videoRef.current!.pause();
      stream.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setCameraFinished(true);
    }
    setFinishing(false);
  }

  async function startCamera() {
    const motionPermission = target === "lamp" ? requestMotionPermission() : null;
    if (motionPermission) void motionPermission.then(setMotionAllowed);
    startSelectedTarget();
    stopSource();
    setError("");
    try {
      // Keep a portrait camera's full frame instead of cropping it to the preferred landscape size.
      const video: MediaTrackConstraints & { resizeMode: string } = {
        facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 }, resizeMode: "none",
      };
      const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      streamRef.current = stream;
      videoRef.current!.srcObject = stream;
      await videoRef.current!.play();
      loopRef.current!.start();
      setSource("camera");
      setSourceName("");
    } catch (e) {
      setSource(null);
      const name = e instanceof DOMException ? e.name : "";
      setError(name === "NotAllowedError" ? "Camera permission denied" : `Camera failed: ${errorMessage(e)}`);
    }
  }

  /** A chosen file waits for the user to pick how to analyze it. */
  const startVideoFile = (file: File) => {
    startSelectedTarget();
    stopSource();
    setSource(null);
    setError("");
    setPendingFile(file);
  };

  function playLive(file: File) {
    setPendingFile(null);
    void startVideo(URL.createObjectURL(file), file.name);
  }

  /** Analyzes every frame of the file first, then plays it with the stored results (FrameLoop.analyze). */
  async function analyzeFirst(file: File) {
    setPendingFile(null);
    stopSource();
    setError("");
    const v = videoRef.current!;
    v.src = URL.createObjectURL(file);
    setSource("file");
    setSourceName(file.name);
    const abort = new AbortController();
    analysisAbort.current = abort;
    setAnalysis({ done: 0, total: 0 });
    try {
      await loadedMetadata(v);
      if (!Number.isFinite(v.duration)) throw new Error("the video's length is unknown, so it can't be analyzed first; choose Live");
      const ok = await loopRef.current!.analyze((done, total) => {
        if (analysisAbort.current === abort) setAnalysis({ done, total });
      }, abort.signal);
      if (analysisAbort.current !== abort) return; // another source replaced this one
      if (!ok) {
        if (!abort.signal.aborted) setError("A model or setting changed during the analysis. Choose the video again.");
        stopSource();
        setSource(null);
        setPendingFile(file);
        return;
      }
      analysisAbort.current = null;
      setAnalysis(null);
      setAnalyzedFrames(Math.floor(v.duration * ANALYZE_FPS) + 1);
      setSummaries(readSummaries());
      v.currentTime = 0;
      loopRef.current!.start();
      await v.play();
    } catch (e) {
      if (analysisAbort.current !== abort) return;
      setAnalysis(null);
      setError(playError(file.name, e));
    }
  }

  async function startVideo(src: string, name: string) {
    stopSource();
    setError("");
    const v = videoRef.current!;
    v.src = src;
    setSource("file");
    setSourceName(name);
    try {
      await v.play();
      loopRef.current!.start();
    } catch (e) {
      setError(playError(name, e));
    }
  }

  const active = models.filter((m) => m.on && m.runtime);
  const gpuNotices = active.flatMap((m) => m.gpuVerification?.notice ? [`${active.length > 1 ? `${m.name}: ` : ""}${m.gpuVerification.notice}`] : []);
  // A decode error already gives the same advice, with the actual reason.
  const showGuessWarnings = !(error && error === decodeErrorText);
  const warnings = active.flatMap((m) => {
    const prefix = active.length > 1 ? `${m.name}: ` : "";
    const out: string[] = [];
    if (m.decoderGuessed && !pipeline?.decode && showGuessWarnings) {
      out.push(
        `${prefix}We don't know how to read this model's output, so we're trying "${m.config!.decoder}". ` +
          "If the results look wrong, open Advanced: in Model config, pick another decoder or load a Config .json; in Pipeline, load a Pipeline .js with its own decode.",
      );
    }
    const shape = checkInputShape(m.config!, m.runtime!.input);
    if (shape) out.push(prefix + shape);
    return out;
  });

  const hasSelection = !!target || active.length > 0 || loading;
  const examples = EXAMPLE_VIDEOS.filter((x) => target === "vortex" ? x.models !== "particles" : target === "particles" && x.models === "particles");

  function readFrameSize() {
    const { videoWidth: width, videoHeight: height } = videoRef.current!;
    if (!width || !height) return;
    setAspect(width / height);
    setFrameSize({ width, height });
  }

  return (
    <div className="app">
      <header>
        <h1>Model Studio</h1>
        <p className="hint">Choose what you are looking at. Everything runs on your device.</p>
      </header>
      <div className="target-picker" role="group" aria-label="What are you looking at?">
        {TARGETS.map((t) => (
          <button key={t.id} aria-pressed={target === t.id} disabled={!models.length || loading} onClick={() => chooseTarget(t.id)}>
            {t.name}
          </button>
        ))}
      </div>
      {hasSelection && (
        <div className="source-bar">
          <div className="row">
            <button onClick={startCamera} disabled={!models.length} className={source === "camera" ? "active" : ""}>Camera</button>
            <FileButton label="Video file" accept="video/*" onFile={startVideoFile} active={source === "file"} disabled={!models.length} />
          </div>
          {target === "lamp" && !source && <p className="hint">Hold the phone upright, with the lamp filling about 70% of the frame width.</p>}
          {target === "particles" && !loading && !models.some((m) => m.sample?.url.split("/").pop() === PARTICLES && m.on) && <p className="hint">Particles model (39 MB) downloads when you start.</p>}
          {examples.length > 0 && (
            <details className="examples">
              <summary>Try an example</summary>
              <div className="samples">
                {examples.map((x) => <button key={x.file} disabled={loading} title={x.file} aria-label={`${x.label}: ${x.file}`} onClick={() => chooseExample(x)}>{x.label}</button>)}
              </div>
            </details>
          )}
        </div>
      )}
      <main>
        {(error || warnings.length > 0 || gpuNotices.length > 0) && (
          <div className="status" role="status">
            {error && <div className="err">{error}</div>}
            {warnings.map((w) => <div key={w} className="warn">{w}</div>)}
            {gpuNotices.map((notice) => <p key={notice} className="gpu-notice hint warn">{notice}</p>)}
          </div>
        )}
        <div className={`analysis-view${summaries.length ? " finished" : ""}`}>
          <div className="stage" style={{ aspectRatio: aspect, maxWidth: `calc(var(--stage-height) * ${aspect})` }}>
            <video
              ref={videoRef}
              muted
              playsInline
              // Pipelines measure a single pass; looping would restart their measurements.
              loop={!anyPipeline}
              controls={source === "file" && !analysis}
              onLoadedMetadata={readFrameSize}
              onResize={readFrameSize}
              onEnded={() => { if (source === "file") setSummaries(readSummaries()); }}
              // Results analyzed in advance cover the whole video, so its summary stays.
              onPlay={() => { if (!analyzedFrames) setSummaries([]); }}
              onSeeking={() => { if (!analyzedFrames) setSummaries([]); }}
            />
            <canvas ref={overlayRef} />
            <CalibrationLine
              width={frameSize.width}
              height={frameSize.height}
              line={calLine}
              drawing={drawingLine}
              onLine={(l) => { setCalLine(l); setDrawingLine(false); }}
            />
            {!source && !pendingFile && <div className="placeholder">{hasSelection ? "Choose Camera or Video file to see the analysis." : "Start by choosing a subject above."}</div>}
            {!source && pendingFile && (
              <div className="mode-choice">
                <p>How should <b>{pendingFile.name}</b> be analyzed?</p>
                <div className="row">
                  <button onClick={() => void analyzeFirst(pendingFile)} disabled={loading || !active.length}>Analyze first</button>
                  <button onClick={() => playLive(pendingFile)}>Live</button>
                </div>
                <p className="hint"><b>Analyze first:</b> every frame is analyzed before the video plays, so playback is smooth on any device; a slow device takes longer to prepare.</p>
                <p className="hint"><b>Live:</b> plays right away and analyzes as it goes; a slow device skips frames and the drawing can lag.</p>
                {(loading || !active.length) && <p className="hint">Analyze first is available once the model has loaded.</p>}
              </div>
            )}
            {analysis && (
              <div className="analysis-progress" role="status">
                <p>Analyzing {analysis.total ? `${analysis.done} / ${analysis.total} frames (${Math.floor((100 * analysis.done) / analysis.total)}%)` : "..."}</p>
                <progress value={analysis.done} max={analysis.total || 1} />
                <button onClick={() => analysisAbort.current?.abort()}>Cancel</button>
              </div>
            )}
            <div className="stage-notices">
              {cameraFinished && <div className="session-finished" role="status">Session finished</div>}
              {loading && <div className="loading-status" role="status">Loading {models.filter((m) => m.loading).map((m) => m.name).join(", ")}...</div>}
              {target === "lamp" && source === "camera" && !cameraFinished && <FramingGuide
                locked={stats?.metrics?.["vortex orientation"] === "vertical" || stats?.metrics?.["vortex orientation"] === "horizontal"}
                motionAllowed={motionAllowed}
              />}
            </div>
          </div>
          {source && <div className="analysis-readings">
            {stats && active.length > 0
              ? <KeyMetrics metrics={stats.metrics} detections={stats.detections} showParticles={active.some((m) => m.config?.decoder === "yolo-detect")} trend={<Sparkline buffer={trend.current} />} />
              : <p className="hint analysis-wait">{loading ? "Your analysis will appear when the model is ready."
                : analysis ? "Analyzing the video first; it plays when done." : "Waiting for the first analyzed frame..."}</p>}
            {analyzedFrames > 0 && <p className="hint">Analyzed in advance: {analyzedFrames} frames at {ANALYZE_FPS} per second.</p>}
            {summaries.length > 0 && <SummaryCard models={summaries} title={source === "camera" ? "Camera session summary" : "Video summary"} />}
            <div className="reading-actions">
              {source === "camera" && !cameraFinished && !loading && stats && modelPipelines.some((def, i) => def && models[i]?.on && models[i]?.runtime) &&
                <button onClick={finishCamera} disabled={finishing}>{finishing ? "Finishing..." : "Finish"}</button>}
              <button className="download" onClick={downloadResults} disabled={finishing}>Download results (.json)</button>
            </div>
          </div>}
        </div>
        {stats && active.length > 0 && (
          <details className="diagnostics">
            <summary>All measurements and timing</summary>
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
          </details>
        )}
      </main>
      <details className="advanced">
        <summary>Advanced</summary>
        <div className="advanced-grid">
          <section>
            <h2>Models and backend</h2>
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
            <button className="small" disabled={loading} onClick={() => void checkGpuAgain()}>Check GPU again</button>
            <p className="hint">Small models are checked against the CPU once per GPU and model. Large models, including particles, are not checked; choose WASM if results look wrong.</p>
            <p className="hint">Tick the models to run; they all run on the same frames. A model downloads the first time it is ticked.</p>
            {models.map((m, i) => (
              <label className="check" key={m.key}>
                <input type="checkbox" checked={m.on} disabled={loading} onChange={(e) => { setTarget(null); setStats(null); void toggleModel(i, e.target.checked); }} />
                <span>
                  <b>{m.name}</b>
                  {m.loading ? " · loading…" : m.runtime && <> on <b className={m.runtime.backend}>{m.runtime.backend}</b></>}
                  {!m.loading && m.runtime && <span className="gpu-verification"> · {m.gpuVerification?.label}</span>}
                  {modelPipelines[i] && <> · pipeline <b>{modelPipelines[i]!.name}</b></>}
                </span>
              </label>
            ))}
            <FileButton label="Load .onnx" accept=".onnx" onFile={addModelFile} disabled={loading} small />
          </section>

          <section>
            <h2>Model config</h2>
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
            <h2>Pipeline</h2>
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
            <h2>Calibration</h2>
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
        </div>
      </details>
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

/** Resolves once the video's length and size are known; rejects when the browser can't read the file. */
function loadedMetadata(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= 1) return Promise.resolve();
  return new Promise((resolve, reject) => {
    video.addEventListener("loadedmetadata", () => resolve(), { once: true });
    video.addEventListener("error", () => reject(video.error?.code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
      ? new DOMException("unsupported format", "NotSupportedError") : video.error), { once: true });
  });
}

function playError(name: string, e: unknown): string {
  // The browser can't decode this codec (e.g. MPEG-4 Part 2, H.265, or an .avi container).
  const unsupported = e instanceof DOMException && e.name === "NotSupportedError";
  return unsupported
    ? `Could not play ${name}: the browser doesn't support this video's format. Convert it to H.264 MP4, ` +
        `e.g. ffmpeg -i "${name}" -c:v libx264 -crf 16 -pix_fmt yuv420p out.mp4`
    : `Could not play video: ${errorMessage(e)}`;
}
