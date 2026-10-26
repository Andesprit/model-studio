package com.andesprit.detectors

import android.Manifest
import android.app.AlertDialog
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.MediaController
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import java.util.concurrent.Executors
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * One screen: pick a detector (vortex lamp, rig vortex, particles) and where the vortex models run, then the camera or
 * a video. A video runs live (every frame as fast as the model allows, each result shown as it comes) or, for the
 * vortex detectors, "Analyze first": frames at ANALYZE_RATES per second of video are analysed with a progress bar
 * (Stop cancels), then the video plays with the stored results drawn at its current time, and the summary.
 */
class MainActivity : ComponentActivity() {
    companion object {
        const val MODEL_TIMES = 30
        val ANALYZE_RATES = listOf(10, 15, 30)
        /** Camera frames for the models: the models see 416 x 232 at most, so more pixels only cost time. */
        val ANALYSIS_SIZE = Size(640, 480)
        /** The status line is redrawn at most this often: each change of its text lays the screen out again. */
        const val STATUS_EVERY_MS = 250L
    }

    private enum class Kind { LAMP, RIG, PARTICLES }

    /** A frame analysed in advance: its time, the result without its mask, and the mask run-length encoded. */
    private class Stored(val t: Double, val result: RopeResult, val rle: IntArray?)

    private val worker = Executors.newSingleThreadExecutor() // every model call runs here
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var overlay: OverlayView
    private lateinit var video: VideoView
    private lateinit var preview: PreviewView
    private lateinit var progress: ProgressBar
    private lateinit var status: TextView
    private lateinit var gpuBox: CheckBox
    @Volatile private var kind = Kind.LAMP
    @Volatile private var running = false
    @Volatile private var backend = Backend.CPU_INT8
    private var lamp: VortexDetector? = null
    private var rig: RigDetector? = null
    private var particles: ParticleDetector? = null
    /** One model run (ms) on the last MODEL_TIMES frames, for the median shown. */
    private val modelTimes = ArrayDeque<Double>()
    private var camera: ProcessCameraProvider? = null
    private var lastFrameAt = 0L
    private var cameraStart = -1L
    private var playback: Runnable? = null
    @Volatile private var statusShownAt = 0L

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else status.text = "camera permission refused"
    }
    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) chooseVideoMode(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val which = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        for ((k, label) in listOf(Kind.LAMP to "Vortex lamp", Kind.RIG to "Rig vortex", Kind.PARTICLES to "Particles")) {
            which.addView(RadioButton(this).apply { text = label; id = 1 + k.ordinal })
        }
        which.check(1 + kind.ordinal)
        which.setOnCheckedChangeListener { _, id ->
            kind = Kind.entries[id - 1]
            worker.execute { lamp?.reset(); rig?.reset(); modelTimes.clear() }
        }

        // where the vortex models run: changing it reloads them and starts the timing over
        val backends = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        for (b in Backend.entries) backends.addView(RadioButton(this).apply { text = b.label; id = 100 + b.ordinal })
        backends.check(100 + backend.ordinal)
        backends.setOnCheckedChangeListener { _, id ->
            backend = Backend.entries[id - 100]
            worker.execute { lamp?.close(); lamp = null; rig?.close(); rig = null; modelTimes.clear() }
        }
        gpuBox = CheckBox(this).apply { text = "GPU (particles)" }
        gpuBox.setOnCheckedChangeListener { _, _ -> worker.execute { particles?.close(); particles = null } }

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        buttons.addView(Button(this).apply { text = "Camera"; setOnClickListener { onCamera() } })
        buttons.addView(Button(this).apply { text = "Video"; setOnClickListener { pickVideo.launch(arrayOf("video/*")) } })
        buttons.addView(Button(this).apply { text = "Stop"; setOnClickListener { stop() } })
        buttons.addView(gpuBox)

        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { visibility = View.GONE }
        status = TextView(this).apply { text = "pick a detector and a source"; setPadding(16, 8, 16, 8) }
        // the camera's live picture, or the video (Analyze first playback), under the overlay; all fitted and centred
        // the same way
        val stage = FrameLayout(this)
        preview = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE // a TextureView: the overlay draws over it
            visibility = View.GONE
        }
        stage.addView(preview, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        video = VideoView(this).apply { visibility = View.GONE }
        stage.addView(video, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT, Gravity.CENTER))
        overlay = OverlayView(this)
        stage.addView(overlay, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

        root.addView(which, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(backends, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(buttons, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(progress, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(stage, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        // keep clear of the status bar, the camera cutout and the gesture bar (Android 15 draws apps edge to edge),
        // with a small margin on top
        val margin = (16 * resources.displayMetrics.density).toInt()
        root.setPadding(0, margin, 0, 0)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            v.setPadding(bars.left, bars.top + margin, bars.right, bars.bottom)
            insets
        }
        setContentView(root)
        ViewCompat.requestApplyInsets(root)
    }

    private fun onCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    /**
     * The camera: its live picture full speed, and small frames (ANALYSIS_SIZE, the same 4:3 view) for the models, the
     * latest one each time the model is free; the results are drawn over the live picture as they come.
     */
    private fun startCamera() {
        stop()
        running = true
        worker.execute { lamp?.reset(); rig?.reset(); cameraStart = -1L }
        preview.visibility = View.VISIBLE
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get().also { camera = it }
            val selector = { size: Size ->
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy(size, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                    .build()
            }
            val live = Preview.Builder().setResolutionSelector(selector(Size(1280, 960))).build()
            live.surfaceProvider = preview.surfaceProvider
            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(selector(ANALYSIS_SIZE))
                .build()
            analysis.setAnalyzer(worker) { image ->
                val bmp = image.toBitmap()
                val deg = image.imageInfo.rotationDegrees
                val ns = image.imageInfo.timestamp
                image.close()
                if (cameraStart < 0) cameraStart = ns
                if (running) process(if (deg == 0) bmp else rotate(bmp, deg), (ns - cameraStart) / 1e9, overPreview = true)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, live, analysis)
        }, mainExecutor)
    }

    /** Analyze first at a rate, or live; the particle detector runs live. */
    private fun chooseVideoMode(uri: Uri) {
        if (kind == Kind.PARTICLES) return startLive(uri)
        val options = ANALYZE_RATES.map { "Analyze first, $it frames per second" } + "Live"
        AlertDialog.Builder(this)
            .setTitle("How should this video be analysed?")
            .setItems(options.toTypedArray()) { _, i -> if (i < ANALYZE_RATES.size) analyzeFirst(uri, ANALYZE_RATES[i]) else startLive(uri) }
            .show()
    }

    /** The video's frame count, length, rotation tag and display size, and a frame turned by the tag if not yet. */
    private class VideoInfo(r: MediaMetadataRetriever) {
        val frames = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toInt() ?: 0
        val seconds = (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDouble() ?: 0.0) / 1000
        val fps = if (seconds > 0) frames / seconds else 30.0
        val rotation = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
        private val codedW = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 0
        private val codedH = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toInt() ?: 0
        private val sideways = rotation == 90 || rotation == 270
        val width = if (sideways) codedH else codedW
        val height = if (sideways) codedW else codedH

        fun upright(frame: Bitmap): Bitmap =
            if (sideways && frame.width == codedW) Bitmap.createBitmap(frame, 0, 0, frame.width, frame.height,
                Matrix().apply { postRotate(rotation.toFloat()) }, true) else frame
    }

    /**
     * Live: every frame of the video in order, as fast as the model allows (not in real time). Its time is the frame
     * index over the video's mean frame rate.
     */
    private fun startLive(uri: Uri) {
        stop()
        running = true
        worker.execute {
            lamp?.reset(); rig?.reset()
            val r = MediaMetadataRetriever()
            try {
                r.setDataSource(this, uri)
                val v = VideoInfo(r)
                var i = 0
                while (running && i < v.frames) {
                    val frame = r.getFrameAtIndex(i) ?: break
                    process(v.upright(frame), i / v.fps)
                    i++
                }
                setStatus("$lastStatus\nvideo done: $i of ${v.frames} frames", force = true)
            } catch (e: Exception) {
                runOnUiThread { status.text = "could not read the video: ${e.message}" }
            } finally {
                r.release()
            }
        }
    }

    /**
     * Analyze first: the frame shown at every 1/rate s of the video, analysed in order (the measurements cover the
     * whole video), then the video plays with the stored results; Stop cancels.
     */
    private fun analyzeFirst(uri: Uri, rate: Int) {
        stop()
        running = true
        overlay.clear()
        progress.progress = 0
        progress.visibility = View.VISIBLE
        worker.execute {
            val d = ropeDetector()
            d.reset()
            modelTimes.clear()
            val r = MediaMetadataRetriever()
            try {
                r.setDataSource(this, uri)
                val v = VideoInfo(r)
                val total = floor(v.seconds * rate).toInt() + 1
                val stored = ArrayList<Stored>(total)
                val t0 = System.nanoTime()
                runOnUiThread { progress.max = total }
                // Frames are read in order, every one of them: reading only every few would make the retriever decode
                // again from the last keyframe each time (seconds per frame); only the wanted ones are analysed.
                var next = 0
                var frame: Bitmap? = null // the last frame read: two times can fall on the same frame
                for (i in 0 until total) {
                    if (!running) break
                    val t = min(i.toDouble() / rate, v.seconds)
                    val want = min(v.frames - 1, (t * v.fps).roundToInt())
                    while (next <= want && running) frame = r.getFrameAtIndex(next++) ?: break
                    if (frame == null || !running) break
                    val res = d.detectRope(v.upright(frame), t)
                    modelTime(res)
                    val (packed, rle) = res.packed()
                    stored.add(Stored(t, packed, rle))
                    val shown = i + 1
                    runOnUiThread { progress.progress = shown }
                    setStatus("Analysing $shown / $total frames (${100 * shown / total} %), ${d.backendNote}")
                }
                val done = stored.size
                if (!running || done == 0) {
                    runOnUiThread { progress.visibility = View.GONE; status.text = "analysis stopped after $done of $total frames" }
                    return@execute
                }
                val seconds = (System.nanoTime() - t0) / 1e9
                val about = "Analysed first: $done frames at $rate per second in %.1f s, %s".format(seconds, modelLine(d))
                val summary = listOf(about, "", "Summary:") + d.summary()
                runOnUiThread { progress.visibility = View.GONE; play(uri, stored, rate, v.width, v.height, summary) }
            } catch (e: Exception) {
                runOnUiThread { progress.visibility = View.GONE; status.text = "could not read the video: ${e.message}" }
            } finally {
                r.release()
            }
        }
    }

    /** Plays the video and draws the stored result shown at its current time (the last analysed frame up to it). */
    private fun play(uri: Uri, stored: List<Stored>, rate: Int, width: Int, height: Int, summary: List<String>) {
        video.visibility = View.VISIBLE
        video.setMediaController(MediaController(this).apply { setAnchorView(video) })
        video.setVideoURI(uri)
        video.setOnPreparedListener { it.isLooping = true; video.start() }
        var shown = -1
        val tick = object : Runnable {
            override fun run() {
                val now = video.currentPosition / 1000.0 + 0.5 / rate
                var lo = 0; var hi = stored.size - 1
                while (lo < hi) { val mid = (lo + hi + 1) / 2; if (stored[mid].t <= now) lo = mid else hi = mid - 1 }
                if (lo != shown) {
                    shown = lo
                    val s = stored[lo]
                    overlay.showRope(null, s.result.unpacked(s.rle), width, height)
                    status.text = (listOf(s.result.header, readings(s.result)) + listOf("") + summary).joinToString("\n")
                }
                ui.postDelayed(this, 33)
            }
        }
        playback = tick
        ui.post(tick)
    }

    private fun stop() {
        running = false
        camera?.unbindAll()
        preview.visibility = View.GONE
        playback?.let { ui.removeCallbacks(it) }
        playback = null
        if (video.visibility == View.VISIBLE) {
            video.stopPlayback()
            video.visibility = View.GONE
        }
        progress.visibility = View.GONE
    }

    /** Runs on the worker thread. */
    private fun ropeDetector(): RopeDetector = when (kind) {
        Kind.RIG -> rig ?: RigDetector(this, backend).also { rig = it; modelTimes.clear() }
        else -> lamp ?: VortexDetector(this, backend).also { lamp = it; modelTimes.clear() }
    }

    /**
     * Runs on the worker thread; t in seconds. overPreview: the camera's live picture is under the overlay, so only the
     * results are drawn (fitted to the frame's size, the same view); else the analysed frame is drawn under them.
     */
    private fun process(frame: Bitmap, t: Double, overPreview: Boolean = false) {
        val t0 = System.nanoTime()
        val text: String
        if (kind != Kind.PARTICLES) {
            val d = ropeDetector()
            val r = d.detectRope(frame, t)
            modelTime(r)
            runOnUiThread { overlay.showRope(if (overPreview) null else frame, r, frame.width, frame.height) }
            text = "${r.header}\n${readings(r)}\n${modelLine(d)}"
        } else {
            val d = particles ?: ParticleDetector(this, gpuBox.isChecked).also { particles = it }
            val r = d.detect(frame)
            runOnUiThread { overlay.showParticles(if (overPreview) null else frame, r, frame.width, frame.height) }
            text = "particles: ${r.boxes.size} (${d.backend}${if (r.shrunk) ", frame shrunk to fit" else ""})\n" +
                "network %.0f ms".format(r.networkMs)
        }
        val now = System.nanoTime()
        val total = (now - t0) / 1e6
        val fps = if (lastFrameAt > 0) 1e9 / (now - lastFrameAt) else 0.0
        lastFrameAt = now
        setStatus("$text, frame %.0f ms, %.1f fps, %d x %d".format(total, fps, frame.width, frame.height))
    }

    @Volatile private var lastStatus = ""

    /** Sets the status line, at most every STATUS_EVERY_MS unless forced (the last word of a run). Any thread. */
    private fun setStatus(text: String, force: Boolean = false) {
        lastStatus = text
        val now = System.currentTimeMillis()
        if (!force && now - statusShownAt < STATUS_EVERY_MS) return
        statusShownAt = now
        runOnUiThread { status.text = text }
    }

    /** The rope's readings, named as model studio names them for the lamp or the rig. */
    private fun readings(r: RopeResult): String {
        val m = r.step?.metrics.orEmpty()
        fun v(k: String) = m[k] ?: "-"
        val position = if (r.header.startsWith("rig")) "center y ${v("center")}, vertical speed ${v("drift speed")} (${v("moving")})"
            else "center ${v("center")}, drift ${v("drift speed")}"
        return "thickness ${v("thickness")}, breathing ${v("breathing")} at ${v("breathing Hz")} Hz, waviness ${v("waviness")}\n" +
            "$position, area ${v("area")} (${v("size")})"
    }

    private fun modelTime(r: RopeResult) {
        modelTimes.addLast(r.networkMs / r.runs)
        while (modelTimes.size > MODEL_TIMES) modelTimes.removeFirst()
    }

    /** The last model run and the median over the last MODEL_TIMES frames, on the backend that ran them. */
    private fun modelLine(d: RopeDetector): String {
        val median = modelTimes.sorted().getOrNull(modelTimes.size / 2) ?: 0.0
        return "model %.0f ms per run, median %.0f ms over %d frames (${d.backendNote})".format(modelTimes.lastOrNull() ?: 0.0, median, modelTimes.size)
    }

    private fun rotate(b: Bitmap, deg: Int): Bitmap =
        Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, true)

    override fun onDestroy() {
        stop()
        worker.execute { lamp?.close(); rig?.close(); particles?.close() }
        worker.shutdown()
        super.onDestroy()
    }
}
