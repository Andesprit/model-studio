package com.andesprit.detectors

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.util.Size
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.CheckBox
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import java.util.concurrent.Executors

/** One screen: pick a detector, then the camera or a video; the analysed frames are shown with what was found. */
class MainActivity : ComponentActivity() {
    companion object {
        const val LAMP_TIMES = 30
    }

    private val worker = Executors.newSingleThreadExecutor() // every model call runs here
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private lateinit var gpuBox: CheckBox
    @Volatile private var useVortex = true
    @Volatile private var running = false
    private var vortex: VortexDetector? = null
    @Volatile private var lampBackend = LampBackend.CPU_FP32
    /** One lamp model run (ms) on the last LAMP_TIMES frames, for the median shown. */
    private val lampTimes = ArrayDeque<Double>()
    private var particles: ParticleDetector? = null
    private var camera: ProcessCameraProvider? = null
    private var lastFrameAt = 0L
    private var cameraStart = -1L

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else status.text = "camera permission refused"
    }
    private val pickVideo = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) startVideo(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val which = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        val vortexButton = RadioButton(this).apply { text = "Vortex"; id = 1 }
        val particleButton = RadioButton(this).apply { text = "Particles"; id = 2 }
        which.addView(vortexButton); which.addView(particleButton)
        which.check(1)
        which.setOnCheckedChangeListener { _, id ->
            useVortex = id == 1
            worker.execute { vortex?.reset() }
        }
        gpuBox = CheckBox(this).apply { text = "GPU (particles)" }
        gpuBox.setOnCheckedChangeListener { _, _ -> worker.execute { particles?.close(); particles = null } }
        which.addView(gpuBox)

        // where the lamp model runs: changing it reloads the model and starts the timing over
        val backends = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        for (b in LampBackend.entries) backends.addView(RadioButton(this).apply { text = b.label; id = 100 + b.ordinal })
        backends.check(100 + lampBackend.ordinal)
        backends.setOnCheckedChangeListener { _, id ->
            lampBackend = LampBackend.entries[id - 100]
            worker.execute { vortex?.close(); vortex = null; lampTimes.clear() }
        }

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        buttons.addView(Button(this).apply { text = "Camera"; setOnClickListener { onCamera() } })
        buttons.addView(Button(this).apply { text = "Video"; setOnClickListener { pickVideo.launch(arrayOf("video/*")) } })
        buttons.addView(Button(this).apply { text = "Stop"; setOnClickListener { stop() } })

        status = TextView(this).apply { text = "pick a detector and a source"; setPadding(16, 8, 16, 8) }
        overlay = OverlayView(this)
        root.addView(which, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(backends, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(buttons, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(status, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(overlay, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    private fun onCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        stop()
        running = true
        worker.execute { vortex?.reset(); cameraStart = -1L }
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get().also { camera = it }
            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setResolutionSelector(
                    ResolutionSelector.Builder().setResolutionStrategy(
                        ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                    ).build()
                ).build()
            analysis.setAnalyzer(worker) { image ->
                val bmp = image.toBitmap()
                val deg = image.imageInfo.rotationDegrees
                val ns = image.imageInfo.timestamp
                image.close()
                if (cameraStart < 0) cameraStart = ns
                if (running) process(if (deg == 0) bmp else rotate(bmp, deg), (ns - cameraStart) / 1e9)
            }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
        }, mainExecutor)
    }

    /**
     * Every frame of the video in order, as fast as the model allows (not in real time). Its time is the frame index
     * over the video's mean frame rate. A frame not yet turned by the file's rotation tag is turned here.
     */
    private fun startVideo(uri: Uri) {
        stop()
        running = true
        worker.execute {
            vortex?.reset()
            val r = MediaMetadataRetriever()
            try {
                r.setDataSource(this, uri)
                val n = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toInt() ?: 0
                val seconds = (r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toDouble() ?: 0.0) / 1000
                val fps = if (seconds > 0) n / seconds else 30.0
                val rotation = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toInt() ?: 0
                val codedWidth = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toInt() ?: 0
                var i = 0
                while (running && i < n) {
                    var frame = r.getFrameAtIndex(i) ?: break
                    if ((rotation == 90 || rotation == 270) && frame.width == codedWidth) frame = rotate(frame, rotation)
                    process(frame, i / fps)
                    i++
                }
                runOnUiThread { status.append("\nvideo done: $i of $n frames") }
            } catch (e: Exception) {
                runOnUiThread { status.text = "could not read the video: ${e.message}" }
            } finally {
                r.release()
            }
        }
    }

    private fun stop() {
        running = false
        camera?.unbindAll()
    }

    /** Runs on the worker thread; t in seconds. */
    private fun process(frame: Bitmap, t: Double) {
        val t0 = System.nanoTime()
        val text: String
        if (useVortex) {
            val d = vortex ?: VortexDetector(this, lampBackend).also { vortex = it; lampTimes.clear() }
            val r = d.detect(frame, t)
            runOnUiThread { overlay.showVortex(frame, r) }
            val m = r.step?.metrics.orEmpty()
            text = (if (r.lampMissing) "no lamp in view\n" else "") +
                "vortex: ${r.orientation}, rope strength %.3f\n".format(r.strength) +
                "thickness ${m["thickness"] ?: "-"}, breathing ${m["breathing"] ?: "-"} at ${m["breathing Hz"] ?: "-"} Hz, " +
                "waviness ${m["waviness"] ?: "-"}\n" +
                "center ${m["center"] ?: "-"}, drift ${m["drift speed"] ?: "-"}, area ${m["area"] ?: "-"} (${m["size"] ?: "-"})\n" +
                lampTiming(d, r)
        } else {
            val d = particles ?: ParticleDetector(this, gpuBox.isChecked).also { particles = it }
            val r = d.detect(frame)
            runOnUiThread { overlay.showParticles(frame, r) }
            text = "particles: ${r.boxes.size} (${d.backend}${if (r.shrunk) ", frame shrunk to fit" else ""})\n" +
                "network %.0f ms".format(r.networkMs)
        }
        val now = System.nanoTime()
        val total = (now - t0) / 1e6
        val fps = if (lastFrameAt > 0) 1e9 / (now - lastFrameAt) else 0.0
        lastFrameAt = now
        val line = "$text, frame %.0f ms, %.1f fps, %d x %d".format(total, fps, frame.width, frame.height)
        runOnUiThread { status.text = line }
    }

    /** One model run on this frame and the median over the last LAMP_TIMES frames, on the backend that ran it. */
    private fun lampTiming(d: VortexDetector, r: VortexDetector.Result): String {
        val one = r.networkMs / r.branches
        lampTimes.addLast(one)
        while (lampTimes.size > LAMP_TIMES) lampTimes.removeFirst()
        val median = lampTimes.sorted()[lampTimes.size / 2]
        return "lamp model %.0f ms per run, median %.0f ms over %d frames (${d.backendNote})".format(one, median, lampTimes.size)
    }

    private fun rotate(b: Bitmap, deg: Int): Bitmap =
        Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, true)

    override fun onDestroy() {
        stop()
        worker.execute { vortex?.close(); particles?.close() }
        worker.shutdown()
        super.onDestroy()
    }
}
