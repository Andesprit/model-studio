package com.andesprit.detectors

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The vortex lamp rope network (vortex_lamp_rope.onnx, tools/export_lamp_model.py) and model studio's
 * lamp.pipeline.js for a model with no water gate:
 * - a landscape picture is turned 90 deg clockwise first (the model's canvas is portrait), as model studio does;
 * - the network knows a rope that runs left to right; runBranch runs it on the canvas as is (horizontal) or turned
 *   90 deg counter-clockwise with the mask turned back (vertical), as model studio's file does inside;
 * - while deciding, both branches run and each is scored by its rope strength (mean rope probability x share of the
 *   rope in its largest piece); after DECIDE_FRAMES frames with a rope the stronger orientation is locked and one
 *   branch runs per frame; it decides again when the picture's shape changes or after REDECIDE_FRAMES frames with no
 *   rope, and warns "no lamp in view" after LAMP_MISSING_FRAMES;
 * - only the largest rope piece is kept, and RopeAnalyzer measures it.
 * Not ported: the tilt warning (this model was trained on lamps tilted up to 45 deg).
 *
 * `backend` picks the model file and where it runs (LampBackend). There is one session per canvas shape (the frame as
 * is, 416 x 232, and turned, 232 x 416), each with its shape fixed: NNAPI handles fixed shapes, not free ones. If
 * NNAPI can't be set up the CPU (fp32) runs instead, and `backendNote` says so.
 */
class VortexDetector(context: Context, backend: LampBackend = LampBackend.CPU_FP32) : AutoCloseable {
    companion object {
        const val H = 416
        const val W = 232
        const val DECIDE_FRAMES = 10
        const val ROPE_MIN = 0.005 // rope strength of a lamp in view: lamp frames 0.010-0.10, rooms with no lamp ~0.001
        const val LAMP_MISSING_FRAMES = 5
        const val REDECIDE_FRAMES = 15
    }

    /**
     * The rope mask is H x W cells of the canvas (picture pixel = cell / scale); the picture is the frame turned by
     * `turn` degrees clockwise. The drawing is in picture pixels.
     */
    class Result(
        val rope: ByteArray?,
        val scale: Float,
        val turn: Int,
        val orientation: String,
        val strength: Double,
        val lampMissing: Boolean,
        val step: RopeAnalyzer.Step?,
        val networkMs: Double,
        /** Branches run this frame (both while deciding, one after): networkMs / branches is one model run. */
        val branches: Int,
    )

    private val env = OrtEnvironment.getEnvironment()
    /** What runs the model: the chosen backend, or the CPU fallback with the reason. */
    var backendNote: String = backend.label
        private set
    private val sessions: Map<Int, OrtSession> // by canvas height: H (as is) or W (turned)

    init {
        val open = { b: LampBackend ->
            val bytes = context.assets.open(b.asset).use { it.readBytes() }
            listOf(H to W, W to H).associate { (h, w) ->
                h to env.createSession(bytes, OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(4)
                    setSymbolicDimensionValue("h", h.toLong())
                    setSymbolicDimensionValue("w", w.toLong())
                    // fp16 lets the GPU or AI chip compute in half precision, as model studio's GPU file does
                    if (b == LampBackend.NNAPI) addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                })
            }
        }
        sessions = try {
            open(backend)
        } catch (e: Exception) {
            if (backend != LampBackend.NNAPI) throw e
            backendNote = "${LampBackend.CPU_FP32.label} (NNAPI failed: ${e.message?.take(80)})"
            open(LampBackend.CPU_FP32)
        }
    }
    private val analyzer = RopeAnalyzer()

    private var sumVertical = 0.0
    private var sumHorizontal = 0.0
    private var seen = 0
    private var locked: String? = null
    private var missingRope = 0
    private var shape: String? = null

    private fun decideAgain() {
        sumVertical = 0.0; sumHorizontal = 0.0; seen = 0; locked = null
    }

    /** A new video or camera start: decide the orientation again and start the measurements over. */
    fun reset() {
        decideAgain(); missingRope = 0; shape = null; analyzer.reset()
    }

    private fun better(): String = if (sumVertical >= sumHorizontal) "vertical" else "horizontal"

    /** BT.601 luma, fit into H x W at the top-left, zero padding, /255. Returns the input and the scale. */
    fun preprocess(frame: Bitmap): Pair<FloatArray, Float> {
        val s = min(W.toFloat() / frame.width, H.toFloat() / frame.height)
        val sw = (frame.width * s).roundToInt().coerceIn(1, W)
        val sh = (frame.height * s).roundToInt().coerceIn(1, H)
        val small = Bitmap.createScaledBitmap(frame, sw, sh, true)
        val px = IntArray(sw * sh)
        small.getPixels(px, 0, sw, 0, 0, sw, sh)
        val x = FloatArray(H * W)
        for (y in 0 until sh) for (c in 0 until sw) {
            val p = px[y * sw + c]
            x[y * W + c] = (0.299f * (p shr 16 and 255) + 0.587f * (p shr 8 and 255) + 0.114f * (p and 255)) / 255f
        }
        return x to s
    }

    /** The network's rope logits on the H x W canvas, as is or turned (vertical: the rope runs top to bottom). */
    fun runBranch(input: FloatArray, vertical: Boolean): FloatArray {
        if (!vertical) return network(input, H, W)
        // as model studio's file: transpose, flip the rows (90 deg counter-clockwise), run, transpose back, flip the columns
        val turned = FloatArray(H * W)
        for (i in 0 until W) for (j in 0 until H) turned[i * H + j] = input[j * W + (W - 1 - i)]
        val y = network(turned, W, H)
        val out = FloatArray(H * W)
        for (a in 0 until H) for (b in 0 until W) out[a * W + b] = y[(W - 1 - b) * H + a]
        return out
    }

    /** Model studio's modes, for the smoke test: 1 = vertical ("logits"), 2 = as is ("logits_asis"). */
    fun run(input: FloatArray, mode: Int): FloatArray = runBranch(input, mode == 1)

    private fun network(x: FloatArray, h: Int, w: Int): FloatArray =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(x), longArrayOf(1, 1, h.toLong(), w.toLong())).use { t ->
            sessions.getValue(h).run(mapOf("x" to t)).use { out ->
                val b = (out.get(0) as OnnxTensor).floatBuffer
                FloatArray(b.remaining()).also { b.get(it) }
            }
        }

    /** One frame at `t` seconds (of the video, or since the camera started). */
    fun detect(frame: Bitmap, t: Double): Result {
        val turn = if (frame.width > frame.height) 90 else 0
        val picture = if (turn == 0) frame else Bitmap.createBitmap(frame, 0, 0, frame.width, frame.height,
            Matrix().apply { postRotate(turn.toFloat()) }, true)
        val pictureShape = "${picture.width}x${picture.height}"
        if (shape != null && shape != pictureShape) decideAgain() // the phone turned: decide from the new view
        shape = pictureShape

        val (input, s) = preprocess(picture)
        val masks = HashMap<String, ByteArray>()
        val strengths = HashMap<String, Double>()
        val t0 = System.nanoTime()
        for (which in locked?.let { listOf(it) } ?: listOf("vertical", "horizontal")) {
            val logits = runBranch(input, which == "vertical")
            val mask = ByteArray(H * W) { if (logits[it] > 0f) 1 else 0 }
            strengths[which] = ropeStrength(logits, mask)
            keepLargest(mask)
            masks[which] = mask
        }
        val ms = (System.nanoTime() - t0) / 1e6

        val strength = strengths.values.max()
        missingRope = if (strength >= ROPE_MIN) 0 else missingRope + 1
        if (locked != null && missingRope >= REDECIDE_FRAMES) decideAgain()
        if (locked == null && strength >= ROPE_MIN) {
            sumVertical += strengths["vertical"] ?: 0.0
            sumHorizontal += strengths["horizontal"] ?: 0.0
            if (++seen >= DECIDE_FRAMES) locked = better()
        }
        val which = locked ?: better()
        val rope = masks[which] // null right after deciding again with only the old branch run: the next frame has it
        val step = rope?.let { analyzer.step(it, H, W, which == "vertical", s, picture.width, picture.height, t) }
        val orientation = if (locked != null) which else "$which (deciding $seen/$DECIDE_FRAMES)"
        return Result(rope, s, turn, orientation, strength, missingRope >= LAMP_MISSING_FRAMES, step, ms, masks.size)
    }

    override fun close() = sessions.values.forEach { it.close() }
}

/** Where the lamp model runs, and which file: tools/export_lamp_model.py writes both. */
enum class LampBackend(val label: String, val asset: String) {
    CPU_FP32("CPU fp32", "vortex_lamp_rope.onnx"),
    CPU_INT8("CPU INT8", "vortex_lamp_rope_int8.onnx"),
    NNAPI("NNAPI (GPU / AI chip)", "vortex_lamp_rope.onnx"),
}

/** Mean rope probability over the canvas x share of the rope cells in the largest piece (before keepLargest). */
fun ropeStrength(logits: FloatArray, mask: ByteArray): Double {
    var sum = 0.0
    for (v in logits) sum += 1 / (1 + exp(-v.toDouble().coerceIn(-30.0, 30.0)))
    val cells = mask.count { it.toInt() != 0 }
    if (cells == 0) return 0.0
    val largest = keepLargest(mask.copyOf())
    return sum / logits.size * largest / cells
}

/** Keeps only the largest 4-connected piece of a H x W mask, in place; returns its size. */
fun keepLargest(mask: ByteArray, w: Int = VortexDetector.W, h: Int = VortexDetector.H): Int {
    val label = IntArray(mask.size)
    val stack = IntArray(mask.size)
    var id = 0; var bestId = 0; var bestSize = 0
    for (i in mask.indices) {
        if (mask[i].toInt() == 0 || label[i] != 0) continue
        id++
        var top = 0; var size = 0
        label[i] = id; stack[top++] = i
        while (top > 0) {
            val p = stack[--top]; size++
            val x = p % w; val y = p / w
            fun visit(q: Int) { if (mask[q].toInt() != 0 && label[q] == 0) { label[q] = id; stack[top++] = q } }
            if (x > 0) visit(p - 1)
            if (x < w - 1) visit(p + 1)
            if (y > 0) visit(p - w)
            if (y < h - 1) visit(p + w)
        }
        if (size > bestSize) { bestSize = size; bestId = id }
    }
    for (i in mask.indices) if (label[i] != bestId) mask[i] = 0
    return bestSize
}
