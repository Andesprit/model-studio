package com.andesprit.detectors

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The rig vortex model, as model studio's "Rig vortex" runs it (vortex_detector_fp32_with_pipeline.onnx, its INT8
 * original vortex_detector_int8.onnx): the frame scaled by SCALE (not fitted), gray (BT.601), top-left on the
 * H x W canvas with zero padding, /255; rope = logits > 0, all of it (no largest piece, as model studio). Measured by
 * RopeAnalyzer as vortex.pipeline.js does (the rope runs left to right). A rig frame is about 795 x 271.
 */
class RigDetector(context: Context, backend: Backend = Backend.CPU_INT8) : RopeDetector {
    companion object {
        const val H = 160
        const val W = 416
        const val SCALE = 0.5f
    }

    private val env = OrtEnvironment.getEnvironment()
    override var backendNote: String = backend.label
        private set
    private val session: OrtSession
    private val analyzer = RopeAnalyzer()

    init {
        val open = { b: Backend ->
            val asset = if (b == Backend.CPU_INT8) "rig_vortex_int8.onnx" else "rig_vortex_fp32.onnx"
            env.createSession(context.assets.open(asset).use { it.readBytes() }, OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(4)
                if (b == Backend.NNAPI) addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            })
        }
        session = try {
            open(backend)
        } catch (e: Exception) {
            if (backend != Backend.NNAPI) throw e
            backendNote = "${Backend.CPU_FP32.label} (NNAPI failed: ${e.message?.take(80)})"
            open(Backend.CPU_FP32)
        }
    }

    override fun reset() = analyzer.reset()

    /** BT.601 luma of the frame scaled by SCALE, top-left on the canvas (cut at its edges), zero padding, /255. */
    fun preprocess(frame: Bitmap): FloatArray {
        val sw = (frame.width * SCALE).roundToInt().coerceAtLeast(1)
        val sh = (frame.height * SCALE).roundToInt().coerceAtLeast(1)
        val small = Bitmap.createScaledBitmap(frame, sw, sh, true)
        val cw = min(sw, W); val ch = min(sh, H)
        val px = IntArray(cw * ch)
        small.getPixels(px, 0, cw, 0, 0, cw, ch)
        val x = FloatArray(H * W)
        for (y in 0 until ch) for (c in 0 until cw) {
            val p = px[y * cw + c]
            x[y * W + c] = (0.299f * (p shr 16 and 255) + 0.587f * (p shr 8 and 255) + 0.114f * (p and 255)) / 255f
        }
        return x
    }

    /** The rope logits, H x W. */
    fun run(input: FloatArray): FloatArray =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 1, H.toLong(), W.toLong())).use { t ->
            session.run(mapOf("image" to t)).use { out ->
                val b = (out.get(0) as OnnxTensor).floatBuffer
                FloatArray(b.remaining()).also { b.get(it) }
            }
        }

    override fun detectRope(frame: Bitmap, t: Double): RopeResult {
        val input = preprocess(frame)
        val t0 = System.nanoTime()
        val logits = run(input)
        val ms = (System.nanoTime() - t0) / 1e6
        val mask = ByteArray(H * W) { if (logits[it] > 0f) 1 else 0 }
        val step = analyzer.step(mask, H, W, false, SCALE, frame.width, frame.height, t)
        return RopeResult(mask, W, H, SCALE, 0, step, "rig vortex", ms, 1)
    }

    override fun summary() = analyzer.summary()

    override fun close() = session.close()
}
