package com.andesprit.detectors

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * The vortex lamp model (vortex_lamp_auto_fp32.onnx, as model studio runs it) and the orientation part of
 * vortex_detector/model_studio/lamp.pipeline.js: water region, vertical/horizontal decision over DECIDE_FRAMES
 * frames with water, then one rope branch per frame. The rope measurements of lamp.pipeline.js are not ported yet.
 */
class VortexDetector(context: Context) : AutoCloseable {
    companion object {
        const val H = 416
        const val W = 232
        const val OUTSIDE = -10000f
        const val DECIDE_FRAMES = 10
        const val WATER_MIN_FRACTION = 0.01
    }

    /** Masks are H x W cells of the model canvas; frame pixel = cell / scale. */
    class Result(
        val rope: ByteArray?,
        val water: ByteArray?,
        val waterFound: Boolean,
        val orientation: String,
        val scale: Float,
        val networkMs: Double,
    )

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(
        context.assets.open("vortex_lamp.onnx").use { it.readBytes() },
        OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) },
    )

    private var sumVertical = 0.0
    private var sumHorizontal = 0.0
    private var seen = 0
    private var locked: String? = null

    /** A new video or camera start: decide the orientation again. */
    fun reset() {
        sumVertical = 0.0; sumHorizontal = 0.0; seen = 0; locked = null
    }

    private fun better(): String = locked ?: if (sumVertical >= sumHorizontal) "vertical" else "horizontal"

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

    /** The raw network output of one branch (mode 1 "logits", mode 2 "logits_asis"), H x W. */
    fun run(input: FloatArray, mode: Int): FloatArray {
        val name = if (mode == 1) "logits" else "logits_asis"
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 1, H.toLong(), W.toLong())).use { image ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(mode.toFloat())), longArrayOf(1)).use { m ->
                session.run(mapOf("image" to image, "mode" to m), setOf(name)).use { out ->
                    val t = out.get(0) as OnnxTensor
                    val b = t.floatBuffer
                    return FloatArray(b.remaining()).also { b.get(it) }
                }
            }
        }
    }

    fun detect(frame: Bitmap): Result {
        val (input, s) = preprocess(frame)
        val which = better()
        val t0 = System.nanoTime()
        val logits = run(input, if (which == "horizontal") 2 else 1)
        val ms = (System.nanoTime() - t0) / 1e6

        // cells of the canvas that hold the frame (the rest is padding)
        val sw = (frame.width * s).roundToInt().coerceIn(1, W)
        val sh = (frame.height * s).roundToInt().coerceIn(1, H)
        val water = ByteArray(H * W)
        val rope = ByteArray(H * W)
        for (y in 0 until sh) for (c in 0 until sw) {
            val v = logits[y * W + c]
            if (v != OUTSIDE) water[y * W + c] = 1
            if (v > 0f) rope[y * W + c] = 1
        }
        val waterPixels = keepLargest(water)
        keepLargest(rope)
        val found = waterPixels > 0 && waterPixels >= WATER_MIN_FRACTION * sw * sh

        if (locked == null && found) {
            val a = axisAngle(water)
            if (a != null) {
                val r = Math.toRadians(a)
                sumVertical += sin(r) * sin(r)
                sumHorizontal += cos(r) * cos(r)
            }
            if (++seen >= DECIDE_FRAMES) locked = better()
        }
        val orientation = locked ?: "${better()} (deciding $seen/$DECIDE_FRAMES)"
        // After a switch of the deciding branch, this frame's mask belongs to the old branch: show it anyway.
        return Result(rope, water, found, orientation, s, ms)
    }

    override fun close() = session.close()
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

/** Main axis of the mask cells against +x, degrees in (-90, 90]; null under 10 cells. Cells are square here. */
fun axisAngle(mask: ByteArray, w: Int = VortexDetector.W): Double? {
    var n = 0; var sx = 0.0; var sy = 0.0; var sxx = 0.0; var syy = 0.0; var sxy = 0.0
    for (i in mask.indices) {
        if (mask[i].toInt() == 0) continue
        val x = (i % w).toDouble(); val y = (i / w).toDouble()
        n++; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y
    }
    if (n < 10) return null
    val vx = sxx / n - (sx / n) * (sx / n)
    val vy = syy / n - (sy / n) * (sy / n)
    val cxy = sxy / n - (sx / n) * (sy / n)
    return Math.toDegrees(0.5 * atan2(2 * cxy, vx - vy))
}
