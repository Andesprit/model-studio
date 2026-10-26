package com.andesprit.detectors

import android.graphics.PointF
import java.util.Locale
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * The rope measurements of model studio's lamp.pipeline.js (vortexAnalyzer, measureRope, breathing), in pixels of the
 * analysed picture: measured slice by slice across the rope (columns for a horizontal rope, rows for a vertical one).
 *   area, area rate (against TREND_FRAMES frames before), size, thickness (across the rope), center (from the top for a
 *   horizontal rope, from the left for a vertical one), drift speed, waviness, breathing (over BREATHING_WINDOW frames).
 */
class RopeAnalyzer {
    companion object {
        const val BREATHING_WINDOW = 60
        const val TREND_FRAMES = 5
    }

    private class Sample(val t: Double, val present: Double, val area: Double, val thickness: Double, val center: Double, val waviness: Double)

    /** One analysed frame: the metrics, and what to draw in picture pixels (rope edges, centreline, center point). */
    class Step(val metrics: Map<String, String>, val top: List<PointF>, val bottom: List<PointF>, val mid: List<PointF>, val center: PointF?)

    private val series = ArrayList<Sample>()

    fun reset() = series.clear()

    /**
     * `mask`: H x W cells of the model canvas, frame pixel = cell / scale; `vertical`: the rope runs top to bottom (the
     * mask is measured transposed and the drawing turned back); pictureW x pictureH: the analysed picture; t: seconds.
     */
    fun step(mask: ByteArray, h: Int, w: Int, vertical: Boolean, scale: Float, pictureW: Int, pictureH: Int, t: Double): Step {
        val rope = if (vertical) measure(transpose(mask, h, w), w, h, scale, pictureH, pictureW) else measure(mask, h, w, scale, pictureW, pictureH)
        if (series.isEmpty() || t > series.last().t) series.add(Sample(t, rope.present, rope.area, rope.thickness, rope.centerY, rope.waviness))
        val s = series.last()
        val ref = series.getOrNull(series.size - 1 - TREND_FRAMES)
        val dt = if (ref != null) s.t - ref.t else 0.0
        val areaRate = if (dt > 0) (s.area - ref!!.area) / dt else null
        val drift = if (dt > 0) -(s.center - ref!!.center) / dt else null
        val b = breathing(series.takeLast(BREATHING_WINDOW))
        val metrics = linkedMapOf(
            "present" to fmt("%.1f %%", s.present * 100),
            "area" to fmt("%.0f px²", s.area),
            "area rate" to (areaRate?.let { fmt("%.0f px²/s", it) } ?: "-"),
            "size" to (areaRate?.let { trend(it * dt, s.area * 0.01) } ?: "-"),
            "thickness" to fmt("%.2f px", s.thickness),
            "center" to fmt("%.2f px", s.center),
            "drift speed" to (drift?.let { fmt("%.2f px/s", it) } ?: "-"),
            "waviness" to fmt("%.2f px", s.waviness),
            "breathing" to (b?.let { fmt("%.2f px", it.amplitude) } ?: "-"),
            "breathing Hz" to (b?.frequency?.let { fmt("%.2f", it) } ?: "-"),
        )
        // drawing: every few slices, in picture pixels (turned back for a vertical rope)
        val stride = max(1, rope.slices.size / 120)
        val cols = rope.slices.filterIndexed { i, _ -> i % stride == 0 }
        fun pt(x: Float, y: Float) = if (vertical) PointF(y, x) else PointF(x, y)
        return Step(
            metrics,
            cols.map { pt(it.x, it.top) },
            cols.map { pt(it.x, it.bottom) },
            cols.map { pt(it.x, (it.top + it.bottom) / 2) },
            if (rope.present > 0) pt(rope.centerX.toFloat(), rope.centerY.toFloat()) else null,
        )
    }

    private class Slice(val x: Float, val top: Float, val bottom: Float)
    private class Rope(val slices: List<Slice>, val present: Double, val area: Double, val thickness: Double,
                       val centerY: Double, val centerX: Double, val waviness: Double)

    /** Per-column top and bottom edge of the rope (the longest run of mask cells in each column), in picture pixels. */
    private fun measure(mask: ByteArray, h: Int, w: Int, scale: Float, pictureW: Int, pictureH: Int): Rope {
        val cell = 1f / scale
        val slices = ArrayList<Slice>()
        var colsInPicture = 0
        var area = 0.0
        for (x in 0 until w) {
            val cx = (x + 0.5f) * cell
            if (cx >= pictureW) continue
            colsInPicture++
            var best = 0; var bestStart = -1; var run = 0; var start = 0
            for (y in 0 until h) {
                val cy = (y + 0.5f) * cell
                if (mask[y * w + x].toInt() != 0 && cy < pictureH) {
                    if (run == 0) start = y
                    run++
                    if (run > best) { best = run; bestStart = start }
                } else run = 0
            }
            if (best > 0) {
                slices.add(Slice(cx, bestStart * cell, (bestStart + best) * cell))
                area += best * cell * cell.toDouble()
            }
        }
        val thick = slices.map { (it.bottom - it.top).toDouble() }
        val center = slices.map { ((it.top + it.bottom) / 2).toDouble() }
        return Rope(
            slices,
            if (colsInPicture > 0) slices.size.toDouble() / colsInPicture else 0.0,
            area,
            if (thick.isNotEmpty()) thick.average() else 0.0,
            if (center.isNotEmpty()) center.average() else 0.0,
            if (slices.isNotEmpty()) slices.map { it.x.toDouble() }.average() else 0.0,
            if (center.isNotEmpty()) (quantile(center, 0.95) - quantile(center, 0.05)) / 2 else 0.0,
        )
    }

    private class Breathing(val amplitude: Double, val frequency: Double?)

    /** Half the p10-p90 range of the thickness, and expansions per second from crossings of its mean. */
    private fun breathing(window: List<Sample>): Breathing? {
        val pts = window.filter { it.present > 0 }
        if (pts.size < 6) return null
        val v = pts.map { it.thickness }
        val m = v.average()
        var crossings = 0
        for (i in 1 until v.size) if ((v[i - 1] - m) * (v[i] - m) < 0) crossings++
        val span = pts.last().t - pts.first().t
        return Breathing((quantile(v, 0.9) - quantile(v, 0.1)) / 2, if (span > 0 && crossings > 0) crossings / 2.0 / span else null)
    }

    private fun trend(change: Double, deadBand: Double) = when {
        abs(change) <= deadBand -> "steady"
        change > 0 -> "growing"
        else -> "shrinking"
    }
}

/** The H x W mask turned to W x H (cell (y, x) to (x, y)): a vertical rope then runs left to right. */
fun transpose(mask: ByteArray, h: Int, w: Int): ByteArray {
    val t = ByteArray(mask.size)
    for (y in 0 until h) for (x in 0 until w) t[x * h + y] = mask[y * w + x]
    return t
}

/** Numbers with a point whatever the phone's language, as model studio shows them. */
private fun fmt(pattern: String, vararg args: Any?) = String.format(Locale.ROOT, pattern, *args)

/** Linear interpolation between the closest ranks, as the pipeline's quantile. */
fun quantile(xs: List<Double>, q: Double): Double {
    val s = xs.sorted()
    val i = (s.size - 1) * q
    val lo = floor(i).toInt()
    return s[lo] + (s[min(lo + 1, s.size - 1)] - s[lo]) * (i - lo)
}
