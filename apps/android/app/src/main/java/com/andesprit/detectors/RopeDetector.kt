package com.andesprit.detectors

import android.graphics.Bitmap

/** The two vortex rope detectors (VortexDetector for the lamp, RigDetector for the rig), as the screen uses them. */
interface RopeDetector : AutoCloseable {
    /** A new video or camera start: start the measurements (and the lamp's orientation) over. */
    fun reset()

    /** One frame at `t` seconds. */
    fun detectRope(frame: Bitmap, t: Double): RopeResult

    /** The measurements over every frame since reset(), as lines for the screen. */
    fun summary(): List<String>
}

/**
 * A frame's rope: the mask is maskH x maskW cells of the model's canvas (picture pixel = cell / scale), the picture is
 * the frame turned by `turn` degrees clockwise, the drawing is in picture pixels. `header`: the detector's own state
 * (the lamp's orientation, "no lamp in view"); runs: model runs on this frame, networkMs: all of them.
 */
class RopeResult(
    val mask: ByteArray?,
    val maskW: Int,
    val maskH: Int,
    val scale: Float,
    val turn: Int,
    val step: RopeAnalyzer.Step?,
    val header: String,
    val networkMs: Double,
    val runs: Int,
) {
    /** The same result with the mask run-length encoded (value, length pairs): a mask is mostly background. */
    fun packed(): Pair<RopeResult, IntArray?> {
        val m = mask ?: return this to null
        val runs = ArrayList<Int>()
        var i = 0
        while (i < m.size) {
            var j = i + 1
            while (j < m.size && m[j] == m[i]) j++
            runs.add(m[i].toInt()); runs.add(j - i)
            i = j
        }
        return RopeResult(null, maskW, maskH, scale, turn, step, header, networkMs, this.runs) to runs.toIntArray()
    }

    fun unpacked(rle: IntArray?): RopeResult {
        if (rle == null) return this
        val m = ByteArray(maskW * maskH)
        var i = 0
        for (r in rle.indices step 2) {
            m.fill(rle[r].toByte(), i, i + rle[r + 1])
            i += rle[r + 1]
        }
        return RopeResult(m, maskW, maskH, scale, turn, step, header, networkMs, runs)
    }
}
