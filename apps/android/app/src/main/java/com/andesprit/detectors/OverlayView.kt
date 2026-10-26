package com.andesprit.detectors

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.view.View
import kotlin.math.min

/**
 * What was found, fitted to the view: the vortex rope (mask, edges, centreline, center point) or the particle boxes.
 * Live, it draws the analysed frame under them; over a playing video (Analyze first) it draws only them, fitted to
 * the video's size the same way the video is.
 */
class OverlayView(context: Context) : View(context) {
    private var frame: Bitmap? = null
    private var contentW = 0
    private var contentH = 0
    private var mask: Bitmap? = null
    private var rope: RopeResult? = null
    private var boxes: List<ParticleDetector.Box> = emptyList()
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; color = Color.rgb(255, 212, 0); strokeWidth = 2f }
    private val maskPaint = Paint().apply { isFilterBitmap = false }
    private val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Color.rgb(0, 229, 255) }
    private val midPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Color.rgb(255, 212, 0) }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(255, 59, 59) }

    /** The rope on `frame`, or on a picture of contentW x contentH shown under this view (frame null: the camera's live picture or the playing video). */
    fun showRope(frame: Bitmap?, r: RopeResult, contentW: Int = frame?.width ?: 0, contentH: Int = frame?.height ?: 0) {
        val px = IntArray(r.maskW * r.maskH)
        r.mask?.let { m -> for (i in px.indices) if (m[i].toInt() == 1) px[i] = Color.argb(110, 0, 229, 255) }
        this.frame = frame
        this.contentW = contentW
        this.contentH = contentH
        mask = Bitmap.createBitmap(px, r.maskW, r.maskH, Bitmap.Config.ARGB_8888)
        rope = r
        boxes = emptyList()
        invalidate()
    }

    /** The boxes on `frame` (live), or on the camera's live picture of contentW x contentH under this view (frame null). */
    fun showParticles(frame: Bitmap?, r: ParticleDetector.Result, contentW: Int = frame?.width ?: 0, contentH: Int = frame?.height ?: 0) {
        this.frame = frame
        this.contentW = contentW
        this.contentH = contentH
        mask = null
        rope = null
        boxes = r.boxes
        invalidate()
    }

    fun clear() {
        frame = null; mask = null; rope = null; boxes = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (contentW <= 0 || contentH <= 0) return
        val s = min(width.toFloat() / contentW, height.toFloat() / contentH)
        canvas.save()
        canvas.translate((width - contentW * s) / 2, (height - contentH * s) / 2)
        canvas.scale(s, s)
        frame?.let { canvas.drawBitmap(it, 0f, 0f, null) }
        rope?.let { r ->
            // the results are in the picture (the frame turned by r.turn clockwise): turn them back onto the frame
            val back = Matrix()
            if (r.turn == 90) { back.postRotate(-90f); back.postTranslate(0f, contentH.toFloat()) }
            canvas.save()
            canvas.concat(back)
            mask?.let { canvas.drawBitmap(it, Matrix().apply { setScale(1 / r.scale, 1 / r.scale) }, maskPaint) }
            r.step?.let { step ->
                edgePaint.strokeWidth = 2f / s
                midPaint.strokeWidth = 2f / s
                canvas.drawPath(path(step.top), edgePaint)
                canvas.drawPath(path(step.bottom), edgePaint)
                canvas.drawPath(path(step.mid), midPaint)
                step.center?.let { canvas.drawCircle(it.x, it.y, 4f / s, pointPaint) }
            }
            canvas.restore()
        }
        boxPaint.strokeWidth = 2f / s
        for (b in boxes) canvas.drawRect(b.rect, boxPaint)
        canvas.restore()
    }

    private fun path(points: List<PointF>) = Path().apply {
        points.firstOrNull()?.let { moveTo(it.x, it.y) }
        for (p in points.drop(1)) lineTo(p.x, p.y)
    }
}
