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

/** The analysed frame, fitted to the view, with the vortex rope or the particle boxes drawn on it. */
class OverlayView(context: Context) : View(context) {
    private var frame: Bitmap? = null
    private var mask: Bitmap? = null
    private var vortex: VortexDetector.Result? = null
    private var boxes: List<ParticleDetector.Box> = emptyList()
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; color = Color.rgb(255, 212, 0); strokeWidth = 2f }
    private val maskPaint = Paint().apply { isFilterBitmap = false }
    private val edgePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Color.rgb(0, 229, 255) }
    private val midPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; color = Color.rgb(255, 212, 0) }
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(255, 59, 59) }

    fun showVortex(frame: Bitmap, r: VortexDetector.Result) {
        val px = IntArray(VortexDetector.H * VortexDetector.W)
        r.rope?.let { rope -> for (i in px.indices) if (rope[i].toInt() == 1) px[i] = Color.argb(110, 0, 229, 255) }
        this.frame = frame
        mask = Bitmap.createBitmap(px, VortexDetector.W, VortexDetector.H, Bitmap.Config.ARGB_8888)
        vortex = r
        boxes = emptyList()
        invalidate()
    }

    fun showParticles(frame: Bitmap, r: ParticleDetector.Result) {
        this.frame = frame
        mask = null
        vortex = null
        boxes = r.boxes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val f = frame ?: return
        val s = min(width.toFloat() / f.width, height.toFloat() / f.height)
        canvas.save()
        canvas.translate((width - f.width * s) / 2, (height - f.height * s) / 2)
        canvas.scale(s, s)
        canvas.drawBitmap(f, 0f, 0f, null)
        vortex?.let { r ->
            // the results are in the picture (the frame turned by r.turn clockwise): turn them back onto the frame
            val back = Matrix()
            if (r.turn == 90) { back.postRotate(-90f); back.postTranslate(0f, f.height.toFloat()) }
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
