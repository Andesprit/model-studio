package com.andesprit.detectors

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.view.View
import kotlin.math.min

/** The analysed frame, fitted to the view, with the vortex masks or the particle boxes drawn on it. */
class OverlayView(context: Context) : View(context) {
    private var frame: Bitmap? = null
    private var mask: Bitmap? = null
    private var maskScale = 1f
    private var boxes: List<ParticleDetector.Box> = emptyList()
    private val boxPaint = Paint().apply { style = Paint.Style.STROKE; color = Color.rgb(255, 212, 0); strokeWidth = 2f }
    private val maskPaint = Paint().apply { isFilterBitmap = false }

    fun showVortex(frame: Bitmap, r: VortexDetector.Result) {
        val px = IntArray(VortexDetector.H * VortexDetector.W)
        for (i in px.indices) px[i] = when {
            r.rope?.get(i)?.toInt() == 1 -> Color.argb(150, 0, 229, 255)
            r.water?.get(i)?.toInt() == 1 -> Color.argb(50, 74, 222, 128)
            else -> 0
        }
        this.frame = frame
        mask = Bitmap.createBitmap(px, VortexDetector.W, VortexDetector.H, Bitmap.Config.ARGB_8888)
        maskScale = r.scale
        boxes = emptyList()
        invalidate()
    }

    fun showParticles(frame: Bitmap, r: ParticleDetector.Result) {
        this.frame = frame
        mask = null
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
        mask?.let { canvas.drawBitmap(it, Matrix().apply { setScale(1 / maskScale, 1 / maskScale) }, maskPaint) }
        boxPaint.strokeWidth = 2f / s
        for (b in boxes) canvas.drawRect(b.rect, boxPaint)
        canvas.restore()
    }
}
