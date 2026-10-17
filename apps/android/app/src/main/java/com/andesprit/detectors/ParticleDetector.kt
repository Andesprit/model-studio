package com.andesprit.detectors

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * The particle detector (particle_detector_int8.tflite), with the steps of particle_detector/run_model.py:
 * CLAHE on the lightness channel, letterbox to 1280 x 384 on grey 114 (RGB, 0-1, NCHW), the network
 * (cx, cy, w, h, score per anchor, as fractions of the input), score >= CONF, NMS at IOU, back to frame pixels.
 */
class ParticleDetector(context: Context, gpu: Boolean) : AutoCloseable {
    companion object {
        const val CONF = 0.20f
        const val IOU = 0.35f
        const val MAX_DET = 1000
        const val THREADS = 4
    }

    class Box(val rect: RectF, val score: Float)
    class Result(val boxes: List<Box>, val networkMs: Double, val shrunk: Boolean)

    private var delegate: GpuDelegate? = null
    private val interpreter: Interpreter
    /** "gpu" or "cpu": the GPU delegate falls back to the CPU when it cannot take the model. */
    val backend: String
    private val inH: Int
    private val inW: Int
    private val anchors: Int
    private val input: ByteBuffer
    private val output: ByteBuffer
    private val clahe = run {
        check(OpenCVLoader.initLocal()) { "OpenCV did not load" }
        Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    }

    init {
        val model = context.assets.openFd("particle_detector_int8.tflite").use { fd ->
            FileInputStream(fd.fileDescriptor).channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        var itp: Interpreter? = null
        if (gpu) {
            try {
                delegate = GpuDelegate()
                itp = Interpreter(model, Interpreter.Options().addDelegate(delegate))
            } catch (e: Throwable) {
                delegate?.close(); delegate = null
            }
        }
        backend = if (itp != null) "gpu" else "cpu"
        interpreter = itp ?: Interpreter(model, Interpreter.Options().setNumThreads(THREADS))
        val inT = interpreter.getInputTensor(0)
        val outT = interpreter.getOutputTensor(0)
        check(inT.dataType() == DataType.FLOAT32 && outT.dataType() == DataType.FLOAT32) { "expected float32 in and out" }
        inH = inT.shape()[2]; inW = inT.shape()[3] // NCHW
        anchors = outT.shape()[2] // (1, 5, anchors)
        input = ByteBuffer.allocateDirect(4 * 3 * inH * inW).order(ByteOrder.nativeOrder())
        output = ByteBuffer.allocateDirect(4 * 5 * anchors).order(ByteOrder.nativeOrder())
    }

    fun detect(frame: Bitmap): Result {
        val rgba = Mat()
        Utils.bitmapToMat(frame, rgba)
        val bgr = Mat(); Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        val lab = Mat(); Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
        val ch = ArrayList<Mat>(); Core.split(lab, ch)
        clahe.apply(ch[0], ch[0])
        Core.merge(ch, lab)
        Imgproc.cvtColor(lab, bgr, Imgproc.COLOR_Lab2BGR)

        // letterbox as Ultralytics pads it; Math.rint is Python's round (half to even)
        val scale = min(inH.toDouble() / bgr.rows(), inW.toDouble() / bgr.cols())
        val w = Math.rint(bgr.cols() * scale).toInt()
        val h = Math.rint(bgr.rows() * scale).toInt()
        val left = Math.rint((inW - w) / 2.0 - 0.1).toInt()
        val top = Math.rint((inH - h) / 2.0 - 0.1).toInt()
        val canvas = Mat(inH, inW, CvType.CV_8UC3, Scalar(114.0, 114.0, 114.0))
        val resized = Mat(); Imgproc.resize(bgr, resized, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        resized.copyTo(canvas.submat(Rect(left, top, w, h)))

        val bytes = ByteArray(inH * inW * 3)
        canvas.get(0, 0, bytes)
        input.rewind()
        val plane = inH * inW
        val fb = input.asFloatBuffer()
        for (c in 0 until 3) { // RGB planes from BGR pixels
            val src = 2 - c
            for (i in 0 until plane) fb.put(c * plane + i, (bytes[i * 3 + src].toInt() and 255) / 255f)
        }
        listOf(rgba, bgr, lab, canvas, resized).plus(ch).forEach { it.release() }

        output.rewind()
        val t0 = System.nanoTime()
        interpreter.run(input, output)
        val ms = (System.nanoTime() - t0) / 1e6
        output.rewind()
        val o = output.asFloatBuffer()

        val keep = ArrayList<Int>()
        for (a in 0 until anchors) if (o.get(4 * anchors + a) >= CONF) keep.add(a)
        val n = keep.size
        val x1 = FloatArray(n); val y1 = FloatArray(n); val x2 = FloatArray(n); val y2 = FloatArray(n); val sc = FloatArray(n)
        for ((j, a) in keep.withIndex()) {
            val cx = o.get(a) * inW; val cy = o.get(anchors + a) * inH
            val bw = o.get(2 * anchors + a) * inW; val bh = o.get(3 * anchors + a) * inH
            x1[j] = cx - bw / 2; y1[j] = cy - bh / 2; x2[j] = cx + bw / 2; y2[j] = cy + bh / 2; sc[j] = o.get(4 * anchors + a)
        }
        val boxes = nms(x1, y1, x2, y2, sc).map { i ->
            fun fx(v: Float) = ((v - left) / scale).toFloat().coerceIn(0f, frame.width.toFloat())
            fun fy(v: Float) = ((v - top) / scale).toFloat().coerceIn(0f, frame.height.toFloat())
            Box(RectF(fx(x1[i]), fy(y1[i]), fx(x2[i]), fy(y2[i])), sc[i])
        }
        return Result(boxes, ms, scale < 1)
    }

    /** Indices kept, best score first; equal scores in anchor order (stable sort), as run_model.nms. */
    private fun nms(x1: FloatArray, y1: FloatArray, x2: FloatArray, y2: FloatArray, sc: FloatArray): List<Int> {
        val order = sc.indices.sortedByDescending { sc[it] }.toMutableList() // sortedBy is stable
        val area = FloatArray(sc.size) { (x2[it] - x1[it]) * (y2[it] - y1[it]) }
        val keep = ArrayList<Int>()
        val removed = BooleanArray(sc.size)
        for (i in order) {
            if (removed[i]) continue
            keep.add(i)
            if (keep.size >= MAX_DET) break
            for (j in order) {
                if (removed[j] || j == i) continue
                val iw = max(0f, min(x2[i], x2[j]) - max(x1[i], x1[j]))
                val ih = max(0f, min(y2[i], y2[j]) - max(y1[i], y1[j]))
                val inter = iw * ih
                if (inter > IOU * (area[i] + area[j] - inter)) removed[j] = true
            }
            removed[i] = true
        }
        return keep
    }

    override fun close() {
        interpreter.close(); delegate?.close()
    }
}
