package com.andesprit.detectors

import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.max
import kotlin.math.min

/**
 * Both models on one test image each, against what the desktop runtimes give (tools/make_reference.py).
 *   ./gradlew connectedDebugAndroidTest    (a phone over adb or an emulator)
 * Times go to logcat under the tag SmokeTest.
 */
@RunWith(AndroidJUnit4::class)
class SmokeTest {
    private val app = InstrumentationRegistry.getInstrumentation().targetContext
    private val test = InstrumentationRegistry.getInstrumentation().context
    private val ref = JSONObject(test.assets.open("reference.json").bufferedReader().readText())

    @Test
    fun vortexMatchesDesktop() {
        val frame = test.assets.open("lamp.png").use { BitmapFactory.decodeStream(it) }
        VortexDetector(app).use { d ->
            // the desktop's own input: the runtimes must agree; the app's resize (Android bilinear, not cv2's)
            // moves a few boundary cells of a small mask, hence the lower bar
            val desktop = test.assets.open("lamp_input.bin").readBytes().let { b ->
                java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    .let { f -> FloatArray(f.remaining()).also { f.get(it) } }
            }
            for ((input, bar) in listOf("desktop input" to desktop to 0.99, "app input" to d.preprocess(frame).first to 0.95)) {
                for (mode in 1..2) {
                    val out = d.run(input.second, mode)
                    val want = test.assets.open("lamp_rope_mode$mode.bin").readBytes()
                    val water = test.assets.open("lamp_water_mode$mode.bin").readBytes()
                    val rope = iou(ByteArray(out.size) { if (out[it] > 0f) 1 else 0 }, want)
                    val w = iou(ByteArray(out.size) { if (out[it] != VortexDetector.OUTSIDE) 1 else 0 }, water)
                    Log.i("SmokeTest", "vortex ${input.first}, mode $mode: rope IoU %.4f, water IoU %.4f".format(rope, w))
                    assertTrue("rope IoU $rope, ${input.first}, mode $mode", rope > bar)
                    assertTrue("water IoU $w, ${input.first}, mode $mode", w > bar)
                }
            }
            val ms = (0 until 10).map { d.detect(frame).networkMs }.sorted()
            Log.i("SmokeTest", "vortex network median %.1f ms".format(ms[5]))
        }
    }

    @Test
    fun particlesMatchDesktop() {
        val frame = test.assets.open("particles.png").use { BitmapFactory.decodeStream(it) }
        val refBoxes = ref.getJSONArray("particles").let { a -> (0 until a.length()).map { a.getJSONArray(it) } }
        for (gpu in listOf(false, true)) {
            ParticleDetector(app, gpu).use { d ->
                val r = d.detect(frame)
                // one-to-one greedy matches at IoU 0.5, as an F1 against the fp32 desktop boxes
                val used = BooleanArray(refBoxes.size)
                var hits = 0
                for (b in r.boxes) {
                    var best = -1; var bestIou = 0.5f
                    for ((j, q) in refBoxes.withIndex()) {
                        if (used[j]) continue
                        val iw = max(0f, min(b.rect.right, q.getDouble(2).toFloat()) - max(b.rect.left, q.getDouble(0).toFloat()))
                        val ih = max(0f, min(b.rect.bottom, q.getDouble(3).toFloat()) - max(b.rect.top, q.getDouble(1).toFloat()))
                        val inter = iw * ih
                        val ua = b.rect.width() * b.rect.height() +
                            (q.getDouble(2) - q.getDouble(0)).toFloat() * (q.getDouble(3) - q.getDouble(1)).toFloat() - inter
                        if (inter / ua >= bestIou) { bestIou = inter / ua; best = j }
                    }
                    if (best >= 0) { used[best] = true; hits++ }
                }
                val f1 = 2.0 * hits / (r.boxes.size + refBoxes.size)
                val ms = (0 until 5).map { d.detect(frame).networkMs }.sorted()
                Log.i("SmokeTest", "particles ${d.backend}: ${r.boxes.size} boxes vs ${refBoxes.size}, F1 %.3f, network median %.0f ms"
                    .format(f1, ms[2]))
                assertTrue("particle F1 $f1 on ${d.backend}", f1 > 0.85)
            }
        }
    }

    private fun iou(a: ByteArray, b: ByteArray): Double {
        var inter = 0; var union = 0
        for (i in a.indices) {
            val x = a[i].toInt() == 1; val y = b[i].toInt() == 1
            if (x && y) inter++
            if (x || y) union++
        }
        return if (union == 0) 1.0 else inter.toDouble() / union
    }
}
