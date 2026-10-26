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
        VortexDetector(app).use { d ->
            for (image in listOf("lamp", "lamp_upright")) {
                val frame = test.assets.open("$image.png").use { BitmapFactory.decodeStream(it) }
                // the desktop's own input: the runtimes must agree (model studio's fp16 file, the app's fp32 network);
                // the app's resize (Android bilinear, not cv2's) moves a few boundary cells, hence the lower bar
                val desktop = test.assets.open("${image}_input.bin").readBytes().let { b ->
                    java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                        .let { f -> FloatArray(f.remaining()).also { f.get(it) } }
                }
                for ((input, bar) in listOf("desktop input" to desktop to 0.98, "app input" to d.preprocess(frame).first to 0.9)) {
                    for (mode in 1..2) {
                        val out = d.run(input.second, mode)
                        val want = test.assets.open("${image}_rope_mode$mode.bin").readBytes()
                        val rope = iou(ByteArray(out.size) { if (out[it] > 0f) 1 else 0 }, want)
                        Log.i("SmokeTest", "vortex $image, ${input.first}, mode $mode: rope IoU %.4f (${want.count { it.toInt() == 1 }} cells)".format(rope))
                        // an almost empty reference (the wrong branch): only a bar on the count, a few cells either way
                        if (want.count { it.toInt() == 1 } < 300) assertTrue("$image mode $mode", out.count { it > 0f } < 600)
                        else assertTrue("rope IoU $rope, $image, ${input.first}, mode $mode", rope > bar)
                    }
                }
                d.reset()
                val ms = (0 until 12).map { d.detect(frame, it / 30.0).networkMs }
                val r = d.detect(frame, 0.5)
                Log.i("SmokeTest", "vortex $image: ${r.orientation}, strength %.3f, thickness ${r.step?.metrics?.get("thickness")}, network median %.1f ms (both branches while deciding, one after)"
                    .format(r.strength, ms.sorted()[6]))
                // both end up vertical: the lying lamp's picture is landscape, so the app turns it upright first
                assertTrue("$image orientation ${r.orientation}, turn ${r.turn}", r.orientation == "vertical" && r.turn == (if (image == "lamp") 90 else 0))
            }
        }
    }

    @Test
    fun ropeMeasurements() {
        // a straight band 20 cells thick at rows 100-119, at scale 1 (a cell is a pixel)
        val h = VortexDetector.H; val w = VortexDetector.W
        val band = ByteArray(h * w) { if (it / w in 100..119) 1 else 0 }
        val lying = RopeAnalyzer().step(band, h, w, false, 1f, w, h, 0.0).metrics
        assertTrue("$lying", lying["thickness"] == "20.00 px" && lying["center"] == "110.00 px" && lying["present"] == "100.0 %")
        // the same band standing up (columns 100-119), measured as a vertical rope
        val upright = ByteArray(h * w) { if (it % w in 100..119) 1 else 0 }
        val standing = RopeAnalyzer().step(upright, h, w, true, 1f, w, h, 0.0).metrics
        assertTrue("$standing", standing["thickness"] == "20.00 px" && standing["center"] == "110.00 px" && standing["present"] == "100.0 %")
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
