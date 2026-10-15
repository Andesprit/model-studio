// Example decoder-only pipeline. Load it with "Pipeline .js", then load yolov8n.onnx.
// A pipeline is an ES module whose default export is { name, defaults?, decode?(outputs, ctx), create?(ctx) }.
//   outputs: { [outputName]: { dims: number[], data: Float32Array } }
//   ctx:     { config, transform, options, labels, utils: { nms, sigmoid, softmax, toFrameRect, toFramePoint } }
// decode must return a Result: { boxes?, masks?, segmentation?, poses?, skeleton?, classes? } in frame pixels.
export default {
  name: "people-counter",
  defaults: { scoreThreshold: 0.4, iouThreshold: 0.5 },
  decode(outputs, ctx) {
    const pred = Object.values(outputs)[0]; // YOLOv8: [1, 84, 8400]
    const [, channels, anchors] = pred.dims;
    const d = pred.data;
    const boxes = [];
    for (let i = 0; i < anchors; i++) {
      const score = d[4 * anchors + i]; // class 0 = person
      if (score < ctx.options.scoreThreshold) continue;
      const cx = d[i], cy = d[anchors + i], w = d[2 * anchors + i], h = d[3 * anchors + i];
      const rect = ctx.utils.toFrameRect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, ctx.transform);
      boxes.push({ ...rect, classId: 0, score, label: "person" });
    }
    const kept = ctx.utils.nms(boxes, ctx.options.iouThreshold);
    return {
      boxes: kept,
      classes: [{ classId: 0, score: 1, label: `${kept.length} people (from ${channels - 4} classes)` }],
    };
  },
};
