import { computeTransform, PAD_VALUE, type Frame, type InputConfig, type Transform } from "@detect/core";

/** Uses the browser's native canvas scaling to place a video frame onto the model input canvas. */
export class FrameGrabber {
  private canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  private scaled = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : null;
  private scaledCtx = this.scaled?.getContext("2d");

  grab(source: HTMLVideoElement, input: InputConfig): { frame: Frame; transform: Transform } {
    const sw = source.videoWidth;
    const sh = source.videoHeight;
    if (!sw || !sh) throw new Error("Video has no frame yet");
    // Preserve CPU capture up to 1 MP; GPU downscaling saves full-frame copies on large videos.
    const fast = sw * sh > 1_000_000 && this.scaledCtx;
    const canvas = fast ? this.scaled! : this.canvas;
    const ctx = fast ? this.scaledCtx! : this.ctx;
    const { width, height } = input;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const t = computeTransform(sw, sh, input);
    const pad = input.padValue ?? PAD_VALUE;
    ctx.fillStyle = `rgb(${pad},${pad},${pad})`;
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(source, t.padX, t.padY, sw * t.scaleX, sh * t.scaleY);
    const img = ctx.getImageData(0, 0, width, height);
    return { frame: { width, height, data: img.data }, transform: t };
  }
}
