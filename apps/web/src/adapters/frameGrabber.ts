import { computeTransform, PAD_VALUE, type Frame, type InputConfig, type Transform } from "@detect/core";

/** Uses the browser's native canvas scaling to place a video frame onto the model input canvas. */
export class FrameGrabber {
  private canvas: OffscreenCanvas | HTMLCanvasElement;
  private ctx: OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

  constructor() {
    // Let the GPU shrink large sources before reading back only the model-sized pixels.
    const canvas = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : null;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      this.canvas = canvas;
      this.ctx = ctx;
    } else {
      this.canvas = document.createElement("canvas");
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    }
  }

  grab(source: HTMLVideoElement, input: InputConfig): { frame: Frame; transform: Transform } {
    const sw = source.videoWidth;
    const sh = source.videoHeight;
    if (!sw || !sh) throw new Error("Video has no frame yet");
    const { width, height } = input;
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    const t = computeTransform(sw, sh, input);
    const pad = input.padValue ?? PAD_VALUE;
    this.ctx.fillStyle = `rgb(${pad},${pad},${pad})`;
    this.ctx.fillRect(0, 0, width, height);
    this.ctx.drawImage(source, t.padX, t.padY, sw * t.scaleX, sh * t.scaleY);
    const img = this.ctx.getImageData(0, 0, width, height);
    return { frame: { width, height, data: img.data }, transform: t };
  }
}
