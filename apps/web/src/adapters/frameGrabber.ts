import { computeTransform, PAD_VALUE, type Frame, type InputConfig, type Transform } from "@detect/core";

/** Uses the browser's native canvas scaling to place a video frame onto the model input canvas. */
export class FrameGrabber {
  private canvas = document.createElement("canvas");
  private ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
  private scaled = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : null;
  private scaledCtx = this.scaled?.getContext("2d");

  /**
   * `turn` (degrees, clockwise) turns the picture before it is placed: results are then in the turned picture's
   * coordinates (frameWidth x frameHeight of the turned picture) and the renderer turns them back.
   */
  grab(source: HTMLVideoElement, input: InputConfig, turn = 0): { frame: Frame; transform: Transform } {
    const sw = source.videoWidth;
    const sh = source.videoHeight;
    if (!sw || !sh) throw new Error("Video has no frame yet");
    if (turn) return this.grabTurned(source, input, sw, sh, turn);
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

  private grabTurned(source: HTMLVideoElement, input: InputConfig, sw: number, sh: number, turn: number) {
    const { fw, fh } = turnedSize(sw, sh, turn);
    const { width, height } = input;
    const canvas = this.canvas;
    const ctx = this.ctx;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const t = computeTransform(fw, fh, input);
    const pad = input.padValue ?? PAD_VALUE;
    ctx.fillStyle = `rgb(${pad},${pad},${pad})`;
    ctx.fillRect(0, 0, width, height);
    ctx.save();
    ctx.translate(t.padX + (fw * t.scaleX) / 2, t.padY + (fh * t.scaleY) / 2);
    ctx.rotate((turn * Math.PI) / 180);
    ctx.drawImage(source, (-sw * t.scaleX) / 2, (-sh * t.scaleY) / 2, sw * t.scaleX, sh * t.scaleY);
    ctx.restore();
    const img = ctx.getImageData(0, 0, width, height);
    return { frame: { width, height, data: img.data }, transform: t };
  }
}

/** Size of the bounding box of a sw x sh picture turned by `turn` degrees. */
export function turnedSize(sw: number, sh: number, turn: number) {
  const r = (turn * Math.PI) / 180;
  const c = Math.abs(Math.cos(r));
  const s = Math.abs(Math.sin(r));
  return { fw: Math.round(sw * c + sh * s), fh: Math.round(sw * s + sh * c) };
}
