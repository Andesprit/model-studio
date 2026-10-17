import { afterEach, describe, expect, it, vi } from "vitest";
import { FrameGrabber } from "../src/adapters/frameGrabber";
import { DEFAULT_INPUT } from "@detect/core";

afterEach(() => vi.unstubAllGlobals());

function canvas() {
  const data = new Uint8ClampedArray(16);
  const ctx = { fillStyle: "", fillRect: vi.fn(), drawImage: vi.fn(), getImageData: vi.fn(() => ({ data })) };
  return { width: 1, height: 1, getContext: vi.fn(() => ctx), ctx, data };
}

describe("frame grab canvas", () => {
  it("uses OffscreenCanvas 2D without the frequent-read flag and preserves resize and padding", () => {
    const off = canvas();
    vi.stubGlobal("OffscreenCanvas", class { constructor() { return off; } });
    const createElement = vi.fn(() => canvas());
    vi.stubGlobal("document", { createElement });
    const input = { ...DEFAULT_INPUT, width: 4, height: 4, padValue: 10 };
    const source = { videoWidth: 8, videoHeight: 4 } as HTMLVideoElement;
    const result = new FrameGrabber().grab(source, input);
    expect(off.getContext).toHaveBeenCalledWith("2d");
    expect(createElement).not.toHaveBeenCalled();
    expect([off.width, off.height]).toEqual([4, 4]);
    expect(off.ctx.fillStyle).toBe("rgb(10,10,10)");
    expect(off.ctx.fillRect).toHaveBeenCalledWith(0, 0, 4, 4);
    expect(off.ctx.drawImage).toHaveBeenCalledWith(source, 0, 1, 4, 2);
    expect(result.frame.data).toBe(off.data);
    expect(result.transform).toMatchObject({ scaleX: 0.5, scaleY: 0.5, padX: 0, padY: 1 });
  });

  it.each(["missing", "no 2D context"])("keeps the HTML canvas path when OffscreenCanvas is %s", (reason) => {
    const fallback = canvas();
    vi.stubGlobal("OffscreenCanvas", reason === "missing" ? undefined : class { getContext() { return null; } });
    const createElement = vi.fn(() => fallback);
    vi.stubGlobal("document", { createElement });
    const result = new FrameGrabber().grab({ videoWidth: 8, videoHeight: 4 } as HTMLVideoElement, { ...DEFAULT_INPUT, width: 4, height: 4 });
    expect(createElement).toHaveBeenCalledWith("canvas");
    expect(fallback.getContext).toHaveBeenCalledWith("2d", { willReadFrequently: true });
    expect(result.frame.data).toBe(fallback.data);
  });
});
