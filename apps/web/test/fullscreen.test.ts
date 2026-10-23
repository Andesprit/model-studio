import { describe, expect, it, vi } from "vitest";
import { enterFullscreen, exitFullscreen, FULLSCREEN_TIMEOUT_MS, fullscreenElement } from "../src/ui/fullscreen";

const element = (request?: () => Promise<void>) => ({ requestFullscreen: request }) as unknown as Element;

describe("stage full screen", () => {
  it("uses the browser's full screen when it is allowed", async () => {
    const request = vi.fn(async () => {});
    expect(await enterFullscreen(element(request), { fullscreenEnabled: true } as never)).toBe("native");
    expect(request).toHaveBeenCalledOnce();
  });

  it("fills the page where only a bare video may go full screen (iPhone Safari)", async () => {
    const request = vi.fn(async () => {});
    expect(await enterFullscreen(element(request), { fullscreenEnabled: false } as never)).toBe("page");
    expect(await enterFullscreen(element(), {} as never)).toBe("page");
    expect(request).not.toHaveBeenCalled();
  });

  it("fills the page when the browser refuses", async () => {
    const request = vi.fn(async () => { throw new TypeError("not allowed"); });
    expect(await enterFullscreen(element(request), { fullscreenEnabled: true } as never)).toBe("page");
  });

  it("fills the page when the browser never answers", async () => {
    vi.useFakeTimers();
    try {
      const mode = enterFullscreen(element(() => new Promise(() => {})), { fullscreenEnabled: true } as never);
      await vi.advanceTimersByTimeAsync(FULLSCREEN_TIMEOUT_MS);
      expect(await mode).toBe("page");
    } finally {
      vi.useRealTimers();
    }
  });

  it("uses the prefixed API of older Safari", async () => {
    const webkitRequestFullscreen = vi.fn();
    const el = { webkitRequestFullscreen } as unknown as Element;
    expect(await enterFullscreen(el, { webkitFullscreenEnabled: true } as never)).toBe("native");
    const webkitExitFullscreen = vi.fn();
    const doc = { webkitFullscreenElement: el, webkitExitFullscreen } as never;
    expect(fullscreenElement(doc)).toBe(el);
    await exitFullscreen(doc);
    expect(webkitExitFullscreen).toHaveBeenCalledOnce();
  });

  it("exits only when something is in full screen", async () => {
    const exit = vi.fn(async () => {});
    await exitFullscreen({ fullscreenElement: null, exitFullscreen: exit } as never);
    expect(exit).not.toHaveBeenCalled();
  });
});
