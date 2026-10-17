import { afterEach, describe, expect, it, vi } from "vitest";
import { requestMotionPermission } from "../src/ui/FramingGuide";

afterEach(() => vi.unstubAllGlobals());

describe("motion permission", () => {
  it("works without a motion API or without a permission prompt", async () => {
    vi.stubGlobal("DeviceMotionEvent", undefined);
    expect(await requestMotionPermission()).toBe(false);
    vi.stubGlobal("DeviceMotionEvent", class {});
    expect(await requestMotionPermission()).toBe(true);
  });

  it("calls the permission API synchronously, before the first await", async () => {
    const requestPermission = vi.fn(async () => "granted");
    vi.stubGlobal("DeviceMotionEvent", { requestPermission });
    const pending = requestMotionPermission();
    expect(requestPermission).toHaveBeenCalledOnce();
    expect(await pending).toBe(true);
  });

  it("treats denial as an unavailable level", async () => {
    vi.stubGlobal("DeviceMotionEvent", { requestPermission: async () => "denied" });
    expect(await requestMotionPermission()).toBe(false);
  });

  it("keeps camera use working when the sensor prompt fails", async () => {
    vi.stubGlobal("DeviceMotionEvent", { requestPermission: async () => { throw new Error("blocked"); } });
    expect(await requestMotionPermission()).toBe(false);
    vi.stubGlobal("DeviceMotionEvent", { requestPermission: () => { throw new Error("unsupported"); } });
    expect(await requestMotionPermission()).toBe(false);
  });
});
