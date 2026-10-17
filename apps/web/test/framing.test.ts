import { describe, expect, it } from "vitest";
import { rollFromGravity, smoothRoll, tiltColour } from "../src/ui/framing";

function gravity(roll: number, screenAngle = 0) {
  const radians = (roll - screenAngle) * Math.PI / 180;
  return { x: 9.81 * Math.sin(radians), y: 9.81 * Math.cos(radians), z: 0 };
}

describe("phone roll from gravity", () => {
  it.each([0, 90, 180, 270])("accounts for screen rotation %s", (screenAngle) => {
    for (const angle of [0, 12, -12, 25, -25]) {
      expect(rollFromGravity(gravity(angle, screenAngle), screenAngle)).toBeCloseTo(angle, 6);
    }
  });

  it("keeps the same axis when the phone is upside down", () => {
    expect(rollFromGravity(gravity(180))).toBeCloseTo(0);
    expect(rollFromGravity(gravity(192))).toBeCloseTo(12);
  });

  it.each([9.81, -9.81])("refuses mostly out-of-plane gravity, z=%s", (z) => {
    expect(rollFromGravity({ x: 0.12, y: 0.2, z })).toBeNull();
  });

  it("accepts a forward-facing camera but rejects a steep pitch", () => {
    expect(rollFromGravity({ x: 0, y: 8, z: 5 })).toBe(0);
    expect(rollFromGravity({ x: 0, y: 4, z: 9 })).toBeNull();
  });

  it("ignores absent, incomplete and invalid sensor readings", () => {
    for (const value of [null, { x: null, y: 9.81, z: 0 }, { x: 0, y: null, z: 0 }, { x: 0, y: 0, z: null }, { x: NaN, y: 1, z: 0 }, { x: 0, y: Infinity, z: 0 }, { x: 0, y: 0, z: 0 }]) {
      expect(rollFromGravity(value)).toBeNull();
    }
  });
});

describe("level colours", () => {
  it("uses the 10 and 15 degree boundaries on both sides", () => {
    for (const sign of [1, -1]) {
      expect(tiltColour(sign * 10)).toBe("green");
      expect(tiltColour(sign * 10.1)).toBe("amber");
      expect(tiltColour(sign * 15)).toBe("amber");
      expect(tiltColour(sign * 15.1)).toBe("red");
      expect(tiltColour(sign * 25)).toBe("red");
    }
  });
});

describe("roll smoothing", () => {
  it("starts immediately then damps sensor jitter", () => {
    expect(smoothRoll(null, 12)).toBe(12);
    expect(smoothRoll(0, 12)).toBeCloseTo(3);
    let angle = 0;
    for (let i = 0; i < 20; i++) angle = smoothRoll(angle, 25);
    expect(angle).toBeCloseTo(25, 0);
  });

  it("smooths across the upside-down wrap without a false upright reading", () => {
    expect(Math.abs(smoothRoll(89, -89))).toBeGreaterThan(88);
  });
});
