type Gravity = { x: number | null; y: number | null; z: number | null };

const axisAngle = (angle: number) => ((angle + 270) % 180 + 180) % 180 - 90;

/** Roll relative to the displayed screen; a nearly flat phone cannot give a reliable level. */
export function rollFromGravity(gravity: Gravity | null, screenAngle = 0): number | null {
  if (!gravity) return null;
  const { x, y, z } = gravity;
  if (x === null || y === null || z === null || ![x, y, z].every(Number.isFinite)) return null;
  const total = Math.hypot(x, y, z);
  if (!total || Math.hypot(x, y) / total < 0.5) return null;
  const radians = screenAngle * Math.PI / 180;
  const screenX = x * Math.cos(radians) + y * Math.sin(radians);
  const screenY = -x * Math.sin(radians) + y * Math.cos(radians);
  return axisAngle(Math.atan2(screenX, screenY) * 180 / Math.PI);
}

export function tiltColour(angle: number): "green" | "amber" | "red" {
  return Math.abs(angle) <= 10 ? "green" : Math.abs(angle) <= 15 ? "amber" : "red";
}

/** Follow the nearest equivalent axis, avoiding a jump when upside-down roll wraps. */
export function smoothRoll(previous: number | null, next: number): number {
  return previous === null ? next : axisAngle(previous + 0.25 * axisAngle(next - previous));
}
