import { useEffect, useRef } from "react";
import { rollFromGravity, smoothRoll, tiltColour } from "./framing";

type MotionAPI = typeof DeviceMotionEvent & { requestPermission?: () => Promise<PermissionState> };

/** Call synchronously from the Camera tap, before awaiting camera access. */
export async function requestMotionPermission(): Promise<boolean> {
  try {
    const motion = globalThis.DeviceMotionEvent as MotionAPI | undefined;
    if (!motion) return false;
    return !motion.requestPermission || await motion.requestPermission() === "granted";
  } catch {
    return false;
  }
}

/** Mounted only on the lamp camera path; sensor updates do not render React or touch inference. */
export function FramingGuide({ locked, motionAllowed }: { locked: boolean; motionAllowed: boolean }) {
  const levelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const level = levelRef.current!;
    level.hidden = true;
    const motion = globalThis.DeviceMotionEvent as MotionAPI | undefined;
    if (!motion || (motion.requestPermission && !motionAllowed)) return;
    let previous: number | null = null;
    let lastUpdate = -Infinity;
    let lastScreenAngle: number | undefined;
    const onMotion = (event: DeviceMotionEvent) => {
      const now = performance.now();
      if (now - lastUpdate < 100) return;
      lastUpdate = now;
      const gravity = event.accelerationIncludingGravity;
      if (!gravity || ![gravity.x, gravity.y, gravity.z].every((v) => typeof v === "number" && Number.isFinite(v))) {
        level.hidden = true;
        previous = null;
        return;
      }
      const screenAngle = screen.orientation?.angle ?? (window as Window & { orientation?: number }).orientation ?? 0;
      if (screenAngle !== lastScreenAngle) previous = null;
      lastScreenAngle = screenAngle;
      const roll = rollFromGravity(gravity, screenAngle);
      level.hidden = false;
      let colour = "unreadable";
      let text = "Tilt unavailable: point camera forward";
      if (roll === null) {
        previous = null;
      } else {
        previous = smoothRoll(previous, roll);
        const shown = Math.round(previous);
        colour = tiltColour(shown);
        text = `Phone tilt ${shown}\u00b0`;
      }
      if (level.dataset.colour !== colour) level.dataset.colour = colour;
      if (level.textContent !== text) level.textContent = text;
    };
    window.addEventListener("devicemotion", onMotion);
    return () => window.removeEventListener("devicemotion", onMotion);
  }, [motionAllowed]);

  return <>
    {!locked && <div className="framing-guide" aria-hidden="true">
      <div className="framing-lines" />
      <div className="framing-caption">Fit the lamp between the lines</div>
    </div>}
    <div ref={levelRef} className="phone-level" hidden title="Phone sensor level; the model measures the rope's tilt separately." />
  </>;
}
