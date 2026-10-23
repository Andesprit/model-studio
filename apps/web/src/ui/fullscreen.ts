/**
 * Full screen for the stage (the video and the overlay drawn over it). The video's own full-screen button shows
 * the bare video without the overlay, so the app puts the stage's container in full screen instead. iPhone Safari
 * only allows a bare video in full screen: there the stage fills the page ("page"), which keeps the overlay.
 */
export type FullscreenMode = "native" | "page";

type FullscreenElement = Element & { webkitRequestFullscreen?: () => Promise<void> | void };
type FullscreenDocument = Pick<Document, "fullscreenElement" | "exitFullscreen"> & {
  fullscreenEnabled?: boolean;
  webkitFullscreenEnabled?: boolean;
  webkitFullscreenElement?: Element | null;
  webkitExitFullscreen?: () => Promise<void> | void;
};

/** A full-screen request with no answer by then is treated as refused (some embedded browsers never answer). */
export const FULLSCREEN_TIMEOUT_MS = 1500;

/** Puts `el` in full screen when the browser allows it, else asks the caller to fill the page. */
export async function enterFullscreen(el: FullscreenElement, doc: FullscreenDocument = document): Promise<FullscreenMode> {
  const enabled = doc.fullscreenEnabled ?? doc.webkitFullscreenEnabled ?? false;
  const request = el.requestFullscreen ?? el.webkitRequestFullscreen;
  if (!enabled || !request) return "page";
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const answered = await Promise.race([
      Promise.resolve(request.call(el)).then(() => true),
      new Promise<false>((resolve) => { timer = setTimeout(() => resolve(false), FULLSCREEN_TIMEOUT_MS); }),
    ]);
    return answered ? "native" : "page";
  } catch {
    return "page";
  } finally {
    clearTimeout(timer);
  }
}

export async function exitFullscreen(doc: FullscreenDocument = document): Promise<void> {
  if (!fullscreenElement(doc)) return;
  const exit = doc.exitFullscreen ?? doc.webkitExitFullscreen;
  await exit?.call(doc);
}

export function fullscreenElement(doc: FullscreenDocument = document): Element | null {
  return doc.fullscreenElement ?? doc.webkitFullscreenElement ?? null;
}
