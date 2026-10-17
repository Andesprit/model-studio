export type Target = "lamp" | "vortex" | "particles";

const STORAGE_KEY = "model-studio.subject.v1";
const isTarget = (value: string | null): value is Target => value === "lamp" || value === "vortex" || value === "particles";

/** A shared link wins over the last subject chosen on this device. */
export function readInitialTarget(): Target | null {
  const url = new URL(location.href);
  const linked = url.searchParams.get("target");
  if (isTarget(linked)) return linked;
  if (linked !== null) {
    url.searchParams.delete("target");
    history.replaceState(history.state, "", url);
  }
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    return isTarget(saved) ? saved : null;
  } catch {
    return null;
  }
}

export function rememberTarget(target: Target) {
  try {
    localStorage.setItem(STORAGE_KEY, target);
  } catch { /* A shared link still works when storage is blocked. */ }
  const url = new URL(location.href);
  url.searchParams.set("target", target);
  history.replaceState(history.state, "", url);
}
