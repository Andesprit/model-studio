import { afterEach, describe, expect, it, vi } from "vitest";
import { readInitialTarget, rememberTarget } from "../src/subjects";

const KEY = "model-studio.subject.v1";

function browser(url = "https://example.com/model-studio/", saved: string | null = null) {
  const storage = { getItem: vi.fn(() => saved), setItem: vi.fn() };
  const history = { state: { existing: true }, replaceState: vi.fn() };
  vi.stubGlobal("location", { href: url });
  vi.stubGlobal("localStorage", storage);
  vi.stubGlobal("history", history);
  return { storage, history };
}

afterEach(() => vi.unstubAllGlobals());

describe("initial subject", () => {
  it("leaves a fresh visit unselected", () => {
    browser();
    expect(readInitialTarget()).toBeNull();
  });

  it.each(["lamp", "vortex", "particles"])("accepts a %s link before storage", (target) => {
    const { storage, history } = browser(`https://example.com/model-studio/?target=${target}`, "vortex");
    expect(readInitialTarget()).toBe(target);
    expect(storage.getItem).not.toHaveBeenCalled();
    expect(history.replaceState).not.toHaveBeenCalled();
  });

  it.each(["lamp", "vortex", "particles"])("remembers %s without a link", (target) => {
    const { storage } = browser(undefined, target);
    expect(readInitialTarget()).toBe(target);
    expect(storage.getItem).toHaveBeenCalledWith(KEY);
  });

  it("ignores an invalid link and uses the remembered subject", () => {
    const { history } = browser("https://example.com/model-studio/?other=value&target=invalid#video", "lamp");
    expect(readInitialTarget()).toBe("lamp");
    expect(history.replaceState).toHaveBeenCalledWith(history.state, "", new URL("https://example.com/model-studio/?other=value#video"));
  });

  it.each(["bogus", "", "LAMP", "../lamp"])("removes ignored target %s even when storage is blocked", (target) => {
    const { storage, history } = browser(`https://example.com/model-studio/?target=${target}&other=value#video`);
    storage.getItem.mockImplementation(() => { throw new Error("blocked"); });
    expect(readInitialTarget()).toBeNull();
    expect(history.replaceState).toHaveBeenCalledWith(history.state, "", new URL("https://example.com/model-studio/?other=value#video"));
  });

  it.each(["", "invalid", "LAMP", "../lamp"])("ignores invalid stored value %s", (saved) => {
    browser("https://example.com/model-studio/?target=invalid", saved);
    expect(readInitialTarget()).toBeNull();
  });

  it("works when storage reads throw", () => {
    const { storage } = browser();
    storage.getItem.mockImplementation(() => { throw new Error("blocked"); });
    expect(readInitialTarget()).toBeNull();
  });

  it("works when storage is unavailable", () => {
    browser();
    vi.stubGlobal("localStorage", undefined);
    expect(readInitialTarget()).toBeNull();
  });
});

describe("remembering a subject", () => {
  it("preserves the URL prefix, other parameters, hash and history state", () => {
    const { storage, history } = browser("https://example.com/model-studio/?other=value&target=vortex#video");
    rememberTarget("lamp");
    expect(storage.setItem).toHaveBeenCalledWith(KEY, "lamp");
    expect(history.replaceState).toHaveBeenCalledWith(history.state, "", new URL("https://example.com/model-studio/?other=value&target=lamp#video"));
  });

  it("still updates the link when saving to storage throws", () => {
    const { storage, history } = browser();
    storage.setItem.mockImplementation(() => { throw new Error("blocked"); });
    expect(() => rememberTarget("particles")).not.toThrow();
    expect(history.replaceState).toHaveBeenCalledWith(history.state, "", new URL("https://example.com/model-studio/?target=particles"));
  });
});
