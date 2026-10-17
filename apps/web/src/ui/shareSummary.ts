interface ShareBrowser {
  share?: (data: { title: string; text: string }) => Promise<void>;
  clipboard?: { writeText(text: string): Promise<void> };
}

/** Invoke directly from the tap handler to retain Web Share's user activation. */
export async function shareSummary(data: { title: string; text: string }, browser: ShareBrowser = navigator): Promise<string> {
  if (browser.share) {
    try {
      await browser.share(data);
      return "";
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") return "";
    }
  }
  try {
    await browser.clipboard!.writeText(data.text);
    return "Copied";
  } catch {
    return "Could not share or copy. Use Download results (.json).";
  }
}
