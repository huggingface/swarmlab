// Sanity render of a swarmlab view.html (not part of pytest).
// usage: node screenshot.js RUN_DIR/view.html out.png [round] [--dark] [--reveal]
const { chromium } = require("playwright");
const path = require("path");

(async () => {
  const [page_, out, round = "1", ...flags] = process.argv.slice(2);
  if (!page_ || !out) {
    console.error("usage: node screenshot.js view.html out.png [round] [--dark] [--reveal]");
    process.exit(2);
  }
  const browser = await chromium.launch({
    executablePath: process.env.PLAYWRIGHT_BROWSERS_PATH +
      "/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell",
  });
  const page = await browser.newPage({
    viewport: { width: 1400, height: 1000 },
    colorScheme: flags.includes("--dark") ? "dark" : "light",
  });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("request", (r) => { if (!r.url().startsWith("file:")) errors.push("external request: " + r.url()); });
  await page.goto("file://" + path.resolve(page_) + "#r=" + round);
  if (flags.includes("--reveal")) await page.check("#reveal");
  await page.keyboard.press("ArrowRight");
  await page.keyboard.press("ArrowLeft");
  const label = await page.textContent("#roundLabel");
  await page.screenshot({ path: out, fullPage: true });
  await browser.close();
  console.log(JSON.stringify({ label, errors }));
  process.exit(errors.length ? 1 : 0);
})();
