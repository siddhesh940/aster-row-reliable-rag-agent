"""Capture the web UI demo GIF from the REAL running application.

Launches headless Edge via CDP, drives the demo autoplay (?demo=1) which
sends real messages through the real backend, waits for each state to
finish (window.__demoDone), and screenshots. Frames are assembled into
docs/demo.gif with reading-paced durations (2-4 minutes total).

Prereqs: websocket-client (already present). Run:
    python web/server.py --port 8125     # in another shell
    python scripts/capture_web_demo.py --url http://127.0.0.1:8125
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]

FRAMES = [
    {"name": "welcome", "url": "/", "expect_msgs": 0},
    {"name": "kb-citation", "url": "/?demo=1&steps=1", "expect_msgs": 2},
    {"name": "order-card", "url": "/?demo=1&steps=2", "expect_msgs": 4},
    {"name": "multiturn", "url": "/?demo=1&steps=3", "expect_msgs": 6},
    {"name": "grounded-refusal", "url": "/?demo=1&steps=4", "expect_msgs": 8},
    {"name": "conflict", "url": "/?demo=1&steps=5", "expect_msgs": 10},
    {"name": "evaluation", "url": "/?demo=eval", "expect_msgs": 0},
]


def find_browser() -> str:
    for path in EDGE_CANDIDATES:
        if Path(path).is_file():
            return path
    raise RuntimeError("Edge/Chrome not found")


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self._id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method,
                                 "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                return msg.get("result", {})

    def evaluate(self, expression: str):
        res = self.call("Runtime.evaluate",
                        {"expression": expression, "returnByValue": True})
        return res.get("result", {}).get("value")

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def wait_for_debugger(port: int, deadline_s: float = 20.0) -> list[dict]:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json", timeout=2
            ) as res:
                targets = json.loads(res.read().decode("utf-8"))
            pages = [t for t in targets if t.get("type") == "page"]
            if pages:
                return pages
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("browser debugger endpoint never came up")


def capture_frame(cdp: CDP, url: str, expect_msgs: int, out: Path,
                  max_wait: float = 40.0) -> None:
    cdp.call("Page.navigate", {"url": url})
    time.sleep(0.6)
    end = time.time() + max_wait
    while time.time() < end:
        ready = cdp.evaluate(
            "window.__demoDone === true || "
            "(window.__demoDone === undefined && "
            f"document.querySelectorAll('.msg').length >= {expect_msgs})"
        )
        if ready:
            break
        time.sleep(0.4)
    else:
        raise RuntimeError(f"timeout waiting for {url}")
    time.sleep(0.8)  # let animations settle
    shot = cdp.call("Page.captureScreenshot", {"format": "png"})
    out.write_bytes(base64.b64decode(shot["data"]))
    print(f"captured {out.name} ({out.stat().st_size // 1024} KB)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=9333,
                        help="CDP debugging port")
    parser.add_argument("--out", default=str(ROOT / "docs" / "demo.gif"))
    args = parser.parse_args()

    browser = find_browser()
    tmp_profile = tempfile.mkdtemp(prefix="ar-capture-")
    proc = subprocess.Popen([
        browser,
        f"--remote-debugging-port={args.port}",
        "--remote-allow-origins=*",
        f"--user-data-dir={tmp_profile}",
        "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=1", "--window-size=1180,860",
        "--no-first-run", "--no-default-browser-check",
        "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames: list[Image.Image] = []
    try:
        targets = wait_for_debugger(args.port)
        page = next(t for t in targets
                    if t.get("url", "").startswith(("about", args.url)))
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")

        tmp_dir = Path(tempfile.mkdtemp(prefix="ar-frames-"))
        for spec in FRAMES:
            out = tmp_dir / f"{spec['name']}.png"
            capture_frame(cdp, args.url + spec["url"],
                          spec["expect_msgs"], out)
            frames.append(Image.open(out).convert("RGB"))
        cdp.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # Reading-paced durations: ~45 ms per rendered pixel-row of content is
    # overkill; use fixed per-frame durations scaled to hit 2-4 minutes.
    n = len(frames)
    durations = [6000] + [26000] * (n - 2) + [14000]
    total = sum(durations)
    if not 125_000 <= total <= 235_000:
        target = 165_000
        durations = [int(d * target / total) for d in durations]

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    print(f"Wrote {out_path} ({n} frames, {sum(durations)/1000:.0f}s runtime, "
          f"{out_path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
