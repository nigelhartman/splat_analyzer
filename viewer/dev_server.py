#!/usr/bin/env python3
"""Local static server. /r2/<key> is served from a cache filled via `wrangler r2 object get`."""
from __future__ import annotations

import http.server
import mimetypes
import socketserver
import subprocess
import sys
from pathlib import Path

PORT = 8000
BUCKET = "splat-analyzer"
CACHE_DIR = Path(__file__).resolve().parent / ".r2-cache"
VIEWER_DIR = Path(__file__).resolve().parent


def ensure_cached(key: str) -> Path | None:
    """Download key from remote R2 into .r2-cache/ if missing. Returns path or None."""
    if not key or "/" in key or ".." in key or key.startswith("."):
        return None

    CACHE_DIR.mkdir(exist_ok=True)
    dest = CACHE_DIR / key
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    print(f"[r2] downloading {key} …", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        result = subprocess.run(
            [
                "npx",
                "wrangler",
                "r2",
                "object",
                "get",
                f"{BUCKET}/{key}",
                "--remote",
                "--file",
                str(tmp),
            ],
            cwd=str(VIEWER_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, flush=True)
            if tmp.exists():
                tmp.unlink()
            return None
        tmp.replace(dest)
        print(f"[r2] cached {key} ({dest.stat().st_size} bytes)", flush=True)
        return dest
    except Exception as e:
        print(f"[r2] failed: {e}", file=sys.stderr, flush=True)
        if tmp.exists():
            tmp.unlink()
        return None


class DevHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith("/r2/"):
            self._serve_r2()
            return
        super().do_GET()

    def do_HEAD(self):
        if self.path.startswith("/r2/"):
            self._serve_r2(head_only=True)
            return
        super().do_HEAD()

    def _serve_r2(self, head_only=False):
        key = self.path[len("/r2/") :].split("?", 1)[0]
        path = ensure_cached(key)
        if path is None:
            self.send_error(404, f"R2 object not found: {key}")
            return

        ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
        if key.endswith(".json"):
            ctype = "application/json"
        elif key.endswith(".rad"):
            ctype = "application/octet-stream"

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        if not head_only:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 256)
                    if not chunk:
                        break
                    self.wfile.write(chunk)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), DevHandler) as httpd:
        print(f"Serving on http://localhost:{PORT}/")
        print(f"  /r2/* → remote R2 bucket '{BUCKET}' (cached in .r2-cache/)")
        httpd.serve_forever()
