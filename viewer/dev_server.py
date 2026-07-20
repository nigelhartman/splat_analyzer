#!/usr/bin/env python3
"""Local static server for the viewer (no build step, no external bucket)."""
from __future__ import annotations

import http.server
import socketserver

PORT = 8000


class DevHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


if __name__ == "__main__":
    with ReusableTCPServer(("", PORT), DevHandler) as httpd:
        print(f"Serving on http://localhost:{PORT}/")
        httpd.serve_forever()
