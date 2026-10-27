"""Local-only generated research workbench. Run with python -m strokeweave.app."""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .workflow import LIMITS, RECIPES, WorkflowStore, canonical, plan_task, tool_cards

WEB = Path(__file__).with_name("web")


def make_server(store: WorkflowStore, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_):
            pass

        def respond(self, status, value, kind="application/json; charset=utf-8", name=None):
            body = value if isinstance(value, bytes) else canonical(value)
            self.send_response(status)
            headers = {"Content-Type": kind, "Content-Length": str(len(body)), "Cache-Control": "no-store",
                       "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                       "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"}
            if name:
                headers["Content-Disposition"] = 'attachment; filename="' + name + '"'
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def error(self, status, message):
            self.respond(status, {"error": message, "patient_execution": False})

        def local_request(self):
            hosts = {"127.0.0.1:" + str(self.server.server_port), "localhost:" + str(self.server.server_port)}
            if self.headers.get("Host") not in hosts:
                self.error(403, "Invalid local host")
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + h for h in hosts}:
                self.error(403, "Cross-origin access disabled")
                return False
            target = urlsplit(self.path)
            if target.scheme or target.netloc or target.query or target.fragment:
                self.error(400, "Only exact local endpoints are supported")
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            try:
                path = urlsplit(self.path).path
                static = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if path in static:
                    name, kind = static[path]
                    self.respond(200, (WEB / name).read_bytes(), kind)
                elif path == "/api/health":
                    self.respond(200, {"name": "StrokeWeave", "lineage": "TACTIC", "mode": "generated-only",
                                       "patient_execution": False, "training_enabled": False,
                                       "protected_labels_sealed": True, "natural_language": "rule_based_router_not_LLM",
                                       "worker_running": bool(store.thread and store.thread.is_alive()),
                                       "clinical_ready": False})
                elif path == "/api/tools":
                    self.respond(200, {"tools": tool_cards(), "recipes": RECIPES, "limits": LIMITS})
                elif path == "/api/jobs":
                    self.respond(200, {"jobs": store.listing()})
                else:
                    match = re.fullmatch(r"/api/jobs/([a-f0-9]{32})(?:/(report\.json|report\.html|export))?", path)
                    if match is None:
                        self.error(404, "Unknown or disallowed endpoint")
                        return
                    job, action = match.groups()
                    if action == "report.html":
                        self.respond(200, store.report_html(job), "text/html; charset=utf-8", "strokeweave-audit.html")
                    elif action == "report.json":
                        self.respond(200, store.report(job), name="strokeweave-audit.json")
                    elif action == "export":
                        self.respond(200, store.export(job), "application/zip", "strokeweave-audit.zip")
                    else:
                        self.respond(200, store.detail(job))
            except KeyError:
                self.error(404, "Unknown job")
            except Exception:
                self.error(500, "Integrity or storage error; original records retained")

        def do_POST(self):
            if not self.local_request():
                return
            if self.path not in ("/api/plans", "/api/batches"):
                self.error(404, "Unknown endpoint")
                return
            if (self.headers.get("X-StrokeWeave-Request") != "local-research-v1"
                    or self.headers.get("Content-Type", "").split(";")[0] != "application/json"):
                self.error(403, "Explicit local JSON header required")
                return
            if self.headers.get("Transfer-Encoding"):
                self.error(400, "Transfer encoding not supported")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= LIMITS["request_bytes"]:
                    self.error(413, "Request size limit")
                    return
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete body")
                value = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
                if self.path == "/api/plans":
                    if not isinstance(value, dict) or set(value) != {"text"}:
                        raise ValueError("Only task text is accepted")
                    self.respond(200, plan_task(value["text"]))
                else:
                    result = store.submit(value)
                    self.respond(200 if result["reused"] else 201, result)
            except FileExistsError as exc:
                self.error(409, str(exc))
            except (ValueError, TypeError, KeyError) as exc:
                self.error(400, str(exc))
            except Exception:
                self.error(500, "Storage error; original records retained")

        def do_OPTIONS(self):
            self.error(405, "CORS is disabled")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", "--state-dir", dest="runtime", type=Path, default=Path("runtime"))
    parser.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1",
                        help="Loopback binding only; remote access is intentionally disabled")
    parser.add_argument("--port", type=int, default=8801)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    store = WorkflowStore(args.runtime)
    server = make_server(store, args.port)
    store.start()
    print(f"StrokeWeave research preview: http://127.0.0.1:{server.server_port}/ (generated inputs only)", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
