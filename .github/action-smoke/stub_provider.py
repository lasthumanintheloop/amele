#!/usr/bin/env python3
"""Loopback OpenAI-compatible stub for the `action` CI job.

The job runs the repository's own composite action (`uses: ./`) so action.yml
cannot rot while CI stays green. Reaching a real provider would need a secret
and network, so this stands in for one: it answers /v1/chat/completions with a
deterministic completion built from the request, which lets the workflow assert
that the action's `config`, `task` and `model` inputs all reached `amele run`
and that its stdout came back as the `answer` output.

Usage: stub_provider.py [port]   (default 8099, bound to 127.0.0.1 only)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Echo format: "<model>|<last user message>". Both halves come from the
# request, so a broken argv in action.yml shows up as a failed assertion
# rather than as a still-green job.
ANSWER = "{model}|{task}"


class Handler(BaseHTTPRequestHandler):
    """Answers exactly one route; anything else is a 404."""

    def do_POST(self):  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        if self.path != "/v1/chat/completions":
            self.send_error(404, "unexpected path")
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length) or b"{}")
        task = ""
        for message in req.get("messages", []):
            if message.get("role") == "user":
                task = message.get("content", "")
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": ANSWER.format(
                                model=req.get("model", ""), task=task
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - readiness probe for the workflow
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args):
        """Silence per-request logging; the workflow log is the product."""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099  # ci.yml ile aynı kalmalı
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
