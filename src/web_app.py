from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from predict import (
    CHANNEL_CHOICES,
    IMPACT_SCOPE_CHOICES,
    REQUESTER_ROLE_CHOICES,
    TARGETS,
    build_model_text,
    predict_target,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
HOST = "127.0.0.1"
PORT = 8000
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}
REQUIRED_FIELDS = {
    "text",
    "impact_scope",
    "requester_role",
    "channel",
}


class HelpdeskTriageHandler(BaseHTTPRequestHandler):
    server_version = "HelpdeskTriageHTTP/1.0"

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.write_static_file(WEB_DIR / "index.html")
            return

        requested_path = WEB_DIR / unquote(self.path.lstrip("/"))
        resolved_path = requested_path.resolve()
        if not resolved_path.is_relative_to(WEB_DIR.resolve()):
            self.write_json({"error": "Invalid path."}, HTTPStatus.BAD_REQUEST)
            return
        if not resolved_path.exists() or not resolved_path.is_file():
            self.write_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        self.write_static_file(resolved_path)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.write_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        payload = self.read_json_body()
        self.validate_payload(payload)
        model_text = build_model_text(
            inquiry_text=payload["text"],
            impact_scope=payload["impact_scope"],
            requester_role=payload["requester_role"],
            channel=payload["channel"],
        )
        results = [
            predict_target(target=target, model_text=model_text, top_k=3)
            for target in TARGETS
        ]
        self.write_json(
            {
                "input": payload,
                "predictions": [result.__dict__ for result in results],
            },
            HTTPStatus.OK,
        )

    def read_json_body(self) -> dict[str, str]:
        content_length = self.headers["Content-Length"]
        raw_body = self.rfile.read(int(content_length))
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def validate_payload(self, payload: dict[str, str]) -> None:
        missing_fields = sorted(REQUIRED_FIELDS - payload.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        text = payload["text"].strip()
        if not text:
            raise ValueError("text must not be empty.")
        payload["text"] = text

        validate_choice("impact_scope", payload["impact_scope"], IMPACT_SCOPE_CHOICES)
        validate_choice("requester_role", payload["requester_role"], REQUESTER_ROLE_CHOICES)
        validate_choice("channel", payload["channel"], CHANNEL_CHOICES)

    def write_static_file(self, path: Path) -> None:
        content = path.read_bytes()
        content_type = CONTENT_TYPES[path.suffix]
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def write_json(self, body: dict[str, object], status: HTTPStatus) -> None:
        content = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (json.JSONDecodeError, KeyError, ValueError) as error:
            self.write_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as error:
            self.write_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        return


def validate_choice(field_name: str, value: str, choices: list[str]) -> None:
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}.")


def main() -> None:
    if not (WEB_DIR / "index.html").exists():
        raise FileNotFoundError("web/index.html does not exist.")

    server = ThreadingHTTPServer((HOST, PORT), HelpdeskTriageHandler)
    print(f"Serving Helpdesk Triage UI at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
