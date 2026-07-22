from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from urllib.parse import parse_qs, unquote, urlparse

from notion_sync import (
    NotionConfigurationError,
    NotionSyncError,
    NotionSyncResult,
    sync_feedback_to_notion,
    sync_prediction_to_notion,
)
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
STORAGE_DIR = ROOT_DIR / "storage"
FEEDBACK_PATH = STORAGE_DIR / "prediction_feedback.csv"
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
CATEGORY_CHOICES = [
    "勤怠",
    "請求",
    "権限",
    "システム障害",
    "ネットワーク",
    "アカウント",
    "データ連携",
    "端末",
]
PRIORITY_CHOICES = ["High", "Middle", "Low"]
DEPARTMENT_CHOICES = ["総務", "経理", "情シス", "開発", "インフラ"]
HISTORY_FIELDS = [
    "prediction_id",
    "created_at",
    "text",
    "impact_scope",
    "requester_role",
    "channel",
    "predicted_category",
    "predicted_priority",
    "predicted_department",
    "category_confidence",
    "priority_confidence",
    "department_confidence",
    "corrected_category",
    "corrected_priority",
    "corrected_department",
    "note",
    "feedback_saved_at",
    "notion_page_id",
    "notion_sync_status",
    "notion_sync_error",
    "notion_synced_at",
]
TARGET_TO_PREDICTED_FIELD = {
    "category": "predicted_category",
    "priority": "predicted_priority",
    "department": "predicted_department",
}
HISTORY_LOCK = RLock()


class HelpdeskTriageHandler(BaseHTTPRequestHandler):
    server_version = "HelpdeskTriageHTTP/1.0"

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/api/history":
            query = parse_qs(parsed_url.query)
            limit = parse_history_limit(query)
            self.write_json({"items": read_history(limit)}, HTTPStatus.OK)
            return

        if parsed_url.path == "/" or parsed_url.path == "/index.html":
            self.write_static_file(WEB_DIR / "index.html")
            return

        requested_path = WEB_DIR / unquote(parsed_url.path.lstrip("/"))
        resolved_path = requested_path.resolve()
        if not resolved_path.is_relative_to(WEB_DIR.resolve()):
            self.write_json({"error": "Invalid path."}, HTTPStatus.BAD_REQUEST)
            return
        if not resolved_path.exists() or not resolved_path.is_file():
            self.write_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        self.write_static_file(resolved_path)

    def do_POST(self) -> None:
        if self.path == "/api/feedback":
            payload = self.read_json_body()
            self.validate_feedback_payload(payload)
            updated_record = save_feedback(payload)
            notion_result = attempt_feedback_sync(updated_record)
            updated_record = save_notion_sync_result(
                updated_record["prediction_id"], notion_result
            )
            self.write_json(
                {
                    "item": updated_record,
                    "notion_sync": notion_result.__dict__,
                },
                HTTPStatus.OK,
            )
            return

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
        history_record = save_prediction(payload, results)
        notion_result = attempt_prediction_sync(history_record)
        history_record = save_notion_sync_result(
            history_record["prediction_id"], notion_result
        )
        self.write_json(
            {
                "prediction_id": history_record["prediction_id"],
                "input": payload,
                "predictions": [result.__dict__ for result in results],
                "notion_sync": notion_result.__dict__,
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

        text = validate_string_field("text", payload["text"])
        if not text:
            raise ValueError("text must not be empty.")
        payload["text"] = text

        validate_choice("impact_scope", payload["impact_scope"], IMPACT_SCOPE_CHOICES)
        validate_choice("requester_role", payload["requester_role"], REQUESTER_ROLE_CHOICES)
        validate_choice("channel", payload["channel"], CHANNEL_CHOICES)

    def validate_feedback_payload(self, payload: dict[str, str]) -> None:
        required_fields = {
            "prediction_id",
            "corrected_category",
            "corrected_priority",
            "corrected_department",
            "note",
        }
        missing_fields = sorted(required_fields - payload.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        prediction_id = validate_string_field("prediction_id", payload["prediction_id"])
        if not prediction_id:
            raise ValueError("prediction_id must not be empty.")
        payload["prediction_id"] = prediction_id
        payload["note"] = validate_string_field("note", payload["note"])

        validate_choice("corrected_category", payload["corrected_category"], CATEGORY_CHOICES)
        validate_choice("corrected_priority", payload["corrected_priority"], PRIORITY_CHOICES)
        validate_choice("corrected_department", payload["corrected_department"], DEPARTMENT_CHOICES)

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
    validate_string_field(field_name, value)
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}.")


def validate_string_field(field_name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    return value.strip()


def parse_history_limit(query: dict[str, list[str]]) -> int:
    if "limit" not in query:
        return 10
    raw_limit = query["limit"][0]
    limit = int(raw_limit)
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    return limit


def read_history(limit: int) -> list[dict[str, str]]:
    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        rows.sort(key=lambda row: row["created_at"], reverse=True)
        return rows[:limit]


def save_prediction(payload: dict[str, str], results: list[object]) -> dict[str, str]:
    with HISTORY_LOCK:
        ensure_history_file()
        prediction_map = {
            TARGET_TO_PREDICTED_FIELD[result.target]: result.label
            for result in results
        }
        confidence_map = {
            f"{result.target}_confidence": f"{result.confidence:.6f}"
            for result in results
        }
        record = {
            "prediction_id": str(uuid.uuid4()),
            "created_at": current_timestamp(),
            "text": payload["text"],
            "impact_scope": payload["impact_scope"],
            "requester_role": payload["requester_role"],
            "channel": payload["channel"],
            "predicted_category": prediction_map["predicted_category"],
            "predicted_priority": prediction_map["predicted_priority"],
            "predicted_department": prediction_map["predicted_department"],
            "category_confidence": confidence_map["category_confidence"],
            "priority_confidence": confidence_map["priority_confidence"],
            "department_confidence": confidence_map["department_confidence"],
            "corrected_category": "",
            "corrected_priority": "",
            "corrected_department": "",
            "note": "",
            "feedback_saved_at": "",
            "notion_page_id": "",
            "notion_sync_status": "",
            "notion_sync_error": "",
            "notion_synced_at": "",
        }
        append_history_record(record)
        return record


def save_feedback(payload: dict[str, str]) -> dict[str, str]:
    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        updated_record: dict[str, str] | None = None
        for row in rows:
            if row["prediction_id"] == payload["prediction_id"]:
                row["corrected_category"] = payload["corrected_category"]
                row["corrected_priority"] = payload["corrected_priority"]
                row["corrected_department"] = payload["corrected_department"]
                row["note"] = payload["note"]
                row["feedback_saved_at"] = current_timestamp()
                updated_record = row
                break

        if updated_record is None:
            raise ValueError(f"prediction_id not found: {payload['prediction_id']}")

        write_history(rows)
        return updated_record


def attempt_prediction_sync(record: dict[str, str]) -> NotionSyncResult:
    try:
        return sync_prediction_to_notion(record)
    except (NotionConfigurationError, NotionSyncError) as error:
        return NotionSyncResult(status="failed", error=str(error))


def attempt_feedback_sync(record: dict[str, str]) -> NotionSyncResult:
    try:
        return sync_feedback_to_notion(record)
    except (NotionConfigurationError, NotionSyncError) as error:
        return NotionSyncResult(
            status="failed",
            page_id=record["notion_page_id"],
            error=str(error),
        )


def save_notion_sync_result(
    prediction_id: str,
    result: NotionSyncResult,
) -> dict[str, str]:
    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        updated_record: dict[str, str] | None = None
        for row in rows:
            if row["prediction_id"] == prediction_id:
                if result.page_id:
                    row["notion_page_id"] = result.page_id
                row["notion_sync_status"] = result.status
                row["notion_sync_error"] = result.error
                if result.status == "synced":
                    row["notion_synced_at"] = current_timestamp()
                updated_record = row
                break

        if updated_record is None:
            raise ValueError(f"prediction_id not found: {prediction_id}")

        write_history(rows)
        return updated_record


def ensure_history_file() -> None:
    with HISTORY_LOCK:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        if FEEDBACK_PATH.exists():
            migrate_history_schema()
            return
        write_history([])


def migrate_history_schema() -> None:
    with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        if reader.fieldnames == HISTORY_FIELDS:
            return
    write_history(rows)


def append_history_record(record: dict[str, str]) -> None:
    with HISTORY_LOCK:
        with FEEDBACK_PATH.open("a", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=HISTORY_FIELDS)
            writer.writerow(record)


def write_history(rows: list[dict[str, str]]) -> None:
    with HISTORY_LOCK:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=HISTORY_FIELDS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in HISTORY_FIELDS}
                for row in rows
            )


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    if not (WEB_DIR / "index.html").exists():
        raise FileNotFoundError("web/index.html does not exist.")

    server = ThreadingHTTPServer((HOST, PORT), HelpdeskTriageHandler)
    print(f"Serving Helpdesk Triage UI at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
