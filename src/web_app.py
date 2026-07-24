from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import dataclass
from datetime import date, datetime
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
from retrain_from_feedback import read_status, start_retraining_if_ready
from slack_notify import (
    SlackConfigurationError,
    SlackNotificationError,
    SlackNotificationResult,
    notify_prediction,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "web"
STORAGE_DIR = ROOT_DIR / "storage"
FEEDBACK_PATH = STORAGE_DIR / "prediction_feedback.csv"
HISTORY_EVENT_PATH = STORAGE_DIR / "history_events.csv"
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
    "その他・対象外",
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
    "review_required",
    "routing_status",
    "review_reasons",
    "corrected_category",
    "corrected_priority",
    "corrected_department",
    "note",
    "reviewer_id",
    "feedback_saved_at",
    "notion_page_id",
    "notion_sync_status",
    "notion_sync_error",
    "notion_synced_at",
    "slack_notification_status",
    "slack_notification_error",
    "slack_message_ts",
    "slack_notified_at",
    "deleted_at",
    "deleted_by",
    "delete_reason",
]
HISTORY_EVENT_FIELDS = [
    "event_id",
    "prediction_id",
    "event_type",
    "actor_id",
    "reason",
    "created_at",
]
HISTORY_SEARCH_FIELDS = [
    "prediction_id",
    "text",
    "note",
    "reviewer_id",
    "predicted_category",
    "predicted_priority",
    "predicted_department",
    "corrected_category",
    "corrected_priority",
    "corrected_department",
]
HISTORY_EXPORT_COLUMNS = [
    ("prediction_id", "予測ID"),
    ("created_at", "登録日時"),
    ("text", "問い合わせ文"),
    ("impact_scope", "影響範囲"),
    ("requester_role", "依頼者"),
    ("channel", "受付経路"),
    ("predicted_category", "予測カテゴリ"),
    ("predicted_priority", "予測優先度"),
    ("predicted_department", "予測担当部署"),
    ("category_confidence", "カテゴリ信頼度"),
    ("priority_confidence", "優先度信頼度"),
    ("department_confidence", "担当部署信頼度"),
    ("routing_status", "振り分け状態"),
    ("review_reasons", "要確認理由"),
    ("corrected_category", "修正カテゴリ"),
    ("corrected_priority", "修正優先度"),
    ("corrected_department", "修正担当部署"),
    ("reviewer_id", "確認者ID"),
    ("note", "修正理由"),
    ("feedback_saved_at", "修正日時"),
    ("notion_sync_status", "Notion同期状態"),
    ("slack_notification_status", "Slack通知状態"),
    ("deleted_at", "削除日時"),
    ("deleted_by", "削除者ID"),
    ("delete_reason", "削除理由"),
]
TARGET_TO_PREDICTED_FIELD = {
    "category": "predicted_category",
    "priority": "predicted_priority",
    "department": "predicted_department",
}
HISTORY_LOCK = RLock()


@dataclass(frozen=True)
class HistoryQuery:
    keyword: str = ""
    date_from: str = ""
    date_to: str = ""
    category: str = ""
    priority: str = ""
    department: str = ""
    routing_status: str = ""
    feedback_status: str = "all"
    notion_status: str = ""
    slack_status: str = ""
    deleted_status: str = "active"
    sort_order: str = "newest"
    page: int = 1
    page_size: int = 10


class HelpdeskTriageHandler(BaseHTTPRequestHandler):
    server_version = "HelpdeskTriageHTTP/1.0"

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        if parsed_url.path == "/api/retraining":
            self.write_json({"retraining": read_status().__dict__}, HTTPStatus.OK)
            return

        if parsed_url.path == "/api/history":
            query = parse_qs(parsed_url.query)
            history_query = parse_history_query(query)
            self.write_json(query_history(history_query), HTTPStatus.OK)
            return

        if parsed_url.path == "/api/history/export.csv":
            query = parse_qs(parsed_url.query)
            history_query = parse_history_query(query, paginate=False)
            self.write_csv(build_history_export(history_query))
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
        parsed_path = urlparse(self.path).path
        restore_prediction_id = history_action_prediction_id(
            parsed_path,
            suffix="/restore",
        )
        if restore_prediction_id:
            payload = self.read_json_body()
            self.validate_history_action_payload(payload, action="restore")
            restored_record = restore_history_record(
                restore_prediction_id,
                payload["actor_id"],
                payload["reason"],
            )
            self.write_json({"item": restored_record}, HTTPStatus.OK)
            return

        if parsed_path == "/api/feedback":
            payload = self.read_json_body()
            self.validate_feedback_payload(payload)
            updated_record = save_feedback(payload)
            notion_result = attempt_feedback_sync(updated_record)
            updated_record = save_notion_sync_result(
                updated_record["prediction_id"], notion_result
            )
            retraining_status = start_retraining_if_ready()
            self.write_json(
                {
                    "item": updated_record,
                    "notion_sync": notion_result.__dict__,
                    "retraining": retraining_status.__dict__,
                },
                HTTPStatus.OK,
            )
            return

        if parsed_path != "/api/predict":
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
        slack_result = attempt_slack_notification(history_record)
        history_record = save_slack_notification_result(
            history_record["prediction_id"], slack_result
        )
        self.write_json(
            {
                "prediction_id": history_record["prediction_id"],
                "input": payload,
                "predictions": [result.__dict__ for result in results],
                "routing": routing_result(results),
                "notion_sync": notion_result.__dict__,
                "slack_notification": slack_result.__dict__,
            },
            HTTPStatus.OK,
        )

    def do_DELETE(self) -> None:
        parsed_path = urlparse(self.path).path
        prediction_id = history_action_prediction_id(parsed_path)
        if not prediction_id:
            self.write_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        payload = self.read_json_body()
        self.validate_history_action_payload(payload, action="delete")
        deleted_record = delete_history_record(
            prediction_id,
            payload["actor_id"],
            payload["reason"],
        )
        self.write_json({"item": deleted_record}, HTTPStatus.OK)

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
            "reviewer_id",
        }
        missing_fields = sorted(required_fields - payload.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        prediction_id = validate_string_field("prediction_id", payload["prediction_id"])
        if not prediction_id:
            raise ValueError("prediction_id must not be empty.")
        payload["prediction_id"] = prediction_id
        payload["note"] = validate_string_field("note", payload["note"])
        payload["reviewer_id"] = validate_string_field(
            "reviewer_id", payload["reviewer_id"]
        )
        if not payload["note"]:
            raise ValueError("note must explain why the feedback is correct.")
        if not payload["reviewer_id"]:
            raise ValueError("reviewer_id must not be empty.")

        validate_choice("corrected_category", payload["corrected_category"], CATEGORY_CHOICES)
        validate_choice("corrected_priority", payload["corrected_priority"], PRIORITY_CHOICES)
        validate_choice("corrected_department", payload["corrected_department"], DEPARTMENT_CHOICES)

    def validate_history_action_payload(
        self,
        payload: dict[str, str],
        action: str,
    ) -> None:
        required_fields = {"actor_id", "reason"}
        missing_fields = sorted(required_fields - payload.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        payload["actor_id"] = validate_string_field("actor_id", payload["actor_id"])
        payload["reason"] = validate_string_field("reason", payload["reason"])
        if not payload["actor_id"]:
            raise ValueError("actor_id must not be empty.")
        if not payload["reason"]:
            raise ValueError(f"reason must explain the {action} action.")

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

    def write_csv(self, content: bytes) -> None:
        filename = f"helpdesk-history-{date.today().isoformat()}.csv"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{filename}"',
        )
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


def parse_history_query(
    query: dict[str, list[str]],
    paginate: bool = True,
) -> HistoryQuery:
    page_size = parse_history_limit(query) if "limit" in query else 10
    if "page_size" in query:
        page_size = parse_positive_query_integer(
            query,
            "page_size",
            maximum=50,
        )

    page = parse_positive_query_integer(query, "page", maximum=100000)
    if not paginate:
        page = 1

    history_query = HistoryQuery(
        keyword=query_value(query, "q"),
        date_from=query_value(query, "date_from"),
        date_to=query_value(query, "date_to"),
        category=query_value(query, "category"),
        priority=query_value(query, "priority"),
        department=query_value(query, "department"),
        routing_status=query_value(query, "routing_status"),
        feedback_status=query_value(query, "feedback_status") or "all",
        notion_status=query_value(query, "notion_status"),
        slack_status=query_value(query, "slack_status"),
        deleted_status=query_value(query, "deleted_status") or "active",
        sort_order=query_value(query, "sort_order") or "newest",
        page=page,
        page_size=page_size,
    )
    validate_history_query(history_query)
    return history_query


def query_value(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not values:
        return ""
    return values[0].strip()


def parse_positive_query_integer(
    query: dict[str, list[str]],
    name: str,
    maximum: int,
) -> int:
    raw_value = query_value(query, name)
    if not raw_value:
        return 1
    value = int(raw_value)
    if value < 1 or value > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")
    return value


def validate_history_query(history_query: HistoryQuery) -> None:
    validate_optional_choice("category", history_query.category, CATEGORY_CHOICES)
    validate_optional_choice("priority", history_query.priority, PRIORITY_CHOICES)
    validate_optional_choice(
        "department",
        history_query.department,
        DEPARTMENT_CHOICES,
    )
    validate_optional_choice(
        "routing_status",
        history_query.routing_status,
        ["自動振り分け", "要確認（一次受付）"],
    )
    validate_optional_choice(
        "feedback_status",
        history_query.feedback_status,
        ["all", "with_feedback", "without_feedback"],
    )
    validate_optional_choice(
        "notion_status",
        history_query.notion_status,
        ["synced", "disabled", "failed", "not_linked"],
    )
    validate_optional_choice(
        "slack_status",
        history_query.slack_status,
        ["sent", "skipped", "disabled", "failed"],
    )
    validate_optional_choice(
        "deleted_status",
        history_query.deleted_status,
        ["active", "deleted", "all"],
    )
    validate_optional_choice(
        "sort_order",
        history_query.sort_order,
        ["newest", "oldest"],
    )
    validate_history_dates(history_query.date_from, history_query.date_to)


def validate_optional_choice(
    field_name: str,
    value: str,
    choices: list[str],
) -> None:
    if value and value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}.")


def validate_history_dates(date_from: str, date_to: str) -> None:
    start_date = date.fromisoformat(date_from) if date_from else None
    end_date = date.fromisoformat(date_to) if date_to else None
    if start_date and end_date and start_date > end_date:
        raise ValueError("date_from must be on or before date_to.")


def read_history(limit: int) -> list[dict[str, str]]:
    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50.")
    return query_history(HistoryQuery(page_size=limit))["items"]


def query_history(history_query: HistoryQuery) -> dict[str, object]:
    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
    filtered_rows = filter_history_rows(rows, history_query)
    filtered_rows.sort(
        key=lambda row: row["created_at"],
        reverse=history_query.sort_order == "newest",
    )
    total = len(filtered_rows)
    total_pages = max(1, (total + history_query.page_size - 1) // history_query.page_size)
    start = (history_query.page - 1) * history_query.page_size
    end = start + history_query.page_size
    return {
        "items": filtered_rows[start:end],
        "total": total,
        "page": history_query.page,
        "page_size": history_query.page_size,
        "total_pages": total_pages,
    }


def filter_history_rows(
    rows: list[dict[str, str]],
    history_query: HistoryQuery,
) -> list[dict[str, str]]:
    keyword = history_query.keyword.casefold()
    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        is_deleted = bool(row["deleted_at"])
        if history_query.deleted_status == "active" and is_deleted:
            continue
        if history_query.deleted_status == "deleted" and not is_deleted:
            continue
        if keyword and not any(
            keyword in row[field].casefold() for field in HISTORY_SEARCH_FIELDS
        ):
            continue
        created_date = row["created_at"][:10]
        if history_query.date_from and created_date < history_query.date_from:
            continue
        if history_query.date_to and created_date > history_query.date_to:
            continue
        if history_query.category and effective_label(row, "category") != history_query.category:
            continue
        if history_query.priority and effective_label(row, "priority") != history_query.priority:
            continue
        if (
            history_query.department
            and effective_label(row, "department") != history_query.department
        ):
            continue
        if (
            history_query.routing_status
            and row["routing_status"] != history_query.routing_status
        ):
            continue
        has_feedback = bool(row["feedback_saved_at"])
        if history_query.feedback_status == "with_feedback" and not has_feedback:
            continue
        if history_query.feedback_status == "without_feedback" and has_feedback:
            continue
        if (
            history_query.notion_status
            and row["notion_sync_status"] != history_query.notion_status
        ):
            continue
        if (
            history_query.slack_status
            and row["slack_notification_status"] != history_query.slack_status
        ):
            continue
        filtered_rows.append(row)
    return filtered_rows


def effective_label(row: dict[str, str], target: str) -> str:
    corrected_label = row[f"corrected_{target}"]
    if corrected_label:
        return corrected_label
    return row[f"predicted_{target}"]


def history_action_prediction_id(path: str, suffix: str = "") -> str:
    prefix = "/api/history/"
    if not path.startswith(prefix):
        return ""
    prediction_id = path[len(prefix) :]
    if suffix:
        if not prediction_id.endswith(suffix):
            return ""
        prediction_id = prediction_id[: -len(suffix)]
    elif "/" in prediction_id:
        return ""
    prediction_id = unquote(prediction_id).strip()
    if not prediction_id or "/" in prediction_id:
        return ""
    return prediction_id


def delete_history_record(
    prediction_id: str,
    actor_id: str,
    reason: str,
) -> dict[str, str]:
    return change_history_deletion_state(
        prediction_id=prediction_id,
        actor_id=actor_id,
        reason=reason,
        event_type="deleted",
    )


def restore_history_record(
    prediction_id: str,
    actor_id: str,
    reason: str,
) -> dict[str, str]:
    return change_history_deletion_state(
        prediction_id=prediction_id,
        actor_id=actor_id,
        reason=reason,
        event_type="restored",
    )


def change_history_deletion_state(
    prediction_id: str,
    actor_id: str,
    reason: str,
    event_type: str,
) -> dict[str, str]:
    if event_type not in {"deleted", "restored"}:
        raise ValueError(f"Unsupported history event: {event_type}")

    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        updated_record: dict[str, str] | None = None
        for row in rows:
            if row["prediction_id"] != prediction_id:
                continue
            if event_type == "deleted":
                if row["deleted_at"]:
                    raise ValueError(f"prediction_id is already deleted: {prediction_id}")
                row["deleted_at"] = current_timestamp()
                row["deleted_by"] = actor_id
                row["delete_reason"] = reason
            else:
                if not row["deleted_at"]:
                    raise ValueError(f"prediction_id is not deleted: {prediction_id}")
                row["deleted_at"] = ""
                row["deleted_by"] = ""
                row["delete_reason"] = ""
            updated_record = row
            break

        if updated_record is None:
            raise ValueError(f"prediction_id not found: {prediction_id}")

        write_history(rows)
        append_history_event(
            prediction_id=prediction_id,
            event_type=event_type,
            actor_id=actor_id,
            reason=reason,
        )
        return updated_record


def append_history_event(
    prediction_id: str,
    event_type: str,
    actor_id: str,
    reason: str,
) -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not HISTORY_EVENT_PATH.exists()
    with HISTORY_EVENT_PATH.open("a", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HISTORY_EVENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "event_id": str(uuid.uuid4()),
                "prediction_id": prediction_id,
                "event_type": event_type,
                "actor_id": actor_id,
                "reason": reason,
                "created_at": current_timestamp(),
            }
        )


def build_history_export(history_query: HistoryQuery) -> bytes:
    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
    filtered_rows = filter_history_rows(rows, history_query)
    filtered_rows.sort(
        key=lambda row: row["created_at"],
        reverse=history_query.sort_order == "newest",
    )

    output = io.StringIO(newline="")
    headers = [header for _, header in HISTORY_EXPORT_COLUMNS]
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for row in filtered_rows:
        writer.writerow(
            {
                header: csv_safe_value(row[field])
                for field, header in HISTORY_EXPORT_COLUMNS
            }
        )
    return output.getvalue().encode("utf-8-sig")


def csv_safe_value(value: str) -> str:
    if value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


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
        routing = routing_result(results)
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
            "review_required": str(routing["review_required"]).lower(),
            "routing_status": str(routing["status"]),
            "review_reasons": " / ".join(routing["reasons"]),
            "corrected_category": "",
            "corrected_priority": "",
            "corrected_department": "",
            "note": "",
            "reviewer_id": "",
            "feedback_saved_at": "",
            "notion_page_id": "",
            "notion_sync_status": "",
            "notion_sync_error": "",
            "notion_synced_at": "",
            "slack_notification_status": "",
            "slack_notification_error": "",
            "slack_message_ts": "",
            "slack_notified_at": "",
            "deleted_at": "",
            "deleted_by": "",
            "delete_reason": "",
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
                if row["deleted_at"]:
                    raise ValueError(
                        f"prediction_id is deleted: {payload['prediction_id']}"
                    )
                row["corrected_category"] = payload["corrected_category"]
                row["corrected_priority"] = payload["corrected_priority"]
                row["corrected_department"] = payload["corrected_department"]
                row["note"] = payload["note"]
                row["reviewer_id"] = payload["reviewer_id"]
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


def attempt_slack_notification(record: dict[str, str]) -> SlackNotificationResult:
    try:
        return notify_prediction(record)
    except (SlackConfigurationError, SlackNotificationError) as error:
        return SlackNotificationResult(status="failed", error=str(error))


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


def save_slack_notification_result(
    prediction_id: str,
    result: SlackNotificationResult,
) -> dict[str, str]:
    with HISTORY_LOCK:
        ensure_history_file()
        with FEEDBACK_PATH.open("r", encoding="utf-8-sig", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        updated_record: dict[str, str] | None = None
        for row in rows:
            if row["prediction_id"] == prediction_id:
                row["slack_notification_status"] = result.status
                row["slack_notification_error"] = result.error
                row["slack_message_ts"] = result.message_ts
                if result.status == "sent":
                    row["slack_notified_at"] = current_timestamp()
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


def routing_result(results: list[object]) -> dict[str, object]:
    review_results = [result for result in results if result.requires_review]
    reasons = [
        (
            f"{result.target}: 信頼度{result.confidence:.1%}が"
            f"基準{result.threshold:.1%}未満"
        )
        for result in review_results
    ]
    return {
        "review_required": bool(review_results),
        "status": "要確認（一次受付）" if review_results else "自動振り分け",
        "reasons": reasons,
    }


def main() -> None:
    if not (WEB_DIR / "index.html").exists():
        raise FileNotFoundError("web/index.html does not exist.")

    ensure_history_file()
    server = ThreadingHTTPServer((HOST, PORT), HelpdeskTriageHandler)
    print(f"Serving Helpdesk Triage UI at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
