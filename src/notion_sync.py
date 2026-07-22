from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


NOTION_API_BASE_URL = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"
NOTION_TIMEOUT_SECONDS = 10
RICH_TEXT_CHUNK_SIZE = 2_000
RICH_TEXT_MAX_CHUNKS = 100

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
IMPACT_SCOPE_CHOICES = ["個人", "部署", "複数部署", "全社"]
REQUESTER_ROLE_CHOICES = ["社員", "管理者", "経理担当", "開発者"]
CHANNEL_CHOICES = ["Slack", "問い合わせフォーム", "メール", "電話"]

REQUIRED_SCHEMA = {
    "問い合わせ文": {"rich_text": {}},
    "カテゴリ": {"select": {"options": [{"name": value} for value in CATEGORY_CHOICES]}},
    "優先度": {"select": {"options": [{"name": value} for value in PRIORITY_CHOICES]}},
    "担当部署": {"select": {"options": [{"name": value} for value in DEPARTMENT_CHOICES]}},
    "影響範囲": {"select": {"options": [{"name": value} for value in IMPACT_SCOPE_CHOICES]}},
    "依頼者": {"select": {"options": [{"name": value} for value in REQUESTER_ROLE_CHOICES]}},
    "受付経路": {"select": {"options": [{"name": value} for value in CHANNEL_CHOICES]}},
    "予測ID": {"rich_text": {}},
    "登録日時": {"date": {}},
    "カテゴリ信頼度": {"number": {"format": "percent"}},
    "優先度信頼度": {"number": {"format": "percent"}},
    "担当部署信頼度": {"number": {"format": "percent"}},
    "修正カテゴリ": {"select": {"options": [{"name": value} for value in CATEGORY_CHOICES]}},
    "修正優先度": {"select": {"options": [{"name": value} for value in PRIORITY_CHOICES]}},
    "修正担当部署": {"select": {"options": [{"name": value} for value in DEPARTMENT_CHOICES]}},
    "フィードバックメモ": {"rich_text": {}},
    "修正日時": {"date": {}},
}


class NotionConfigurationError(ValueError):
    pass


class NotionSyncError(RuntimeError):
    pass


@dataclass(frozen=True)
class NotionConfig:
    api_token: str
    database_id: str
    data_source_id: str

    @classmethod
    def from_environment(cls) -> NotionConfig | None:
        api_token = os.environ.get("NOTION_API_TOKEN", "").strip()
        database_id = os.environ.get("NOTION_DATABASE_ID", "").strip()
        data_source_id = os.environ.get("NOTION_DATA_SOURCE_ID", "").strip()

        if not api_token and not database_id and not data_source_id:
            return None
        if not api_token:
            raise NotionConfigurationError("NOTION_API_TOKEN is required for Notion sync.")
        if not database_id and not data_source_id:
            raise NotionConfigurationError(
                "NOTION_DATABASE_ID or NOTION_DATA_SOURCE_ID is required for Notion sync."
            )

        return cls(
            api_token=api_token,
            database_id=normalize_notion_id(database_id) if database_id else "",
            data_source_id=normalize_notion_id(data_source_id) if data_source_id else "",
        )


@dataclass(frozen=True)
class NotionSyncResult:
    status: str
    page_id: str = ""
    error: str = ""


class NotionClient:
    def __init__(
        self,
        api_token: str,
        timeout_seconds: int = NOTION_TIMEOUT_SECONDS,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        self.api_token = api_token
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        request_body = None
        if payload is not None:
            request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        request = Request(
            url=f"{NOTION_API_BASE_URL}{path}",
            data=request_body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Notion-Version": NOTION_API_VERSION,
                "Content-Type": "application/json",
            },
        )

        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")
            raise NotionSyncError(format_http_error(error.code, error_body)) from error
        except URLError as error:
            raise NotionSyncError(f"Notion API connection failed: {error.reason}") from error

        decoded = json.loads(response_body)
        if not isinstance(decoded, dict):
            raise NotionSyncError("Notion API returned an unexpected response format.")
        return decoded


class NotionSyncService:
    def __init__(self, config: NotionConfig, client: NotionClient | None = None) -> None:
        self.config = config
        self.client = client or NotionClient(config.api_token)
        self.data_source_id = config.data_source_id
        self.title_property_name = ""
        self.schema_ready = False

    def create_prediction(self, record: dict[str, str]) -> str:
        self.ensure_schema()
        response = self.client.request(
            "POST",
            "/pages",
            {
                "parent": {
                    "type": "data_source_id",
                    "data_source_id": self.data_source_id,
                },
                "properties": prediction_properties(record, self.title_property_name),
            },
        )
        page_id = response.get("id")
        if not isinstance(page_id, str) or not page_id:
            raise NotionSyncError("Notion API did not return a page ID.")
        return page_id

    def update_feedback(self, page_id: str, record: dict[str, str]) -> None:
        self.ensure_schema()
        self.client.request(
            "PATCH",
            f"/pages/{normalize_notion_id(page_id)}",
            {"properties": feedback_properties(record)},
        )

    def ensure_schema(self) -> None:
        if self.schema_ready:
            return
        if not self.data_source_id:
            self.data_source_id = self.resolve_data_source_id()

        data_source = self.client.request("GET", f"/data_sources/{self.data_source_id}")
        properties = data_source.get("properties")
        if not isinstance(properties, dict):
            raise NotionSyncError("Notion data source properties are missing.")

        title_properties = [
            name
            for name, value in properties.items()
            if isinstance(value, dict) and value.get("type") == "title"
        ]
        if len(title_properties) != 1:
            raise NotionSyncError("Notion data source must have exactly one title property.")
        self.title_property_name = title_properties[0]

        missing_properties: dict[str, object] = {}
        for name, specification in REQUIRED_SCHEMA.items():
            existing = properties.get(name)
            expected_type = next(iter(specification))
            if existing is None:
                missing_properties[name] = specification
                continue
            if not isinstance(existing, dict) or existing.get("type") != expected_type:
                raise NotionSyncError(
                    f"Notion property '{name}' must use the '{expected_type}' type."
                )

        if missing_properties:
            self.client.request(
                "PATCH",
                f"/data_sources/{self.data_source_id}",
                {"properties": missing_properties},
            )
        self.schema_ready = True

    def resolve_data_source_id(self) -> str:
        database = self.client.request("GET", f"/databases/{self.config.database_id}")
        data_sources = database.get("data_sources")
        if not isinstance(data_sources, list) or not data_sources:
            raise NotionSyncError("No data source was found in the Notion database.")
        if len(data_sources) != 1:
            raise NotionSyncError(
                "Multiple data sources were found. Set NOTION_DATA_SOURCE_ID explicitly."
            )
        data_source_id = data_sources[0].get("id")
        if not isinstance(data_source_id, str) or not data_source_id:
            raise NotionSyncError("The Notion data source ID is missing.")
        return normalize_notion_id(data_source_id)


SERVICE_LOCK = RLock()
SERVICE_CACHE: tuple[NotionConfig, NotionSyncService] | None = None


def sync_prediction_to_notion(record: dict[str, str]) -> NotionSyncResult:
    service = get_service()
    if service is None:
        return NotionSyncResult(status="disabled")
    page_id = service.create_prediction(record)
    return NotionSyncResult(status="synced", page_id=page_id)


def sync_feedback_to_notion(record: dict[str, str]) -> NotionSyncResult:
    service = get_service()
    if service is None:
        return NotionSyncResult(status="disabled")
    page_id = record["notion_page_id"]
    if not page_id:
        return NotionSyncResult(
            status="not_linked",
            error="The prediction has no linked Notion page.",
        )
    service.update_feedback(page_id, record)
    return NotionSyncResult(status="synced", page_id=page_id)


def get_service() -> NotionSyncService | None:
    config = NotionConfig.from_environment()
    if config is None:
        return None

    global SERVICE_CACHE
    with SERVICE_LOCK:
        if SERVICE_CACHE is None or SERVICE_CACHE[0] != config:
            SERVICE_CACHE = (config, NotionSyncService(config))
        return SERVICE_CACHE[1]


def prediction_properties(record: dict[str, str], title_property_name: str) -> dict[str, object]:
    title = record["text"][:120]
    return {
        title_property_name: title_value(title),
        "問い合わせ文": rich_text_value(record["text"]),
        "カテゴリ": select_value(record["predicted_category"]),
        "優先度": select_value(record["predicted_priority"]),
        "担当部署": select_value(record["predicted_department"]),
        "影響範囲": select_value(record["impact_scope"]),
        "依頼者": select_value(record["requester_role"]),
        "受付経路": select_value(record["channel"]),
        "予測ID": rich_text_value(record["prediction_id"]),
        "登録日時": date_value(record["created_at"]),
        "カテゴリ信頼度": number_value(record["category_confidence"]),
        "優先度信頼度": number_value(record["priority_confidence"]),
        "担当部署信頼度": number_value(record["department_confidence"]),
    }


def feedback_properties(record: dict[str, str]) -> dict[str, object]:
    return {
        "修正カテゴリ": select_value(record["corrected_category"]),
        "修正優先度": select_value(record["corrected_priority"]),
        "修正担当部署": select_value(record["corrected_department"]),
        "フィードバックメモ": rich_text_value(record["note"]),
        "修正日時": date_value(record["feedback_saved_at"]),
    }


def title_value(value: str) -> dict[str, object]:
    return {"title": rich_text_objects(value)}


def rich_text_value(value: str) -> dict[str, object]:
    return {"rich_text": rich_text_objects(value)}


def rich_text_objects(value: str) -> list[dict[str, object]]:
    chunks = [
        value[index : index + RICH_TEXT_CHUNK_SIZE]
        for index in range(0, len(value), RICH_TEXT_CHUNK_SIZE)
    ]
    if len(chunks) > RICH_TEXT_MAX_CHUNKS:
        raise NotionSyncError("Text is too long for a Notion rich-text property.")
    return [
        {"type": "text", "text": {"content": chunk}}
        for chunk in chunks
    ]


def select_value(value: str) -> dict[str, object]:
    return {"select": {"name": value}}


def number_value(value: str) -> dict[str, object]:
    return {"number": float(value)}


def date_value(value: str) -> dict[str, object]:
    return {"date": {"start": value}}


def normalize_notion_id(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise NotionConfigurationError(f"Invalid Notion ID: {value}") from error


def format_http_error(status_code: int, response_body: str) -> str:
    message = response_body
    try:
        decoded = json.loads(response_body)
        if isinstance(decoded, dict) and isinstance(decoded.get("message"), str):
            message = decoded["message"]
    except json.JSONDecodeError:
        pass
    return f"Notion API returned HTTP {status_code}: {message[:500]}"
