from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import pandas as pd

from predict import MODEL_ACCESS_LOCK, load_target
from train_baseline import (
    DATA_PATH,
    MODEL_DIR,
    RANDOM_SEED,
    TARGETS,
    LabelEncoder,
    SoftmaxLogisticRegression,
    TfidfVectorizer,
    build_text,
    classification_report,
    save_model_artifacts,
    stratified_split,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT_DIR / "storage" / "prediction_feedback.csv"
STATE_PATH = ROOT_DIR / "storage" / "retraining_state.json"
STATUS_PATH = ROOT_DIR / "storage" / "retraining_status.json"
REPORT_ROOT = ROOT_DIR / "reports" / "retraining"
MIN_PENDING_FEEDBACK = 10
MAX_MACRO_F1_REGRESSION = 0.0
RETRAINING_LOCK = Lock()
REQUIRED_FEEDBACK_FIELDS = {
    "prediction_id",
    "text",
    "impact_scope",
    "requester_role",
    "channel",
    "corrected_category",
    "corrected_priority",
    "corrected_department",
    "feedback_saved_at",
}


@dataclass
class RetrainingStatus:
    status: str
    complete_feedback_count: int
    pending_feedback_count: int
    threshold: int
    message: str
    model_version: str = ""
    started_at: str = ""
    completed_at: str = ""
    metrics: dict[str, object] | None = None
    error: str = ""


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_complete_feedback(path: Path | None = None) -> list[dict[str, str]]:
    if path is None:
        path = FEEDBACK_PATH
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Feedback CSV has no header: {path}")
        missing_fields = sorted(REQUIRED_FEEDBACK_FIELDS - set(reader.fieldnames))
        if missing_fields:
            raise ValueError(f"Feedback CSV is missing fields: {missing_fields}")
        rows = list(reader)

    complete_rows = [
        row
        for row in rows
        if row["corrected_category"]
        and row["corrected_priority"]
        and row["corrected_department"]
        and row["feedback_saved_at"]
    ]
    prediction_ids = [row["prediction_id"] for row in complete_rows]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Feedback CSV contains duplicate prediction_id values.")
    return sorted(complete_rows, key=lambda row: row["prediction_id"])


def load_processed_versions(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = STATE_PATH
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    versions = state["processed_feedback_versions"]
    if not isinstance(versions, dict):
        raise ValueError("processed_feedback_versions must be an object.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in versions.items()):
        raise ValueError("processed_feedback_versions must contain string pairs.")
    return versions


def pending_feedback_rows(
    rows: list[dict[str, str]],
    processed_versions: dict[str, str],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["prediction_id"] not in processed_versions
        or processed_versions[row["prediction_id"]] != feedback_version(row)
    ]


def feedback_version(row: dict[str, str]) -> str:
    version_fields = [
        "text",
        "impact_scope",
        "requester_role",
        "channel",
        "corrected_category",
        "corrected_priority",
        "corrected_department",
        "feedback_saved_at",
    ]
    payload = {field: row[field] for field in version_fields}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def feedback_to_dataframe(rows: list[dict[str, str]]) -> pd.DataFrame:
    records = [
        {
            "ticket_id": f"feedback-{row['prediction_id']}",
            "inquiry_text": row["text"],
            "impact_scope": row["impact_scope"],
            "requester_role": row["requester_role"],
            "channel": row["channel"],
            "label_category": row["corrected_category"],
            "label_priority": row["corrected_priority"],
            "label_department": row["corrected_department"],
        }
        for row in rows
    ]
    data = pd.DataFrame.from_records(records)
    data["model_text"] = data.apply(build_text, axis=1)
    return data


def evaluate_artifacts(
    test_data: pd.DataFrame,
    target_name: str,
    target_column: str,
    model_dir: Path,
) -> dict[str, object]:
    vectorizer, model, classes = load_target(target_name, model_dir=model_dir)
    x_test = vectorizer.transform(test_data["model_text"].tolist())
    predicted_indexes = model.predict(x_test)
    y_pred = [classes[index] for index in predicted_indexes]
    y_true = test_data[target_column].tolist()
    report = classification_report(y_true, y_pred)
    if target_name == "priority":
        report["high_recall"] = report["classes"]["High"]["recall"]
    return report


def train_candidate_target(
    train_data: pd.DataFrame,
    target_name: str,
    target_column: str,
    model_dir: Path,
) -> None:
    vectorizer = TfidfVectorizer(min_n=1, max_n=4)
    x_train = vectorizer.fit_transform(train_data["model_text"].tolist())
    label_encoder = LabelEncoder().fit(train_data[target_column].tolist())
    y_train = label_encoder.transform(train_data[target_column].tolist())
    if label_encoder.classes_ is None:
        raise RuntimeError("Label encoder is not fitted.")

    model = SoftmaxLogisticRegression()
    model.fit(x_train, y_train, class_count=len(label_encoder.classes_))
    save_model_artifacts(
        target_name,
        target_column,
        vectorizer,
        label_encoder,
        model,
        model_dir=model_dir,
    )


def build_candidate(
    feedback_rows: list[dict[str, str]],
    candidate_dir: Path,
) -> dict[str, object]:
    base_data = pd.read_csv(DATA_PATH, encoding="utf-8")
    base_data["model_text"] = base_data.apply(build_text, axis=1)
    feedback_data = feedback_to_dataframe(feedback_rows)
    metrics: dict[str, object] = {}

    for target_name, target_column in TARGETS.items():
        invalid_labels = sorted(
            set(feedback_data[target_column]) - set(base_data[target_column])
        )
        if invalid_labels:
            raise ValueError(
                f"Unknown feedback labels for {target_name}: {invalid_labels}"
            )
        train_indexes, test_indexes = stratified_split(base_data[target_column].tolist())
        base_train = base_data.iloc[train_indexes].copy()
        test_data = base_data.iloc[test_indexes].copy()
        candidate_train = pd.concat([base_train, feedback_data], ignore_index=True)

        train_candidate_target(
            candidate_train,
            target_name,
            target_column,
            candidate_dir,
        )
        current_report = evaluate_artifacts(
            test_data,
            target_name,
            target_column,
            MODEL_DIR,
        )
        candidate_report = evaluate_artifacts(
            test_data,
            target_name,
            target_column,
            candidate_dir,
        )
        metrics[target_name] = {
            "current_macro_f1": current_report["macro_f1"],
            "candidate_macro_f1": candidate_report["macro_f1"],
            "current_accuracy": current_report["accuracy"],
            "candidate_accuracy": candidate_report["accuracy"],
        }
        if target_name == "priority":
            metrics[target_name]["current_high_recall"] = current_report["high_recall"]
            metrics[target_name]["candidate_high_recall"] = candidate_report["high_recall"]

    metrics["random_seed"] = RANDOM_SEED
    metrics["feedback_count"] = len(feedback_rows)
    return metrics


def candidate_passes(metrics: dict[str, object]) -> bool:
    for target_name in TARGETS:
        target_metrics = metrics[target_name]
        if not isinstance(target_metrics, dict):
            raise ValueError(f"Invalid metrics for target: {target_name}")
        current_macro_f1 = float(target_metrics["current_macro_f1"])
        candidate_macro_f1 = float(target_metrics["candidate_macro_f1"])
        if candidate_macro_f1 < current_macro_f1 - MAX_MACRO_F1_REGRESSION:
            return False

    priority_metrics = metrics["priority"]
    if not isinstance(priority_metrics, dict):
        raise ValueError("Invalid priority metrics.")
    return float(priority_metrics["candidate_high_recall"]) >= float(
        priority_metrics["current_high_recall"]
    )


def promote_candidate(candidate_dir: Path, model_version: str) -> None:
    archive_root = MODEL_DIR.parent / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / model_version
    if archive_dir.exists():
        raise FileExistsError(f"Archive directory already exists: {archive_dir}")

    with MODEL_ACCESS_LOCK:
        MODEL_DIR.rename(archive_dir)
        try:
            candidate_dir.rename(MODEL_DIR)
        except Exception:
            archive_dir.rename(MODEL_DIR)
            raise


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def write_status(status: RetrainingStatus, path: Path | None = None) -> None:
    if path is None:
        path = STATUS_PATH
    write_json_atomic(path, asdict(status))


def read_status() -> RetrainingStatus:
    complete_rows = load_complete_feedback()
    processed_versions = load_processed_versions()
    pending_count = len(pending_feedback_rows(complete_rows, processed_versions))
    if not STATUS_PATH.exists():
        return RetrainingStatus(
            status="waiting",
            complete_feedback_count=len(complete_rows),
            pending_feedback_count=pending_count,
            threshold=MIN_PENDING_FEEDBACK,
            message=f"再学習まであと{max(0, MIN_PENDING_FEEDBACK - pending_count)}件です。",
        )

    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status = RetrainingStatus(**payload)
    if status.status == "running" and not RETRAINING_LOCK.locked():
        status.status = "failed"
        status.message = "前回の再学習はアプリ終了により中断されました。"
        status.completed_at = current_timestamp()
        status.error = "Retraining worker was interrupted."
        write_status(status)
    return status


def start_retraining_if_ready() -> RetrainingStatus:
    complete_rows = load_complete_feedback()
    processed_versions = load_processed_versions()
    pending_rows = pending_feedback_rows(complete_rows, processed_versions)
    if len(pending_rows) < MIN_PENDING_FEEDBACK:
        status = RetrainingStatus(
            status="waiting",
            complete_feedback_count=len(complete_rows),
            pending_feedback_count=len(pending_rows),
            threshold=MIN_PENDING_FEEDBACK,
            message=f"再学習まであと{MIN_PENDING_FEEDBACK - len(pending_rows)}件です。",
        )
        write_status(status)
        return status

    if not RETRAINING_LOCK.acquire(blocking=False):
        return read_status()

    started_at = current_timestamp()
    status = RetrainingStatus(
        status="running",
        complete_feedback_count=len(complete_rows),
        pending_feedback_count=len(pending_rows),
        threshold=MIN_PENDING_FEEDBACK,
        message="候補モデルをバックグラウンドで学習しています。",
        started_at=started_at,
    )
    try:
        write_status(status)
        worker = Thread(
            target=run_retraining,
            args=(complete_rows, started_at),
            daemon=True,
            name="feedback-retraining",
        )
        worker.start()
    except Exception:
        RETRAINING_LOCK.release()
        raise
    return status


def run_retraining(feedback_rows: list[dict[str, str]], started_at: str) -> None:
    model_version = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    candidate_dir = MODEL_DIR.parent / f"candidate-{model_version}"
    report_dir = REPORT_ROOT / model_version
    promoted = False
    try:
        metrics = build_candidate(feedback_rows, candidate_dir)
        report_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(report_dir / "metrics.json", metrics)
        promoted = candidate_passes(metrics)
        if promoted:
            promote_candidate(candidate_dir, model_version)
            final_status = "promoted"
            message = "評価基準を満たした候補モデルを採用しました。"
        else:
            shutil.rmtree(candidate_dir)
            final_status = "rejected"
            message = "精度低下を検出したため、現行モデルを維持しました。"

        processed_versions = {
            row["prediction_id"]: feedback_version(row)
            for row in feedback_rows
        }
        write_json_atomic(
            STATE_PATH,
            {
                "processed_feedback_versions": processed_versions,
                "last_completed_at": current_timestamp(),
                "last_result": final_status,
                "model_version": model_version if promoted else "",
            },
        )
        write_status(
            RetrainingStatus(
                status=final_status,
                complete_feedback_count=len(feedback_rows),
                pending_feedback_count=0,
                threshold=MIN_PENDING_FEEDBACK,
                message=message,
                model_version=model_version if promoted else "",
                started_at=started_at,
                completed_at=current_timestamp(),
                metrics=metrics,
            )
        )
    except Exception as error:
        traceback.print_exc()
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        failure_message = (
            "モデル切替後の状態保存に失敗しました。管理者による確認が必要です。"
            if promoted
            else "再学習に失敗しました。現行モデルは変更していません。"
        )
        write_status(
            RetrainingStatus(
                status="failed",
                complete_feedback_count=len(feedback_rows),
                pending_feedback_count=len(feedback_rows),
                threshold=MIN_PENDING_FEEDBACK,
                message=failure_message,
                started_at=started_at,
                completed_at=current_timestamp(),
                error=str(error),
            )
        )
    finally:
        RETRAINING_LOCK.release()
