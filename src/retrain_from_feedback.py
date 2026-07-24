from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import traceback
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import joblib
import pandas as pd

from modeling import (
    DESIRED_ACCEPTED_PRECISION,
    HOLDOUT_DATA_PATH,
    MODEL_OUTPUT_DIR,
    TARGETS,
    TRAIN_DATA_PATH,
    ModelSpec,
    choose_confidence_threshold,
    evaluate_pipeline,
    fit_and_measure,
    load_dataset,
    selective_metrics,
    split_development_data,
)
from predict import MODEL_ACCESS_LOCK, clear_model_cache, load_manifest
from train_baseline import build_text


ROOT_DIR = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = ROOT_DIR / "storage" / "prediction_feedback.csv"
STATE_PATH = ROOT_DIR / "storage" / "retraining_state.json"
STATUS_PATH = ROOT_DIR / "storage" / "retraining_status.json"
REPORT_ROOT = ROOT_DIR / "reports" / "retraining"
MODEL_DIR = MODEL_OUTPUT_DIR
FIXED_EVALUATION_PATH = HOLDOUT_DATA_PATH
MIN_PENDING_FEEDBACK = 20
MIN_FEEDBACK_PER_LABEL = 2
MIN_REVIEWERS = 2
MAX_MACRO_F1_REGRESSION = 0.0
MAX_COVERAGE_REGRESSION = 0.15
MIN_ACCEPTED_COVERAGE = 0.20
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
    "note",
    "reviewer_id",
    "feedback_saved_at",
    "deleted_at",
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
    gate_reasons: list[str] | None = None
    decision_reasons: list[str] | None = None
    error: str = ""


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_complete_feedback(path: Path | None = None) -> list[dict[str, str]]:
    path = path or FEEDBACK_PATH
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
        and not row["deleted_at"]
    ]
    prediction_ids = [row["prediction_id"] for row in complete_rows]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Feedback CSV contains duplicate prediction_id values.")
    return sorted(complete_rows, key=lambda row: row["prediction_id"])


def load_processed_versions(path: Path | None = None) -> dict[str, str]:
    path = path or STATE_PATH
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
        "note",
        "reviewer_id",
        "feedback_saved_at",
    ]
    payload = {field: row[field] for field in version_fields}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def feedback_gate_reasons(
    complete_rows: list[dict[str, str]],
    pending_count: int,
) -> list[str]:
    reasons: list[str] = []
    if pending_count < MIN_PENDING_FEEDBACK:
        reasons.append(
            f"新規・更新フィードバックが{MIN_PENDING_FEEDBACK}件必要です"
            f"（現在{pending_count}件）"
        )

    missing_reason_count = sum(not row["note"].strip() for row in complete_rows)
    if missing_reason_count:
        reasons.append(f"修正理由が未入力のフィードバックが{missing_reason_count}件あります")

    reviewers = {row["reviewer_id"].strip() for row in complete_rows if row["reviewer_id"].strip()}
    if len(reviewers) < MIN_REVIEWERS:
        reasons.append(
            f"確認者が{MIN_REVIEWERS}名以上必要です（現在{len(reviewers)}名）"
        )

    base_data = load_dataset(TRAIN_DATA_PATH)
    feedback_fields = {
        "category": "corrected_category",
        "priority": "corrected_priority",
        "department": "corrected_department",
    }
    for target_name, target_column in TARGETS.items():
        expected_labels = sorted(set(base_data[target_column]))
        counts = Counter(row[feedback_fields[target_name]] for row in complete_rows)
        insufficient = [
            f"{label}={counts[label]}件"
            for label in expected_labels
            if counts[label] < MIN_FEEDBACK_PER_LABEL
        ]
        if insufficient:
            reasons.append(
                f"{target_name}は各ラベル{MIN_FEEDBACK_PER_LABEL}件以上必要です: "
                + ", ".join(insufficient)
            )
    return reasons


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
            "template_group": f"approved-feedback-{row['prediction_id']}",
            "data_origin": "approved_feedback",
        }
        for row in rows
    ]
    data = pd.DataFrame.from_records(records)
    data["model_text"] = data.apply(build_text, axis=1)
    return data


def model_spec_from_metadata(metadata: dict[str, object]) -> ModelSpec:
    specification = metadata["model_spec"]
    if not isinstance(specification, dict):
        raise ValueError("model_spec must be an object.")
    return ModelSpec(
        name=str(specification["name"]),
        feature_kind=str(specification["feature_kind"]),
        estimator_kind=str(specification["estimator_kind"]),
        class_weight=specification["class_weight"],
    )


def metric_snapshot(
    metrics: dict[str, object],
    threshold: float,
    target_name: str,
) -> dict[str, object]:
    selective = selective_metrics(
        metrics["y_true"],
        metrics["y_pred"],
        metrics["probabilities"],
        threshold,
    )
    snapshot: dict[str, object] = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        **selective,
    }
    if target_name == "priority":
        snapshot["high_recall"] = metrics["per_class"]["High"]["recall"]
        snapshot["high_false_negatives"] = int(
            ((metrics["y_true"] == "High") & (metrics["y_pred"] != "High")).sum()
        )
    if target_name == "department":
        snapshot["misroutes"] = int(
            (metrics["y_true"] != metrics["y_pred"]).sum()
        )
    return snapshot


def build_candidate(
    feedback_rows: list[dict[str, str]],
    candidate_dir: Path,
) -> dict[str, object]:
    base_data = load_dataset(TRAIN_DATA_PATH)
    fixed_evaluation = load_dataset(FIXED_EVALUATION_PATH)
    development, validation, calibration = split_development_data(base_data)
    feedback_data = feedback_to_dataframe(feedback_rows)
    current_manifest = load_manifest(MODEL_DIR)
    candidate_targets: dict[str, object] = {}
    metrics: dict[str, object] = {}
    candidate_dir.mkdir(parents=True, exist_ok=False)

    for target_name, target_column in TARGETS.items():
        invalid_labels = sorted(set(feedback_data[target_column]) - set(base_data[target_column]))
        if invalid_labels:
            raise ValueError(f"Unknown feedback labels for {target_name}: {invalid_labels}")

        current_metadata = current_manifest["targets"][target_name]
        spec = model_spec_from_metadata(current_metadata)
        threshold_training = pd.concat(
            [development, validation, feedback_data],
            ignore_index=True,
        )
        threshold_model, _ = fit_and_measure(spec, threshold_training, target_column)
        calibration_metrics = evaluate_pipeline(threshold_model, calibration, target_column)
        threshold_result = choose_confidence_threshold(
            calibration_metrics["y_true"],
            calibration_metrics["y_pred"],
            calibration_metrics["probabilities"],
        )
        candidate_threshold = float(threshold_result["threshold"])

        candidate_training = pd.concat([base_data, feedback_data], ignore_index=True)
        candidate_model, _ = fit_and_measure(spec, candidate_training, target_column)
        model_filename = f"{target_name}.joblib"
        joblib.dump(candidate_model, candidate_dir / model_filename)

        current_model = joblib.load(MODEL_DIR / current_metadata["model_file"])
        current_metrics = evaluate_pipeline(current_model, fixed_evaluation, target_column)
        candidate_metrics = evaluate_pipeline(candidate_model, fixed_evaluation, target_column)
        current_snapshot = metric_snapshot(
            current_metrics,
            float(current_metadata["confidence_threshold"]),
            target_name,
        )
        candidate_snapshot = metric_snapshot(
            candidate_metrics,
            candidate_threshold,
            target_name,
        )
        metrics[target_name] = {
            "current": current_snapshot,
            "candidate": candidate_snapshot,
        }
        candidate_targets[target_name] = {
            **current_metadata,
            "model_file": model_filename,
            "classes": [str(value) for value in candidate_model.classes_],
            "confidence_threshold": candidate_threshold,
            "threshold_calibration": threshold_result,
        }

    origins = sorted(set(fixed_evaluation["data_origin"]))
    metrics["feedback_count"] = len(feedback_rows)
    metrics["fixed_evaluation_path"] = str(FIXED_EVALUATION_PATH)
    metrics["fixed_evaluation_origins"] = origins
    metrics["evaluation_is_real_data"] = not all(
        origin.startswith("synthetic_") for origin in origins
    )
    candidate_manifest = {
        **current_manifest,
        "targets": candidate_targets,
        "retrained_at": current_timestamp(),
        "feedback_count": len(feedback_rows),
    }
    (candidate_dir / "manifest.json").write_text(
        json.dumps(candidate_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def candidate_rejection_reasons(metrics: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    for target_name in TARGETS:
        target_metrics = metrics[target_name]
        current = target_metrics["current"]
        candidate = target_metrics["candidate"]
        if float(candidate["macro_f1"]) < float(current["macro_f1"]) - MAX_MACRO_F1_REGRESSION:
            reasons.append(f"{target_name}のMacro F1が低下しました")
        if float(candidate["accepted_precision"]) < DESIRED_ACCEPTED_PRECISION:
            reasons.append(
                f"{target_name}の自動振り分け正答率が"
                f"{DESIRED_ACCEPTED_PRECISION:.0%}未満です"
            )
        if float(candidate["coverage"]) < MIN_ACCEPTED_COVERAGE:
            reasons.append(f"{target_name}の自動振り分け対象が少なすぎます")
        if float(candidate["coverage"]) < float(current["coverage"]) - MAX_COVERAGE_REGRESSION:
            reasons.append(f"{target_name}の自動振り分け範囲が大きく低下しました")

    priority = metrics["priority"]
    if float(priority["candidate"]["high_recall"]) < float(priority["current"]["high_recall"]):
        reasons.append("優先度HighのRecallが低下しました")
    if int(priority["candidate"]["high_false_negatives"]) > int(
        priority["current"]["high_false_negatives"]
    ):
        reasons.append("優先度Highの見逃し件数が増加しました")

    department = metrics["department"]
    if int(department["candidate"]["misroutes"]) > int(
        department["current"]["misroutes"]
    ):
        reasons.append("担当部署の誤振り分け件数が増加しました")
    return reasons


def candidate_passes(metrics: dict[str, object]) -> bool:
    return not candidate_rejection_reasons(metrics)


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
            clear_model_cache()
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
    write_json_atomic(path or STATUS_PATH, asdict(status))


def waiting_status(
    complete_rows: list[dict[str, str]],
    pending_count: int,
) -> RetrainingStatus:
    reasons = feedback_gate_reasons(complete_rows, pending_count)
    message = reasons[0] if reasons else "再学習の品質条件を満たしています。"
    return RetrainingStatus(
        status="waiting",
        complete_feedback_count=len(complete_rows),
        pending_feedback_count=pending_count,
        threshold=MIN_PENDING_FEEDBACK,
        message=message,
        gate_reasons=reasons,
    )


def read_status() -> RetrainingStatus:
    complete_rows = load_complete_feedback()
    processed_versions = load_processed_versions()
    pending_count = len(pending_feedback_rows(complete_rows, processed_versions))
    if not STATUS_PATH.exists():
        return waiting_status(complete_rows, pending_count)
    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    status = RetrainingStatus(**payload)
    if status.status == "waiting":
        status = waiting_status(complete_rows, pending_count)
        write_status(status)
        return status
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
    gate_reasons = feedback_gate_reasons(complete_rows, len(pending_rows))
    if gate_reasons:
        status = waiting_status(complete_rows, len(pending_rows))
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
        message="品質条件を満たしたフィードバックで候補モデルを学習しています。",
        started_at=started_at,
        gate_reasons=[],
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
        rejection_reasons = candidate_rejection_reasons(metrics)
        decision_reasons = rejection_reasons or ["すべての品質基準を満たしました"]
        report_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(
            report_dir / "decision.json",
            {
                "accepted": not rejection_reasons,
                "reasons": decision_reasons,
                "metrics": metrics,
            },
        )
        promoted = not rejection_reasons
        if promoted:
            promote_candidate(candidate_dir, model_version)
            final_status = "promoted"
            message = "固定評価の品質基準を満たした候補モデルを採用しました。"
        else:
            shutil.rmtree(candidate_dir)
            final_status = "rejected"
            message = "品質低下を検出したため、現行モデルを維持しました。"

        write_json_atomic(
            STATE_PATH,
            {
                "processed_feedback_versions": {
                    row["prediction_id"]: feedback_version(row) for row in feedback_rows
                },
                "last_completed_at": current_timestamp(),
                "last_result": final_status,
                "model_version": model_version if promoted else "",
                "decision_reasons": decision_reasons,
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
                gate_reasons=[],
                decision_reasons=decision_reasons,
            )
        )
    except Exception as error:
        traceback.print_exc()
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        write_status(
            RetrainingStatus(
                status="failed",
                complete_feedback_count=len(feedback_rows),
                pending_feedback_count=len(feedback_rows),
                threshold=MIN_PENDING_FEEDBACK,
                message=(
                    "モデル切替後の状態保存に失敗しました。管理者による確認が必要です。"
                    if promoted
                    else "再学習に失敗しました。現行モデルは変更していません。"
                ),
                started_at=started_at,
                completed_at=current_timestamp(),
                error=str(error),
            )
        )
    finally:
        RETRAINING_LOCK.release()
