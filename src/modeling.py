from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from train_baseline import TARGETS, build_text


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = ROOT_DIR / "data" / "tickets_dummy.csv"
HOLDOUT_DATA_PATH = ROOT_DIR / "data" / "tickets_holdout.csv"
MODEL_OUTPUT_DIR = ROOT_DIR / "models" / "improved"
REPORT_OUTPUT_DIR = ROOT_DIR / "reports" / "improved"
RANDOM_SEED = 42
DESIRED_ACCEPTED_PRECISION = 0.90
MIN_ACCEPTED_RATIO = 0.10


@dataclass(frozen=True)
class ModelSpec:
    name: str
    feature_kind: str
    estimator_kind: str
    class_weight: str | dict[str, float] | None


MODEL_SPECS = (
    ModelSpec("char_logistic", "char", "logistic", None),
    ModelSpec("hybrid_logistic", "hybrid", "logistic", None),
    ModelSpec("hybrid_logistic_balanced", "hybrid", "logistic", "balanced"),
    ModelSpec("hybrid_linear_svc", "hybrid", "linear_svc", None),
    ModelSpec("hybrid_linear_svc_balanced", "hybrid", "linear_svc", "balanced"),
    ModelSpec(
        "char_logistic_high_weighted",
        "char",
        "logistic",
        {"High": 2.5, "Middle": 1.0, "Low": 1.0},
    ),
    ModelSpec(
        "hybrid_logistic_high_weighted",
        "hybrid",
        "logistic",
        {"High": 2.5, "Middle": 1.0, "Low": 1.0},
    ),
)
MODEL_SPEC_ORDER = {spec.name: index for index, spec in enumerate(MODEL_SPECS)}


def load_dataset(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8")
    required_columns = {
        "inquiry_text",
        "impact_scope",
        "requester_role",
        "channel",
        "template_group",
        *TARGETS.values(),
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {missing_columns}")
    data = data.copy()
    data["model_text"] = data.apply(build_text, axis=1)
    return data


def split_development_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=RANDOM_SEED,
    )
    development_indexes, temporary_indexes = next(
        first_split.split(data, groups=data["template_group"])
    )
    development = data.iloc[development_indexes].copy()
    temporary = data.iloc[temporary_indexes].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED + 1,
    )
    validation_indexes, calibration_indexes = next(
        second_split.split(temporary, groups=temporary["template_group"])
    )
    validation = temporary.iloc[validation_indexes].copy()
    calibration = temporary.iloc[calibration_indexes].copy()

    group_sets = [
        set(development["template_group"]),
        set(validation["template_group"]),
        set(calibration["template_group"]),
    ]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("Template groups overlap across development splits.")
    return development, validation, calibration


def build_pipeline(spec: ModelSpec) -> Pipeline:
    if spec.feature_kind == "char":
        features = TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 4),
            sublinear_tf=True,
            min_df=1,
        )
    elif spec.feature_kind == "hybrid":
        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(1, 4),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown feature kind: {spec.feature_kind}")

    if spec.estimator_kind == "logistic":
        estimator = LogisticRegression(
            C=4.0,
            class_weight=spec.class_weight,
            max_iter=2_000,
            random_state=RANDOM_SEED,
        )
    elif spec.estimator_kind == "linear_svc":
        estimator = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight=spec.class_weight,
                random_state=RANDOM_SEED,
            ),
            method="sigmoid",
            cv=3,
        )
    else:
        raise ValueError(f"Unknown estimator kind: {spec.estimator_kind}")

    return Pipeline([("features", features), ("classifier", estimator)])


def fit_and_measure(
    spec: ModelSpec,
    train_data: pd.DataFrame,
    target_column: str,
) -> tuple[Pipeline, float]:
    pipeline = build_pipeline(spec)
    started_at = time.perf_counter()
    pipeline.fit(train_data["model_text"], train_data[target_column])
    train_seconds = time.perf_counter() - started_at
    return pipeline, train_seconds


def evaluate_pipeline(
    pipeline: Pipeline,
    data: pd.DataFrame,
    target_column: str,
) -> dict[str, object]:
    y_true = data[target_column].astype(str).to_numpy()
    started_at = time.perf_counter()
    y_pred = pipeline.predict(data["model_text"])
    probabilities = pipeline.predict_proba(data["model_text"])
    inference_seconds = time.perf_counter() - started_at
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            label: {
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1-score"]),
                "support": int(values["support"]),
            }
            for label, values in report.items()
            if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}
        },
        "inference_ms_per_row": (
            inference_seconds * 1_000 / len(data) if len(data) else 0.0
        ),
        "y_true": y_true,
        "y_pred": np.asarray(y_pred),
        "probabilities": probabilities,
    }


def choose_confidence_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    desired_precision: float = DESIRED_ACCEPTED_PRECISION,
) -> dict[str, float | int]:
    confidences = probabilities.max(axis=1)
    minimum_accepted = max(3, math.ceil(len(y_true) * MIN_ACCEPTED_RATIO))
    candidates = sorted(set(float(value) for value in confidences))
    valid_results: list[dict[str, float | int]] = []
    for threshold in candidates:
        accepted = confidences >= threshold
        accepted_count = int(accepted.sum())
        if accepted_count < minimum_accepted:
            continue
        accepted_precision = float((y_true[accepted] == y_pred[accepted]).mean())
        if accepted_precision >= desired_precision:
            valid_results.append(
                {
                    "threshold": threshold,
                    "accepted_precision": accepted_precision,
                    "coverage": accepted_count / len(y_true),
                    "accepted_count": accepted_count,
                }
            )
    if not valid_results:
        raise RuntimeError(
            "No confidence threshold satisfies the required precision and coverage."
        )
    return max(
        valid_results,
        key=lambda result: (float(result["coverage"]), float(result["accepted_precision"])),
    )


def selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    confidences = probabilities.max(axis=1)
    accepted = confidences >= threshold
    accepted_count = int(accepted.sum())
    if accepted_count == 0:
        return {
            "accepted_precision": 0.0,
            "coverage": 0.0,
            "accepted_count": 0,
            "review_count": len(y_true),
        }
    return {
        "accepted_precision": float((y_true[accepted] == y_pred[accepted]).mean()),
        "coverage": accepted_count / len(y_true),
        "accepted_count": accepted_count,
        "review_count": len(y_true) - accepted_count,
    }


def select_model_row(
    target_name: str,
    target_rows: list[dict[str, object]],
) -> dict[str, object]:
    if target_name == "priority":
        return max(
            target_rows,
            key=lambda row: (
                float(row["validation_macro_f1"]),
                float(row["validation_high_recall"]),
                float(row["validation_high_mean_probability"]),
                float(row["validation_accuracy"]),
                -MODEL_SPEC_ORDER[str(row["model"])],
            ),
        )
    return max(
        target_rows,
        key=lambda row: (
            float(row["validation_macro_f1"]),
            float(row["validation_accuracy"]),
            -MODEL_SPEC_ORDER[str(row["model"])],
        ),
    )


def compare_models(
    development: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[dict[str, ModelSpec], list[dict[str, object]]]:
    selected_specs: dict[str, ModelSpec] = {}
    rows: list[dict[str, object]] = []
    for target_name, target_column in TARGETS.items():
        target_rows: list[dict[str, object]] = []
        applicable_specs = [
            spec
            for spec in MODEL_SPECS
            if "high_weighted" not in spec.name or target_name == "priority"
        ]
        for spec in applicable_specs:
            pipeline, train_seconds = fit_and_measure(spec, development, target_column)
            metrics = evaluate_pipeline(pipeline, validation, target_column)
            high_recall = ""
            high_mean_probability = ""
            if target_name == "priority":
                high_recall = metrics["per_class"]["High"]["recall"]
                high_index = list(pipeline.classes_).index("High")
                true_high = metrics["y_true"] == "High"
                high_mean_probability = float(
                    metrics["probabilities"][true_high, high_index].mean()
                )
            row = {
                "target": target_name,
                "target_column": target_column,
                "model": spec.name,
                "feature_kind": spec.feature_kind,
                "estimator_kind": spec.estimator_kind,
                "class_weight": (
                    json.dumps(spec.class_weight, ensure_ascii=False, sort_keys=True)
                    if isinstance(spec.class_weight, dict)
                    else spec.class_weight or "none"
                ),
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_high_recall": high_recall,
                "validation_high_mean_probability": high_mean_probability,
                "train_seconds": train_seconds,
                "inference_ms_per_row": metrics["inference_ms_per_row"],
            }
            rows.append(row)
            target_rows.append(row)
        winner = select_model_row(target_name, target_rows)
        selected_specs[target_name] = next(
            spec for spec in MODEL_SPECS if spec.name == winner["model"]
        )
    return selected_specs, rows


def train_selected_models(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    calibration: pd.DataFrame,
    full_training_data: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_specs: dict[str, ModelSpec],
    model_dir: Path = MODEL_OUTPUT_DIR,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_targets: dict[str, object] = {}
    final_summary: list[dict[str, object]] = []
    calibration_train = pd.concat([development, validation], ignore_index=True)

    for target_name, target_column in TARGETS.items():
        spec = selected_specs[target_name]
        calibration_model, _ = fit_and_measure(spec, calibration_train, target_column)
        calibration_metrics = evaluate_pipeline(
            calibration_model,
            calibration,
            target_column,
        )
        threshold_result = choose_confidence_threshold(
            calibration_metrics["y_true"],
            calibration_metrics["y_pred"],
            calibration_metrics["probabilities"],
        )
        threshold = float(threshold_result["threshold"])

        production_model, train_seconds = fit_and_measure(
            spec,
            full_training_data,
            target_column,
        )
        joblib.dump(production_model, model_dir / f"{target_name}.joblib")
        holdout_metrics = evaluate_pipeline(production_model, holdout, target_column)
        holdout_selective = selective_metrics(
            holdout_metrics["y_true"],
            holdout_metrics["y_pred"],
            holdout_metrics["probabilities"],
            threshold,
        )
        labels = [str(label) for label in production_model.classes_]
        manifest_targets[target_name] = {
            "target_column": target_column,
            "model_file": f"{target_name}.joblib",
            "classes": labels,
            "model_spec": asdict(spec),
            "confidence_threshold": threshold,
            "threshold_calibration": threshold_result,
        }
        high_false_negatives = ""
        if target_name == "priority":
            y_true = holdout_metrics["y_true"]
            y_pred = holdout_metrics["y_pred"]
            high_false_negatives = int(((y_true == "High") & (y_pred != "High")).sum())
        department_misroutes = ""
        if target_name == "department":
            department_misroutes = int(
                (holdout_metrics["y_true"] != holdout_metrics["y_pred"]).sum()
            )
        final_summary.append(
            {
                "target": target_name,
                "model": spec.name,
                "holdout_accuracy": holdout_metrics["accuracy"],
                "holdout_macro_f1": holdout_metrics["macro_f1"],
                "accepted_precision": holdout_selective["accepted_precision"],
                "coverage": holdout_selective["coverage"],
                "review_count": holdout_selective["review_count"],
                "confidence_threshold": threshold,
                "high_false_negatives": high_false_negatives,
                "department_misroutes": department_misroutes,
                "train_seconds": train_seconds,
                "inference_ms_per_row": holdout_metrics["inference_ms_per_row"],
                "per_class": holdout_metrics["per_class"],
                "confusion_matrix": confusion_matrix(
                    holdout_metrics["y_true"],
                    holdout_metrics["y_pred"],
                    labels=labels,
                ).tolist(),
                "labels": labels,
            }
        )

    manifest = {
        "format": "sklearn-joblib-v1",
        "random_seed": RANDOM_SEED,
        "desired_accepted_precision": DESIRED_ACCEPTED_PRECISION,
        "selection_data": "grouped validation split from tickets_dummy.csv",
        "threshold_data": "grouped calibration split from tickets_dummy.csv",
        "final_evaluation_data": "tickets_holdout.csv",
        "targets": manifest_targets,
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest, final_summary


def learning_curve_rows(
    training_data: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_specs: dict[str, ModelSpec],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for per_category_count in (20, 40, 60):
        subset_parts = []
        for _, category_rows in training_data.groupby("label_category", sort=True):
            sampled = category_rows.sample(
                n=per_category_count,
                random_state=RANDOM_SEED + per_category_count,
            )
            subset_parts.append(sampled)
        subset = pd.concat(subset_parts, ignore_index=True)
        for target_name, target_column in TARGETS.items():
            model, train_seconds = fit_and_measure(
                selected_specs[target_name],
                subset,
                target_column,
            )
            metrics = evaluate_pipeline(model, holdout, target_column)
            rows.append(
                {
                    "rows_per_category": per_category_count,
                    "training_count": len(subset),
                    "target": target_name,
                    "model": selected_specs[target_name].name,
                    "holdout_accuracy": metrics["accuracy"],
                    "holdout_macro_f1": metrics["macro_f1"],
                    "train_seconds": train_seconds,
                }
            )
    return rows
