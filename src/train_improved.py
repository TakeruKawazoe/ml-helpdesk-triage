from __future__ import annotations

import csv
import json
from pathlib import Path

from modeling import (
    HOLDOUT_DATA_PATH,
    MODEL_OUTPUT_DIR,
    REPORT_OUTPUT_DIR,
    TRAIN_DATA_PATH,
    compare_models,
    learning_curve_rows,
    load_dataset,
    split_development_data,
    train_selected_models,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(normalized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    training_data = load_dataset(TRAIN_DATA_PATH)
    holdout = load_dataset(HOLDOUT_DATA_PATH)
    if set(training_data["inquiry_text"]) & set(holdout["inquiry_text"]):
        raise ValueError("Training and holdout inquiry texts must not overlap.")

    development, validation, calibration = split_development_data(training_data)
    selected_specs, comparison_rows = compare_models(development, validation)
    manifest, final_summary = train_selected_models(
        development,
        validation,
        calibration,
        training_data,
        holdout,
        selected_specs,
        model_dir=MODEL_OUTPUT_DIR,
    )
    curve_rows = learning_curve_rows(training_data, holdout, selected_specs)

    write_csv(REPORT_OUTPUT_DIR / "model_comparison.csv", comparison_rows)
    write_csv(REPORT_OUTPUT_DIR / "summary.csv", final_summary)
    write_csv(REPORT_OUTPUT_DIR / "learning_curve.csv", curve_rows)
    write_json(REPORT_OUTPUT_DIR / "manifest.json", manifest)
    write_json(REPORT_OUTPUT_DIR / "final_metrics.json", final_summary)

    print("selected models")
    for row in final_summary:
        print(
            f"{row['target']}: model={row['model']} "
            f"holdout_macro_f1={float(row['holdout_macro_f1']):.4f} "
            f"accepted_precision={float(row['accepted_precision']):.4f} "
            f"coverage={float(row['coverage']):.4f} "
            f"threshold={float(row['confidence_threshold']):.4f}"
        )


if __name__ == "__main__":
    main()
