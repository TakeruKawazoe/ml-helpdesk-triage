from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "tickets_dummy.csv"
REPORT_DIR = ROOT_DIR / "reports" / "baseline"
MODEL_DIR = ROOT_DIR / "models" / "baseline"
RANDOM_SEED = 42


TEXT_COLUMNS = [
    "inquiry_text",
    "impact_scope",
    "requester_role",
    "channel",
]

TARGETS = {
    "category": "label_category",
    "priority": "label_priority",
    "department": "label_department",
}


def build_text(row: pd.Series) -> str:
    return (
        f"{row['inquiry_text']} "
        f"影響範囲={row['impact_scope']} "
        f"依頼者={row['requester_role']} "
        f"経路={row['channel']}"
    )


def char_ngrams(text: str, min_n: int = 2, max_n: int = 4) -> list[str]:
    normalized = "".join(str(text).split())
    features: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(normalized) < n:
            continue
        features.extend(normalized[i : i + n] for i in range(len(normalized) - n + 1))
    return features


@dataclass
class TfidfVectorizer:
    vocab_: dict[str, int] | None = None
    idf_: np.ndarray | None = None
    min_n: int = 2
    max_n: int = 4

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        document_frequency: Counter[str] = Counter()
        for text in texts:
            document_frequency.update(set(char_ngrams(text, self.min_n, self.max_n)))

        tokens = sorted(document_frequency)
        self.vocab_ = {token: index for index, token in enumerate(tokens)}
        doc_count = len(texts)
        self.idf_ = np.array(
            [
                np.log((1 + doc_count) / (1 + document_frequency[token])) + 1
                for token in tokens
            ],
            dtype=np.float64,
        )
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.vocab_ is None or self.idf_ is None:
            raise RuntimeError("TfidfVectorizer must be fitted before transform.")

        matrix = np.zeros((len(texts), len(self.vocab_)), dtype=np.float64)
        for row_index, text in enumerate(texts):
            counts = Counter(char_ngrams(text, self.min_n, self.max_n))
            total = sum(counts.values())
            if total == 0:
                continue

            for token, count in counts.items():
                column_index = self.vocab_.get(token)
                if column_index is None:
                    continue
                matrix[row_index, column_index] = count / total

        matrix *= self.idf_
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return matrix / norms

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


@dataclass
class LabelEncoder:
    classes_: list[str] | None = None
    class_to_index_: dict[str, int] | None = None

    def fit(self, labels: list[str]) -> "LabelEncoder":
        self.classes_ = sorted(set(labels))
        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}
        return self

    def transform(self, labels: list[str]) -> np.ndarray:
        if self.class_to_index_ is None:
            raise RuntimeError("LabelEncoder must be fitted before transform.")
        return np.array([self.class_to_index_[label] for label in labels], dtype=np.int64)

    def inverse_transform(self, label_indexes: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder must be fitted before inverse_transform.")
        return np.array([self.classes_[index] for index in label_indexes])


@dataclass
class SoftmaxLogisticRegression:
    learning_rate: float = 0.8
    epochs: int = 2500
    l2: float = 0.001
    weights_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, class_count: int) -> "SoftmaxLogisticRegression":
        x_train_with_bias = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
        weights = np.zeros((x_train_with_bias.shape[1], class_count), dtype=np.float64)
        y_one_hot = np.eye(class_count)[y_train]

        for _ in range(self.epochs):
            logits = x_train_with_bias @ weights
            probabilities = softmax(logits)
            gradient = x_train_with_bias.T @ (probabilities - y_one_hot) / x_train.shape[0]
            gradient[:-1] += self.l2 * weights[:-1]
            weights -= self.learning_rate * gradient

        self.weights_ = weights
        return self

    def predict_proba(self, x_values: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before predict_proba.")
        x_values_with_bias = np.hstack([x_values, np.ones((x_values.shape[0], 1))])
        return softmax(x_values_with_bias @ self.weights_)

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.predict_proba(x_values).argmax(axis=1)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def stratified_split(labels: list[str], test_ratio: float = 0.25) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(RANDOM_SEED)
    class_to_indexes: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        class_to_indexes[label].append(index)

    train_indexes: list[int] = []
    test_indexes: list[int] = []
    for indexes in class_to_indexes.values():
        shuffled = np.array(indexes)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * test_ratio)))
        test_indexes.extend(shuffled[:test_count].tolist())
        train_indexes.extend(shuffled[test_count:].tolist())

    rng.shuffle(train_indexes)
    rng.shuffle(test_indexes)
    return train_indexes, test_indexes


def classification_report(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    classes = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
    report: dict[str, object] = {
        "accuracy": correct / total if total else 0,
        "total": total,
        "classes": {},
    }

    f1_values: list[float] = []
    for class_name in classes:
        tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == class_name and predicted == class_name)
        fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual != class_name and predicted == class_name)
        fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == class_name and predicted != class_name)
        support = sum(1 for actual in y_true if actual == class_name)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        f1_values.append(f1)
        report["classes"][class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    report["macro_f1"] = float(np.mean(f1_values)) if f1_values else 0
    return report


def confusion_matrix_rows(y_true: list[str], y_pred: list[str]) -> list[dict[str, object]]:
    classes = sorted(set(y_true) | set(y_pred))
    rows: list[dict[str, object]] = []
    for actual in classes:
        row: dict[str, object] = {"actual": actual}
        for predicted in classes:
            row[predicted] = sum(
                1 for actual_value, predicted_value in zip(y_true, y_pred)
                if actual_value == actual and predicted_value == predicted
            )
        rows.append(row)
    return rows


def train_target(data: pd.DataFrame, target_name: str, target_column: str) -> dict[str, object]:
    labels = data[target_column].tolist()
    train_indexes, test_indexes = stratified_split(labels)

    train_data = data.iloc[train_indexes].copy()
    test_data = data.iloc[test_indexes].copy()

    vectorizer = TfidfVectorizer(min_n=1, max_n=4)
    x_train = vectorizer.fit_transform(train_data["model_text"].tolist())
    x_test = vectorizer.transform(test_data["model_text"].tolist())

    label_encoder = LabelEncoder().fit(train_data[target_column].tolist())
    y_train = label_encoder.transform(train_data[target_column].tolist())

    if label_encoder.classes_ is None:
        raise RuntimeError("Label encoder is not fitted.")

    model = SoftmaxLogisticRegression()
    model.fit(x_train, y_train, class_count=len(label_encoder.classes_))
    save_model_artifacts(target_name, target_column, vectorizer, label_encoder, model)

    predicted_indexes = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    y_pred = label_encoder.inverse_transform(predicted_indexes).tolist()
    y_true = test_data[target_column].tolist()

    report = classification_report(y_true, y_pred)
    report["target"] = target_name
    report["target_column"] = target_column
    report["train_count"] = len(train_data)
    report["test_count"] = len(test_data)
    if vectorizer.vocab_ is None:
        raise RuntimeError("Vectorizer is not fitted.")
    report["vocab_size"] = len(vectorizer.vocab_)
    if target_name == "priority":
        priority_classes = report["classes"]
        report["high_recall"] = priority_classes["High"]["recall"]

    write_json(REPORT_DIR / f"{target_name}_metrics.json", report)
    write_csv(REPORT_DIR / f"{target_name}_confusion_matrix.csv", confusion_matrix_rows(y_true, y_pred))

    prediction_rows = []
    for row, predicted, probability in zip(test_data.to_dict("records"), y_pred, probabilities.max(axis=1)):
        prediction_rows.append(
            {
                "ticket_id": row["ticket_id"],
                "inquiry_text": row["inquiry_text"],
                "actual": row[target_column],
                "predicted": predicted,
                "confidence": round(float(probability), 4),
            }
        )
    write_csv(REPORT_DIR / f"{target_name}_predictions.csv", prediction_rows)

    return report


def save_model_artifacts(
    target_name: str,
    target_column: str,
    vectorizer: TfidfVectorizer,
    label_encoder: LabelEncoder,
    model: SoftmaxLogisticRegression,
    model_dir: Path = MODEL_DIR,
) -> None:
    if vectorizer.vocab_ is None or vectorizer.idf_ is None:
        raise RuntimeError("Vectorizer is not fitted.")
    if label_encoder.classes_ is None:
        raise RuntimeError("Label encoder is not fitted.")
    if model.weights_ is None:
        raise RuntimeError("Model is not fitted.")

    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "target": target_name,
        "target_column": target_column,
        "text_columns": TEXT_COLUMNS,
        "min_n": vectorizer.min_n,
        "max_n": vectorizer.max_n,
        "vocab": vectorizer.vocab_,
        "classes": label_encoder.classes_,
    }
    write_json(model_dir / f"{target_name}_metadata.json", metadata)
    np.savez_compressed(
        model_dir / f"{target_name}_arrays.npz",
        idf=vectorizer.idf_,
        weights=model.weights_,
    )


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    missing_columns = [column for column in [*TEXT_COLUMNS, *TARGETS.values()] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data["model_text"] = data.apply(build_text, axis=1)

    reports = []
    for target_name, target_column in TARGETS.items():
        reports.append(train_target(data, target_name, target_column))

    summary_rows = [
        {
            "target": report["target"],
            "target_column": report["target_column"],
            "accuracy": round(float(report["accuracy"]), 4),
            "macro_f1": round(float(report["macro_f1"]), 4),
            "high_recall": round(float(report["high_recall"]), 4)
            if report["target"] == "priority"
            else "",
            "train_count": report["train_count"],
            "test_count": report["test_count"],
            "vocab_size": report["vocab_size"],
        }
        for report in reports
    ]
    write_csv(REPORT_DIR / "summary.csv", summary_rows)

    for row in summary_rows:
        output_parts = [
            f"{row['target']}: "
            f"accuracy={row['accuracy']}",
            f"macro_f1={row['macro_f1']}",
        ]
        if row["target"] == "priority":
            output_parts.append(f"high_recall={row['high_recall']}")
        output_parts.extend(
            [
                f"train={row['train_count']}",
                f"test={row['test_count']}",
            ]
        )
        print(" ".join(output_parts))


if __name__ == "__main__":
    main()
