from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.ml.tickets.data import TICKET_CATEGORIES

MINIMUM_TEXT_LENGTH = 10


class SpanishTextCleaner(BaseEstimator, TransformerMixin):
    """Normalize Spanish text while preserving accents and the letter enye."""

    _url_pattern = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
    _invalid_character_pattern = re.compile(r"[^a-záéíóúüñ0-9\s]", flags=re.IGNORECASE)
    _whitespace_pattern = re.compile(r"\s+")

    def fit(self, texts: Any, target: Any = None) -> SpanishTextCleaner:
        return self

    def transform(self, texts: Any) -> list[str]:
        return [self.clean(text) for text in texts]

    @classmethod
    def clean(cls, text: Any) -> str:
        normalized = unicodedata.normalize("NFKC", str(text)).lower()
        without_urls = cls._url_pattern.sub(" ", normalized)
        allowed_characters = cls._invalid_character_pattern.sub(" ", without_urls)
        return cls._whitespace_pattern.sub(" ", allowed_characters).strip()


def validate_ticket_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("La descripcion del ticket debe ser texto")
    normalized_text = text.strip()
    if len(normalized_text) < MINIMUM_TEXT_LENGTH:
        raise ValueError(f"La descripcion debe tener al menos {MINIMUM_TEXT_LENGTH} caracteres")
    return normalized_text


def build_candidate_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    def vectorizer() -> TfidfVectorizer:
        return TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10_000,
            min_df=2,
            sublinear_tf=True,
        )

    return {
        "logistic_regression": Pipeline(
            steps=[
                ("clean_text", SpanishTextCleaner()),
                ("tfidf", vectorizer()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "multinomial_naive_bayes": Pipeline(
            steps=[
                ("clean_text", SpanishTextCleaner()),
                ("tfidf", vectorizer()),
                ("classifier", MultinomialNB(alpha=0.5)),
            ]
        ),
    }


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    metadata: dict[str, Any]


def train_ticket_classifier(
    dataset: pd.DataFrame,
    random_state: int = 42,
) -> TrainingResult:
    descriptions = dataset["description"].astype(str)
    categories = dataset["category"].astype(str)

    train_descriptions, test_descriptions, train_categories, test_categories = train_test_split(
        descriptions,
        categories,
        test_size=0.2,
        stratify=categories,
        random_state=random_state,
    )

    cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    candidates = build_candidate_pipelines(random_state=random_state)
    comparison: dict[str, dict[str, float]] = {}
    for model_name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            train_descriptions,
            train_categories,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
        )
        comparison[model_name] = {
            metric: float(np.mean(scores[f"test_{metric}"])) for metric in scoring
        }

    selected_model_name = max(
        comparison,
        key=lambda model_name: comparison[model_name]["f1_macro"],
    )
    selected_pipeline = candidates[selected_model_name]
    selected_pipeline.fit(train_descriptions, train_categories)

    predictions = selected_pipeline.predict(test_descriptions)
    labels = list(TICKET_CATEGORIES)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        test_categories,
        predictions,
        labels=labels,
        zero_division=0,
    )
    per_category = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1_score": float(f1_score[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    matrix = confusion_matrix(test_categories, predictions, labels=labels)

    metadata: dict[str, Any] = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "random_state": random_state,
        "dataset_rows": int(len(dataset)),
        "dataset_source": (
            str(dataset["source"].iloc[0]) if "source" in dataset.columns else "provided"
        ),
        "categories": labels,
        "selected_model": selected_model_name,
        "selection_metric": "f1_macro",
        "cross_validation": {
            "folds": 5,
            "training_rows": int(len(train_descriptions)),
            "models": comparison,
        },
        "test_evaluation": {
            "test_rows": int(len(test_descriptions)),
            "accuracy": float(accuracy_score(test_categories, predictions)),
            "per_category": per_category,
            "classification_report": classification_report(
                test_categories,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": matrix.tolist(),
        },
    }

    selected_pipeline.fit(descriptions, categories)
    return TrainingResult(pipeline=selected_pipeline, metadata=metadata)


def save_training_outputs(
    result: TrainingResult,
    model_path: Path,
    metrics_path: Path,
    confusion_matrix_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {"pipeline": result.pipeline, "metadata": result.metadata}
    joblib.dump(artifact, model_path)
    metrics_path.write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = [str(label) for label in result.metadata["categories"]]
    matrix = result.metadata["test_evaluation"]["confusion_matrix"]
    confusion_matrix_path.write_text(
        _render_confusion_matrix_svg(matrix, labels),
        encoding="utf-8",
    )


def _render_confusion_matrix_svg(matrix: list[list[int]], labels: list[str]) -> str:
    """Render a dependency-free, accessible confusion-matrix visualization."""
    cell_size = 82
    left_margin = 150
    top_margin = 110
    width = left_margin + cell_size * len(labels) + 40
    height = top_margin + cell_size * len(labels) + 80
    maximum = max(max(row) for row in matrix) or 1

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Matriz de confusion del clasificador de tickets">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="50%" y="35" text-anchor="middle" font-family="Arial" '
        'font-size="20" font-weight="bold">Matriz de confusion - tickets</text>',
        f'<text x="{left_margin + cell_size * len(labels) / 2}" y="65" '
        'text-anchor="middle" font-family="Arial" font-size="14">Prediccion</text>',
        f'<text x="24" y="{top_margin + cell_size * len(labels) / 2}" '
        'text-anchor="middle" font-family="Arial" font-size="14" '
        'transform="rotate(-90 24 '
        f'{top_margin + cell_size * len(labels) / 2})">Categoria real</text>',
    ]

    for index, label in enumerate(labels):
        x = left_margin + index * cell_size + cell_size / 2
        y = top_margin + index * cell_size + cell_size / 2 + 5
        elements.append(
            f'<text x="{x}" y="92" text-anchor="middle" font-family="Arial" '
            f'font-size="13">{escape(label)}</text>'
        )
        elements.append(
            f'<text x="{left_margin - 12}" y="{y}" text-anchor="end" '
            f'font-family="Arial" font-size="13">{escape(label)}</text>'
        )

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            intensity = value / maximum
            blue = int(245 - 135 * intensity)
            red_green = int(247 - 182 * intensity)
            text_color = "white" if intensity > 0.55 else "#102a43"
            x = left_margin + column_index * cell_size
            y = top_margin + row_index * cell_size
            elements.extend(
                (
                    f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                    f'fill="rgb({red_green},{red_green},{blue})" stroke="white"/>',
                    f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2 + 6}" '
                    f'text-anchor="middle" font-family="Arial" font-size="18" '
                    f'font-weight="bold" fill="{text_color}">{value}</text>',
                )
            )

    elements.append("</svg>")
    return "\n".join(elements)


class TicketClassifier:
    def __init__(self, model_path: Path) -> None:
        artifact = joblib.load(model_path)
        self.pipeline: Pipeline = artifact["pipeline"]
        self.metadata: dict[str, Any] = artifact["metadata"]

    def predict(self, text: str) -> dict[str, Any]:
        valid_text = validate_ticket_text(text)
        predicted_category = str(self.pipeline.predict([valid_text])[0])
        probabilities = self.pipeline.predict_proba([valid_text])[0]
        classifier = self.pipeline.named_steps["classifier"]

        return {
            "category": predicted_category,
            "probabilities": {
                str(category): float(probability)
                for category, probability in zip(classifier.classes_, probabilities, strict=True)
            },
            "model_name": self.metadata["selected_model"],
            "model_version": self.metadata["model_version"],
        }
