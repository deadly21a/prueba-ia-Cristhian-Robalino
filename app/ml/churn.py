from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "num_tickets",
    "avg_satisfaction",
    "charge_per_tenure",
    "support_pressure",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
BASE_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "contract_type",
    "payment_method",
    "num_tickets",
    "avg_satisfaction",
]


def engineer_churn_features(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = frame.copy()
    engineered["charge_per_tenure"] = engineered["total_charges"] / engineered[
        "tenure_months"
    ].clip(lower=1)
    engineered["support_pressure"] = engineered["num_tickets"] / engineered["tenure_months"].clip(
        lower=1
    )
    return engineered


def generate_churn_dataset(rows: int = 600, random_state: int = 42) -> pd.DataFrame:
    generator = random.Random(random_state)
    records = []
    for _ in range(rows):
        tenure = generator.randint(1, 72)
        monthly = round(generator.uniform(20, 130), 2)
        tickets = generator.randint(0, 10)
        satisfaction = round(generator.uniform(1, 5), 1)
        contract = generator.choices(
            ["month-to-month", "one-year", "two-year"], weights=[0.55, 0.28, 0.17]
        )[0]
        payment = generator.choice(["card", "bank-transfer", "cash"])
        total = round(monthly * tenure * generator.uniform(0.92, 1.03), 2)
        risk_logit = (
            -1.5
            + 1.35 * (contract == "month-to-month")
            + 0.15 * tickets
            - 0.035 * tenure
            - 0.6 * (satisfaction - 3)
            + 0.009 * (monthly - 60)
        )
        probability = 1 / (1 + np.exp(-risk_logit))
        churn = int(generator.random() < probability)
        records.append(
            {
                "tenure_months": tenure,
                "monthly_charge": monthly,
                "total_charges": total,
                "contract_type": contract,
                "payment_method": payment,
                "num_tickets": tickets,
                "avg_satisfaction": satisfaction,
                "churn": churn,
            }
        )
    return pd.DataFrame(records)


def train_churn_model(dataset: pd.DataFrame, model_path: Path, report_path: Path) -> dict[str, Any]:
    from sklearn.model_selection import train_test_split

    dataset = engineer_churn_features(dataset)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        dataset["churn"],
        test_size=0.2,
        stratify=dataset["churn"],
        random_state=42,
    )
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    precision_values, recall_values, _ = precision_recall_curve(y_test, probabilities)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    importance = sorted(
        zip(feature_names.tolist(), np.abs(coefficients).tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    report = {
        "model_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_rows": len(dataset),
        "churn_rate": float(dataset["churn"].mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "precision_recall_curve": {
            "precision": precision_values.tolist(),
            "recall": recall_values.tolist(),
        },
        "roc_curve": {
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
        },
        "numeric_correlations": dataset.select_dtypes(include="number")
        .corr()["churn"]
        .sort_values(ascending=False)
        .to_dict(),
        "feature_importance": [
            {"feature": feature, "importance": value} for feature, value in importance
        ],
        "derived_features": ["charge_per_tenure", "support_pressure"],
    }
    pipeline.fit(dataset[NUMERIC_FEATURES + CATEGORICAL_FEATURES], dataset["churn"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": report}, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


class ChurnPredictor:
    def __init__(self, model_path: Path) -> None:
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.metadata = artifact["metadata"]

    def predict(self, values: dict[str, Any]) -> dict[str, Any]:
        frame = engineer_churn_features(pd.DataFrame([{key: values[key] for key in BASE_FEATURES}]))
        probability = float(
            self.pipeline.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[0, 1]
        )
        risk = "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"
        return {"churn_probability": probability, "risk_level": risk, "model_version": "1.0.0"}
