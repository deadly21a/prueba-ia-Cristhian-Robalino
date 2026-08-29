from pathlib import Path

from app.ml.churn import generate_churn_dataset, train_churn_model
from app.ml.deep_learning import train_deep_learning_models


def main() -> None:
    churn_data = generate_churn_dataset()
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    churn_data.to_csv("data/raw/churn_train.csv", index=False)
    churn_report = train_churn_model(
        churn_data,
        Path("artifacts/models/churn_predictor.joblib"),
        Path("reports/churn_metrics.json"),
    )
    print(f"Churn AUC-ROC: {churn_report['roc_auc']:.4f}")
    deep_metrics = train_deep_learning_models(Path("artifacts/models"), Path("reports"))
    print(f"Sentiment accuracy: {deep_metrics['sentiment']['accuracy']:.4f}")
    print(f"Resolution MAE: {deep_metrics['resolution_time']['mae']:.4f}")


if __name__ == "__main__":
    main()
