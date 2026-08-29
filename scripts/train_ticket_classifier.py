from __future__ import annotations

import argparse
from pathlib import Path

from app.ml.tickets.data import (
    generate_ticket_dataset,
    load_ticket_dataset,
    save_ticket_dataset,
)
from app.ml.tickets.model import save_training_outputs, train_ticket_classifier

DEFAULT_DATASET_PATH = Path("data/raw/tickets_train.csv")
DEFAULT_MODEL_PATH = Path("artifacts/models/ticket_classifier.joblib")
DEFAULT_METRICS_PATH = Path("reports/ticket_classifier_metrics.json")
DEFAULT_MATRIX_PATH = Path("reports/ticket_confusion_matrix.svg")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y evalua el clasificador de tickets de soporte."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--samples-per-category", type=int, default=80)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--regenerate-synthetic-data",
        action="store_true",
        help="Reemplaza el CSV existente por un dataset sintetico reproducible.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.regenerate_synthetic_data or not arguments.dataset.exists():
        dataset = generate_ticket_dataset(
            samples_per_category=arguments.samples_per_category,
            random_state=arguments.random_state,
        )
        save_ticket_dataset(dataset, arguments.dataset)

    dataset = load_ticket_dataset(arguments.dataset)
    result = train_ticket_classifier(dataset, random_state=arguments.random_state)
    save_training_outputs(
        result,
        model_path=arguments.model_output,
        metrics_path=arguments.metrics_output,
        confusion_matrix_path=arguments.matrix_output,
    )

    selected_model = result.metadata["selected_model"]
    test_accuracy = result.metadata["test_evaluation"]["accuracy"]
    print(f"Dataset: {arguments.dataset} ({len(dataset)} filas)")
    print(f"Modelo seleccionado: {selected_model}")
    print(f"Accuracy en test: {test_accuracy:.4f}")
    print(f"Modelo guardado en: {arguments.model_output}")
    print(f"Metricas guardadas en: {arguments.metrics_output}")
    print(f"Matriz guardada en: {arguments.matrix_output}")


if __name__ == "__main__":
    main()
