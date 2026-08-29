from pathlib import Path

import pytest

from app.ml.tickets.data import TICKET_CATEGORIES, generate_ticket_dataset
from app.ml.tickets.model import (
    SpanishTextCleaner,
    TicketClassifier,
    save_training_outputs,
    train_ticket_classifier,
    validate_ticket_text,
)


def test_synthetic_dataset_is_balanced_and_reproducible() -> None:
    first_dataset = generate_ticket_dataset(samples_per_category=5, random_state=7)
    second_dataset = generate_ticket_dataset(samples_per_category=5, random_state=7)

    assert first_dataset.equals(second_dataset)
    assert first_dataset["category"].value_counts().to_dict() == {
        category: 5 for category in TICKET_CATEGORIES
    }


def test_spanish_cleaner_preserves_accents_and_enye() -> None:
    cleaned_text = SpanishTextCleaner.clean("¡CONEXIÓN lenta! Niño: https://example.com")

    assert cleaned_text == "conexión lenta niño"


@pytest.mark.parametrize("invalid_text", ["", " corto ", "123456789"])
def test_ticket_description_requires_ten_characters(invalid_text: str) -> None:
    with pytest.raises(ValueError, match="al menos 10 caracteres"):
        validate_ticket_text(invalid_text)


def test_trained_classifier_returns_category_and_all_probabilities(tmp_path: Path) -> None:
    dataset = generate_ticket_dataset(samples_per_category=10, random_state=11)
    result = train_ticket_classifier(dataset, random_state=11)
    model_path = tmp_path / "ticket_classifier.joblib"

    save_training_outputs(
        result,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        confusion_matrix_path=tmp_path / "matrix.svg",
    )
    classifier = TicketClassifier(model_path)
    prediction = classifier.predict(
        "Mi conexión de internet está muy lenta y el módem pierde la señal"
    )

    assert prediction["category"] == "TECH"
    assert set(prediction["probabilities"]) == set(TICKET_CATEGORIES)
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert model_path.exists()


def test_billing_status_phrase_is_not_confused_with_cancellation(tmp_path: Path) -> None:
    dataset = generate_ticket_dataset(samples_per_category=10, random_state=17)
    result = train_ticket_classifier(dataset, random_state=17)
    model_path = tmp_path / "ticket_classifier.joblib"
    save_training_outputs(
        result,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        confusion_matrix_path=tmp_path / "matrix.svg",
    )

    prediction = TicketClassifier(model_path).predict("Quiero saber mi estado de facturacion")

    assert prediction["category"] == "BILL"
    assert prediction["classification_source"] == "domain_rule"
    assert prediction["probabilities"]["BILL"] == pytest.approx(0.8)
