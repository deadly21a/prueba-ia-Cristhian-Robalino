from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

from app.ml.tickets.model import _render_confusion_matrix_svg

SENTIMENT_LABELS = ("negative", "neutral", "positive")


def build_sentiment_model(vocabulary_size: int = 10_000, max_length: int = 200):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(max_length,), name="text_tokens")
    embedded = tf.keras.layers.Embedding(vocabulary_size, 64, mask_zero=True)(inputs)
    encoded = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))(embedded)
    regularized = tf.keras.layers.Dropout(0.3)(encoded)
    hidden = tf.keras.layers.Dense(32, activation="relu")(regularized)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(hidden)
    model = tf.keras.Model(inputs, outputs, name="sentiment_classifier")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_resolution_model(vocabulary_size: int = 10_000, max_length: int = 200):
    import tensorflow as tf

    text_input = tf.keras.Input(shape=(max_length,), name="description")
    category_input = tf.keras.Input(shape=(5,), name="category")
    numeric_input = tf.keras.Input(shape=(3,), name="numeric")
    embedded = tf.keras.layers.Embedding(vocabulary_size, 32, mask_zero=True)(text_input)
    encoded = tf.keras.layers.GRU(16)(embedded)
    combined = tf.keras.layers.Concatenate()([encoded, category_input, numeric_input])
    hidden = tf.keras.layers.Dense(32, activation="relu")(combined)
    hidden = tf.keras.layers.Dropout(0.25)(hidden)
    output = tf.keras.layers.Dense(1, name="resolution_hours")(hidden)
    model = tf.keras.Model([text_input, category_input, numeric_input], output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "root_mean_squared_error"])
    return model


def sentiment_heuristic(text: str) -> dict[str, Any]:
    lowered = text.lower()
    negative = ("molesto", "pésimo", "terrible", "frustrado", "enojado", "nunca funciona")
    positive = ("gracias", "excelente", "contento", "perfecto", "muy bien")
    label = (
        "negative"
        if any(word in lowered for word in negative)
        else ("positive" if any(word in lowered for word in positive) else "neutral")
    )
    probabilities = {item: 0.1 for item in SENTIMENT_LABELS}
    probabilities[label] = 0.8
    return {"sentiment": label, "probabilities": probabilities, "source": "keyword_fallback"}


@lru_cache
def _load_sentiment_assets():
    import tensorflow as tf

    model = tf.keras.models.load_model("artifacts/models/sentiment_classifier.keras")
    tokenizer_data = Path("artifacts/models/sentiment_tokenizer.json").read_text(encoding="utf-8")
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
    return model, tokenizer


def analyze_sentiment(text: str) -> dict[str, Any]:
    try:
        import tensorflow as tf

        model, tokenizer = _load_sentiment_assets()
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            tokenizer.texts_to_sequences([text]), maxlen=200, padding="post", truncating="post"
        )
        values = model.predict(sequence, verbose=0)[0]
        probabilities = {
            label: float(values[index]) for index, label in enumerate(SENTIMENT_LABELS)
        }
        sentiment = max(probabilities, key=probabilities.get)
        return {"sentiment": sentiment, "probabilities": probabilities, "source": "keras"}
    except Exception:
        return sentiment_heuristic(text)


def _history_svg(history: dict[str, list[float]], title: str) -> str:
    width, height, margin = 720, 420, 55
    series = {
        key: values
        for key, values in history.items()
        if key in {"loss", "val_loss", "accuracy", "val_accuracy"}
    }
    maximum = max(max(values) for values in series.values()) or 1
    colors = {
        "loss": "#d62728",
        "val_loss": "#ff9896",
        "accuracy": "#1f77b4",
        "val_accuracy": "#9ecae1",
    }
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="360" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
    ]
    for index, (name, values) in enumerate(series.items()):
        points = []
        for epoch, value in enumerate(values):
            x = margin + epoch * (width - 2 * margin) / max(1, len(values) - 1)
            y = height - margin - value * (height - 2 * margin) / maximum
            points.append(f"{x:.1f},{y:.1f}")
        elements.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[name]}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{margin + index * 150}" y="{height - 15}" font-family="Arial" fill="{colors[name]}">{name}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def export_architecture_summaries(output_path: Path) -> None:
    sentiment = build_sentiment_model()
    resolution = build_resolution_model()
    summary = {
        "sentiment": {
            "parameters": sentiment.count_params(),
            "input_length": 200,
            "vocabulary": 10000,
        },
        "resolution": {"parameters": resolution.count_params(), "mixed_inputs": True},
        "callbacks": ["EarlyStopping(patience=3)", "ModelCheckpoint", "ReduceLROnPlateau"],
        "resolution_metrics": ["MAE", "RMSE", "R2 (evaluated externally)"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def train_deep_learning_models(artifacts_dir: Path, reports_dir: Path) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    phrases = {
        0: [
            "Estoy muy molesto porque el servicio nunca funciona",
            "La atención fue pésima y sigo sin conexión",
            "Es terrible que no resuelvan mi problema de internet",
            "Estoy frustrado por tantos cobros incorrectos",
            "El soporte fue lento y no solucionó absolutamente nada",
        ],
        1: [
            "Necesito consultar el estado actual de mi solicitud",
            "Quiero información sobre los planes disponibles",
            "La factura llegó hoy y deseo revisar sus detalles",
            "Tengo una consulta relacionada con mi cuenta",
            "Por favor indiquen el horario normal de atención",
        ],
        2: [
            "Muchas gracias por resolver mi problema rápidamente",
            "La atención fue excelente y todo funciona muy bien",
            "Estoy contento con la solución proporcionada",
            "Perfecto ahora la conexión funciona correctamente",
            "El agente fue amable y solucionó todo enseguida",
        ],
    }
    texts: list[str] = []
    labels: list[int] = []
    for label, examples in phrases.items():
        for repetition in range(8):
            for example in examples:
                texts.append(f"{example} caso {repetition}")
                labels.append(label)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10_000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(
        tokenizer.texts_to_sequences(texts), maxlen=200, padding="post", truncating="post"
    )
    labels_array = np.asarray(labels)
    permutation = np.random.default_rng(42).permutation(len(labels_array))
    sequences = sequences[permutation]
    labels_array = labels_array[permutation]
    sentiment_model = build_sentiment_model()
    sentiment_checkpoint = artifacts_dir / "sentiment_classifier.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(sentiment_checkpoint, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]
    history = sentiment_model.fit(
        sequences,
        labels_array,
        validation_split=0.2,
        epochs=8,
        batch_size=16,
        callbacks=callbacks,
        verbose=0,
    )
    sentiment_model.save(sentiment_checkpoint)
    (reports_dir / "sentiment_training_curves.svg").write_text(
        _history_svg(history.history, "Entrenamiento de sentimiento"), encoding="utf-8"
    )
    predictions = np.argmax(sentiment_model.predict(sequences, verbose=0), axis=1)
    matrix = confusion_matrix(labels_array, predictions, labels=[0, 1, 2]).tolist()
    (reports_dir / "sentiment_confusion_matrix.svg").write_text(
        _render_confusion_matrix_svg(matrix, list(SENTIMENT_LABELS)), encoding="utf-8"
    )
    (artifacts_dir / "sentiment_tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")

    generator = np.random.default_rng(42)
    rows = 140
    text_inputs = sequences[np.arange(rows) % len(sequences)]
    categories = np.eye(5, dtype=np.float32)[generator.integers(0, 5, rows)]
    priority = generator.integers(0, 4, rows)
    hour = generator.integers(0, 24, rows)
    weekday = generator.integers(0, 7, rows)
    numeric = np.column_stack((priority / 3, hour / 23, weekday / 6)).astype(np.float32)
    targets = (
        1.5
        + priority * 4
        + categories[:, 0] * 5
        + categories[:, 3] * 2
        + generator.normal(0, 0.5, rows)
    ).astype(np.float32)
    resolution_model = build_resolution_model()
    resolution_path = artifacts_dir / "resolution_time.keras"
    resolution_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(resolution_path, save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]
    resolution_history = resolution_model.fit(
        {"description": text_inputs, "category": categories, "numeric": numeric},
        targets,
        validation_split=0.2,
        epochs=30,
        batch_size=16,
        callbacks=resolution_callbacks,
        verbose=0,
    )
    resolution_model.save(resolution_path)
    (reports_dir / "resolution_training_curves.svg").write_text(
        _history_svg(resolution_history.history, "Entrenamiento de tiempo de resolucion"),
        encoding="utf-8",
    )
    resolution_predictions = resolution_model.predict(
        {"description": text_inputs, "category": categories, "numeric": numeric}, verbose=0
    ).reshape(-1)
    metrics = {
        "sentiment": {
            "accuracy": float(np.mean(predictions == labels_array)),
            "epochs": len(history.history["loss"]),
            "final_loss": float(history.history["loss"][-1]),
            "architecture": "Embedding + Bidirectional GRU + Dropout + Dense",
            "vocabulary_max": 10_000,
            "sequence_length": 200,
        },
        "resolution_time": {
            "mae": float(mean_absolute_error(targets, resolution_predictions)),
            "rmse": float(mean_squared_error(targets, resolution_predictions) ** 0.5),
            "r2": float(r2_score(targets, resolution_predictions)),
            "epochs": len(resolution_history.history["loss"]),
            "mixed_inputs": [
                "category_one_hot",
                "priority",
                "description_embedding",
                "hour",
                "weekday",
            ],
        },
        "callbacks": ["EarlyStopping(patience=3)", "ModelCheckpoint", "ReduceLROnPlateau"],
    }
    (reports_dir / "deep_learning_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics
