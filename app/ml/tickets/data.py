from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

TICKET_CATEGORIES = ("TECH", "BILL", "PLAN", "CNCL", "OTHR")

_TEMPLATES: dict[str, tuple[str, ...]] = {
    "TECH": (
        "Mi conexión a internet está demasiado lenta y no puedo trabajar",
        "El módem tiene una luz roja y desde ayer no tengo conexión",
        "La señal de wifi se desconecta constantemente en toda la casa",
        "No puedo navegar aunque ya reinicié el router varias veces",
        "La velocidad contratada no coincide con la que recibo actualmente",
        "El servicio de televisión muestra la pantalla congelada y sin audio",
        "Necesito soporte porque el internet se cae cada pocos minutos",
        "El router no enciende después del último corte de energía",
    ),
    "BILL": (
        "Mi factura tiene un cobro adicional que no reconozco este mes",
        "Necesito una explicación del valor total facturado en mi cuenta",
        "Ya pagué la mensualidad pero todavía aparece como una deuda pendiente",
        "Solicito una copia de la factura con el detalle de todos los cargos",
        "Me cobraron dos veces el mismo servicio en la tarjeta de crédito",
        "El descuento ofrecido no fue aplicado en mi última factura",
        "Quiero actualizar el método de pago asociado con mi cuenta",
        "La fecha límite de pago indicada en la factura parece incorrecta",
    ),
    "PLAN": (
        "Quiero cambiar mi plan actual por uno con mayor velocidad de internet",
        "Necesito conocer los planes disponibles y sus beneficios mensuales",
        "Deseo agregar televisión y telefonía a mi paquete de servicios",
        "Quiero reducir la velocidad para contratar un plan más económico",
        "Solicito activar un paquete adicional de canales deportivos",
        "Necesito migrar mi contrato al nuevo plan de fibra óptica",
        "Quisiera retirar la telefonía y mantener solamente el servicio de internet",
        "Deseo renovar mi plan y conocer las promociones vigentes para clientes",
    ),
    "CNCL": (
        "Quiero cancelar definitivamente el servicio porque ya no lo necesito",
        "Solicito terminar mi contrato y conocer el proceso de cancelación",
        "Me voy a mudar y deseo dar de baja todos los servicios contratados",
        "Necesito cancelar la cuenta antes del próximo ciclo de facturación",
        "No estoy conforme con el servicio y quiero finalizar el contrato",
        "Deseo retirar todos los equipos y cerrar mi cuenta de cliente",
        "Quiero anular la renovación y cancelar el plan al finalizar este mes",
        "Solicito la baja inmediata del internet y de la televisión contratada",
    ),
    "OTHR": (
        "Necesito actualizar el correo electrónico registrado en mi perfil",
        "Quiero cambiar el nombre del titular asociado con la cuenta",
        "¿Cuál es el horario de atención de la oficina ubicada en el centro?",
        "Necesito solicitar una copia del contrato firmado cuando instalé el servicio",
        "Quiero registrar un nuevo número telefónico para recibir notificaciones",
        "Deseo conocer la dirección de la sucursal más cercana a mi domicilio",
        "Necesito corregir un dato personal que aparece mal escrito en el sistema",
        "Quisiera dejar un comentario sobre la atención recibida por el asesor",
    ),
}

_PREFIXES = (
    "Hola, por favor ayúdenme.",
    "Buenos días, necesito asistencia.",
    "Buenas tardes.",
    "Estimado equipo de atención al cliente:",
    "Por favor revisen mi solicitud.",
)

_SUFFIXES = (
    "Agradezco una respuesta lo antes posible.",
    "¿Podrían indicarme qué debo hacer?",
    "El problema continúa hasta este momento.",
    "Quedo atento a su pronta respuesta.",
    "Necesito resolverlo durante el día de hoy.",
)


def generate_ticket_dataset(
    samples_per_category: int = 80,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a balanced and reproducible Spanish support-ticket dataset."""
    if samples_per_category < 5:
        raise ValueError("samples_per_category debe ser al menos 5 para validacion cruzada")

    random_generator = random.Random(random_state)
    rows: list[dict[str, str]] = []

    for category in TICKET_CATEGORIES:
        templates = _TEMPLATES[category]
        for index in range(samples_per_category):
            description = " ".join(
                (
                    random_generator.choice(_PREFIXES),
                    random_generator.choice(templates),
                    random_generator.choice(_SUFFIXES),
                    f"Referencia {category}-{index + 1:03d}.",
                )
            )
            rows.append(
                {
                    "description": description,
                    "category": category,
                    "source": "synthetic",
                }
            )

    random_generator.shuffle(rows)
    return pd.DataFrame(rows, columns=["description", "category", "source"])


def save_ticket_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, encoding="utf-8")


def load_ticket_dataset(input_path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(input_path, encoding="utf-8")
    required_columns = {"description", "category"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")

    invalid_categories = set(dataset["category"].dropna().unique()) - set(TICKET_CATEGORIES)
    if invalid_categories:
        raise ValueError(f"Categorias no soportadas: {sorted(invalid_categories)}")

    if dataset[list(required_columns)].isna().any().any():
        raise ValueError("El dataset contiene descripciones o categorias nulas")

    category_counts = dataset["category"].value_counts()
    categories_with_few_samples = category_counts[category_counts < 5].index.tolist()
    if categories_with_few_samples:
        raise ValueError(
            "Cada categoria necesita al menos 5 ejemplos; insuficientes: "
            f"{categories_with_few_samples}"
        )
    return dataset
