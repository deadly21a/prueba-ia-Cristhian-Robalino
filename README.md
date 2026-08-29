# Sistema Inteligente de Atencion al Cliente

Prueba tecnica de IA/ML para una empresa de telecomunicaciones.

El proyecto integra modelos de machine learning, modelos de deep learning, un agente
conversacional con LangGraph y una API REST con FastAPI y compatibilidad MCP.

## Estado del desarrollo

El proyecto se implementa y valida por secciones:

1. Base del proyecto y configuracion.
2. Clasificacion de tickets con Scikit-learn.
3. Prediccion de churn con Scikit-learn.
4. Modelos de sentimiento y tiempo de resolucion con TensorFlow/Keras.
5. Agente conversacional con LangGraph.
6. API REST, autenticacion y persistencia.
7. Protocolo MCP e integracion.
8. Pruebas, Docker y documentacion final.

## Requisitos

- Python 3.12
- Docker y Docker Compose (opcional para desarrollo local)

## Ejecucion local

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

La documentacion interactiva estara disponible en:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health check: `http://localhost:8000/health`

## Ejecucion con Docker

```bash
docker compose up --build
```

Docker Compose utiliza los valores seguros de desarrollo cuando no existe un archivo
`.env`. Para personalizar la configuracion, cree `.env` a partir de `.env.example`.

## Pruebas

```bash
pytest
```

## Estructura

```text
app/
  agent/       # Grafo conversacional y memoria
  api/         # Endpoints REST y dependencias HTTP
  core/        # Configuracion y seguridad
  db/          # ORM, sesiones y repositorios
  mcp/         # Capacidades, recursos y herramientas MCP
  ml/          # Datos, entrenamiento, artefactos e inferencia
tests/         # Pruebas unitarias y de integracion
scripts/       # Entrenamiento y utilidades del proyecto
sql/           # Funciones y scripts SQL
```

## Autor

Cristhian Robalino

Las decisiones tecnicas, tiempos dedicados, dificultades y credenciales de prueba se
documentaran al completar cada seccion.
