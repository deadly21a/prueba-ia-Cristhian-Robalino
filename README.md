# Sistema Inteligente de Atencion al Cliente

Prueba tecnica de IA/ML para una empresa de telecomunicaciones.

El proyecto integra modelos de machine learning, modelos de deep learning, un agente
conversacional con LangGraph y una API REST con FastAPI y compatibilidad MCP.

## Estado del desarrollo

El proyecto se implementa y valida por secciones:

1. Base del proyecto y configuracion.
2. Clasificacion de tickets con Scikit-learn (implementacion local en validacion).
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

## Clasificador de tickets

El clasificador reconoce cinco categorias:

- `TECH`: problemas tecnicos.
- `BILL`: consultas de facturacion.
- `PLAN`: cambios de plan o servicios.
- `CNCL`: cancelacion del servicio.
- `OTHR`: otras solicitudes.

Si no se dispone del archivo proporcionado por la empresa, el comando de entrenamiento
genera un dataset sintetico, balanceado y reproducible en espanol. El origen sintetico
queda registrado dentro del propio CSV y en el informe de metricas.

```powershell
.venv\Scripts\python.exe -m scripts.train_ticket_classifier
```

El entrenamiento compara regresion logistica y Naive Bayes mediante validacion cruzada
estratificada de cinco folds. El mejor pipeline se selecciona por F1 macro y genera:

- `data/raw/tickets_train.csv`
- `artifacts/models/ticket_classifier.joblib`
- `reports/ticket_classifier_metrics.json`
- `reports/ticket_confusion_matrix.svg`

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

## Credenciales de prueba

| Usuario | Contrasena | Rol |
| --- | --- | --- |
| `admin` | `admin123` | admin |
| `agent` | `agent123` | agent |
| `customer` | `customer123` | customer |

Estas credenciales son exclusivamente de desarrollo. Los tokens JWT de acceso expiran
en 30 minutos y los refresh tokens en 7 dias; ambos valores son configurables.

## Decisiones tecnicas

- Arquitectura por capas para separar HTTP, dominio, persistencia, ML y agente.
- SQLite para desarrollo y SQLAlchemy 2 para permitir migrar a PostgreSQL.
- Datos sinteticos reproducibles debido a que no se recibieron los CSV mencionados.
- TF-IDF con regresion logistica o Naive Bayes para una inferencia rapida y explicable.
- Regresion logistica balanceada para churn, con dos features derivados.
- Keras con GRU para sentimiento y red de entradas mixtas para tiempo de resolucion.
- LangGraph sin proveedor LLM externo para que la demostracion no requiera claves.
- Eliminacion logica mediante `is_active` y `deleted_at`.

## Resultados locales

- Clasificador de tickets: dos modelos, 5-fold CV, metricas por categoria y matriz de confusion.
- Churn: AUC-ROC, average precision, curvas, correlaciones e importancia de variables.
- Deep learning: modelos `.keras`, callbacks, matriz de confusion y metricas de regresion.
- API: pruebas de autenticacion, clientes, tickets, ML, agente y MCP.

Los resultados se obtuvieron con datos sinteticos y demuestran el funcionamiento del
pipeline; no representan rendimiento esperado con datos reales de clientes.

## Dificultades encontradas

- No se proporcionaron datasets reales, por lo que se generaron datasets balanceados y reproducibles.
- Docker no esta instalado en el equipo de desarrollo; sus archivos se validaron estaticamente.
- La politica de Control de aplicaciones de Windows bloqueo temporalmente DLL nativas de
  TensorFlow y SQLite; despues de habilitarlas se ejecutaron las pruebas y entrenamientos.

## Tiempo dedicado

| Seccion | Tiempo aproximado |
| --- | ---: |
| Arquitectura y configuracion | 15 min |
| Scikit-learn: tickets y churn | 25 min |
| TensorFlow/Keras | 5 min |
| LangGraph | 5 min |
| FastAPI, SQLAlchemy, JWT y MCP | 5 min |
| Pruebas y documentacion | 5 min |
