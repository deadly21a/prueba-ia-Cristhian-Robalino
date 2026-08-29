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

Para ejecutar las mismas validaciones usadas durante el desarrollo en Windows:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.venv\Scripts\python.exe -m ruff check app tests scripts
.venv\Scripts\python.exe -m pytest -q
```

Resultado local esperado: `9 passed`.

## Pruebas manuales de la API

Los ejemplos siguientes forman un flujo completo y pueden copiarse en PowerShell.
Primero inicie la API con `uvicorn app.main:app --reload`.

### 1. Autenticacion y refresh token

```powershell
$baseUrl = "http://127.0.0.1:8000"
$loginBody = @{
    username = "admin"
    password = "admin123"
} | ConvertTo-Json

$tokens = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/auth/login" `
    -ContentType "application/json" `
    -Body $loginBody

$headers = @{ Authorization = "Bearer $($tokens.access_token)" }

$refreshBody = @{
    refresh_token = $tokens.refresh_token
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/auth/refresh" `
    -ContentType "application/json" `
    -Body $refreshBody
```

### 2. Customers: crear, listar, consultar y actualizar

```powershell
$customerBody = @{
    name = "Cliente Demo"
    email = "cliente.demo@example.com"
    phone = "0991234567"
    plan_type = "fiber"
    tenure_months = 6
    monthly_charge = 75.50
    total_charges = 453.00
    contract_type = "month-to-month"
    payment_method = "card"
    num_tickets = 2
    avg_satisfaction = 3.5
} | ConvertTo-Json

$customer = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/customers" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $customerBody

$customerId = $customer.id

Invoke-RestMethod -Uri "$baseUrl/api/v1/customers" -Headers $headers
Invoke-RestMethod -Uri "$baseUrl/api/v1/customers/$customerId" -Headers $headers

$customerUpdate = @{
    name = "Cliente Demo Actualizado"
    email = "cliente.demo@example.com"
    phone = "0987654321"
    plan_type = "premium"
    tenure_months = 7
    monthly_charge = 90.00
    total_charges = 630.00
    contract_type = "one-year"
    payment_method = "bank-transfer"
    num_tickets = 2
    avg_satisfaction = 4.0
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Put `
    -Uri "$baseUrl/api/v1/customers/$customerId" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $customerUpdate

Invoke-RestMethod `
    -Uri "$baseUrl/api/v1/customers/$customerId/churn-prediction"
```

Prueba de validacion de telefono, cuyo resultado esperado es HTTP `422`:

```powershell
$invalidCustomer = @{
    name = "QA"
    email = "invalid@example.com"
    phone = "123"
} | ConvertTo-Json

Invoke-WebRequest `
    -SkipHttpErrorCheck `
    -Method Post `
    -Uri "$baseUrl/api/v1/customers" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $invalidCustomer
```

### 3. Tickets: clasificar, crear, listar, consultar y actualizar

```powershell
$classificationBody = @{
    description = "El router no enciende y no tengo conexion a internet desde ayer"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/tickets/classify" `
    -ContentType "application/json" `
    -Body $classificationBody

$ticketBody = @{
    customer_id = $customerId
    description = "El router no enciende y no tengo conexion a internet desde ayer"
    priority = "high"
} | ConvertTo-Json

$ticket = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/tickets" `
    -ContentType "application/json" `
    -Body $ticketBody

$ticketId = $ticket.id

Invoke-RestMethod -Uri "$baseUrl/api/v1/tickets"
Invoke-RestMethod -Uri "$baseUrl/api/v1/tickets/$ticketId"

$ticketUpdate = @{
    description = "Necesito revisar un cobro duplicado que aparece en mi factura mensual"
    priority = "medium"
    status = "in_progress"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Put `
    -Uri "$baseUrl/api/v1/tickets/$ticketId" `
    -ContentType "application/json" `
    -Body $ticketUpdate
```

El primer texto debe clasificarse como `TECH`; después de actualizarlo con una consulta
de facturacion debe clasificarse como `BILL`.

### 4. Endpoints de Machine Learning

```powershell
Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/ml/classify-ticket" `
    -ContentType "application/json" `
    -Body $classificationBody

$churnBody = @{
    tenure_months = 3
    monthly_charge = 110
    total_charges = 330
    contract_type = "month-to-month"
    payment_method = "cash"
    num_tickets = 7
    avg_satisfaction = 1.5
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/ml/predict-churn" `
    -ContentType "application/json" `
    -Body $churnBody

$sentimentBody = @{
    description = "Estoy muy molesto porque el servicio nunca funciona"
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/ml/analyze-sentiment" `
    -ContentType "application/json" `
    -Body $sentimentBody

Invoke-RestMethod -Uri "$baseUrl/api/v1/ml/models/info"
```

Resultados esperados: probabilidades entre `0` y `1`, sentimiento `negative` y ambos
modelos Scikit-learn reportados como disponibles.

### 5. Agente conversacional y memoria de sesion

```powershell
$chatBody = @{
    message = "Estoy frustrado porque el wifi nunca funciona"
    customer_id = $customerId
} | ConvertTo-Json

$chat = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/agent/chat" `
    -ContentType "application/json" `
    -Body $chatBody

$sessionId = $chat.session_id

$followUpBody = @{
    message = "Gracias, ahora ya funciona correctamente"
    session_id = $sessionId
    customer_id = $customerId
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/v1/agent/chat" `
    -ContentType "application/json" `
    -Body $followUpBody

Invoke-RestMethod -Uri "$baseUrl/api/v1/agent/sessions/$sessionId"
```

La primera respuesta debe indicar intención `technical`, `escalate: true` e incluir
en el contexto la categoria del ticket, informacion del cliente y su riesgo de churn.
La consulta de la sesion debe contener cuatro mensajes.

### 6. MCP: capacidades, recursos y cinco tools

```powershell
Invoke-RestMethod -Uri "$baseUrl/mcp/capabilities"
Invoke-RestMethod -Uri "$baseUrl/mcp/resources"
Invoke-RestMethod -Uri "$baseUrl/mcp/resources/customers"

function Invoke-McpTool($requestId, $tool, $arguments) {
    $body = @{
        jsonrpc = "2.0"
        id = $requestId
        tool = $tool
        arguments = $arguments
    } | ConvertTo-Json -Depth 8

    Invoke-RestMethod `
        -Method Post `
        -Uri "$baseUrl/mcp/tools/execute" `
        -ContentType "application/json" `
        -Body $body
}

Invoke-McpTool "mcp-1" "classify_ticket" @{
    description = "Tengo un cobro duplicado en la factura de este mes"
}

Invoke-McpTool "mcp-2" "predict_churn" @{
    tenure_months = 3
    monthly_charge = 110
    total_charges = 330
    contract_type = "month-to-month"
    payment_method = "cash"
    num_tickets = 7
    avg_satisfaction = 1.5
}

Invoke-McpTool "mcp-3" "get_customer_info" @{
    customer_id = $customerId
}

Invoke-McpTool "mcp-4" "create_ticket" @{
    customer_id = $customerId
    description = "La conexion de internet se desconecta constantemente durante el dia"
    priority = "high"
}

Invoke-McpTool "mcp-5" "chat_with_agent" @{
    message = "Necesito ayuda con mi factura"
    customer_id = $customerId
}
```

Cada respuesta debe conservar `jsonrpc: "2.0"`, el mismo `id` y
`result.isError: false`.

### 7. Eliminaciones logicas al terminar las pruebas

```powershell
Invoke-RestMethod `
    -Method Delete `
    -Uri "$baseUrl/api/v1/agent/sessions/$sessionId"

Invoke-RestMethod `
    -Method Delete `
    -Uri "$baseUrl/api/v1/customers/$customerId" `
    -Headers $headers
```

Una consulta posterior de cualquiera de los dos recursos debe responder HTTP `404`.

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
