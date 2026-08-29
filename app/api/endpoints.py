from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.graph import build_agent_graph
from app.api.schemas import (
    AgentChatRequest,
    ChurnRequest,
    CustomerCreate,
    CustomerResponse,
    CustomerUpdate,
    LoginRequest,
    RefreshRequest,
    TicketCreate,
    TicketResponse,
    TicketTextRequest,
    TicketUpdate,
    TokenResponse,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    require_roles,
)
from app.db.models import AgentSession, Customer, Ticket
from app.db.session import get_db
from app.ml.churn import ChurnPredictor
from app.ml.deep_learning import analyze_sentiment as analyze_sentiment_model
from app.ml.tickets.model import TicketClassifier

router = APIRouter()

DEMO_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "agent": {"password": "agent123", "role": "agent"},
    "customer": {"password": "customer123", "role": "customer"},
}
TICKET_MODEL_PATH = Path("artifacts/models/ticket_classifier.joblib")
CHURN_MODEL_PATH = Path("artifacts/models/churn_predictor.joblib")


def ticket_classifier() -> TicketClassifier:
    if not TICKET_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="El clasificador de tickets no esta entrenado")
    return TicketClassifier(TICKET_MODEL_PATH)


def churn_predictor() -> ChurnPredictor:
    if not CHURN_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="El modelo de churn no esta entrenado")
    return ChurnPredictor(CHURN_MODEL_PATH)


@router.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login(payload: LoginRequest) -> TokenResponse:
    user = DEMO_USERS.get(payload.username)
    if not user or user["password"] != payload.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales invalidas"
        )
    return TokenResponse(
        access_token=create_access_token(payload.username, user["role"]),
        refresh_token=create_refresh_token(payload.username, user["role"]),
    )


@router.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
def refresh_token(payload: RefreshRequest) -> TokenResponse:
    try:
        token_data = decode_token(payload.refresh_token, "refresh")
    except ValueError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    return TokenResponse(
        access_token=create_access_token(token_data["sub"], token_data["role"]),
        refresh_token=create_refresh_token(token_data["sub"], token_data["role"]),
    )


@router.get(
    "/customers",
    response_model=list[CustomerResponse],
    tags=["Customers"],
    dependencies=[Depends(require_roles("admin", "agent"))],
)
def list_customers(database: Session = Depends(get_db)) -> list[Customer]:
    return list(database.scalars(select(Customer).where(Customer.is_active.is_(True))))


@router.get(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["Customers"],
    dependencies=[Depends(require_roles("admin", "agent", "customer"))],
)
def get_customer(customer_id: int, database: Session = Depends(get_db)) -> Customer:
    customer = database.scalar(
        select(Customer).where(Customer.id == customer_id, Customer.is_active.is_(True))
    )
    if not customer:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
    return customer


@router.post(
    "/customers",
    response_model=CustomerResponse,
    status_code=201,
    tags=["Customers"],
    dependencies=[Depends(require_roles("admin", "agent"))],
)
def create_customer(payload: CustomerCreate, database: Session = Depends(get_db)) -> Customer:
    if database.scalar(select(Customer).where(Customer.email == str(payload.email))):
        raise HTTPException(status_code=409, detail="El correo ya esta registrado")
    customer = Customer(**payload.model_dump(mode="json"))
    database.add(customer)
    database.commit()
    database.refresh(customer)
    return customer


@router.put(
    "/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["Customers"],
    dependencies=[Depends(require_roles("admin", "agent"))],
)
def update_customer(
    customer_id: int, payload: CustomerUpdate, database: Session = Depends(get_db)
) -> Customer:
    customer = get_customer(customer_id, database)
    for key, value in payload.model_dump(mode="json").items():
        setattr(customer, key, value)
    database.commit()
    database.refresh(customer)
    return customer


@router.delete(
    "/customers/{customer_id}", tags=["Customers"], dependencies=[Depends(require_roles("admin"))]
)
def delete_customer(customer_id: int, database: Session = Depends(get_db)) -> dict[str, bool]:
    customer = get_customer(customer_id, database)
    customer.is_active = False
    customer.deleted_at = datetime.now(UTC)
    database.commit()
    return {"deleted": True}


@router.get("/customers/{customer_id}/churn-prediction", tags=["Customers"])
def customer_churn(customer_id: int, database: Session = Depends(get_db)) -> dict:
    customer = get_customer(customer_id, database)
    values = {key: getattr(customer, key) for key in ChurnRequest.model_fields}
    return churn_predictor().predict(values)


@router.get("/tickets", response_model=list[TicketResponse], tags=["Tickets"])
def list_tickets(database: Session = Depends(get_db)) -> list[Ticket]:
    return list(database.scalars(select(Ticket).where(Ticket.is_active.is_(True))))


@router.get("/tickets/{ticket_id}", response_model=TicketResponse, tags=["Tickets"])
def get_ticket(ticket_id: int, database: Session = Depends(get_db)) -> Ticket:
    ticket = database.scalar(
        select(Ticket).where(Ticket.id == ticket_id, Ticket.is_active.is_(True))
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket no encontrado")
    return ticket


@router.post("/tickets", response_model=TicketResponse, status_code=201, tags=["Tickets"])
def create_ticket(payload: TicketCreate, database: Session = Depends(get_db)) -> Ticket:
    customer = get_customer(payload.customer_id, database)
    category = ticket_classifier().predict(payload.description)["category"]
    ticket = Ticket(**payload.model_dump(), category=category)
    customer.num_tickets += 1
    database.add(ticket)
    database.commit()
    database.refresh(ticket)
    return ticket


@router.put("/tickets/{ticket_id}", response_model=TicketResponse, tags=["Tickets"])
def update_ticket(
    ticket_id: int, payload: TicketUpdate, database: Session = Depends(get_db)
) -> Ticket:
    ticket = get_ticket(ticket_id, database)
    for key, value in payload.model_dump().items():
        setattr(ticket, key, value)
    ticket.category = ticket_classifier().predict(payload.description)["category"]
    database.commit()
    database.refresh(ticket)
    return ticket


@router.post("/tickets/classify", tags=["Tickets"])
def classify_ticket_endpoint(payload: TicketTextRequest) -> dict:
    return ticket_classifier().predict(payload.description)


@router.post("/ml/classify-ticket", tags=["Machine Learning"])
def classify_ticket_ml(payload: TicketTextRequest) -> dict:
    return ticket_classifier().predict(payload.description)


@router.post("/ml/predict-churn", tags=["Machine Learning"])
def predict_churn_ml(payload: ChurnRequest) -> dict:
    return churn_predictor().predict(payload.model_dump())


@router.post("/ml/analyze-sentiment", tags=["Machine Learning"])
def analyze_sentiment(payload: TicketTextRequest) -> dict:
    return analyze_sentiment_model(payload.description)


@router.get("/ml/models/info", tags=["Machine Learning"])
def models_info() -> dict:
    return {
        "ticket_classifier": {
            "available": TICKET_MODEL_PATH.exists(),
            "path": str(TICKET_MODEL_PATH),
        },
        "churn_predictor": {"available": CHURN_MODEL_PATH.exists(), "path": str(CHURN_MODEL_PATH)},
        "sentiment": {
            "available": Path("artifacts/models/sentiment_classifier.keras").exists(),
            "mode": "Keras GRU",
        },
        "resolution_time": {"available": True, "mode": "Keras architecture exported"},
    }


@router.post("/agent/chat", tags=["Conversational Agent"])
def agent_chat(payload: AgentChatRequest, database: Session = Depends(get_db)) -> dict:
    session_id = payload.session_id or str(uuid4())
    session = database.get(AgentSession, session_id)
    messages = json.loads(session.messages_json) if session else []
    messages.append({"role": "user", "content": payload.message})
    context = {"ticket_category": ticket_classifier().predict(payload.message)["category"]}
    if payload.customer_id:
        customer = get_customer(payload.customer_id, database)
        customer_values = {key: getattr(customer, key) for key in ChurnRequest.model_fields}
        context["customer"] = {
            "id": customer.id,
            "name": customer.name,
            "plan_type": customer.plan_type,
        }
        context["churn"] = churn_predictor().predict(customer_values)
    graph = build_agent_graph()
    result = graph.invoke(
        {
            "messages": messages,
            "customer_id": payload.customer_id,
            "context": context,
            "escalate": False,
        }
    )
    messages.append({"role": "assistant", "content": result["response"]})
    if not session:
        session = AgentSession(id=session_id, customer_id=payload.customer_id)
        database.add(session)
    session.messages_json = json.dumps(messages, ensure_ascii=False)
    session.updated_at = datetime.now(UTC)
    database.commit()
    return {
        "session_id": session_id,
        "response": result["response"],
        "intent": result["intent"],
        "escalate": result["escalate"],
        "context": result["context"],
    }


@router.get("/agent/sessions/{session_id}", tags=["Conversational Agent"])
def get_agent_session(session_id: str, database: Session = Depends(get_db)) -> dict:
    session = database.get(AgentSession, session_id)
    if not session or not session.is_active:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    return {
        "session_id": session.id,
        "customer_id": session.customer_id,
        "messages": json.loads(session.messages_json),
    }


@router.delete("/agent/sessions/{session_id}", tags=["Conversational Agent"])
def delete_agent_session(session_id: str, database: Session = Depends(get_db)) -> dict:
    session = database.get(AgentSession, session_id)
    if not session or not session.is_active:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    session.is_active = False
    database.commit()
    return {"deleted": True}
