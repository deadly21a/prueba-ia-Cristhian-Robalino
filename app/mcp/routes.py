from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.endpoints import (
    agent_chat,
    churn_predictor,
    create_ticket,
    get_customer,
    ticket_classifier,
)
from app.api.schemas import AgentChatRequest, MCPExecuteRequest, TicketCreate
from app.db.session import get_db

router = APIRouter(prefix="/mcp", tags=["MCP"])

TOOLS = [
    "predict_churn",
    "classify_ticket",
    "get_customer_info",
    "create_ticket",
    "chat_with_agent",
]


@router.get("/capabilities")
def capabilities() -> dict:
    return {
        "protocol": "MCP-compatible JSON-RPC 2.0",
        "tools": TOOLS,
        "resources": ["customers", "tickets"],
    }


@router.get("/resources")
def resources() -> dict:
    return {
        "resources": [
            {"id": "customers", "uri": "resource://customers"},
            {"id": "tickets", "uri": "resource://tickets"},
        ]
    }


@router.get("/resources/{resource_id}")
def resource(resource_id: str) -> dict:
    if resource_id not in {"customers", "tickets"}:
        raise HTTPException(status_code=404, detail="Recurso MCP no encontrado")
    return {
        "id": resource_id,
        "uri": f"resource://{resource_id}",
        "description": f"Recurso de {resource_id}",
    }


@router.post("/tools/execute")
def execute_tool(payload: MCPExecuteRequest, database: Session = Depends(get_db)) -> dict:
    try:
        if payload.tool == "predict_churn":
            result = churn_predictor().predict(payload.arguments)
        elif payload.tool == "classify_ticket":
            result = ticket_classifier().predict(payload.arguments["description"])
        elif payload.tool == "get_customer_info":
            result = get_customer(int(payload.arguments["customer_id"]), database)
            result = {
                "id": result.id,
                "name": result.name,
                "email": result.email,
                "plan_type": result.plan_type,
            }
        elif payload.tool == "create_ticket":
            ticket = create_ticket(TicketCreate(**payload.arguments), database)
            result = {"ticket_id": ticket.id, "category": ticket.category, "status": ticket.status}
        elif payload.tool == "chat_with_agent":
            result = agent_chat(AgentChatRequest(**payload.arguments), database)
        else:
            raise ValueError(f"Herramienta no soportada: {payload.tool}")
        return {
            "jsonrpc": "2.0",
            "id": payload.id,
            "result": {"content": [result], "isError": False},
        }
    except Exception as error:
        return {
            "jsonrpc": "2.0",
            "id": payload.id,
            "result": {"content": [{"error": str(error)}], "isError": True},
        }
