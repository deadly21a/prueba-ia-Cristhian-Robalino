from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class CustomerBase(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    phone: str
    plan_type: str = "basic"
    tenure_months: int = Field(default=1, ge=0)
    monthly_charge: float = Field(default=30, ge=0)
    total_charges: float = Field(default=30, ge=0)
    contract_type: Literal["month-to-month", "one-year", "two-year"] = "month-to-month"
    payment_method: Literal["card", "bank-transfer", "cash"] = "card"
    num_tickets: int = Field(default=0, ge=0)
    avg_satisfaction: float = Field(default=3, ge=1, le=5)

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, phone: str) -> str:
        if not phone.isdigit() or len(phone) < 10 or not phone.startswith("09"):
            raise ValueError("El telefono debe tener al menos 10 digitos y empezar con 09")
        return phone


class CustomerCreate(CustomerBase):
    pass


class CustomerUpdate(CustomerBase):
    pass


class CustomerResponse(CustomerBase):
    id: int
    is_active: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class TicketBase(BaseModel):
    customer_id: int
    description: str = Field(min_length=20, max_length=500)
    priority: Literal["low", "medium", "high", "critical"] = "medium"


class TicketCreate(TicketBase):
    pass


class TicketUpdate(BaseModel):
    description: str = Field(min_length=20, max_length=500)
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    status: Literal["open", "in_progress", "resolved", "closed"] = "open"


class TicketResponse(TicketBase):
    id: int
    category: str
    status: str
    is_active: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class TicketTextRequest(BaseModel):
    description: str = Field(min_length=10, max_length=500)


class ChurnRequest(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_charge: float = Field(ge=0)
    total_charges: float = Field(ge=0)
    contract_type: Literal["month-to-month", "one-year", "two-year"]
    payment_method: Literal["card", "bank-transfer", "cash"]
    num_tickets: int = Field(ge=0)
    avg_satisfaction: float = Field(ge=1, le=5)


class AgentChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = None
    customer_id: int | None = None


class MCPExecuteRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
