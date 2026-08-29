from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=True)


def auth_headers() -> dict[str, str]:
    response = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_auth_customer_ticket_ml_agent_and_mcp_flow() -> None:
    login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin123"})
    assert login.status_code == 200
    assert login.json()["access_token"]
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    customer_payload = {
        "name": "Cliente Prueba",
        "email": f"qa-{uuid4().hex}@example.com",
        "phone": "0991234567",
        "plan_type": "fiber",
        "tenure_months": 4,
        "monthly_charge": 85,
        "total_charges": 340,
        "contract_type": "month-to-month",
        "payment_method": "card",
        "num_tickets": 3,
        "avg_satisfaction": 2.5,
    }
    customer = client.post("/api/v1/customers", json=customer_payload, headers=headers)
    assert customer.status_code == 201, customer.text
    customer_id = customer.json()["id"]

    ticket = client.post(
        "/api/v1/tickets",
        json={
            "customer_id": customer_id,
            "description": "El internet está muy lento y el router pierde la conexión constantemente",
            "priority": "high",
        },
    )
    assert ticket.status_code == 201, ticket.text
    assert ticket.json()["category"] == "TECH"

    churn = client.get(f"/api/v1/customers/{customer_id}/churn-prediction", headers=headers)
    assert churn.status_code == 200, churn.text
    assert 0 <= churn.json()["churn_probability"] <= 1

    sentiment = client.post(
        "/api/v1/ml/analyze-sentiment",
        json={"description": "Estoy muy molesto porque el servicio nunca funciona"},
    )
    assert sentiment.json()["sentiment"] == "negative"

    chat = client.post(
        "/api/v1/agent/chat",
        json={"message": "Estoy frustrado, el wifi nunca funciona", "customer_id": customer_id},
    )
    assert chat.status_code == 200, chat.text
    assert chat.json()["intent"] == "technical"
    assert chat.json()["escalate"] is True

    mcp = client.post(
        "/mcp/tools/execute",
        json={
            "jsonrpc": "2.0",
            "id": "qa-1",
            "tool": "classify_ticket",
            "arguments": {"description": "Necesito revisar un cobro duplicado de mi factura"},
        },
    )
    assert mcp.status_code == 200
    assert mcp.json()["result"]["isError"] is False


def test_phone_and_ticket_description_validation() -> None:
    headers = auth_headers()
    invalid_customer = client.post(
        "/api/v1/customers",
        json={"name": "QA", "email": "qa@example.com", "phone": "123"},
        headers=headers,
    )
    assert invalid_customer.status_code == 422

    invalid_ticket = client.post(
        "/api/v1/tickets",
        json={"customer_id": 1, "description": "muy corto", "priority": "low"},
    )
    assert invalid_ticket.status_code == 422
