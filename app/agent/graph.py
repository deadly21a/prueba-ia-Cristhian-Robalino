from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.ml.deep_learning import analyze_sentiment


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    customer_id: int | None
    intent: str
    context: dict[str, Any]
    escalate: bool
    response: str


def classify_intent(state: AgentState) -> AgentState:
    text = state["messages"][-1]["content"].lower()
    if any(word in text for word in ("factura", "pago", "cuenta", "cobro")):
        intent = "account"
    elif any(word in text for word in ("internet", "router", "señal", "conexión", "wifi")):
        intent = "technical"
    elif state.get("context", {}).get("ticket_category") == "TECH":
        intent = "technical"
    elif state.get("context", {}).get("ticket_category") in {"BILL", "PLAN", "CNCL"}:
        intent = "account"
    else:
        intent = "general"
    return {**state, "intent": intent}


def route_intent(state: AgentState) -> Literal["account", "technical", "general"]:
    return state.get("intent", "general")  # type: ignore[return-value]


def get_customer_info(state: AgentState) -> AgentState:
    context = dict(state.get("context", {}))
    if state.get("customer_id"):
        context["customer_identified"] = True
    return {**state, "context": context}


def handle_account_query(state: AgentState) -> AgentState:
    return {**state, "response": "Puedo ayudarte a revisar tu factura, pagos y datos de cuenta."}


def handle_technical_support(state: AgentState) -> AgentState:
    return {
        **state,
        "response": "Revisemos la conexión. Reinicia el router por 30 segundos y confirma las luces.",
    }


def handle_general_info(state: AgentState) -> AgentState:
    text = state["messages"][-1]["content"].lower()
    if any(word in text for word in ("hola", "buenos días", "buenas tardes")):
        response = "¡Hola! Soy el asistente virtual. ¿En qué servicio puedo ayudarte?"
    elif any(word in text for word in ("adiós", "hasta luego", "gracias")):
        response = "Gracias por contactarnos. ¡Que tengas un excelente día!"
    else:
        response = "Puedo ayudarte con soporte técnico, facturación, planes y cancelaciones."
    return {**state, "response": response}


def check_escalation(state: AgentState) -> AgentState:
    text = state["messages"][-1]["content"]
    sentiment = analyze_sentiment(text)
    escalate = sentiment["sentiment"] == "negative" or "humano" in text.lower()
    return {
        **state,
        "escalate": escalate,
        "context": {**state.get("context", {}), "sentiment": sentiment},
    }


def generate_response(state: AgentState) -> AgentState:
    response = state.get("response", "No pude procesar la solicitud.")
    if state.get("escalate"):
        response += (
            " Detecté que necesitas atención adicional; escalaré el caso a un agente humano."
        )
    return {**state, "response": response}


def build_agent_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return _FallbackAgentGraph()

    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("get_customer_info", get_customer_info)
    graph.add_node("handle_account_query", handle_account_query)
    graph.add_node("handle_technical_support", handle_technical_support)
    graph.add_node("handle_general_info", handle_general_info)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("generate_response", generate_response)
    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "get_customer_info")
    graph.add_conditional_edges(
        "get_customer_info",
        route_intent,
        {
            "account": "handle_account_query",
            "technical": "handle_technical_support",
            "general": "handle_general_info",
        },
    )
    for node in ("handle_account_query", "handle_technical_support", "handle_general_info"):
        graph.add_edge(node, "check_escalation")
    graph.add_edge("check_escalation", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()


class _FallbackAgentGraph:
    """Execute the same nodes when native optional LangGraph dependencies are blocked."""

    def invoke(self, state: AgentState) -> AgentState:
        state = classify_intent(state)
        state = get_customer_info(state)
        handlers = {
            "account": handle_account_query,
            "technical": handle_technical_support,
            "general": handle_general_info,
        }
        state = handlers[state.get("intent", "general")](state)
        state = check_escalation(state)
        return generate_response(state)
