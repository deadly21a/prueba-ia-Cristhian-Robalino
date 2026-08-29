"""Clasificacion supervisada de tickets de soporte."""

from app.ml.tickets.model import TicketClassifier, train_ticket_classifier

__all__ = ["TicketClassifier", "train_ticket_classifier"]
