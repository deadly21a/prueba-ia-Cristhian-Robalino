from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
try:
    engine = create_engine(settings.database_url, connect_args=connect_args)
except ImportError:
    engine = None


class InMemorySession:
    """Small local fallback for Windows environments that block the sqlite DLL."""

    _store: dict[type, dict[Any, Any]] = {}
    _next_ids: dict[type, int] = {}

    def add(self, instance: Any) -> None:
        model = type(instance)
        collection = self._store.setdefault(model, {})
        identifier = getattr(instance, "id", None)
        if identifier is None:
            identifier = self._next_ids.get(model, 1)
            self._next_ids[model] = identifier + 1
            instance.id = identifier
        now = datetime.now(UTC)
        defaults = {
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "deleted_at": None,
            "status": "open",
            "messages_json": "[]",
        }
        for attribute, value in defaults.items():
            if hasattr(model, attribute) and getattr(instance, attribute, None) is None:
                setattr(instance, attribute, value)
        collection[identifier] = instance

    def get(self, model: type, identifier: Any) -> Any:
        return self._store.get(model, {}).get(identifier)

    def scalars(self, statement: Any) -> list[Any]:
        model = statement.column_descriptions[0]["entity"]
        values = list(self._store.get(model, {}).values())
        for criterion in statement._where_criteria:
            values = [value for value in values if self._matches(value, criterion)]
        return values

    def scalar(self, statement: Any) -> Any:
        values = self.scalars(statement)
        return values[0] if values else None

    @staticmethod
    def _matches(instance: Any, criterion: Any) -> bool:
        left_value = getattr(instance, criterion.left.key)
        right = getattr(criterion.right, "value", None)
        try:
            return bool(criterion.operator(left_value, right))
        except TypeError:
            return bool(left_value is right)

    def commit(self) -> None:
        return None

    def refresh(self, _instance: Any) -> None:
        return None

    def close(self) -> None:
        return None


if engine is not None:
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
else:
    SessionLocal = InMemorySession


def init_db() -> None:
    from app.db import models  # noqa: F401

    if engine is not None:
        Base.metadata.create_all(bind=engine)


def get_db():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
