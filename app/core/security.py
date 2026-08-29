from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings

bearer_scheme = HTTPBearer()


def create_token(subject: str, role: str, token_type: str, expires_delta: timedelta) -> str:
    expires_at = datetime.now(UTC) + expires_delta
    payload = {"sub": subject, "role": role, "type": token_type, "exp": expires_at}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, role: str) -> str:
    return create_token(
        subject,
        role,
        "access",
        timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(subject: str, role: str) -> str:
    return create_token(
        subject,
        role,
        "refresh",
        timedelta(days=settings.refresh_token_expire_days),
    )


def decode_token(token: str, expected_type: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as error:
        raise ValueError("Token invalido o expirado") from error
    if payload.get("type") != expected_type:
        raise ValueError("Tipo de token invalido")
    return payload


def require_roles(*allowed_roles: str):
    def dependency(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ) -> dict[str, Any]:
        try:
            payload = decode_token(credentials.credentials, "access")
        except ValueError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error
        if payload.get("role") not in allowed_roles:
            raise HTTPException(status_code=403, detail="El rol no tiene permiso para esta accion")
        return payload

    return dependency
