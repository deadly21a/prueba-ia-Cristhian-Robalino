from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        version="0.1.0",
        description=(
            "API para clasificacion de tickets, prediccion de churn, analisis de "
            "sentimiento y atencion conversacional."
        ),
    )
    application.include_router(api_router, prefix=settings.api_v1_prefix)

    @application.get("/health", tags=["System"], summary="Verificar estado de la API")
    def health_check() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return application


app = create_application()
