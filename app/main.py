from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.db.session import init_db
from app.mcp.routes import router as mcp_router


def create_application() -> FastAPI:
    init_db()
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
    application.include_router(mcp_router)

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": str(error)}},
        )

    @application.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": "http_error", "message": error.detail}},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Los datos enviados no son validos",
                    "details": jsonable_encoder(error.errors()),
                }
            },
        )

    @application.get("/health", tags=["System"], summary="Verificar estado de la API")
    def health_check() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return application


app = create_application()
