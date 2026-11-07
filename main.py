from pathlib import Path

from cscs_tools.version.services.version_service import VersionService
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from exceptions import InvalidImageSourceError, InvalidVideoSourceError

from config.api_description import API_DESCRIPTION
from config.config import CORS_ALLOWED_ORIGINS
from config.log_config import access_logger_service, error_logger_service
from controllers.controller import api_router
from multi_model.controllers.multi_model_rest_controller import (
    MULTI_MODEL_TAG_DESCRIPTION,
)

access_logger = access_logger_service.get_logger("access_logger")
error_logger = error_logger_service.get_logger("error_logger")

version_service = VersionService()

def get_origins():
    if not CORS_ALLOWED_ORIGINS:
        raise RuntimeError(
            "CORS_ALLOWED_ORIGINS is not set! "
            "Please set it in the environment before starting the application."
        )

    allowed_origins = [origin.strip() for origin in CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]
    if not allowed_origins:
        raise RuntimeError("CORS_ALLOWED_ORIGINS is empty! At least one origin must be specified.")

    access_logger.info(f"Allowed CORS origins: {allowed_origins}")
    return allowed_origins

def add_middleware(application):
    application.add_middleware(
        CORSMiddleware,
        allow_origins=get_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

def start_app():
    application = FastAPI(
        description=API_DESCRIPTION,
        openapi_tags=[
            {
                "name": "multi-model prediction",
                "description": MULTI_MODEL_TAG_DESCRIPTION,
            }
        ],
        version=version_service.get_version().str_repr(),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        swagger_ui_parameters={
            "syntaxHighlight": False,
        },
    )
    application.include_router(api_router, prefix="/api")
    add_middleware(application)
    return application

app = start_app()

async def catch_exceptions_middleware(request: Request, call_next):
    try:
        access_logger.info(f"Access: {request.method} {request.url}")
        return await call_next(request)
    except Exception as e:
        error_logger.exception(f"Unhandled exception in request {request.url}")
        if isinstance(e, (InvalidImageSourceError, InvalidVideoSourceError)):
            message = "Invalid image source" if isinstance(e, InvalidImageSourceError) else "Invalid video source"
            return JSONResponse(
                status_code=400,
                content={
                    "message": message,
                    "status_code": 400,
                    "url": str(request.url),
                    "method": request.method,
                    "detail": str(e),
                },
            )
        message = getattr(e, "message", str(e))
        return JSONResponse(
            status_code=500,
            content={
                "message": "Internal Server Error",
                "status_code": 500,
                "url": str(request.url),
                "method": request.method,
                "detail": message,
            }
        )

app.middleware('http')(catch_exceptions_middleware)

build_dir = Path("build")
if build_dir.exists():
    app.mount("/", StaticFiles(directory=build_dir, html=True), name="frontend")