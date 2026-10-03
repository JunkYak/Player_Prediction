"""FastAPI application serving layer for the NBA Player Performance Rating System."""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import Settings, get_settings
from api.loader import APIException, load_serving_state
from api.routers import health, players

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("propredict.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load parquet artifacts deterministically at startup."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings

    state = load_serving_state(settings)
    app.state.home_players = state.home_players
    app.state.all_players = state.all_players
    app.state.generated_at = state.generated_at
    app.state.game_date = state.game_date
    app.state.artifacts_loaded = state.artifacts_loaded
    app.state.load_error = state.load_error
    app.state.load_error_code = state.load_error_code
    app.state.last_mtime = state.last_mtime

    yield


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title="NBA Player Performance Rating API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Initial default state before lifespan runs or if not triggered
    app.state.settings = resolved_settings
    app.state.home_players = None
    app.state.all_players = None
    app.state.generated_at = None
    app.state.game_date = None
    app.state.artifacts_loaded = False
    app.state.load_error = "Serving artifacts are not loaded"
    app.state.load_error_code = "ARTIFACT_NOT_LOADED"
    app.state.last_mtime = None

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler for APIException
    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code},
        )

    # Register routers
    app.include_router(health.router)
    app.include_router(players.router)

    # Optional static file serving for unified frontend/backend deployment
    if resolved_settings.static_dir and resolved_settings.static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(resolved_settings.static_dir), html=True), name="static")

    return app


# Module-level app instance for ASGI servers (e.g. uvicorn api.main:app)
app = create_app()
