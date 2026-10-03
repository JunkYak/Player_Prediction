"""Health and Readiness endpoints."""

from fastapi import APIRouter, Request

from api.config import Settings, get_settings
from api.loader import APIException, reload_serving_state_if_stale, get_pipeline_diagnostics

router = APIRouter(tags=["health"])


@router.get("/health")
def get_health():
    """Liveness probe.

    Returns HTTP 200 if FastAPI is running.
    Does not check external services or parquet availability.
    """
    return {"status": "ok"}


@router.get("/ready")
def get_ready(request: Request):
    """Readiness probe.

    Returns HTTP 200 if valid serving artifacts are loaded in memory.
    Returns HTTP 503 if artifacts are missing, unreadable, or not loaded.
    """
    reload_serving_state_if_stale(request.app)
    if not getattr(request.app.state, "artifacts_loaded", False):
        raise APIException(
            status_code=503,
            detail=getattr(request.app.state, "load_error", "Serving artifacts are not loaded")
            or "Serving artifacts are not loaded",
            code=getattr(request.app.state, "load_error_code", "ARTIFACT_NOT_LOADED")
            or "ARTIFACT_NOT_LOADED",
        )

    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    home_players = getattr(request.app.state, "home_players", []) or []
    all_players = getattr(request.app.state, "all_players", []) or []
    pipeline_diag = get_pipeline_diagnostics(settings)

    return {
        "status": "ready",
        "generated_at": getattr(request.app.state, "generated_at", None),
        "game_date": getattr(request.app.state, "game_date", None),
        "slate_count": len(home_players),
        "all_count": len(all_players),
        "pipeline": pipeline_diag,
    }


@router.get("/status")
def get_status(request: Request):
    """Comprehensive operational diagnostics endpoint."""
    reload_serving_state_if_stale(request.app)
    settings: Settings = getattr(request.app.state, "settings", None) or get_settings()
    home_players = getattr(request.app.state, "home_players", []) or []
    all_players = getattr(request.app.state, "all_players", []) or []
    pipeline_diag = get_pipeline_diagnostics(settings)
    artifacts_loaded = getattr(request.app.state, "artifacts_loaded", False)

    return {
        "status": "ok" if artifacts_loaded and not pipeline_diag["is_stale"] else "degraded",
        "artifacts_loaded": artifacts_loaded,
        "generated_at": getattr(request.app.state, "generated_at", None),
        "game_date": getattr(request.app.state, "game_date", None),
        "slate_count": len(home_players),
        "all_count": len(all_players),
        "pipeline": pipeline_diag,
    }
