"""Player endpoints serving draft slate and league-wide rosters."""

from fastapi import APIRouter, Request

from api.loader import APIException, reload_serving_state_if_stale

router = APIRouter(prefix="/api/v1", tags=["players"])


@router.get("/players")
def get_draft_players(request: Request):
    """Return the current draft slate (equivalent to frontend_home.json).

    If no games are scheduled, returns HTTP 200 with games_scheduled=False and players=[].
    If artifacts are not loaded, returns HTTP 503.
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

    home_players = getattr(request.app.state, "home_players", None)
    if home_players is None:
        raise APIException(
            status_code=503,
            detail="Draft slate artifact not available",
            code="ARTIFACT_NOT_LOADED",
        )

    generated_at = getattr(request.app.state, "generated_at", None)

    if len(home_players) == 0:
        return {
            "game_date": None,
            "generated_at": generated_at,
            "games_scheduled": False,
            "players": [],
        }

    return {
        "game_date": getattr(request.app.state, "game_date", None),
        "generated_at": generated_at,
        "games_scheduled": True,
        "players": home_players,
    }


@router.get("/teams")
def get_all_players(request: Request):
    """Return the league-wide player dataset (equivalent to frontend_all.json).

    Returns flat list of players even when no games are scheduled.
    If artifacts are not loaded, returns HTTP 503.
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

    all_players = getattr(request.app.state, "all_players", None)
    if all_players is None:
        raise APIException(
            status_code=503,
            detail="League player artifact not available",
            code="ARTIFACT_NOT_LOADED",
        )

    return {
        "generated_at": getattr(request.app.state, "generated_at", None),
        "total_players": len(all_players),
        "players": all_players,
    }
