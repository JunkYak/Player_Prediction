"""Schedule endpoints serving upcoming NBA game schedules."""

import logging
from fastapi import APIRouter, Request

from api.loader import APIException, reload_serving_state_if_stale

logger = logging.getLogger("propredict.api")

router = APIRouter(prefix="/api/v1/schedule", tags=["schedule"])


@router.get("/tomorrow")
def get_tomorrow_schedule(request: Request):
    """Return tomorrow's confirmed NBA game schedule.

    Returns HTTP 200 with status CONFIRMED, OFF_DAY, or STALE.
    Returns HTTP 503 if tomorrow's schedule artifact is missing or unavailable.
    """
    reload_serving_state_if_stale(request.app)

    schedule_data = getattr(request.app.state, "tomorrow_schedule", None)
    if schedule_data is None:
        raise APIException(
            status_code=503,
            detail="Tomorrow schedule artifact not available",
            code="SCHEDULE_UNAVAILABLE",
        )

    valid_statuses = {"CONFIRMED", "OFF_DAY", "STALE"}
    status = schedule_data.get("status")
    if status not in valid_statuses:
        raise APIException(
            status_code=503,
            detail=f"Tomorrow schedule is currently unavailable (status: {status})",
            code="SCHEDULE_UNAVAILABLE",
        )

    return schedule_data
