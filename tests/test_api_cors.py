"""
tests/test_api_cors.py

Phase 2E.7 — CORS Integration and Security Hardening Tests
Verifies that:
1. Configured allowed origin (e.g. http://localhost:5173) is accepted with appropriate CORS headers.
2. Preflight OPTIONS requests from allowed origin succeed with 200 and access-control-allow-methods.
3. Unapproved origins (e.g. http://malicious-site.com) are NOT granted access-control-allow-origin.
4. Preflight OPTIONS requests from unapproved origins do not receive allow-origin.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import Settings
from api.main import create_app


def test_cors_allowed_origin_receives_allow_origin_header(tmp_path):
    """Requests with Origin: http://localhost:5173 receive matching Access-Control-Allow-Origin."""
    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_preflight_options_succeeds_for_allowed_origin(tmp_path):
    """Preflight OPTIONS request from http://localhost:5173 returns 200 with allowed methods."""
    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.options(
            "/api/v1/players",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert "GET" in resp.headers.get("access-control-allow-methods", "")


def test_cors_unapproved_origin_is_rejected(tmp_path):
    """Requests with an unapproved Origin do NOT receive Access-Control-Allow-Origin."""
    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://malicious-site.com"})
        assert resp.status_code == 200
        # The CORS middleware MUST NOT set access-control-allow-origin for unauthorized origins
        assert "access-control-allow-origin" not in resp.headers


def test_cors_preflight_options_rejected_for_unapproved_origin(tmp_path):
    """Preflight OPTIONS request from unapproved origin must not receive allow-origin."""
    settings = Settings(
        data_dir=tmp_path,
        allowed_origins=["http://localhost:5173"],
        env="testing",
        reload=False,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        resp = client.options(
            "/api/v1/players",
            headers={
                "Origin": "http://evil-origin.org",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") != "http://evil-origin.org"
