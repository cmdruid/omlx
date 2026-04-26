# SPDX-License-Identifier: Apache-2.0
"""Tests for the /health endpoint's version field."""

from fastapi.testclient import TestClient

from omlx._version import __version__
from omlx.server import app


def test_health_returns_version_field():
    """GET /health must include a `version` field matching omlx._version.__version__."""
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("version") == __version__


def test_health_response_remains_backward_compatible():
    """Existing fields (status, default_model, engine_pool, mcp) must still be present."""
    with TestClient(app) as client:
        resp = client.get("/health")
    body = resp.json()
    assert "status" in body
    assert "default_model" in body
    assert "engine_pool" in body
    assert "mcp" in body
