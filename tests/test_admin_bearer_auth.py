# SPDX-License-Identifier: Apache-2.0
"""Tests for bearer-token auth on admin endpoints."""

import pytest
from fastapi.testclient import TestClient

from omlx.server import app, _server_state


def _set_api_key(key):
    """Snapshot/swap api_key for one test."""
    original = _server_state.api_key
    _server_state.api_key = key
    return original


def test_admin_endpoint_accepts_valid_bearer_token():
    """GET /admin/api/grammar/parsers with Authorization: Bearer <api_key> succeeds (no cookie)."""
    original = _set_api_key("test-key-123")
    try:
        with TestClient(app) as client:
            resp = client.get(
                "/admin/api/grammar/parsers",
                headers={"Authorization": "Bearer test-key-123"},
            )
        assert resp.status_code == 200
    finally:
        _server_state.api_key = original


def test_admin_endpoint_rejects_invalid_bearer_token():
    """A wrong bearer token results in 401."""
    original = _set_api_key("test-key-123")
    try:
        with TestClient(app) as client:
            resp = client.get(
                "/admin/api/grammar/parsers",
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert resp.status_code == 401
    finally:
        _server_state.api_key = original


def test_admin_endpoint_rejects_no_auth():
    """No bearer, no cookie → 401."""
    original = _set_api_key("test-key-123")
    try:
        with TestClient(app) as client:
            resp = client.get("/admin/api/grammar/parsers")
        assert resp.status_code == 401
    finally:
        _server_state.api_key = original


def test_admin_endpoint_still_accepts_session_cookie():
    """Existing cookie-based admin auth must continue to work."""
    from omlx.admin.auth import create_session_token, SESSION_COOKIE_NAME

    original = _set_api_key("test-key-123")
    try:
        token = create_session_token()
        with TestClient(app) as client:
            client.cookies.set(SESSION_COOKIE_NAME, token)
            resp = client.get("/admin/api/grammar/parsers")
        assert resp.status_code == 200
    finally:
        _server_state.api_key = original
