"""Key-free tests for the per-user chat management endpoints and the
user_id validation shared by the REST routes and the WebSocket route."""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api


# ---------------------------------------------------------------------------
# _validate_user_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("vijay", "vijay"),
    ("  vijay  ", "vijay"),          # trimmed
    ("user-1_x", "user-1_x"),
    ("a" * 64, "a" * 64),
])
def test_valid_user_ids(raw, expected):
    assert api._validate_user_id(raw) == expected


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    "a" * 65,                        # too long
    "----",                          # no letter/digit
    "../..",                         # traversal shape, no alnum
    None,
])
def test_invalid_user_ids_422(raw):
    with pytest.raises(HTTPException) as exc:
        api._validate_user_id(raw)
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# GET /users/{user_id}/chats
# ---------------------------------------------------------------------------

def test_list_user_chats(monkeypatch):
    rows = [{"id": "s1", "title": "clay_survey", "complete": True,
             "has_dataset": True, "created_at": "t0", "updated_at": "t1"}]
    seen = []

    def fake_list(uid):
        seen.append(uid)
        return rows

    monkeypatch.setattr(api, "list_chat_sessions", fake_list)
    out = asyncio.run(api.list_user_chats("  vijay "))
    assert out == {"chats": rows}
    assert seen == ["vijay"]  # validated + trimmed before the DB call


def test_list_user_chats_invalid_id():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.list_user_chats("!!!"))
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# POST /users/{user_id}/chats
# ---------------------------------------------------------------------------

def test_create_user_chat_mints_ids(monkeypatch):
    stubs = []

    def fake_stub(session_id, user_id, thread_id):
        stubs.append((session_id, user_id, thread_id))

    monkeypatch.setattr(api, "create_chat_stub", fake_stub)
    out1 = asyncio.run(api.create_user_chat("vijay"))
    out2 = asyncio.run(api.create_user_chat("vijay"))
    assert out1["session_id"] != out2["session_id"]  # unique per chat
    assert len(stubs) == 2
    sid, uid, tid = stubs[0]
    assert sid == out1["session_id"]
    assert uid == "vijay"
    assert tid.startswith("single-agent-")


def test_create_user_chat_invalid_id():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.create_user_chat(""))
    assert exc.value.status_code == 422
