"""Key-free tests for chat persistence + hydration (multi-chat per user):
the row serializer, the ChatSession rebuild from a persisted row, the
in-place store restoration (tool-closure integrity), hydration routing,
and the per-turn persist write point. DB calls are stubbed throughout.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


VALID_LAYERS = {
    "num_layers": 1,
    "layers": [{
        "name": "sandy_loam",
        "thickness_m_min": 0.3, "thickness_m_max": 0.5,
        "sand_pct_min": 30.0, "sand_pct_max": 40.0,
        "clay_pct_min": 5.0, "clay_pct_max": 15.0,
        "theta_v_min": 0.05, "theta_v_max": 0.20,
        "bulk_density_gcm3_min": 1.5, "bulk_density_gcm3_max": 1.6,
        "particle_density_gcm3_min": 2.60, "particle_density_gcm3_max": 2.66,
    }],
}


def _populated_chat():
    chat = api._new_chat_session("persist-session", "vijay")
    chat.started = True
    chat.complete = True
    chat.phase = "complete"
    chat.post_complete_briefed = True
    chat.transcript = [
        {"type": "user_message", "content": "make me a dataset"},
        {"type": "agent_message", "content": "done"},
    ]
    chat.agent_session.store["layers"] = dict(VALID_LAYERS)
    chat.agent_session.store["dataset_config"] = {
        "num_samples": 3, "model_basename": "clay_survey",
    }
    chat.agent_session.seen = 42
    chat.state.update({"layers": dict(VALID_LAYERS), "halted": False})
    chat.viz_flags = {k: True for k in chat.viz_flags}
    chat.last_scene = {"domain": {"provisional": False}}
    chat.dataset_result = {"rows_inserted": 3, "files": [{"filename": "a.in"}]}
    chat.simulation_result = {"succeeded": 3}
    chat.complete_snapshot = chat.agent_session.snapshot()
    chat.regen_block_notice = frozenset({"layers"})
    return chat


def _row_from(chat):
    payload = api._chat_row_payload(chat)
    return SimpleNamespace(
        id=payload["session_id"],
        user_id=payload["user_id"],
        title=payload["title"],
        thread_id=payload["thread_id"],
        complete=payload["complete"],
        has_dataset=payload["has_dataset"],
        session_state=payload["session_state"],
    )


# ---------------------------------------------------------------------------
# Serializer round-trip
# ---------------------------------------------------------------------------

def test_row_payload_promoted_columns():
    chat = _populated_chat()
    payload = api._chat_row_payload(chat)
    assert payload["title"] == "clay_survey"  # basename wins over first message
    assert payload["complete"] is True
    assert payload["has_dataset"] is True
    assert payload["thread_id"] == chat.agent_session.thread_id
    blob = payload["session_state"]
    assert blob["seen"] == 42
    assert blob["regen_block_notice"] == ["layers"]  # frozenset -> list
    assert blob["store"]["layers"] == VALID_LAYERS


def test_title_falls_back_to_first_user_message():
    chat = api._new_chat_session("t2", "vijay")
    chat.transcript = [{"type": "user_message", "content": "x" * 100}]
    assert api._chat_title(chat) == "x" * 60
    chat.transcript = []
    assert api._chat_title(chat) == "New chat"


def test_round_trip_restores_fields_and_coerces_inflight():
    chat = _populated_chat()
    # Simulate a disconnect-persist that captured a mid-turn snapshot.
    chat.busy = True
    chat.simulating = True
    chat.phase = "deterministic"
    row = _row_from(chat)

    restored = api._chat_from_row(row)
    assert restored.session_id == "persist-session"
    assert restored.user_id == "vijay"
    assert restored.transcript == chat.transcript
    assert restored.state["layers"] == VALID_LAYERS
    assert restored.dataset_result == chat.dataset_result
    assert restored.complete_snapshot == chat.complete_snapshot
    assert restored.regen_block_notice == frozenset({"layers"})
    assert restored.viz_flags == chat.viz_flags
    assert restored.last_scene == chat.last_scene
    # In-flight flags can never survive a restart; mid-turn phase coerced.
    assert restored.busy is False
    assert restored.simulating is False
    assert restored.phase == "complete"
    # Live objects were rebuilt, not deserialized.
    assert restored.ws is None
    assert restored.agent_session is not chat.agent_session


def test_phase_coercion_incomplete_chat():
    chat = _populated_chat()
    chat.complete = False
    chat.phase = "routing"
    restored = api._chat_from_row(_row_from(chat))
    assert restored.phase == "agent"


def test_regenerating_survives_restart():
    # Deliberate: mid-regeneration remediation legitimately spans turns and
    # must keep 409ing simulate/upload after a restart.
    chat = _populated_chat()
    chat.complete = False
    chat.regenerating = True
    restored = api._chat_from_row(_row_from(chat))
    assert restored.regenerating is True


# ---------------------------------------------------------------------------
# Store closure integrity + thread resume material
# ---------------------------------------------------------------------------

def test_restored_store_still_backs_the_tools():
    chat = _populated_chat()
    restored = api._chat_from_row(_row_from(chat))
    # The restored values are visible through the closure-bound tool...
    assert restored.agent_session.stage_done("layers")
    # ...and a save through the tool mutates the restored session's store.
    out = restored.agent_session.save_section.invoke({
        "section": "advanced_params", "payload": "{}",
    })
    assert '"ok"' in out
    assert restored.agent_session.store["advanced_params"] is not None
    assert restored.agent_session.stage_done("advanced_params")


def test_thread_id_and_seen_restored_agent_stays_lazy(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "get_checkpointer", lambda: calls.append(1))
    chat = _populated_chat()
    restored = api._chat_from_row(_row_from(chat))
    assert restored.agent_session.thread_id == chat.agent_session.thread_id
    assert restored.agent_session.seen == 42
    assert calls == []  # factory only fires when the agent is actually built


def test_scoped_output_dir_stable_after_hydration():
    # The rebuilt session must author the SAME dataset dir as the original
    # (deterministic user/session scope), or a re-save after restart would
    # silently move the dataset.
    chat = _populated_chat()
    restored = api._chat_from_row(_row_from(chat))
    payload = '{"num_samples": 3, "model_basename": "clay_survey"}'
    chat.agent_session.save_section.invoke(
        {"section": "dataset_config", "payload": payload})
    restored.agent_session.save_section.invoke(
        {"section": "dataset_config", "payload": payload})
    assert (
        chat.agent_session.store["dataset_config"]["output_dir"]
        == restored.agent_session.store["dataset_config"]["output_dir"]
    )


# ---------------------------------------------------------------------------
# Hydration routing
# ---------------------------------------------------------------------------

def test_resolve_unknown_id_returns_none(monkeypatch):
    monkeypatch.setattr(api, "get_chat_session", lambda sid: None)
    assert asyncio.run(api._resolve_chat("nope")) is None
    assert api._resolve_chat_sync("nope") is None


def test_resolve_hydrates_from_db_and_caches(monkeypatch):
    chat = _populated_chat()
    row = _row_from(chat)
    lookups = []

    def fake_get(sid):
        lookups.append(sid)
        return row

    monkeypatch.setattr(api, "get_chat_session", fake_get)
    try:
        restored = asyncio.run(api._resolve_chat("persist-session"))
        assert restored is not None
        assert api.sessions["persist-session"] is restored
        # Second resolve: in-memory hit, no DB round-trip.
        again = asyncio.run(api._resolve_chat("persist-session"))
        assert again is restored
        assert lookups == ["persist-session"]
    finally:
        api.sessions.pop("persist-session", None)


def test_resolve_swallows_db_errors(monkeypatch):
    def boom(sid):
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "get_chat_session", boom)
    assert asyncio.run(api._resolve_chat("x")) is None
    assert api._resolve_chat_sync("x") is None


# ---------------------------------------------------------------------------
# Persist write point + failure isolation
# ---------------------------------------------------------------------------

def test_turn_persists_exactly_once(monkeypatch):
    persisted = []

    async def fake_persist(chat):
        persisted.append(chat.session_id)

    monkeypatch.setattr(api, "_persist_chat", fake_persist)
    chat = api._new_chat_session("turn-session", "vijay")
    chat.ws = FakeWS()
    chat.phase = "agent"
    monkeypatch.setattr(
        chat.agent_session, "invoke", lambda text: {"messages": []}
    )
    asyncio.run(api._handle_user_text(chat, "hello"))
    assert persisted == ["turn-session"]
    # Early-return turns (wrong phase) persist too — the user message landed
    # in the transcript and must not be lost.
    chat.phase = "deterministic"
    asyncio.run(api._handle_user_text(chat, "again"))
    assert persisted == ["turn-session", "turn-session"]


def test_persist_failure_never_breaks_the_turn(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "upsert_chat_session", boom)
    chat = api._new_chat_session("fail-session", "vijay")
    chat.ws = FakeWS()
    chat.phase = "agent"
    monkeypatch.setattr(
        chat.agent_session, "invoke", lambda text: {"messages": []}
    )
    # Must not raise.
    asyncio.run(api._handle_user_text(chat, "hello"))
    asyncio.run(api._persist_chat(chat))
