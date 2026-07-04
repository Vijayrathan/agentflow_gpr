"""Key-free tests for the refresh/reconnect mechanics in backend.api:
transcript recording, dead-socket resilience, and current-socket routing."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend import api


class FakeWS:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send_json(self, payload):
        if self.fail:
            raise RuntimeError("client gone")
        self.sent.append(payload)


def _chat(ws=None):
    chat = api._new_chat_session("test-session")
    chat.ws = ws
    return chat


def test_send_records_chat_visible_events_and_delivers():
    ws = FakeWS()
    chat = _chat(ws)
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "hi"}))
    asyncio.run(api._send(chat, {"type": "stage_change", "stage_name": "Layers"}))
    assert [e["type"] for e in chat.transcript] == ["agent_message", "stage_change"]
    assert ws.sent == chat.transcript


def test_send_does_not_record_transient_events():
    ws = FakeWS()
    chat = _chat(ws)
    asyncio.run(api._send(chat, {"type": "pipeline_busy", "busy": True}))
    asyncio.run(api._send(chat, {"type": "model_update", "scene": {}}))
    assert chat.transcript == []
    assert len(ws.sent) == 2


def test_send_survives_no_socket_and_dead_socket():
    # No socket at all: event still lands in the transcript.
    chat = _chat(ws=None)
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "kept"}))
    # Dead socket: send failure is swallowed, transcript still gets the event.
    chat.ws = FakeWS(fail=True)
    asyncio.run(api._send(chat, {"type": "progress", "content": "done"}))
    assert [e["type"] for e in chat.transcript] == ["agent_message", "progress"]


def test_send_targets_current_socket_after_reconnect():
    old, new = FakeWS(), FakeWS()
    chat = _chat(old)
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "before"}))
    chat.ws = new  # page refresh swapped the socket mid-turn
    asyncio.run(api._send(chat, {"type": "agent_message", "content": "after"}))
    assert [m["content"] for m in old.sent] == ["before"]
    assert [m["content"] for m in new.sent] == ["after"]
    assert [m["content"] for m in chat.transcript] == ["before", "after"]


# ---------------------------------------------------------------------------
# Eager staleness re-sample: a cross-edit to a sampling input after
# layer_sampling ran (e.g. adding the buried cylinder during the waveform
# stage) must re-draw the samples immediately, not wait for advanced_params.
# ---------------------------------------------------------------------------

_CYLINDER = {
    "cylinder": {
        "x_center_min_m": 0.4, "x_center_max_m": 0.6,
        "depth_min_m": 0.2, "depth_max_m": 0.3,
        "radius_min_m": 0.02, "radius_max_m": 0.04,
    }
}


def _resample_recorder(monkeypatch):
    calls = []

    async def fake_run_deterministic(chat, stage_name, fn):
        chat.phase = "deterministic"  # mimic the real node runner's side effect
        calls.append((stage_name, fn))

    monkeypatch.setattr(api, "_run_deterministic", fake_run_deterministic)
    return calls


def test_no_resample_before_sampling_ran(monkeypatch):
    calls = _resample_recorder(monkeypatch)
    chat = _chat(FakeWS())
    chat.agent_session.store["target_ranges"] = dict(_CYLINDER)
    asyncio.run(api._maybe_resample_stale(chat))  # sampling_snapshot is None
    assert calls == []


def test_no_resample_when_inputs_unchanged(monkeypatch):
    calls = _resample_recorder(monkeypatch)
    chat = _chat(FakeWS())
    chat.agent_session.store["target_ranges"] = dict(_CYLINDER)
    chat.state["sampling_snapshot"] = {
        s: chat.agent_session.store.get(s) for s in api.SAMPLING_INPUT_SECTIONS
    }
    asyncio.run(api._maybe_resample_stale(chat))
    assert calls == []


def test_cross_edit_triggers_immediate_resample(monkeypatch):
    calls = _resample_recorder(monkeypatch)
    chat = _chat(FakeWS())
    # Sampling ran while the target was skipped (empty optional section).
    chat.agent_session.store["target_ranges"] = {}
    chat.state["sampling_snapshot"] = {
        s: chat.agent_session.store.get(s) for s in api.SAMPLING_INPUT_SECTIONS
    }
    # Mid-waveform, the user adds the cylinder.
    chat.agent_session.store["target_ranges"] = dict(_CYLINDER)
    chat.phase = "agent"
    asyncio.run(api._maybe_resample_stale(chat))
    assert [fn for _, fn in calls] == [api.layer_sampling_node]
    # The store edit was synced into pipeline state before the node ran.
    assert chat.state["target_ranges"] == _CYLINDER
    # Still mid-collection: the phase must return to "agent" or the user's
    # next chat message would be rejected ("pipeline is not waiting...").
    assert chat.phase == "agent"


def test_incomplete_edit_defers_resample(monkeypatch):
    calls = _resample_recorder(monkeypatch)
    chat = _chat(FakeWS())
    chat.agent_session.store["target_ranges"] = {}
    chat.state["sampling_snapshot"] = {
        s: chat.agent_session.store.get(s) for s in api.SAMPLING_INPUT_SECTIONS
    }
    # A schema-invalid layers payload can't reach the store via save_section,
    # but stage_done must still gate the eager path (defensive).
    chat.agent_session.store["layers"] = {"num_layers": "not-an-int"}
    asyncio.run(api._maybe_resample_stale(chat))
    assert calls == []
