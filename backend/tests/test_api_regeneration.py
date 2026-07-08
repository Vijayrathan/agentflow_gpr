"""Key-free tests for the post-completion edit -> regenerate flow in
backend.api: change detection against the completion snapshot, the
regeneration kickoff, gate routing, node ordering, _finish_dataset resets,
finalize idempotency and the forward-model guard."""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import api


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


# Schema-valid, stage-complete payloads for every section.
COMPLETE_STORE = {
    "dataset_config": {"num_samples": 3, "model_basename": "demo"},
    "layers": {
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
    },
    "target_ranges": {},
    "waveform": {
        "waveform_kind": "ricker",
        "waveform_amplitude": 1.0,
        "waveform_center_freq_hz": 400e6,
        "waveform_name": "ricker_400mhz",
    },
    "antenna": {"tx_rx_offset_m": 0.12, "antenna_axis": "x"},
    "advanced_params": {},
}


def _completed_chat(ws=None):
    """A chat as _finish_dataset leaves it: complete store, snapshot taken."""
    chat = api._new_chat_session("regen-session")
    chat.ws = ws or FakeWS()
    for section, payload in COMPLETE_STORE.items():
        chat.agent_session.store[section] = dict(payload)
    api._sync_sections(chat)
    chat.state["sampling_snapshot"] = {
        s: chat.state.get(s) for s in api.SAMPLING_INPUT_SECTIONS
    }
    chat.complete = True
    chat.phase = "complete"
    chat.complete_snapshot = chat.agent_session.snapshot()
    chat.post_complete_briefed = True
    return chat


def _edit_waveform(chat):
    wf = dict(chat.agent_session.store["waveform"])
    wf["waveform_center_freq_hz"] = 900e6
    chat.agent_session.store["waveform"] = wf


def _regen_recorder(monkeypatch):
    calls = []

    async def fake_start_regeneration(chat, changed):
        calls.append(set(changed))

    monkeypatch.setattr(api, "_start_regeneration", fake_start_regeneration)
    return calls


# ---------------------------------------------------------------------------
# _check_regeneration
# ---------------------------------------------------------------------------

def test_no_change_is_relay_only(monkeypatch):
    calls = _regen_recorder(monkeypatch)
    chat = _completed_chat()
    asyncio.run(api._check_regeneration(chat))
    assert calls == []
    assert chat.complete is True


def test_complete_edit_triggers_regeneration(monkeypatch):
    calls = _regen_recorder(monkeypatch)
    chat = _completed_chat()
    _edit_waveform(chat)
    asyncio.run(api._check_regeneration(chat))
    assert calls == [{"waveform"}]


def test_incomplete_edit_blocks_and_notifies_once(monkeypatch):
    calls = _regen_recorder(monkeypatch)
    ws = FakeWS()
    chat = _completed_chat(ws)
    # Schema-valid but essential content missing => stage_done is False.
    chat.agent_session.store["layers"] = {"num_layers": 0, "layers": []}
    asyncio.run(api._check_regeneration(chat))
    assert calls == []
    assert chat.complete is True
    notices = [m for m in ws.sent if m["type"] == "progress"]
    assert len(notices) == 1 and "incomplete" in notices[0]["content"]
    # Same incomplete set on the next turn: no duplicate notice.
    asyncio.run(api._check_regeneration(chat))
    assert len([m for m in ws.sent if m["type"] == "progress"]) == 1


def test_edit_during_forward_model_is_deferred(monkeypatch):
    calls = _regen_recorder(monkeypatch)
    ws = FakeWS()
    chat = _completed_chat(ws)
    chat.simulating = True
    _edit_waveform(chat)
    asyncio.run(api._check_regeneration(chat))
    assert calls == []
    assert chat.complete is True and chat.regenerating is False
    assert any(
        "forward model is still running" in m.get("content", "")
        for m in ws.sent
    )


def test_start_regeneration_syncs_and_marks(monkeypatch):
    gate_calls = []

    async def fake_gate(chat):
        gate_calls.append(chat)

    monkeypatch.setattr(api, "_run_sample_validation_gate", fake_gate)
    chat = _completed_chat()
    _edit_waveform(chat)
    asyncio.run(api._start_regeneration(chat, {"waveform"}))
    assert chat.complete is False and chat.regenerating is True
    # The edited store landed in pipeline state before any node runs.
    assert chat.state["waveform"]["waveform_center_freq_hz"] == 900e6
    assert gate_calls == [chat]


# ---------------------------------------------------------------------------
# Sample gate routing
# ---------------------------------------------------------------------------

def test_gate_pass_routes_by_regenerating_flag(monkeypatch):
    async def fake_run_deterministic(chat, stage_name, fn):
        chat.state["sample_validation_passed"] = True

    derive_calls, stage_calls = [], []

    async def fake_derive(chat):
        derive_calls.append(chat)

    async def fake_start_stage(chat, section):
        stage_calls.append(section)

    monkeypatch.setattr(api, "_run_deterministic", fake_run_deterministic)
    monkeypatch.setattr(api, "_run_derive_chain", fake_derive)
    monkeypatch.setattr(api, "_start_stage", fake_start_stage)

    chat = _completed_chat()
    chat.regenerating = True
    asyncio.run(api._run_sample_validation_gate(chat))
    assert derive_calls and not stage_calls

    # First-run regression guard: without the flag, collection continues.
    chat2 = _completed_chat()
    chat2.regenerating = False
    asyncio.run(api._run_sample_validation_gate(chat2))
    assert stage_calls == ["advanced_params"]


# ---------------------------------------------------------------------------
# Node ordering through a full regeneration (deterministic nodes stubbed)
# ---------------------------------------------------------------------------

def _pipeline_recorder(monkeypatch):
    """Stub _run_deterministic: record node order, make both gates pass, and
    mimic layer_sampling's snapshot refresh."""
    calls = []

    async def fake_run_deterministic(chat, stage_name, fn):
        chat.phase = "deterministic"
        calls.append(fn)
        if fn is api.sample_validation_node:
            chat.state["sample_validation_passed"] = True
        elif fn is api.layer_sampling_node:
            chat.state["sampling_snapshot"] = {
                s: chat.state.get(s) for s in api.SAMPLING_INPUT_SECTIONS
            }
        elif fn is api.global_validation_node:
            chat.state["global_validation_passed"] = True

    monkeypatch.setattr(api, "_run_deterministic", fake_run_deterministic)

    finishes = []

    async def fake_finish(chat):
        finishes.append(chat)

    monkeypatch.setattr(api, "_finish_dataset", fake_finish)
    return calls, finishes


def test_sampling_input_edit_resamples_in_canonical_order(monkeypatch):
    calls, finishes = _pipeline_recorder(monkeypatch)
    chat = _completed_chat()
    layers = {
        "num_layers": 1,
        "layers": [dict(COMPLETE_STORE["layers"]["layers"][0],
                        thickness_m_max=0.8)],
    }
    chat.agent_session.store["layers"] = layers
    asyncio.run(api._check_regeneration(chat))
    # Exactly the first-run node order: gate, THEN resample before anything
    # that reads the samples, then the derive chain.
    assert calls == [
        api.sample_validation_node,
        api.layer_sampling_node,
        api.peplinski_derive_node,
        api.global_derive_node,
        api.global_validation_node,
    ]
    assert len(finishes) == 1


def test_non_sampling_edit_skips_resample(monkeypatch):
    calls, finishes = _pipeline_recorder(monkeypatch)
    chat = _completed_chat()
    _edit_waveform(chat)
    asyncio.run(api._check_regeneration(chat))
    assert api.layer_sampling_node not in calls
    assert calls == [
        api.sample_validation_node,
        api.peplinski_derive_node,
        api.global_derive_node,
        api.global_validation_node,
    ]
    assert len(finishes) == 1


# ---------------------------------------------------------------------------
# _finish_dataset resets
# ---------------------------------------------------------------------------

def test_finish_dataset_resets_regen_state(tmp_path, monkeypatch):
    async def fake_run_deterministic(chat, stage_name, fn):
        pass

    async def fake_finalize(payload):
        return {"rows_inserted": 3, "status": "complete"}

    briefings = []

    async def fake_invoke(chat, message):
        briefings.append(message)

    monkeypatch.setattr(api, "_run_deterministic", fake_run_deterministic)
    monkeypatch.setattr(api, "_build_finalize_payload", lambda chat: {
        "session_id": "regen-session",
        "user_id": "regen-session",
        "dataset_config": COMPLETE_STORE["dataset_config"],
        "layers": COMPLETE_STORE["layers"],
        "waveform": COMPLETE_STORE["waveform"],
        "antenna": COMPLETE_STORE["antenna"],
    })
    monkeypatch.setattr(api, "finalize_dataset", fake_finalize)

    chat = _completed_chat()
    chat.post_complete_briefed = False
    chat.complete = False
    chat.regenerating = True
    chat.regen_block_notice = frozenset({"layers"})
    chat.simulation_result = {"ok": 1}
    chat.state["dataset_config"] = dict(
        COMPLETE_STORE["dataset_config"], output_dir=str(tmp_path)
    )
    out_files = tmp_path / "out_files"
    out_files.mkdir()
    (out_files / "demo_1.out").write_text("stale")

    monkeypatch.setattr(api, "_invoke_and_handle", fake_invoke)
    asyncio.run(api._finish_dataset(chat))

    assert chat.complete is True and chat.phase == "complete"
    assert chat.regenerating is False
    assert chat.regen_block_notice is None
    assert chat.simulation_result is None
    assert chat.complete_snapshot == chat.agent_session.snapshot()
    assert not out_files.exists()  # stale outputs never attach to new decks
    assert briefings == [api.POST_COMPLETE_BRIEFING]
    assert [m["type"] for m in chat.ws.sent].count("dataset_ready") == 1

    # A regeneration finish must NOT re-inject the briefing.
    chat.complete = False
    chat.regenerating = True
    asyncio.run(api._finish_dataset(chat))
    assert briefings == [api.POST_COMPLETE_BRIEFING]
    assert [m["type"] for m in chat.ws.sent].count("dataset_ready") == 2
    assert chat.regenerating is False


# ---------------------------------------------------------------------------
# Finalize idempotency: old Simulation rows are deleted before re-insert
# ---------------------------------------------------------------------------

def test_finalize_sync_deletes_rows_before_insert(tmp_path, monkeypatch):
    import json

    (tmp_path / "sampled_layers.json").write_text(json.dumps({"samples": []}))
    (tmp_path / "global_derive.json").write_text(json.dumps({
        "source_height_m": 0.25,
        "domain_x_m": 1.2,
        "domain_y_m": 0.9,
        "dx_m": 0.002,
    }))
    (tmp_path / "emitted_files.json").write_text(json.dumps({"files": []}))

    class FakeDB:
        def get(self, cls, key):
            return None

        def add(self, row):
            pass

        def commit(self):
            pass

    class FakeSessionCtx:
        def __enter__(self):
            return FakeDB()

        def __exit__(self, *exc):
            return False

    order = []
    monkeypatch.setattr(api, "get_session", lambda: FakeSessionCtx())
    monkeypatch.setattr(
        api, "delete_simulations_for_session",
        lambda session_id: order.append(("delete", session_id)) or 0,
    )
    monkeypatch.setattr(
        api, "batch_insert_simulations",
        lambda rows: order.append(("insert", len(rows))) or len(rows),
    )

    payload = api.FinalizeDatasetPayload(
        session_id="regen-session",
        user_id="regen-session",
        dataset_config=COMPLETE_STORE["dataset_config"],
        layers=COMPLETE_STORE["layers"],
        waveform=COMPLETE_STORE["waveform"],
        antenna=COMPLETE_STORE["antenna"],
        artifacts={
            "sampled_layers_json": str(tmp_path / "sampled_layers.json"),
            "global_derive_json": str(tmp_path / "global_derive.json"),
            "emitted_files_json": str(tmp_path / "emitted_files.json"),
        },
    )
    api._finalize_dataset_sync(payload)
    # The delete always runs (even with zero rows to insert) and precedes
    # any insert, so a re-finalize can never duplicate (session, sample) rows.
    assert [step for step, _ in order] == ["delete"]

    (tmp_path / "emitted_files.json").write_text(json.dumps({
        "files": [{"sample_id": 1, "filename": "demo_1.in"}],
    }))
    (tmp_path / "sampled_layers.json").write_text(json.dumps({
        "samples": [{"sample_id": 1, "layers": [], "targets": []}],
    }))
    order.clear()
    api._finalize_dataset_sync(payload)
    assert [step for step, _ in order] == ["delete", "insert"]


# ---------------------------------------------------------------------------
# Forward model is blocked mid-regeneration
# ---------------------------------------------------------------------------

def test_simulate_endpoint_409s_while_regenerating(tmp_path):
    chat = api._new_chat_session("regen-guard")
    api.sessions["regen-guard"] = chat
    try:
        in_dir = tmp_path / "in_files"
        in_dir.mkdir()
        (in_dir / "demo_1.in").write_text("#title: demo\n")
        (tmp_path / "emitted_files.json").write_text(
            '{"n_written": 1, "in_dir": "%s", '
            '"files": [{"sample_id": 1, "filename": "demo_1.in"}]}' % in_dir
        )
        chat.state["dataset_config"] = {
            "num_samples": 1,
            "model_basename": "demo",
            "output_dir": str(tmp_path),
        }
        chat.regenerating = True
        with pytest.raises(HTTPException) as exc:
            asyncio.run(api.start_forward_model("regen-guard"))
        assert exc.value.status_code == 409
        assert "regenerating" in exc.value.detail
    finally:
        api.sessions.pop("regen-guard", None)
