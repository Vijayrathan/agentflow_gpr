"""
Key-free tests for the EXPERIMENTAL single-agent pipeline
(`backend/agentflow_single_agent.py`): the in-memory section store, the
save/get tools, stage-completion detection, staleness routing and the
remediation resample detection. No OPENAI_API_KEY needed — the agent is
built lazily and never constructed here.

Run: pytest backend/tests/test_single_agent_store.py -v
"""
import json
import sys
from pathlib import Path

import pytest

# Mirror the runtime path setup: repo root for `backend.*`, `backend/` for the
# bare `schema` / `dataset_sampling` imports, and the inner gprMax package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND), str(_REPO_ROOT / "gprMax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import agentflow_single_agent as sap  # noqa: E402
from langgraph.graph import END  # noqa: E402


@pytest.fixture(autouse=True)
def reset_store():
    for s in sap.SECTION_SCHEMA:
        sap._STORE[s] = None
    yield


VALID_LAYERS = {
    "num_layers": 1,
    "layers": [
        {
            "name": "sandy_loam",
            "thickness_m_min": 0.3,
            "thickness_m_max": 0.5,
            "sand_pct_min": 30.0,
            "sand_pct_max": 40.0,
            "clay_pct_min": 5.0,
            "clay_pct_max": 15.0,
            "theta_v_min": 0.05,
            "theta_v_max": 0.20,
            "bulk_density_gcm3_min": 1.5,
            "bulk_density_gcm3_max": 1.6,
            "particle_density_gcm3_min": 2.60,
            "particle_density_gcm3_max": 2.66,
        }
    ],
}


def _save(section, payload) -> dict:
    if not isinstance(payload, str):
        payload = json.dumps(payload)
    return json.loads(
        sap.save_section.invoke({"section": section, "payload": payload})
    )


# ---------------------------------------------------------------------------
# save_section / get_section
# ---------------------------------------------------------------------------

def test_save_rejects_unknown_section():
    out = _save("nonsense", "{}")
    assert "error" in out


def test_save_rejects_bad_json_and_leaves_store_untouched():
    out = _save("layers", "{not json")
    assert out["error"] == "invalid_json"
    assert sap._STORE["layers"] is None


def test_save_rejects_schema_violation_and_leaves_store_untouched():
    # theta_v_max above the loosest porosity (1 - 1.5/2.66 ~= 0.436)
    bad = json.loads(json.dumps(VALID_LAYERS))
    bad["layers"][0]["theta_v_max"] = 0.9
    out = _save("layers", bad)
    assert out["error"] == "validation_failed"
    assert sap._STORE["layers"] is None
    assert not sap._stage_done("layers")


def test_save_valid_section_stores_and_completes():
    out = _save("layers", VALID_LAYERS)
    assert out["status"] == "ok"
    assert sap._STORE["layers"]["num_layers"] == 1
    assert sap._stage_done("layers")


def test_save_valid_but_incomplete_is_stored_incomplete():
    # Schema-valid (num_layers == len(layers) == 0) but essential content missing.
    out = _save("layers", {"num_layers": 0, "layers": []})
    assert out["status"] == "stored_incomplete"
    assert sap._STORE["layers"] is not None
    assert not sap._stage_done("layers")


def test_dataset_config_defaults_complete():
    out = _save("dataset_config", {"num_samples": 5})
    assert out["status"] == "ok"
    assert sap._stage_done("dataset_config")
    # Server-fixed fields land at their defaults without being collected;
    # output_dir is the per-dataset dir named after the (default) basename.
    assert sap._STORE["dataset_config"]["output_dir"] == "./dataset/soil_sample"
    assert sap._STORE["dataset_config"]["dimensionality"] == "2D"


def test_dataset_config_preserves_explicit_mode_and_fixes_server_paths(monkeypatch):
    monkeypatch.setenv("GPR_ENABLE_EXPERIMENTAL_3D", "1")
    # Even if the agent passes user-supplied values through, the server-fixed
    # fields are forced back; output_dir follows the model_basename.
    out = _save("dataset_config", {
        "num_samples": 5,
        "model_basename": "clay_survey",
        "output_dir": "/somewhere/else",
        "dimensionality": "3D",
        "num_threads": 16,
    })
    assert out["status"] == "ok"
    assert sap._STORE["dataset_config"]["output_dir"] == "./dataset/clay_survey"
    assert sap._STORE["dataset_config"]["dimensionality"] == "3D"
    assert sap._STORE["dataset_config"]["contract_version"] == 2
    assert sap._STORE["dataset_config"]["num_threads"] is None


def test_3d_collection_requires_developer_release_gate(monkeypatch):
    monkeypatch.delenv("GPR_ENABLE_EXPERIMENTAL_3D", raising=False)
    out = _save("dataset_config", {"num_samples": 1, "dimensionality": "3D"})
    assert out["error"] == "validation_failed"
    assert "experimental" in out["detail"]
    assert sap._STORE["dataset_config"] is None


@pytest.mark.parametrize("raw,expected", [
    ("wet sand run 2", "wet_sand_run_2"),        # whitespace -> underscore
    ("../../etc", "etc"),                        # traversal stripped
    ("a/b\\c", "a_b_c"),                         # path separators collapsed
    ("..", "soil_sample"),                       # nothing usable -> default
    ("", "soil_sample"),
    (None, "soil_sample"),
])
def test_dataset_dirname_sanitization(raw, expected):
    assert sap._dataset_dirname(raw) == expected


def test_optional_section_skip_completes():
    out = _save("target_ranges", {})  # skip = empty payload (no objects)
    assert out["status"] == "ok"
    assert sap._stage_done("target_ranges")


@pytest.mark.parametrize("kind", ["custom_antenna", "", "   ", None])
def test_invalid_source_type_does_not_replace_saved_antenna(kind):
    valid = {"antenna_kind": "transmission_line", "resistance": 75, "tx_rx_offset_m": 0.1}
    assert _save("antenna", valid)["status"] == "ok"
    before = sap._store_snapshot()["antenna"]
    out = _save("antenna", {**valid, "antenna_kind": kind})
    assert out["error"] == "validation_failed"
    assert sap._STORE["antenna"] == before


@pytest.mark.parametrize("kind", ["hertzian_dipole", "voltage_source", "transmission_line"])
@pytest.mark.parametrize("separator", ["_", " ", "  ", "\t"])
def test_source_type_is_normalized_and_preserved_in_store(kind, separator):
    out = _save("antenna", {
        "antenna_kind": f" {kind.upper().replace('_', separator)} ",
        "resistance": 75, "tx_rx_offset_m": 0.1,
    })
    assert out["status"] == "ok"
    assert sap._STORE["antenna"]["antenna_kind"] == kind
    assert sap._stage_done("antenna")


@pytest.mark.parametrize("kind", ["VOLTAGE_SOURCE", "TRANSMISSION_LINE"])
@pytest.mark.parametrize("resistance", [None, 0, -1, 376.73, float("inf"), float("nan")])
def test_resistive_source_requires_valid_resistance_after_normalization(kind, resistance):
    out = _save("antenna", {
        "antenna_kind": kind, "resistance": resistance, "tx_rx_offset_m": 0.1,
    })
    assert out["error"] == "validation_failed"
    assert sap._STORE["antenna"] is None


def test_omitted_source_type_uses_schema_default():
    assert _save("antenna", {"tx_rx_offset_m": 0.1})["status"] == "ok"
    assert sap._STORE["antenna"]["antenna_kind"] == "hertzian_dipole"


def test_target_ranges_multi_object_roundtrip():
    payload = {
        "cylinders": [{
            "name": "pipe", "material": "pec",
            "x_offset_min_m": -0.2, "x_offset_max_m": 0.2,
            "depth_min_m": 0.2, "depth_max_m": 0.4,
            "radius_min_m": 0.03, "radius_max_m": 0.07,
        }],
        "boxes": [{
            # static: min == max everywhere
            "name": "slab", "material": "pec",
            "x_offset_min_m": -0.3, "x_offset_max_m": -0.3,
            "depth_min_m": 0.35, "depth_max_m": 0.35,
            "width_min_m": 0.2, "width_max_m": 0.2,
            "height_min_m": 0.06, "height_max_m": 0.06,
        }],
    }
    out = _save("target_ranges", payload)
    assert out["status"] == "ok"
    assert sap._stage_done("target_ranges")
    got = json.loads(sap.get_section.invoke({"section": "target_ranges"}))
    assert len(got["cylinders"]) == 1 and len(got["boxes"]) == 1
    # the old single-cylinder key is a clean break: extra keys are rejected
    # only if the schema forbids them; here `cylinder` is simply ignored by
    # pydantic, so assert the canonical lists instead of its absence.
    assert got["boxes"][0]["width_min_m"] == 0.2


def test_get_section_roundtrip_and_not_populated():
    missing = json.loads(sap.get_section.invoke({"section": "waveform"}))
    assert missing["error"] == "section_not_populated"
    _save("layers", VALID_LAYERS)
    got = json.loads(sap.get_section.invoke({"section": "layers"}))
    assert got["num_layers"] == 1


# ---------------------------------------------------------------------------
# Snapshot diffing / staleness / routing
# ---------------------------------------------------------------------------

def test_changed_sections_diff():
    _save("layers", VALID_LAYERS)
    before = sap._store_snapshot()
    edited = json.loads(json.dumps(VALID_LAYERS))
    edited["layers"][0]["thickness_m_max"] = 0.8
    _save("layers", edited)
    _save("dataset_config", {"num_samples": 3})
    changed = sap._changed_sections(before, sap._store_snapshot())
    assert changed == {"layers", "dataset_config"}


def test_resample_detection_via_snapshot_diff():
    assert bool({"layers"} & sap.RESAMPLE_SECTIONS)
    assert bool({"target_ranges"} & sap.RESAMPLE_SECTIONS)
    assert not bool({"antenna", "waveform"} & sap.RESAMPLE_SECTIONS)


def test_samples_stale():
    state = {
        "layers": VALID_LAYERS,
        "dataset_config": {"num_samples": 3},
        "target_ranges": None,
    }
    # No snapshot yet -> stale (sampling never ran)
    assert sap._samples_stale(state)
    state["sampling_snapshot"] = sap._sampling_inputs(state)
    assert not sap._samples_stale(state)
    # A cross-edit to layers after sampling -> stale
    edited = json.loads(json.dumps(VALID_LAYERS))
    edited["layers"][0]["thickness_m_max"] = 0.8
    state["layers"] = edited
    assert sap._samples_stale(state)


def test_after_sampling_routes_by_gate_marker():
    assert sap._after_sampling({}) == "waveform"
    assert sap._after_sampling({"sample_validation_passed": True}) == "peplinski_derive"
    assert sap._after_sampling({"halted": True}) == END


def test_route_after_advanced():
    state = {
        "layers": VALID_LAYERS,
        "dataset_config": {"num_samples": 3},
        "target_ranges": None,
    }
    state["sampling_snapshot"] = sap._sampling_inputs(state)
    assert sap._route_after_advanced(state) == "peplinski_derive"
    state["dataset_config"] = {"num_samples": 10}
    assert sap._route_after_advanced(state) == "layer_sampling"
    assert sap._route_after_advanced({"halted": True}) == END


# ---------------------------------------------------------------------------
# API-parity eager resampling / sampling remediation in the standalone graph
# ---------------------------------------------------------------------------

def test_eager_resample_during_collect_syncs_snapshot(monkeypatch):
    state = {
        "dataset_config": {"num_samples": 3},
        "layers": VALID_LAYERS,
        "target_ranges": {},
    }
    state["sampling_snapshot"] = sap._sampling_inputs(state)
    for section, payload in state.items():
        if section in sap.SECTION_SCHEMA:
            sap._STORE[section] = json.loads(json.dumps(payload))

    edited_target = {
        "cylinders": [{
            "name": "pipe",
            "material": "pec",
            "x_offset_min_m": -0.2,
            "x_offset_max_m": 0.2,
            "depth_min_m": 0.2,
            "depth_max_m": 0.4,
            "radius_min_m": 0.03,
            "radius_max_m": 0.07,
        }]
    }
    sap._STORE["target_ranges"] = edited_target
    calls = []

    def fake_sampling(state_arg):
        calls.append(json.loads(json.dumps(state_arg)))
        return {"sampling_snapshot": sap._sampling_inputs(state_arg)}

    monkeypatch.setattr(sap, "_run_layer_sampling_or_remediate", fake_sampling)

    halt_reason = sap._maybe_resample_stale_during_collect(state)

    assert halt_reason is None
    assert len(calls) == 1
    assert calls[0]["target_ranges"] == edited_target
    assert state["target_ranges"] == edited_target
    assert state["sampling_snapshot"]["target_ranges"] == edited_target


def test_eager_resample_defers_incomplete_sampling_edit(monkeypatch):
    state = {
        "dataset_config": {"num_samples": 3},
        "layers": VALID_LAYERS,
        "target_ranges": {},
    }
    state["sampling_snapshot"] = sap._sampling_inputs(state)
    sap._STORE.update({
        "dataset_config": {"num_samples": 3},
        "layers": {"num_layers": "not-an-int"},
        "target_ranges": {},
    })
    calls = []
    monkeypatch.setattr(
        sap,
        "_run_layer_sampling_or_remediate",
        lambda state_arg: calls.append(state_arg) or {},
    )

    halt_reason = sap._maybe_resample_stale_during_collect(state)

    assert halt_reason is None
    assert calls == []
    assert state["sampling_snapshot"]["layers"] == VALID_LAYERS


def test_sampling_failure_remediates_and_retries(monkeypatch):
    sap._STORE["dataset_config"] = {"num_samples": 3}
    sap._STORE["layers"] = VALID_LAYERS
    state = {
        "dataset_config": {"num_samples": 3},
        "layers": VALID_LAYERS,
        "target_ranges": None,
    }
    calls = []

    def fake_layer_sampling(state_arg):
        calls.append(json.loads(json.dumps(state_arg)))
        if len(calls) == 1:
            raise ValueError(
                "Could not draw a feasible sample for layer after 200 retries"
            )
        return {"sampling_snapshot": sap._sampling_inputs(state_arg)}

    def fake_remediation(kickoff, display_name):
        assert "Layer + Target Sampling" in kickoff
        assert display_name == "layer sampling remediation"
        sap._STORE["target_ranges"] = {}
        return {"target_ranges"}, None

    monkeypatch.setattr(sap, "layer_sampling_node", fake_layer_sampling)
    monkeypatch.setattr(sap, "_run_remediation", fake_remediation)

    updates = sap._run_layer_sampling_or_remediate(state)

    assert len(calls) == 2
    assert updates["target_ranges"] == {}
    assert updates["sampling_snapshot"]["target_ranges"] == {}


@pytest.mark.parametrize(
    ("state", "expected_next"),
    [
        ({}, "waveform"),
        ({"sample_validation_passed": True}, "peplinski_derive"),
    ],
)
def test_sampling_retry_resumes_at_api_equivalent_stage(
    monkeypatch, state, expected_next
):
    attempts = 0

    def fake_layer_sampling(state_arg):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("Could not draw a feasible sample")
        return {"sampling_snapshot": sap._sampling_inputs(state_arg)}

    monkeypatch.setattr(sap, "layer_sampling_node", fake_layer_sampling)
    monkeypatch.setattr(
        sap,
        "_run_remediation",
        lambda _kickoff, _display_name: ({"layers"}, None),
    )

    updates = sap.layer_sampling_graph_node(state)
    routed_state = {**state, **updates}

    assert attempts == 2
    assert sap._after_sampling(routed_state) == expected_next


def test_sampling_failure_can_halt(monkeypatch):
    def broken_sampling(_state):
        raise ValueError("Could not draw a feasible sample")

    monkeypatch.setattr(sap, "layer_sampling_node", broken_sampling)
    monkeypatch.setattr(
        sap,
        "_run_remediation",
        lambda _kickoff, _display_name: (set(), "user exited during remediation"),
    )

    updates = sap._run_layer_sampling_or_remediate({})

    assert updates["halted"] is True
    assert updates["halt_reason"] == "user exited during remediation"


# ---------------------------------------------------------------------------
# Graph compiles without an API key (agent is lazy)
# ---------------------------------------------------------------------------

def test_build_graph_compiles():
    graph = sap.build_graph()
    assert graph is not None
