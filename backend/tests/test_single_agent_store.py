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


def test_optional_section_skip_completes():
    out = _save("target_ranges", {})  # skip = empty payload (no objects)
    assert out["status"] == "ok"
    assert sap._stage_done("target_ranges")


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
# Graph compiles without an API key (agent is lazy)
# ---------------------------------------------------------------------------

def test_build_graph_compiles():
    graph = sap.build_graph()
    assert graph is not None
