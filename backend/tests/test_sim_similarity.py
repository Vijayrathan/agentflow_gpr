"""Key-free tests for the session-config similarity index.

No Qdrant, no Postgres, no API keys: the Qdrant client is stubbed and all
assertions target the deterministic feature/vector/rescore math plus the
failure-swallowing contract of the module-level conveniences.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import sim_similarity as ss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sections(*, with_targets: bool = True) -> dict:
    """A minimal valid section set (mirrors what chat.state holds)."""
    layer = {
        "name": "topsoil",
        "thickness_m_min": 0.2, "thickness_m_max": 0.4,
        "sand_pct_min": 50.0, "sand_pct_max": 70.0,
        "clay_pct_min": 10.0, "clay_pct_max": 20.0,
        "theta_v_min": 0.05, "theta_v_max": 0.20,
        "bulk_density_gcm3_min": 1.5, "bulk_density_gcm3_max": 1.7,
        "particle_density_gcm3_min": 2.6, "particle_density_gcm3_max": 2.7,
    }
    layer2 = dict(layer, name="subsoil", thickness_m_min=0.5, thickness_m_max=0.8)
    sections = {
        "dataset_config": {"num_samples": 20, "model_basename": "soil"},
        "layers": {"num_layers": 2, "layers": [layer, layer2]},
        "waveform": {
            "waveform_center_freq_hz": 900e6,
            "waveform_name": "ricker900",
        },
        "antenna": {"tx_rx_offset_m": 0.1},
        "target_ranges": None,
        "advanced_params": None,
    }
    if with_targets:
        sections["target_ranges"] = {
            "cylinders": [
                {"x_offset_min_m": -0.1, "x_offset_max_m": 0.1,
                 "depth_min_m": 0.3, "depth_max_m": 0.5,
                 "radius_min_m": 0.04, "radius_max_m": 0.06},
            ],
            "boxes": [],
        }
    return sections


def make_payload(**kwargs) -> dict:
    sections = make_sections(**kwargs)
    return ss.build_feature_payload(
        dataset_config=sections["dataset_config"],
        layers=sections["layers"],
        target_ranges=sections["target_ranges"],
        waveform=sections["waveform"],
        antenna=sections["antenna"],
        advanced_params=sections["advanced_params"],
    )


# ---------------------------------------------------------------------------
# Vector building
# ---------------------------------------------------------------------------

def test_vector_deterministic_and_fixed_length():
    v1 = ss.build_vector(make_payload())
    v2 = ss.build_vector(make_payload())
    assert v1 == v2
    assert len(v1) == ss.VECTOR_DIM


def test_vector_padding_is_zero():
    v = ss.build_vector(make_payload())
    # layers occupy 2 slots of MAX_LAYERS — the rest must be exact zeros
    layer_block = v[: ss.MAX_LAYERS * ss.LAYER_DIMS]
    assert all(x == 0.0 for x in layer_block[2 * ss.LAYER_DIMS:])
    # 1 cylinder of MAX_CYLINDERS, 0 boxes
    cyl_start = ss.MAX_LAYERS * ss.LAYER_DIMS + ss.WA_DIMS
    cyl_block = v[cyl_start: cyl_start + ss.MAX_CYLINDERS * ss.CYL_DIMS]
    assert all(x == 0.0 for x in cyl_block[ss.CYL_DIMS:])
    assert all(x == 0.0 for x in v[cyl_start + ss.MAX_CYLINDERS * ss.CYL_DIMS:])


def test_target_sort_is_input_order_independent():
    base = make_sections()
    c1 = {"x_offset_min_m": -0.1, "x_offset_max_m": 0.1,
          "depth_min_m": 0.3, "depth_max_m": 0.5,
          "radius_min_m": 0.04, "radius_max_m": 0.06}
    c2 = {"x_offset_min_m": 0.2, "x_offset_max_m": 0.3,
          "depth_min_m": 0.6, "depth_max_m": 0.8,
          "radius_min_m": 0.02, "radius_max_m": 0.03}
    base["target_ranges"] = {"cylinders": [c1, c2], "boxes": []}
    p_fwd = ss.build_feature_payload(
        dataset_config=base["dataset_config"], layers=base["layers"],
        target_ranges=base["target_ranges"], waveform=base["waveform"],
        antenna=base["antenna"])
    base["target_ranges"] = {"cylinders": [c2, c1], "boxes": []}
    p_rev = ss.build_feature_payload(
        dataset_config=base["dataset_config"], layers=base["layers"],
        target_ranges=base["target_ranges"], waveform=base["waveform"],
        antenna=base["antenna"])
    assert p_fwd == p_rev
    assert ss.build_vector(p_fwd) == ss.build_vector(p_rev)


def test_over_capacity_raises():
    sections = make_sections()
    layer = sections["layers"]["layers"][0]
    sections["layers"] = {"num_layers": ss.MAX_LAYERS + 1,
                          "layers": [dict(layer) for _ in range(ss.MAX_LAYERS + 1)]}
    with pytest.raises(ValueError):
        ss.build_feature_payload(
            dataset_config=sections["dataset_config"], layers=sections["layers"],
            target_ranges=None, waveform=sections["waveform"],
            antenna=sections["antenna"])


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------

def test_hard_filters_contents():
    f = ss.hard_filters(make_payload())
    assert f == {
        "num_layers": 2,
        "waveform_kind": "ricker",
        "antenna_kind": "hertzian_dipole",
        "antenna_axis": "x",
        "dimensionality": "2D",
        "n_cylinders": 1,
        "n_boxes": 0,
        "has_surface_roughness": False,
        "cells_per_wavelength": 10,
        "pml_cells": 10,
        "buffer_cells": 10,
        "fractal_nbins": 50,
        "high_freq_factor_x100": 300,
    }


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------

def test_rescore_identical_is_one():
    score, breakdown = ss.rescore(make_payload(), make_payload())
    assert score == pytest.approx(1.0)
    assert all(d["sim"] == 1.0 for d in breakdown)


def test_range_sim_interval_iou():
    # inter 0.1, union 0.3
    assert ss._range_sim([0.1, 0.3], [0.2, 0.4], 3.0) == pytest.approx(1 / 3)
    # disjoint ranges
    assert ss._range_sim([0.1, 0.2], [0.3, 0.4], 3.0) == 0.0
    # identical points
    assert ss._range_sim([0.2, 0.2], [0.2, 0.2], 3.0) == 1.0
    # two distinct fixed values -> scalar rule with the field scale
    assert ss._range_sim([0.2, 0.2], [0.3, 0.3], 3.0) == pytest.approx(1 - 0.1 / 3.0)


def test_rescore_log_frequency_rule():
    a, b = make_payload(), make_payload()
    b["waveform"]["center_freq_hz"] = a["waveform"]["center_freq_hz"] * (10 ** 0.15)
    _, breakdown = ss.rescore(a, b)
    freq = next(d for d in breakdown if d["param"] == "waveform.center_freq_hz")
    assert freq["sim"] == pytest.approx(0.5, abs=1e-3)


def test_rescore_one_sided_none_is_zero():
    a, b = make_payload(), make_payload()
    b["antenna"]["source_height_m"] = 0.3   # a's is None
    score, breakdown = ss.rescore(a, b)
    h = next(d for d in breakdown if d["param"] == "antenna.source_height_m")
    assert h["sim"] == 0.0
    assert score < 1.0


def test_rescore_zero_targets_redistributes_weight():
    a, b = make_payload(with_targets=False), make_payload(with_targets=False)
    score, _ = ss.rescore(a, b)
    assert score == pytest.approx(1.0)   # vacuous target agreement never inflates

    # A known single-field diff must renormalize over the remaining groups.
    b["layers"][1]["thickness_m"] = [0.6, 0.9]   # vs [0.5, 0.8]: IoU = 0.2/0.4
    score, _ = ss.rescore(a, b)
    layer2 = 0.75 + ss.LAYER_FIELD_WEIGHTS["thickness_m"] * 0.5
    layers = (1.0 + layer2) / 2
    expected = (0.50 * layers + 0.20 + 0.15) / 0.85
    assert score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# find_similar with a stubbed Qdrant client
# ---------------------------------------------------------------------------

class FakeQdrant:
    def __init__(self, hits):
        self.hits = hits
        self.last_query = None

    def query_points(self, collection_name, query, query_filter, limit, with_payload):
        self.last_query = {"collection": collection_name, "filter": query_filter,
                           "limit": limit}
        return SimpleNamespace(points=self.hits)


def _index_with(hits) -> ss.SimilarityIndex:
    idx = ss.SimilarityIndex.__new__(ss.SimilarityIndex)
    idx.url = "stub"
    idx.client = FakeQdrant(hits)
    return idx


def _hit(session_id, feature_payload, **meta):
    return SimpleNamespace(payload={"session_id": session_id,
                                    "feature_payload": feature_payload, **meta})


def test_find_similar_threshold_gating():
    query = make_payload()
    near = make_payload()                       # rescores to 1.0
    far = make_payload()
    far["waveform"]["center_freq_hz"] = 450e6   # log-freq sim 0 -> well below 0.95
    idx = _index_with([_hit("far", far), _hit("near", near, num_samples=20,
                                              user_id="u1", output_dir="/d",
                                              created_at="2026-01-01")])
    matches = idx.find_similar(query, exclude_session_id="me", threshold=0.95)
    assert [m["source_session_id"] for m in matches] == ["near"]
    assert matches[0]["similarity"] == 1.0
    assert matches[0]["num_samples"] == 20
    assert matches[0]["source_user_id"] == "u1"

    assert idx.find_similar(query, exclude_session_id="me", threshold=1.01) == []


def test_find_similar_orders_by_rescore_not_ann_order():
    query = make_payload()
    close = make_payload()
    closer = make_payload()
    close["antenna"]["tx_rx_offset_m"] = 0.14    # scalar sim 0.96
    closer["antenna"]["tx_rx_offset_m"] = 0.11   # scalar sim 0.99
    idx = _index_with([_hit("close", close), _hit("closer", closer)])
    matches = idx.find_similar(query, exclude_session_id="me", threshold=0.5)
    assert [m["source_session_id"] for m in matches] == ["closer", "close"]


def test_find_similar_excludes_self_via_filter():
    idx = _index_with([])
    idx.find_similar(make_payload(), exclude_session_id="my-session")
    flt = idx.client.last_query["filter"]
    assert any(c.key == "session_id" for c in flt.must_not)
    hard = ss.hard_filters(make_payload())
    assert {c.key for c in flt.must} == set(hard.keys())


# ---------------------------------------------------------------------------
# Failure swallowing
# ---------------------------------------------------------------------------

def test_find_similar_session_swallows_qdrant_failure(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("qdrant down")

    monkeypatch.setattr(ss, "SimilarityIndex", Boom)
    state = make_sections()
    assert ss.find_similar_session(state, session_id="s") == []


def test_index_completed_session_swallows_failure(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("qdrant down")

    monkeypatch.setattr(ss, "SimilarityIndex", Boom)
    assert ss.index_completed_session(
        make_sections(), session_id="s", user_id="u", num_samples=3,
        output_dir="/tmp/x") is False


def test_find_similar_session_handles_missing_sections():
    assert ss.find_similar_session({}, session_id="s") == []


def test_disabled_flag(monkeypatch):
    monkeypatch.setattr(ss, "REUSE_ENABLED", False)
    assert ss.find_similar_session(make_sections(), session_id="s") == []
    assert ss.index_completed_session(
        make_sections(), session_id="s", user_id="u", num_samples=1,
        output_dir="/tmp/x") is False
