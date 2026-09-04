"""
Key-free tests for the frontend scene projection (`backend/viz_projection.py`):
store -> ranges preview (midpoints, Peplinski preview eps, provisional
frequency), manifests -> samples/grid (flag-gated so stale files never leak),
and the sample payload cap. No OPENAI_API_KEY and no FastAPI involved.

Run: pytest backend/tests/test_viz_projection.py -v
"""
import json
import sys
from pathlib import Path

# Mirror the runtime path setup: repo root for `backend.*`, `backend/` for the
# bare `schema` / `dataset_sampling` imports, and the inner gprMax package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"
for _p in (str(_REPO_ROOT), str(_BACKEND), str(_REPO_ROOT / "gprMax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend import viz_projection as vz  # noqa: E402
from backend.validation_tools_new import peak_frequency  # noqa: E402

EMPTY_STORE = {
    s: None
    for s in ("dataset_config", "layers", "target_ranges", "waveform",
              "antenna", "advanced_params")
}

NO_FLAGS = {"sampled": False, "derived": False, "grid": False,
            "placed": False, "emitted": False}

LAYERS_SECTION = {
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

TARGET_SECTION = {
    "cylinders": [
        {
            "name": "pipe",
            "material": "pec",
            "x_offset_min_m": -0.2,
            "x_offset_max_m": 0.2,
            "depth_min_m": 0.2,
            "depth_max_m": 0.4,
            "radius_min_m": 0.03,
            "radius_max_m": 0.07,
        }
    ],
    "boxes": [
        {
            # STATIC box: min == max on every field
            "name": "slab",
            "material": "pec",
            "x_offset_min_m": -0.3,
            "x_offset_max_m": -0.3,
            "depth_min_m": 0.35,
            "depth_max_m": 0.35,
            "width_min_m": 0.2,
            "width_max_m": 0.2,
            "height_min_m": 0.06,
            "height_max_m": 0.06,
        }
    ],
}

WAVEFORM_SECTION = {
    "waveform_kind": "ricker",
    "waveform_name": "src",
    "waveform_center_freq_hz": 7.0e8,
}


def _store(**sections) -> dict:
    store = dict(EMPTY_STORE)
    store.update(sections)
    return store


def _write_manifests(out_dir: Path, num_samples: int = 2, with_target: bool = True):
    samples = []
    for sid in range(1, num_samples + 1):
        samples.append({
            "sample_id": sid,
            "layers": [{
                "name": "sandy_loam",
                "thickness_m": 0.3 + 0.01 * sid,
                "sand_pct": 35.0,
                "clay_pct": 10.0,
                "silt_pct": 55.0,
                "theta_v_min": 0.05,
                "theta_v_max": 0.20,
                "bulk_density_gcm3": 1.55,
                "particle_density_gcm3": 2.63,
            }],
            "targets": [
                {"kind": "cylinder", "name": "pipe", "material": "pec",
                 "x_offset_m": -0.1, "depth_m": 0.3, "radius_m": 0.05},
                {"kind": "box", "name": "slab", "material": "pec",
                 "x_offset_m": -0.3, "depth_m": 0.35,
                 "width_m": 0.2, "height_m": 0.06},
            ] if with_target else [],
        })
    (out_dir / "sampled_layers.json").write_text(json.dumps({
        "num_samples": num_samples, "warnings": [], "samples": samples,
    }))
    (out_dir / "derived_layers.json").write_text(json.dumps({
        "eps_r_aggregate": {"eps_r_max": 12.0, "eps_r_min": 5.0},
        "samples": [
            {"sample_id": s["sample_id"],
             "layers": [{"name": "sandy_loam", "eps_r_dry": 5.0, "eps_r_wet": 12.0}]}
            for s in samples
        ],
    }))
    (out_dir / "global_derive.json").write_text(json.dumps({
        "f_peak_hz": 7.0e8,
        "dx_m": 0.004,
        "domain_x_m": 1.3,
        "domain_y_m": 1.7,
        "depth_z_m": 1.0,
        "ground_y_m": 1.15,
        "source_height_m": 0.45,
        "tx_x_m": 0.6,
        "rx_x_m": 0.72,
        "time_window_s": 4.0e-8,
    }))


# ---------------------------------------------------------------------------
# store -> ranges preview
# ---------------------------------------------------------------------------

def test_empty_store_returns_none():
    assert vz.build_scene(EMPTY_STORE, NO_FLAGS, None) is None


def test_dataset_config_only_gives_named_scene_without_ranges():
    scene = vz.build_scene(
        _store(dataset_config={"num_samples": 5, "model_basename": "soil_run"}),
        NO_FLAGS, None,
    )
    assert scene["project"] == "soil_run"
    assert scene["ranges"] is None
    assert scene["samples"] is None
    assert scene["phase"] == "collect"


def test_layers_ranges_midpoints_and_provisional_eps():
    scene = vz.build_scene(_store(layers=LAYERS_SECTION), NO_FLAGS, None, stage="layers")
    assert scene["stage"] == "layers"
    (layer,) = scene["ranges"]["layers"]
    assert layer["thickness_mid_m"] == 0.4
    assert layer["thickness_min_m"] == 0.3
    assert layer["thickness_max_m"] == 0.5
    assert layer["sand_pct_mid"] == 35.0
    assert layer["silt_pct_mid"] == 55.0
    # gprMax-native Peplinski at midpoint composition over the theta_v band
    assert layer["eps_dry"] <= layer["eps_mid"] <= layer["eps_wet"]
    assert layer["eps_dry"] > 1.0
    assert layer["eps_freq_hz"] == vz.PREVIEW_FREQ_HZ
    assert layer["eps_provisional"] is True
    # provisional domain covers the layer stack
    assert scene["domain"]["provisional"] is True
    assert scene["domain"]["depth_m"] >= 0.4


def test_waveform_fixes_preview_frequency():
    scene = vz.build_scene(
        _store(
            layers=LAYERS_SECTION,
            waveform=WAVEFORM_SECTION,
            dataset_config={"num_samples": 5, "center_freq_is_peak": True},
        ),
        NO_FLAGS, None,
    )
    (layer,) = scene["ranges"]["layers"]
    assert layer["eps_freq_hz"] == peak_frequency(7.0e8, True)
    assert layer["eps_provisional"] is False
    assert scene["acquisition"]["frequency_ghz"] == 0.7
    assert scene["acquisition"]["waveform"] == "ricker"


def test_target_ranges_midpoints_multi():
    scene = vz.build_scene(
        _store(layers=LAYERS_SECTION, target_ranges=TARGET_SECTION), NO_FLAGS, None
    )
    targets = scene["ranges"]["targets"]
    assert len(targets) == 2
    cyl, box = targets  # canonical order: cylinders, then boxes
    assert cyl["kind"] == "cylinder"
    assert cyl["x_offset_mid_m"] == 0.0
    assert cyl["depth_mid_m"] == 0.3
    assert cyl["radius_mid_m"] == 0.05
    assert cyl["material"] == "pec"
    assert cyl["static"] is False
    assert box["kind"] == "box"
    assert box["x_offset_mid_m"] == -0.3
    assert box["width_mid_m"] == 0.2
    assert box["height_mid_m"] == 0.06
    assert box["static"] is True  # min == max on every field
    # provisional width covers the worst offset+extent symmetrically:
    # box |offset| 0.3 + width/2 0.1 + margin 0.2 => >= 2*0.6
    assert scene["domain"]["width_m"] >= 2 * (0.3 + 0.1 + 0.2) - 1e-9


# ---------------------------------------------------------------------------
# manifests -> samples / grid (flag-gated)
# ---------------------------------------------------------------------------

def test_flags_gate_stale_manifests(tmp_path):
    _write_manifests(tmp_path)
    scene = vz.build_scene(
        _store(layers=LAYERS_SECTION), NO_FLAGS, str(tmp_path)
    )
    assert scene["samples"] is None
    assert scene["grid"] is None
    assert scene["phase"] == "collect"

    flags = dict(NO_FLAGS, sampled=True)  # grid still False => stale file hidden
    scene = vz.build_scene(_store(layers=LAYERS_SECTION), flags, str(tmp_path))
    assert scene["samples"] is not None
    assert scene["grid"] is None
    assert scene["phase"] == "sampled"


def test_samples_joined_with_derived_eps_and_grid(tmp_path):
    _write_manifests(tmp_path)
    flags = dict(NO_FLAGS, sampled=True, derived=True, grid=True)
    scene = vz.build_scene(
        _store(layers=LAYERS_SECTION, target_ranges=TARGET_SECTION),
        flags, str(tmp_path),
    )
    assert scene["phase"] == "grid"
    samples = scene["samples"]
    assert samples["total"] == 2 and samples["included"] == 2
    assert samples["truncated"] is False
    item = samples["items"][0]
    assert item["sample_id"] == 1
    layer = item["layers"][0]
    # per-sample eps comes from derived_layers.json, not the range preview
    assert layer["eps_dry"] == 5.0 and layer["eps_wet"] == 12.0
    assert layer["eps_mid"] == 8.5
    t_cyl, t_box = item["targets"]
    assert t_cyl["kind"] == "cylinder"
    assert t_cyl["x_offset_m"] == -0.1
    assert t_cyl["depth_m"] == 0.3
    assert t_cyl["radius_m"] == 0.05
    assert t_box["kind"] == "box"
    assert t_box["width_m"] == 0.2 and t_box["height_m"] == 0.06

    grid = scene["grid"]
    assert grid["domain_x_m"] == 1.3
    assert grid["time_window_ns"] == 40.0
    # domain snaps to the real grid once global_derive ran
    assert scene["domain"] == {
        "width_m": 1.3, "depth_m": 1.0, "dx_m": 0.004, "provisional": False,
    }
    assert scene["acquisition"]["time_window_ns"] == 40.0


def test_sample_cap(tmp_path):
    _write_manifests(tmp_path, num_samples=vz.SAMPLE_CAP + 5)
    flags = dict(NO_FLAGS, sampled=True)
    scene = vz.build_scene(_store(layers=LAYERS_SECTION), flags, str(tmp_path))
    samples = scene["samples"]
    assert samples["total"] == vz.SAMPLE_CAP + 5
    assert samples["included"] == vz.SAMPLE_CAP
    assert samples["truncated"] is True
    assert len(samples["items"]) == vz.SAMPLE_CAP


def test_missing_manifests_do_not_break(tmp_path):
    flags = dict(NO_FLAGS, sampled=True, grid=True)
    scene = vz.build_scene(_store(layers=LAYERS_SECTION), flags, str(tmp_path))
    assert scene is not None
    assert scene["samples"] is None
    assert scene["grid"] is None
