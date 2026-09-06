"""Scene projection for the live frontend visualization.

Pure deterministic helpers that turn (a) the single-agent section store and
(b) the on-disk pipeline manifests into ONE normalized `scene` dict the
frontend canvas renders from. No LLM involvement anywhere: the preview
permittivity reuses gprMax's own Peplinski routine via
`dataset_sampling.peplinski_derive.derive_layer_eps` (in-band
`calculate_er(f).real`, never the raw infinite-frequency `er`).

The scene serves two views at once (they coexist because layer sampling runs
mid-collection):
  - "ranges": midpoint layers + thickness min/max spread, from the raw store;
  - "samples"/"grid": concrete realizations + the global grid, read from the
    manifests but ONLY when the caller's flags say the producing node ran in
    THIS session (stale files from a previous run are never shown).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

_backend_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
for _p in (_backend_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend.dataset_sampling.peplinski_derive import derive_layer_eps  # noqa: E402
from backend.validation_tools_new import peak_frequency  # noqa: E402

# Mid Peplinski calibration band (0.3-1.3 GHz): used for the preview eps until
# the waveform section fixes the real operating frequency.
PREVIEW_FREQ_HZ = 0.9e9
DEFAULT_NBINS = 50
# Max concrete realizations shipped to the frontend per update (payload cap).
SAMPLE_CAP = 200

# phase = the furthest pipeline product visible in the scene (highest wins).
_PHASE_ORDER = ("emitted", "placed", "grid", "derived", "sampled")

_eps_cache: dict[tuple, tuple[float, float]] = {}


def _mid(lo: Any, hi: Any) -> Optional[float]:
    if lo is None or hi is None:
        return None
    return (float(lo) + float(hi)) / 2.0


def _round(v: Optional[float], nd: int = 6) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def _preview_eps(layer: Mapping[str, Any], freq_hz: float, nbins: int) -> tuple:
    """(eps_dry, eps_wet, eps_mid) at midpoint composition over the full theta_v
    band; display-only, so failures degrade to Nones instead of raising."""
    key = (
        round(_mid(layer["sand_pct_min"], layer["sand_pct_max"]), 3),
        round(_mid(layer["clay_pct_min"], layer["clay_pct_max"]), 3),
        round(_mid(layer["bulk_density_gcm3_min"], layer["bulk_density_gcm3_max"]), 4),
        round(_mid(layer["particle_density_gcm3_min"], layer["particle_density_gcm3_max"]), 4),
        round(float(layer["theta_v_min"]), 4),
        round(float(layer["theta_v_max"]), 4),
        nbins,
        round(freq_hz, 0),
    )
    if key not in _eps_cache:
        try:
            _eps_cache[key] = derive_layer_eps(
                layer.get("name") or "soil",
                key[0], key[1], key[2], key[3], key[4], key[5],
                nbins, freq_hz,
            )
        except Exception:
            _eps_cache[key] = (None, None)
    eps_dry, eps_wet = _eps_cache[key]
    eps_mid = None if eps_dry is None else (eps_dry + eps_wet) / 2.0
    return eps_dry, eps_wet, eps_mid


def _preview_frequency(store: Mapping[str, Any]) -> tuple[float, bool]:
    """Operating frequency for the preview eps: the real peak frequency once
    the waveform is saved, else the mid-band fallback (provisional)."""
    wf = store.get("waveform") or {}
    cfg = store.get("dataset_config") or {}
    center = wf.get("waveform_center_freq_hz")
    if not center or center <= 0:
        return PREVIEW_FREQ_HZ, True
    return peak_frequency(float(center), bool(cfg.get("center_freq_is_peak", True))), False


def _project_range_layers(store: Mapping[str, Any]) -> Optional[list[dict]]:
    section = store.get("layers") or {}
    raw = section.get("layers") or []
    if not raw:
        return None
    freq_hz, provisional = _preview_frequency(store)
    nbins = int((store.get("dataset_config") or {}).get("fractal_nbins") or DEFAULT_NBINS)

    out = []
    for i, layer in enumerate(raw):
        sand = _mid(layer["sand_pct_min"], layer["sand_pct_max"])
        clay = _mid(layer["clay_pct_min"], layer["clay_pct_max"])
        eps_dry, eps_wet, eps_mid = _preview_eps(layer, freq_hz, nbins)
        out.append({
            "name": layer.get("name") or f"layer_{i + 1}",
            "thickness_mid_m": _round(_mid(layer["thickness_m_min"], layer["thickness_m_max"])),
            "thickness_min_m": _round(layer["thickness_m_min"]),
            "thickness_max_m": _round(layer["thickness_m_max"]),
            "sand_pct_mid": _round(sand, 2),
            "clay_pct_mid": _round(clay, 2),
            "silt_pct_mid": _round(100.0 - sand - clay, 2),
            "theta_v_mid": _round(_mid(layer["theta_v_min"], layer["theta_v_max"]), 4),
            "eps_mid": _round(eps_mid, 3),
            "eps_dry": _round(eps_dry, 3),
            "eps_wet": _round(eps_wet, 3),
            "eps_freq_hz": freq_hz,
            "eps_provisional": provisional,
        })
    return out


def _is_static_range(obj: dict, pairs: list) -> bool:
    return all(obj.get(lo) == obj.get(hi) for lo, hi in pairs)


def _project_range_targets(store: Mapping[str, Any]) -> list:
    """All buried-object ranges (canonical order: cylinders, then boxes) as
    midpoint entries. x offsets are SIGNED distances from the domain center."""
    tr = store.get("target_ranges") or {}
    out = []
    for cyl in tr.get("cylinders") or []:
        out.append({
            "kind": "cylinder",
            "name": cyl.get("name") or "target",
            "material": cyl.get("material") or "pec",
            "static": _is_static_range(cyl, [
                ("x_offset_min_m", "x_offset_max_m"),
                ("depth_min_m", "depth_max_m"),
                ("radius_min_m", "radius_max_m"),
            ]),
            "x_offset_mid_m": _round(_mid(cyl["x_offset_min_m"], cyl["x_offset_max_m"])),
            "x_offset_min_m": _round(cyl["x_offset_min_m"]),
            "x_offset_max_m": _round(cyl["x_offset_max_m"]),
            "depth_mid_m": _round(_mid(cyl["depth_min_m"], cyl["depth_max_m"])),
            "depth_min_m": _round(cyl["depth_min_m"]),
            "depth_max_m": _round(cyl["depth_max_m"]),
            "radius_mid_m": _round(_mid(cyl["radius_min_m"], cyl["radius_max_m"])),
            "radius_min_m": _round(cyl["radius_min_m"]),
            "radius_max_m": _round(cyl["radius_max_m"]),
        })
    for box in tr.get("boxes") or []:
        out.append({
            "kind": "box",
            "name": box.get("name") or "target",
            "material": box.get("material") or "pec",
            "static": _is_static_range(box, [
                ("x_offset_min_m", "x_offset_max_m"),
                ("depth_min_m", "depth_max_m"),
                ("width_min_m", "width_max_m"),
                ("height_min_m", "height_max_m"),
            ]),
            "x_offset_mid_m": _round(_mid(box["x_offset_min_m"], box["x_offset_max_m"])),
            "x_offset_min_m": _round(box["x_offset_min_m"]),
            "x_offset_max_m": _round(box["x_offset_max_m"]),
            "depth_mid_m": _round(_mid(box["depth_min_m"], box["depth_max_m"])),
            "depth_min_m": _round(box["depth_min_m"]),
            "depth_max_m": _round(box["depth_max_m"]),
            "width_mid_m": _round(_mid(box["width_min_m"], box["width_max_m"])),
            "width_min_m": _round(box["width_min_m"]),
            "width_max_m": _round(box["width_max_m"]),
            "height_mid_m": _round(_mid(box["height_min_m"], box["height_max_m"])),
            "height_min_m": _round(box["height_min_m"]),
            "height_max_m": _round(box["height_max_m"]),
        })
    return out


def _target_half_extents_worst(t: dict) -> tuple:
    """Worst-case (hx, hy) half extents of a range entry, for canvas sizing."""
    if t["kind"] == "cylinder":
        r = t.get("radius_max_m") or 0
        return r, r
    return (t.get("width_max_m") or 0) / 2.0, (t.get("height_max_m") or 0) / 2.0


def _provisional_domain(ranges: Optional[dict]) -> dict:
    """Rough canvas extents before global_derive fixes the real grid.

    Targets use center-relative x offsets, so the width floor is symmetric:
    2 * (worst |x_offset| + half extent + margin)."""
    depth = 0.0
    width = 1.0
    if ranges:
        for layer in ranges.get("layers") or []:
            depth += layer["thickness_mid_m"] or 0.0
        depth *= 1.2
        for target in ranges.get("targets") or []:
            hx, hy = _target_half_extents_worst(target)
            off_worst = max(
                abs(target.get("x_offset_min_m") or 0),
                abs(target.get("x_offset_max_m") or 0),
            )
            depth = max(depth, (target.get("depth_max_m") or 0) + hy + 0.1)
            width = max(width, 2.0 * (off_worst + hx + 0.2))
    return {
        "width_m": _round(max(width, 0.5)),
        "depth_m": _round(max(depth, 0.5)),
        "dx_m": None,
        "provisional": True,
    }


def _load_json(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _project_samples(
    out_dir: Path,
    flags: Mapping[str, bool],
    range_layers: Optional[list[dict]],
) -> Optional[dict]:
    manifest = _load_json(out_dir / "sampled_layers.json")
    if not manifest or not manifest.get("samples"):
        return None

    # Per-sample in-band eps once peplinski_derive ran; joined by sample_id.
    derived_by_id: dict[int, list[dict]] = {}
    if flags.get("derived"):
        derived = _load_json(out_dir / "derived_layers.json") or {}
        for entry in derived.get("samples") or []:
            derived_by_id[int(entry["sample_id"])] = entry.get("layers") or []

    all_samples = manifest["samples"]
    items = []
    for sample in all_samples[:SAMPLE_CAP]:
        sid = int(sample["sample_id"])
        dlayers = derived_by_id.get(sid, [])
        layers = []
        for i, layer in enumerate(sample.get("layers") or []):
            eps_dry = eps_wet = None
            if i < len(dlayers):
                eps_dry = dlayers[i].get("eps_r_dry")
                eps_wet = dlayers[i].get("eps_r_wet")
            elif range_layers and i < len(range_layers):
                eps_dry = range_layers[i]["eps_dry"]
                eps_wet = range_layers[i]["eps_wet"]
            eps_mid = None if eps_dry is None or eps_wet is None else (eps_dry + eps_wet) / 2.0
            layers.append({
                "name": layer.get("name") or f"layer_{i + 1}",
                "thickness_m": _round(layer["thickness_m"]),
                "sand_pct": _round(layer.get("sand_pct"), 2),
                "clay_pct": _round(layer.get("clay_pct"), 2),
                "silt_pct": _round(layer.get("silt_pct"), 2),
                "theta_v_mid": _round(_mid(layer.get("theta_v_min"), layer.get("theta_v_max")), 4),
                "eps_mid": _round(eps_mid, 3),
                "eps_dry": _round(eps_dry, 3),
                "eps_wet": _round(eps_wet, 3),
            })
        targets = []
        for target in sample.get("targets") or []:
            entry = {
                "kind": target.get("kind") or "cylinder",
                "name": target.get("name") or "target",
                "material": target.get("material") or "pec",
                "x_offset_m": _round(target["x_offset_m"]),
                "depth_m": _round(target["depth_m"]),
            }
            if entry["kind"] == "box":
                entry["width_m"] = _round(target["width_m"])
                entry["height_m"] = _round(target["height_m"])
            else:
                entry["radius_m"] = _round(target["radius_m"])
            targets.append(entry)
        items.append({
            "sample_id": sid,
            "layers": layers,
            "targets": targets,
        })
    return {
        "total": len(all_samples),
        "included": len(items),
        "truncated": len(all_samples) > len(items),
        "items": items,
    }


def _project_grid(out_dir: Path) -> Optional[dict]:
    g = _load_json(out_dir / "global_derive.json")
    if not g:
        return None
    return {
        **{k: g.get(k) for k in ("domain_z_m", "soil_depth_m", "dimensionality", "contract_version", "nx", "ny", "nz", "tx_y_m", "rx_y_m", "tx_z_m", "rx_z_m", "dt_s", "iterations")},
        "domain_x_m": g.get("domain_x_m"),
        "domain_y_m": g.get("domain_y_m"),
        "depth_z_m": g.get("depth_z_m"),
        "ground_y_m": g.get("ground_y_m"),
        "dx_m": g.get("dx_m"),
        "source_height_m": g.get("source_height_m"),
        "tx_x_m": g.get("tx_x_m"),
        "rx_x_m": g.get("rx_x_m"),
        "time_window_ns": _round((g.get("time_window_s") or 0) * 1e9, 3) or None,
        "f_peak_hz": g.get("f_peak_hz"),
    }


def build_scene(
    store: Mapping[str, Optional[dict]],
    flags: Mapping[str, bool],
    output_dir: Optional[str],
    stage: Optional[str] = None,
) -> Optional[dict]:
    """Project the section store (+ manifests gated by `flags`) into the
    frontend scene payload. Returns None while nothing is worth drawing."""
    if not any(store.get(s) for s in ("dataset_config", "layers", "target_ranges",
                                      "waveform", "antenna")):
        return None

    range_layers = _project_range_layers(store)
    range_targets = _project_range_targets(store)
    ranges = None
    if range_layers or range_targets:
        ranges = {"layers": range_layers or [], "targets": range_targets}

    samples = None
    grid = None
    if output_dir:
        out_dir = Path(output_dir)
        if flags.get("sampled"):
            samples = _project_samples(out_dir, flags, range_layers)
        if flags.get("grid"):
            grid = _project_grid(out_dir)
        if flags.get("emitted") and samples:
            emitted = _load_json(out_dir / "emitted_files.json") or {}
            resolved = {int(f["sample_id"]): f.get("resolved_scene") for f in emitted.get("files", [])}
            samples["items"] = [s for s in samples["items"] if s["sample_id"] in resolved]
            samples["total"] = len(resolved)
            samples["included"] = len(samples["items"])
            for sample in samples["items"]:
                actual = resolved.get(sample["sample_id"])
                if actual:
                    sample["resolved_scene"] = {key: actual[key] for key in ("sample_id", "title", "targets", "source", "receiver", "status")}
                    sample["resolved_scene"]["layers"] = [{key: layer[key] for key in ("name", "start_m", "end_m", "thickness_m", "terminal_halfspace", "sampled_thickness_m")} for layer in actual["layers"]]
                    for layer, resolved_layer in zip(sample["layers"], actual["layers"]):
                        layer["thickness_m"] = resolved_layer["thickness_m"]
                        layer["sampled_thickness_m"] = resolved_layer["sampled_thickness_m"]
                    for target, r in zip(sample["targets"], actual["targets"]):
                        target["x_offset_m"] = (r["start_m"][0] + r["end_m"][0]) / 2 - grid["domain_x_m"] / 2
                        target["depth_m"] = grid["ground_y_m"] - (r["start_m"][1] + r["end_m"][1]) / 2

    if grid:
        domain = {
            "width_m": grid["domain_x_m"],
            "depth_m": grid["depth_z_m"],
            "dx_m": grid["dx_m"],
            "provisional": False,
        }
    else:
        domain = _provisional_domain(ranges)

    wf = store.get("waveform") or {}
    ant = store.get("antenna") or {}
    center = wf.get("waveform_center_freq_hz")
    acquisition = {
        "frequency_ghz": _round(center / 1e9, 4) if center else None,
        "waveform": wf.get("waveform_kind"),
        "antenna_kind": ant.get("antenna_kind"),
        "txrx_sep_m": ant.get("tx_rx_offset_m"),
        "time_window_ns": grid["time_window_ns"] if grid else None,
    }

    phase = "collect"
    for candidate in _PHASE_ORDER:
        if flags.get(candidate):
            phase = candidate
            break

    return {
        "dimensionality": (store.get("dataset_config") or {}).get("dimensionality", "2D"),
        "coordinate_frame": "x-horizontal_y-up_z-crossline",
        "pml_cells": (store.get("dataset_config") or {}).get("pml_cells", 10),
        "phase": phase,
        "stage": stage,
        "project": (store.get("dataset_config") or {}).get("model_basename") or "untitled",
        "domain": domain,
        "acquisition": acquisition,
        "ranges": ranges,
        "samples": samples,
        "grid": grid,
    }
