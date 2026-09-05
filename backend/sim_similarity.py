"""Session-config similarity index for forward-model reuse.

When a user is about to run the gprMax forward model, we look for a PAST
session whose parameter envelope is nearly identical and whose signals are
already simulated, and offer to adopt that dataset instead of re-simulating.

Engineering constraints (deliberate — do not "simplify" back to embeddings):
- Session configs serialize to near-identical text differing only in numbers,
  so text embeddings cannot discriminate them. There is NO embedding model
  here: vectors are raw numeric features, each dimension normalized by a
  FIXED physical-plausibility constant (never data-derived, so vectors stay
  comparable as the index grows).
- Qdrant is used only for ANN candidate retrieval, with payload hard filters
  for categorically incompatible configs (layer count, waveform/antenna kind,
  target counts, grid policy). The similarity that gates the recommendation
  is an EXACT rescoring in Python: interval IoU over ranges + scalar rules,
  weighted by physical importance.
- Everything is deterministic — no LLM involvement (CLAUDE.md hard boundary).
- Callers must treat every public entry point as best-effort: the module-level
  conveniences swallow ALL failures (Qdrant down, malformed sections) so the
  recommendation can never break the simulate path.

CLI: `python backend/sim_similarity.py backfill` indexes every already
completed session from Postgres (idempotent — point id = session UUID).
"""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Same import bootstrap as api.py: works under `uvicorn backend.api:app`,
# direct execution, and pytest from the repo root.
_backend_dir = str(Path(__file__).resolve().parent)
_project_root = str(Path(__file__).resolve().parent.parent)
for _p in (_backend_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from schema import (  # noqa: E402
    DatasetConfig,
    ExtractedAdvancedParams,
    ExtractedAntenna,
    ExtractedLayers,
    ExtractedTargetRanges,
    ExtractedWaveform,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

COLLECTION_NAME = "sim_sessions"

# Slot capacities. Configs exceeding these cannot be vectorized (ValueError);
# raising a capacity later only requires a collection rebuild (backfill).
MAX_LAYERS = 6
MAX_CYLINDERS = 4
MAX_BOXES = 4

LAYER_DIMS = 12      # 6 range fields x (min, max)
WA_DIMS = 6          # waveform + antenna scalars
CYL_DIMS = 6         # 3 range fields x (min, max)
BOX_DIMS = 8         # 4 range fields x (min, max)
VECTOR_DIM = MAX_LAYERS * LAYER_DIMS + WA_DIMS + MAX_CYLINDERS * CYL_DIMS + MAX_BOXES * BOX_DIMS

REUSE_THRESHOLD = float(os.getenv("SIM_REUSE_THRESHOLD", "0.95"))
REUSE_TOP_K = int(os.getenv("SIM_REUSE_TOPK", "10"))
REUSE_ENABLED = os.getenv("SIM_REUSE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}

# Fixed normalization scales (raw unit per 1.0 of similarity loss for scalars,
# and per 1.0 of vector space for dimensions). ONE table — the single tuning
# point shared by the vector builder and the rescorer so retrieval distance
# and exact score stay aligned.
SCALES = {
    "thickness_m": 3.0,            # plausible 0-3 m per layer
    "sand_pct": 100.0,
    "clay_pct": 100.0,
    "theta_v": 0.5,
    "bulk_density_gcm3": 1.0,      # offset 1.0 (1-2 g/cm3)
    "particle_density_gcm3": 0.6,  # offset 2.3 (2.3-2.9 g/cm3)
    "log10_freq": 3.0,             # vector: (log10 f - 7) / 3  (10 MHz - 10 GHz)
    "log10_amplitude": 2.0,
    "tx_rx_offset_m": 1.0,
    "source_height_m": 1.0,
    "resistance_ohm": 1000.0,
    "source_time_s": 5e-8,         # source on/off gating times
    "x_offset_m": 2.0,             # signed, vector-mapped from -1..1 m
    "depth_m": 2.0,
    "target_size_m": 0.5,          # radius / width / height
}
# Rescoring rule for center frequency: 1 - |dlog10 f| / 0.30 (~3.5 % freq
# offset costs 5 points) — much stricter than the vector scale on purpose;
# frequency drives the whole grid.
LOG_FREQ_RESCORE_SCALE = 0.30

# Rescoring weights. Groups sum to 1; fields sum to 1 within each group.
GROUP_WEIGHTS = {"layers": 0.50, "waveform": 0.20, "antenna": 0.15, "targets": 0.15}
LAYER_FIELD_WEIGHTS = {
    "thickness_m": 0.25,
    "theta_v": 0.25,
    "sand_pct": 0.15,
    "clay_pct": 0.15,
    "bulk_density_gcm3": 0.10,
    "particle_density_gcm3": 0.10,
}
WAVEFORM_FIELD_WEIGHTS = {"center_freq": 0.80, "amplitude": 0.10, "timing": 0.10}
ANTENNA_FIELD_WEIGHTS = {"tx_rx_offset": 0.50, "source_height": 0.30, "resistance": 0.20}

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Canonical feature payload
# ---------------------------------------------------------------------------

def _pair(lo: Any, hi: Any) -> list[float]:
    return [float(lo), float(hi)]


def _target_sort_key(t: dict) -> tuple:
    """Deterministic order for target slots so equivalent configs vectorize
    identically regardless of the order the user listed the objects in."""
    size = t.get("radius_m") or t.get("width_m") or [0.0, 0.0]
    return (
        (t["depth_m"][0] + t["depth_m"][1]) / 2.0,
        (t["x_offset_m"][0] + t["x_offset_m"][1]) / 2.0,
        (size[0] + size[1]) / 2.0,
    )


def build_feature_payload(
    *,
    dataset_config: dict,
    layers: dict,
    target_ranges: Optional[dict],
    waveform: dict,
    antenna: dict,
    advanced_params: Optional[dict] = None,
) -> dict:
    """Canonical JSON-safe numeric dict describing one session's envelope.

    The SAME function feeds indexing, querying, and the backfill CLI — any
    asymmetry between the three would silently break matching. Validates the
    raw section dicts through the schema models so defaults resolve uniformly.
    Raises ValueError when essentials are missing or slot capacities are
    exceeded (callers treat that as "not vectorizable", not an error).
    """
    cfg = DatasetConfig.model_validate(dataset_config)
    lay = ExtractedLayers.model_validate(layers)
    wf = ExtractedWaveform.model_validate(waveform)
    ant = ExtractedAntenna.model_validate(antenna)
    tr = ExtractedTargetRanges.model_validate(target_ranges or {})
    adv = ExtractedAdvancedParams.model_validate(advanced_params or {})

    if lay.num_layers > MAX_LAYERS:
        raise ValueError(f"num_layers {lay.num_layers} exceeds MAX_LAYERS {MAX_LAYERS}")
    if len(tr.cylinders) > MAX_CYLINDERS:
        raise ValueError(f"{len(tr.cylinders)} cylinders exceed MAX_CYLINDERS {MAX_CYLINDERS}")
    if len(tr.boxes) > MAX_BOXES:
        raise ValueError(f"{len(tr.boxes)} boxes exceed MAX_BOXES {MAX_BOXES}")

    layer_entries = [
        {
            "name": l.name,
            "thickness_m": _pair(l.thickness_m_min, l.thickness_m_max),
            "sand_pct": _pair(l.sand_pct_min, l.sand_pct_max),
            "clay_pct": _pair(l.clay_pct_min, l.clay_pct_max),
            "theta_v": _pair(l.theta_v_min, l.theta_v_max),
            "bulk_density_gcm3": _pair(l.bulk_density_gcm3_min, l.bulk_density_gcm3_max),
            "particle_density_gcm3": _pair(
                l.particle_density_gcm3_min, l.particle_density_gcm3_max
            ),
        }
        for l in lay.layers
    ]
    cylinders = sorted(
        (
            {
                "x_offset_m": _pair(c.x_offset_min_m, c.x_offset_max_m),
                "depth_m": _pair(c.depth_min_m, c.depth_max_m),
                "radius_m": _pair(c.radius_min_m, c.radius_max_m),
            }
            for c in tr.cylinders
        ),
        key=_target_sort_key,
    )
    boxes = sorted(
        (
            {
                "x_offset_m": _pair(b.x_offset_min_m, b.x_offset_max_m),
                "depth_m": _pair(b.depth_min_m, b.depth_max_m),
                "width_m": _pair(b.width_min_m, b.width_max_m),
                "height_m": _pair(b.height_min_m, b.height_max_m),
            }
            for b in tr.boxes
        ),
        key=_target_sort_key,
    )

    return {
        "version": 1,
        "num_layers": lay.num_layers,
        "layers": layer_entries,
        "waveform": {
            "kind": wf.waveform_kind or "ricker",
            "center_freq_hz": float(wf.waveform_center_freq_hz),
            "amplitude": float(wf.waveform_amplitude),
            "source_start_time": wf.source_start_time,
            "source_end_time": wf.source_end_time,
        },
        "antenna": {
            "kind": ant.antenna_kind,
            "axis": ant.antenna_axis or "x",
            "tx_rx_offset_m": float(ant.tx_rx_offset_m),
            "source_height_m": ant.source_height_m,
            "resistance": ant.resistance,
        },
        "grid_policy": {
            "dimensionality": cfg.dimensionality,
            "cells_per_wavelength": cfg.cells_per_wavelength,
            "pml_cells": cfg.pml_cells,
            "buffer_cells": cfg.buffer_cells,
            "fractal_nbins": cfg.fractal_nbins,
            "high_freq_factor": cfg.high_freq_factor,
        },
        "cylinders": cylinders,
        "boxes": boxes,
        "has_surface_roughness": adv.surface_roughness is not None,
    }


# ---------------------------------------------------------------------------
# Vector + hard filters
# ---------------------------------------------------------------------------

def build_vector(payload: dict) -> list[float]:
    """Fixed VECTOR_DIM float vector, zero-padded per slot group.

    Padding never distorts the distance between comparable configs because
    every count that controls padding (num_layers, n_cylinders, n_boxes) is
    a hard payload filter — both sides of any comparison pad identically.
    """
    vec: list[float] = []

    for i in range(MAX_LAYERS):
        if i < len(payload["layers"]):
            l = payload["layers"][i]
            vec += [x / SCALES["thickness_m"] for x in l["thickness_m"]]
            vec += [x / SCALES["sand_pct"] for x in l["sand_pct"]]
            vec += [x / SCALES["clay_pct"] for x in l["clay_pct"]]
            vec += [x / SCALES["theta_v"] for x in l["theta_v"]]
            vec += [(x - 1.0) / SCALES["bulk_density_gcm3"] for x in l["bulk_density_gcm3"]]
            vec += [(x - 2.3) / SCALES["particle_density_gcm3"]
                    for x in l["particle_density_gcm3"]]
        else:
            vec += [0.0] * LAYER_DIMS

    wf, ant = payload["waveform"], payload["antenna"]
    vec.append((math.log10(max(wf["center_freq_hz"], 1.0)) - 7.0) / SCALES["log10_freq"])
    amp = min(max(abs(wf["amplitude"]), 0.01), 100.0)
    vec.append(math.log10(amp) / SCALES["log10_amplitude"])
    vec.append(ant["tx_rx_offset_m"] / SCALES["tx_rx_offset_m"])
    h = ant["source_height_m"]
    vec.append(0.0 if h is None else h / SCALES["source_height_m"])
    vec.append(0.0 if h is None else 1.0)  # specified-flag: both-None pairs stay equal
    r = ant["resistance"]
    vec.append(0.0 if r is None else r / SCALES["resistance_ohm"])

    for i in range(MAX_CYLINDERS):
        if i < len(payload["cylinders"]):
            c = payload["cylinders"][i]
            vec += [(x + 1.0) / SCALES["x_offset_m"] for x in c["x_offset_m"]]
            vec += [x / SCALES["depth_m"] for x in c["depth_m"]]
            vec += [x / SCALES["target_size_m"] for x in c["radius_m"]]
        else:
            vec += [0.0] * CYL_DIMS

    for i in range(MAX_BOXES):
        if i < len(payload["boxes"]):
            b = payload["boxes"][i]
            vec += [(x + 1.0) / SCALES["x_offset_m"] for x in b["x_offset_m"]]
            vec += [x / SCALES["depth_m"] for x in b["depth_m"]]
            vec += [x / SCALES["target_size_m"] for x in b["width_m"]]
            vec += [x / SCALES["target_size_m"] for x in b["height_m"]]
        else:
            vec += [0.0] * BOX_DIMS

    assert len(vec) == VECTOR_DIM
    return vec


def hard_filters(payload: dict) -> dict:
    """Exact-match payload conditions. A mismatch on ANY of these makes two
    sessions categorically incomparable (different physics or different
    emitted decks), so they must never reach the numeric rescoring."""
    gp = payload["grid_policy"]
    return {
        "num_layers": payload["num_layers"],
        "waveform_kind": payload["waveform"]["kind"],
        "antenna_kind": payload["antenna"]["kind"],
        "antenna_axis": payload["antenna"]["axis"],
        "dimensionality": gp["dimensionality"],
        "n_cylinders": len(payload["cylinders"]),
        "n_boxes": len(payload["boxes"]),
        "has_surface_roughness": payload["has_surface_roughness"],
        "cells_per_wavelength": gp["cells_per_wavelength"],
        "pml_cells": gp["pml_cells"],
        "buffer_cells": gp["buffer_cells"],
        "fractal_nbins": gp["fractal_nbins"],
        # float -> int encoding for exact matching
        "high_freq_factor_x100": int(round(gp["high_freq_factor"] * 100)),
    }


# ---------------------------------------------------------------------------
# Exact rescoring
# ---------------------------------------------------------------------------

def _scalar_sim(a: float, b: float, scale: float) -> float:
    return max(0.0, 1.0 - abs(a - b) / scale)


def _range_sim(a: list[float], b: list[float], scale: float) -> float:
    """Interval similarity: IoU for genuine ranges, scalar rule when both
    sides are fixed values. A degenerate value vs a range scores ~0 by IoU —
    deliberate: sampling a range and fixing a value are different experiment
    designs even when the value lies inside the range."""
    lo_a, hi_a = a
    lo_b, hi_b = b
    union = max(hi_a, hi_b) - min(lo_a, lo_b)
    if union < _EPS:
        return 1.0  # all four bounds coincide
    if (hi_a - lo_a) < _EPS and (hi_b - lo_b) < _EPS:
        return _scalar_sim((lo_a + hi_a) / 2.0, (lo_b + hi_b) / 2.0, scale)
    inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    return inter / union


def _optional_sim(a: Optional[float], b: Optional[float], scale: float) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    return _scalar_sim(float(a), float(b), scale)


def rescore(a: dict, b: dict) -> tuple[float, list[dict]]:
    """Exact similarity in [0, 1] between two feature payloads, plus the
    per-parameter breakdown [{param, current, candidate, sim}, ...].

    Assumes hard filters already matched (same counts/kinds); stays finite
    and defined even when they didn't (unpaired slots score 0)."""
    breakdown: list[dict] = []

    def leaf(param: str, cur: Any, cand: Any, sim: float) -> float:
        breakdown.append({"param": param, "current": cur, "candidate": cand,
                          "sim": round(sim, 4)})
        return sim

    # -- layers ------------------------------------------------------------
    n = max(len(a["layers"]), len(b["layers"]))
    layer_scores: list[float] = []
    for i in range(n):
        if i >= len(a["layers"]) or i >= len(b["layers"]):
            layer_scores.append(0.0)
            breakdown.append({"param": f"layer{i + 1}", "current": None,
                              "candidate": None, "sim": 0.0})
            continue
        la, lb = a["layers"][i], b["layers"][i]
        s = 0.0
        for field, w in LAYER_FIELD_WEIGHTS.items():
            s += w * leaf(f"layer{i + 1}.{field}", la[field], lb[field],
                          _range_sim(la[field], lb[field], SCALES[field]))
        layer_scores.append(s)
    layers_score = sum(layer_scores) / n if n else 1.0

    # -- waveform ------------------------------------------------------------
    wa, wb = a["waveform"], b["waveform"]
    freq_sim = leaf(
        "waveform.center_freq_hz", wa["center_freq_hz"], wb["center_freq_hz"],
        max(0.0, 1.0 - abs(math.log10(max(wa["center_freq_hz"], 1.0))
                           - math.log10(max(wb["center_freq_hz"], 1.0)))
            / LOG_FREQ_RESCORE_SCALE),
    )
    amp_sim = leaf(
        "waveform.amplitude", wa["amplitude"], wb["amplitude"],
        _scalar_sim(
            math.log10(min(max(abs(wa["amplitude"]), 0.01), 100.0)),
            math.log10(min(max(abs(wb["amplitude"]), 0.01), 100.0)),
            SCALES["log10_amplitude"],
        ),
    )
    timing_sim = leaf(
        "waveform.source_timing",
        [wa["source_start_time"], wa["source_end_time"]],
        [wb["source_start_time"], wb["source_end_time"]],
        (_optional_sim(wa["source_start_time"], wb["source_start_time"], SCALES["source_time_s"])
         + _optional_sim(wa["source_end_time"], wb["source_end_time"], SCALES["source_time_s"]))
        / 2.0,
    )
    waveform_score = (WAVEFORM_FIELD_WEIGHTS["center_freq"] * freq_sim
                      + WAVEFORM_FIELD_WEIGHTS["amplitude"] * amp_sim
                      + WAVEFORM_FIELD_WEIGHTS["timing"] * timing_sim)

    # -- antenna -------------------------------------------------------------
    aa, ab = a["antenna"], b["antenna"]
    off_sim = leaf("antenna.tx_rx_offset_m", aa["tx_rx_offset_m"], ab["tx_rx_offset_m"],
                   _scalar_sim(aa["tx_rx_offset_m"], ab["tx_rx_offset_m"],
                               SCALES["tx_rx_offset_m"]))
    h_sim = leaf("antenna.source_height_m", aa["source_height_m"], ab["source_height_m"],
                 _optional_sim(aa["source_height_m"], ab["source_height_m"],
                               SCALES["source_height_m"]))
    r_sim = leaf("antenna.resistance", aa["resistance"], ab["resistance"],
                 _optional_sim(aa["resistance"], ab["resistance"], SCALES["resistance_ohm"]))
    antenna_score = (ANTENNA_FIELD_WEIGHTS["tx_rx_offset"] * off_sim
                     + ANTENNA_FIELD_WEIGHTS["source_height"] * h_sim
                     + ANTENNA_FIELD_WEIGHTS["resistance"] * r_sim)

    # -- targets -------------------------------------------------------------
    slot_scores: list[float] = []
    for kind, fields in (("cylinders", ("x_offset_m", "depth_m", "radius_m")),
                         ("boxes", ("x_offset_m", "depth_m", "width_m", "height_m"))):
        m = max(len(a[kind]), len(b[kind]))
        for i in range(m):
            if i >= len(a[kind]) or i >= len(b[kind]):
                slot_scores.append(0.0)
                breakdown.append({"param": f"{kind[:-1]}{i + 1}", "current": None,
                                  "candidate": None, "sim": 0.0})
                continue
            ta, tb = a[kind][i], b[kind][i]
            sims = [
                leaf(f"{kind[:-1]}{i + 1}.{f}", ta[f], tb[f],
                     _range_sim(ta[f], tb[f], SCALES[f if f in SCALES else "target_size_m"]))
                for f in fields
            ]
            slot_scores.append(sum(sims) / len(sims))

    groups = {"layers": layers_score, "waveform": waveform_score, "antenna": antenna_score}
    weights = dict(GROUP_WEIGHTS)
    if slot_scores:
        groups["targets"] = sum(slot_scores) / len(slot_scores)
    else:
        # Both target-free: vacuous agreement must not inflate the score —
        # redistribute the target weight over the remaining groups.
        weights.pop("targets")
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

    score = sum(weights[k] * groups[k] for k in weights)
    return min(max(score, 0.0), 1.0), breakdown


# ---------------------------------------------------------------------------
# Qdrant index
# ---------------------------------------------------------------------------

class SimilarityIndex:
    """Thin wrapper over the `sim_sessions` Qdrant collection. Imports
    qdrant_client lazily so key-free tests never need a running Qdrant."""

    def __init__(self, url: Optional[str] = None):
        from qdrant_client import QdrantClient

        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self) -> None:
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not self.client.collection_exists(COLLECTION_NAME):
            # EUCLID, not COSINE: dimensions are absolute normalized values;
            # cosine's scale invariance would equate proportional configs.
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.EUCLID),
            )
        schema_by_key = {
            "session_id": PayloadSchemaType.KEYWORD,
            "waveform_kind": PayloadSchemaType.KEYWORD,
            "antenna_kind": PayloadSchemaType.KEYWORD,
            "antenna_axis": PayloadSchemaType.KEYWORD,
            "dimensionality": PayloadSchemaType.KEYWORD,
            "num_layers": PayloadSchemaType.INTEGER,
            "n_cylinders": PayloadSchemaType.INTEGER,
            "n_boxes": PayloadSchemaType.INTEGER,
            "cells_per_wavelength": PayloadSchemaType.INTEGER,
            "pml_cells": PayloadSchemaType.INTEGER,
            "buffer_cells": PayloadSchemaType.INTEGER,
            "fractal_nbins": PayloadSchemaType.INTEGER,
            "high_freq_factor_x100": PayloadSchemaType.INTEGER,
            "has_surface_roughness": PayloadSchemaType.BOOL,
        }
        for key, schema in schema_by_key.items():
            try:
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME, field_name=key, field_schema=schema
                )
            except Exception:
                pass  # already exists

    def index_session(self, session_id: str, payload: dict, meta: dict) -> None:
        """Upsert one session point. Point id = session UUID string, so
        re-indexing after a re-simulation is idempotent."""
        from qdrant_client.models import PointStruct

        self.ensure_collection()
        point_payload = {
            **hard_filters(payload),
            **meta,
            "session_id": str(session_id),
            # Full canonical dict rides along so rescoring at query time
            # needs no Postgres round-trip.
            "feature_payload": payload,
        }
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=str(session_id), vector=build_vector(payload),
                                payload=point_payload)],
        )

    def find_similar(
        self,
        payload: dict,
        *,
        exclude_session_id: str,
        threshold: float = REUSE_THRESHOLD,
        top_k: int = REUSE_TOP_K,
    ) -> list[dict]:
        """Rescored matches >= threshold, best first. ANN distance only
        selects candidates — ordering and the gate come from `rescore`."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v))
                  for k, v in hard_filters(payload).items()],
            must_not=[FieldCondition(key="session_id",
                                     match=MatchValue(value=str(exclude_session_id)))],
        )
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=build_vector(payload),
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        ).points

        matches: list[dict] = []
        for hit in hits:
            hp = hit.payload or {}
            candidate = hp.get("feature_payload")
            if not candidate:
                continue
            score, breakdown = rescore(payload, candidate)
            if score < threshold:
                continue
            diffs = sorted((d for d in breakdown if d["sim"] < 1.0),
                           key=lambda d: d["sim"])
            matches.append({
                "source_session_id": hp.get("session_id"),
                "similarity": round(score, 4),
                "similarity_pct": round(score * 100, 1),
                "num_samples": hp.get("num_samples"),
                "simulated_at": hp.get("created_at"),
                "source_output_dir": hp.get("output_dir"),
                "source_user_id": hp.get("user_id"),
                "params_diff": diffs[:8],
            })
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches


# ---------------------------------------------------------------------------
# Failure-swallowing conveniences used by api.py
# ---------------------------------------------------------------------------

def _payload_from_state(state: dict) -> dict:
    return build_feature_payload(
        dataset_config=state["dataset_config"],
        layers=state["layers"],
        target_ranges=state.get("target_ranges"),
        waveform=state["waveform"],
        antenna=state["antenna"],
        advanced_params=state.get("advanced_params"),
    )


def index_completed_session(
    state: dict,
    *,
    session_id: str,
    user_id: str,
    num_samples: int,
    output_dir: str,
    created_at: Optional[str] = None,
) -> bool:
    """Index one fully simulated session. Never raises."""
    if not REUSE_ENABLED:
        return False
    try:
        payload = _payload_from_state(state)
        SimilarityIndex().index_session(session_id, payload, meta={
            "user_id": user_id,
            "num_samples": num_samples,
            "output_dir": str(output_dir),
            "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        })
        return True
    except Exception:
        logger.warning("similarity indexing failed for session %s", session_id,
                       exc_info=True)
        return False


def find_similar_session(state: dict, *, session_id: str) -> list[dict]:
    """Rescored matches >= threshold for the session's current config, best
    first; [] on ANY failure (Qdrant down, incomplete sections, over-capacity
    config) — the simulate path must proceed normally in every such case."""
    if not REUSE_ENABLED:
        return []
    try:
        payload = _payload_from_state(state)
        return SimilarityIndex().find_similar(payload, exclude_session_id=session_id)
    except Exception:
        logger.warning("similarity search failed for session %s", session_id,
                       exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Backfill CLI
# ---------------------------------------------------------------------------

def backfill() -> None:
    """Index every already-completed session from Postgres. Idempotent."""
    from db.db import count_incomplete_simulations, list_extraction_sessions

    rows = list_extraction_sessions()
    indexed = skipped = 0
    for row in rows:
        sid = str(row.id)
        lr = row.layers_ranges or {}
        aw = row.antenna_waveform or {}
        mc = row.model_config_data or {}
        try:
            total, incomplete = count_incomplete_simulations(row.id)
            if total == 0 or incomplete > 0:
                print(f"skip {sid}: {incomplete}/{total} simulations incomplete")
                skipped += 1
                continue
            payload = build_feature_payload(
                dataset_config=mc.get("dataset_config") or {},
                layers=lr.get("layers") or {},
                target_ranges=lr.get("target_ranges"),
                waveform=aw.get("waveform") or {},
                antenna=aw.get("antenna") or {},
                advanced_params=row.advanced_params,
            )
            SimilarityIndex().index_session(sid, payload, meta={
                "user_id": row.user_id,
                "num_samples": total,
                "output_dir": str((mc.get("artifacts") or {}).get("output_dir") or ""),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            })
            print(f"indexed {sid}: {total} samples ({row.user_id})")
            indexed += 1
        except Exception as exc:
            print(f"skip {sid}: {exc}")
            skipped += 1
    print(f"done — {indexed} indexed, {skipped} skipped (of {len(rows)})")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "backfill":
        print("usage: python backend/sim_similarity.py backfill")
        sys.exit(1)
    backfill()
