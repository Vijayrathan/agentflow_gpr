"""Emit one input per accepted sample under a frozen dataset contract.

Version 2 first calls the deterministic resolved-scene stage, joins native
material provenance, and validates the canonical manifest. serialize_scene is
pure string assembly: it neither derives, snaps nor repairs physical settings.
Coordinates remain x horizontal, y vertical, z crossline. 2D TMz uses one z cell;
3D uses finite volumes/targets and honors source polarization. Terminal layers
continue to y=0. Native model-build preflight verifies the serialized scene
before field updates. The older writer remains a version-1 compatibility path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from backend.schema import (
    DatasetConfig,
    ExtractedWaveform,
    ExtractedAntenna,
    ExtractedAdvancedParams,
    GlobalDerived,
    SampledSample,
    SampledLayer,
    SampledTarget,
    SurfaceRoughnessConfigSchema,
    SnapshotConfigSchema,
)
from backend.validation_tools_new import PML_GAP_CELLS
from backend.dataset_sampling.global_derive import read_global
from backend.dataset_sampling.layer_sampler import read_samples

# gprMax builtins / reserved identifiers a layer name must not collide with.
RESERVED_MATERIAL_NAMES = {"pec", "free_space", "grass", "water"}

# Fractal-box distribution defaults (isotropic). The pipeline does not collect
# per-layer fractal parameters, so use gprMax's documented example defaults.
FRACTAL_DIM = 1.5
FRACTAL_WEIGHT_X = 1.0
FRACTAL_WEIGHT_Y = 1.0
FRACTAL_WEIGHT_Z = 1.0


def _g(v: float) -> str:
    """Format a float for a gprMax command (compact, high precision)."""
    return f"{v:.17g}"


def _sanitize(name: str) -> str:
    """Make an identifier safe for gprMax (no whitespace)."""
    return "_".join(str(name).split())


# ---------------------------------------------------------------------------
# gprMax line helpers (exact syntax per gprMax/docs/source/input.rst)
# ---------------------------------------------------------------------------

def _waveform_line(wf: ExtractedWaveform, grid: GlobalDerived) -> str:
    """#waveform: <kind> <amp> <f_peak> <name>.

    Uses grid.f_peak_hz — the already-converted peak — NOT the raw collected
    centre frequency (which may be a band-centre value).
    """
    kind = (wf.waveform_kind or "ricker").lower()
    return (
        f"#waveform: {kind} {_g(wf.waveform_amplitude)} "
        f"{_g(grid.f_peak_hz)} {_sanitize(wf.waveform_name)}"
    )


def _soil_peplinski_line(layer: SampledLayer, soil_id: str) -> str:
    """#soil_peplinski: sand_frac clay_frac bulk particle theta_v_min theta_v_max id.

    sand/clay are FRACTIONS (pct/100); the moisture band is passed through as-is.
    """
    return (
        f"#soil_peplinski: {_g(layer.sand_pct / 100.0)} {_g(layer.clay_pct / 100.0)} "
        f"{_g(layer.bulk_density_gcm3)} {_g(layer.particle_density_gcm3)} "
        f"{_g(layer.theta_v_min)} {_g(layer.theta_v_max)} {soil_id}"
    )


def _fractal_box_line(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float,
    nbins: int, soil_id: str, box_id: str, seed: Optional[int],
) -> str:
    """#fractal_box: x1 y1 z1 x2 y2 z2 fdim wx wy wz nbins soil_id box_id [seed]."""
    seed_part = f" {seed}" if seed is not None else ""
    return (
        f"#fractal_box: {_g(x1)} {_g(y1)} {_g(z1)} {_g(x2)} {_g(y2)} {_g(z2)} "
        f"{_g(FRACTAL_DIM)} {_g(FRACTAL_WEIGHT_X)} {_g(FRACTAL_WEIGHT_Y)} "
        f"{_g(FRACTAL_WEIGHT_Z)} {nbins} {soil_id} {box_id}{seed_part}"
    )


def _source_line(ant: ExtractedAntenna, grid: GlobalDerived, wf: ExtractedWaveform) -> str:
    """Transmitter line, polarised along the thin invariant axis (z).

    Preserve the selected Hertzian, voltage or transmission-line source;
    reject unknown kinds instead of substituting a different excitation.
    The optional [start end] timing pair is written only when BOTH are set
    (gprMax requires the pair). Transmission lines require CPU solving.
    """
    kind = ant.antenna_kind
    timing = ""
    if wf.source_start_time is not None and wf.source_end_time is not None:
        timing = f" {_g(wf.source_start_time)} {_g(wf.source_end_time)}"
    pos = f"{_g(grid.tx_x_m)} {_g(grid.tx_y_m)} 0"
    name = _sanitize(wf.waveform_name)
    if kind == "hertzian_dipole":
        return f"#hertzian_dipole: z {pos} {name}{timing}"
    if kind in ("voltage_source", "transmission_line"):
        if ant.resistance is None:
            raise ValueError(f"{kind} requires a resistance value")
        return f"#{kind}: z {pos} {_g(ant.resistance)} {name}{timing}"
    raise ValueError(f"Unsupported antenna_kind {kind!r}; no source was substituted")


def _rx_line(grid: GlobalDerived) -> str:
    return f"#rx: {_g(grid.rx_x_m)} {_g(grid.rx_y_m)} 0"


def _target_cylinder_line(target: SampledTarget, grid: GlobalDerived) -> str:
    """Buried cylinder disc (axis along thin z). Dielectric smoothing OFF ('n') so
    a PEC target fully replaces the underlying fractal materials at its boundary."""
    x_abs = grid.domain_x_m / 2.0 + target.x_offset_m
    y_center = grid.ground_y_m - target.depth_m
    return (
        f"#cylinder: {_g(x_abs)} {_g(y_center)} 0 "
        f"{_g(x_abs)} {_g(y_center)} {_g(grid.dx_m)} "
        f"{_g(target.radius_m)} {_sanitize(target.material)} n"
    )


def _target_box_line(target: SampledTarget, grid: GlobalDerived) -> str:
    """Buried rectangular box (thin z: one cell). Smoothing OFF like the cylinder
    so the PEC fully replaces the fractal soil at its boundary."""
    x_abs = grid.domain_x_m / 2.0 + target.x_offset_m
    y_center = grid.ground_y_m - target.depth_m
    hx, hy = target.width_m / 2.0, target.height_m / 2.0
    return (
        f"#box: {_g(x_abs - hx)} {_g(y_center - hy)} 0 "
        f"{_g(x_abs + hx)} {_g(y_center + hy)} {_g(grid.dx_m)} "
        f"{_sanitize(target.material)} n"
    )


def _target_line(target: SampledTarget, grid: GlobalDerived) -> str:
    if target.kind == "cylinder":
        return _target_cylinder_line(target, grid)
    if target.kind == "box":
        return _target_box_line(target, grid)
    raise ValueError(f"Unknown target kind '{target.kind}'")


def _surface_roughness_line(
    rough: SurfaceRoughnessConfigSchema, grid: GlobalDerived, box_id: str, ground_y: float,
) -> str:
    """#add_surface_roughness on the top (y = ground_y) face of the surface box.

    f10 f11 are the absolute y-limits over which the surface height varies.
    """
    y_lo = ground_y - rough.amplitude_m
    y_hi = ground_y + rough.amplitude_m
    seed_part = f" {rough.seed}" if rough.seed is not None else ""
    return (
        f"#add_surface_roughness: 0 {_g(ground_y)} 0 "
        f"{_g(grid.domain_x_m)} {_g(ground_y)} {_g(grid.dx_m)} "
        f"{_g(rough.fractal_dim)} {_g(rough.weight_x)} {_g(rough.weight_y)} "
        f"{_g(y_lo)} {_g(y_hi)} {box_id}{seed_part}"
    )


def _snapshot_line(snap: SnapshotConfigSchema, grid: GlobalDerived) -> str:
    x2 = snap.x2 if snap.x2 is not None else grid.domain_x_m
    y2 = snap.y2 if snap.y2 is not None else grid.domain_y_m
    z2 = snap.z2 if snap.z2 is not None else grid.dx_m
    sdx = snap.dx if snap.dx is not None else grid.dx_m
    sdy = snap.dy if snap.dy is not None else grid.dx_m
    sdz = snap.dz if snap.dz is not None else grid.dx_m
    return (
        f"#snapshot: {_g(snap.x1)} {_g(snap.y1)} {_g(snap.z1)} "
        f"{_g(x2)} {_g(y2)} {_g(z2)} {_g(sdx)} {_g(sdy)} {_g(sdz)} "
        f"{_g(snap.time_s)} {_sanitize(snap.filename)}"
    )


def _geometry_view_line(grid: GlobalDerived, name: str) -> str:
    return (
        f"#geometry_view: 0 0 0 {_g(grid.domain_x_m)} {_g(grid.domain_y_m)} "
        f"{_g(grid.dx_m)} {_g(grid.dx_m)} {_g(grid.dx_m)} {_g(grid.dx_m)} {name} n"
    )


# ---------------------------------------------------------------------------
# Identifier bookkeeping
# ---------------------------------------------------------------------------

def _unique_soil_id(raw: str, index: int, used: set) -> str:
    """A gprMax-safe, unique mixing-model identifier for a layer."""
    base = _sanitize(raw) or f"layer{index + 1}"
    if base.lower() in RESERVED_MATERIAL_NAMES:
        base = f"{base}_soil"
    ident = base
    n = index + 1
    while ident in used:
        ident = f"{base}_{n}"
        n += 1
    used.add(ident)
    return ident


# ---------------------------------------------------------------------------
# Per-sample .in text
# ---------------------------------------------------------------------------

@dataclass
class LayerLabel:
    name: Optional[str]
    thickness_m: float   # SNAPPED thickness (matches the emitted geometry)
    y_top_m: float
    y_bottom_m: float


def build_in_text(
    sample: SampledSample,
    grid: GlobalDerived,
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    adv: Optional[ExtractedAdvancedParams],
) -> tuple[str, List[LayerLabel]]:
    """Build the .in text for one sample and the snapped per-layer labels."""
    if cfg.contract_version >= 2:
        from backend.dataset_sampling.scene import resolve_scene, token
        scene = resolve_scene(sample, grid, cfg, wf, ant, adv)
        return serialize_scene(scene, grid, cfg, wf), [LayerLabel(l["name"], l["thickness_m"], l["y_top_m"], l["y_bottom_m"]) for l in scene["layers"]]
    dx = grid.dx_m
    title = f"{_sanitize(cfg.model_basename)}_{sample.sample_id}"

    lines: List[str] = []

    # --- header ---
    lines.append(f"#title: {title}")
    if cfg.num_threads is not None:
        lines.append(f"#num_threads: {cfg.num_threads}")
    lines.append(f"#domain: {_g(grid.domain_x_m)} {_g(grid.domain_y_m)} {_g(dx)}")
    lines.append(f"#dx_dy_dz: {_g(dx)} {_g(dx)} {_g(dx)}")
    lines.append(f"#time_window: {_g(grid.time_window_s)}")
    # 6-value PML form comes from DatasetConfig so extraction-time validation
    # and emission use the same rule. In 2D, z faces must be zero because nz=1.
    pml = cfg.gprmax_pml_cells()
    lines.append(f"#pml_cells: {' '.join(str(p) for p in pml)}")
    lines.append("")

    # --- waveform (before any source) ---
    lines.append(_waveform_line(wf, grid))
    lines.append("")

    # --- soil layers (snapped to whole cells; deepest extended to y=0) ---
    ground_cells = round(grid.ground_y_m / dx)
    ground_y = ground_cells * dx
    top_cells = ground_cells
    used_ids: set = set()
    labels: List[LayerLabel] = []
    top_box_id: Optional[str] = None

    n_layers = len(sample.layers)
    for i, layer in enumerate(sample.layers):
        is_last = i == n_layers - 1
        if is_last:
            bottom_cells = 0
        else:
            t_cells = max(1, round(layer.thickness_m / dx))
            bottom_cells = top_cells - t_cells

        y2 = top_cells * dx
        y1 = bottom_cells * dx
        # Guards: no inverted / zero-height / below-floor boxes.
        if bottom_cells < 0 or y2 <= y1:
            raise ValueError(
                f"sample {sample.sample_id} layer {i} ('{layer.name}') produced an "
                f"invalid box y=[{y1:.4f}, {y2:.4f}] (cells {bottom_cells}..{top_cells}). "
                "The non-final layer stack does not fit under the ground surface — "
                "check the global derive (ground_y = pad + depth_z >= sum thickness)."
            )

        soil_id = _unique_soil_id(layer.name or f"layer{i + 1}", i, used_ids)
        box_id = f"{soil_id}_fbox"
        seed = sample.sample_id * 1000 + (i + 1)   # deterministic, reproducible
        if i == 0:
            top_box_id = box_id

        lines.append(_soil_peplinski_line(layer, soil_id))
        lines.append(
            _fractal_box_line(
                0.0, y1, 0.0, grid.domain_x_m, y2, dx,
                cfg.fractal_nbins, soil_id, box_id, seed,
            )
        )
        labels.append(LayerLabel(layer.name, y2 - y1, y2, y1))
        top_cells = bottom_cells
    lines.append("")

    # --- optional surface roughness on the surface layer's top face ---
    if adv is not None and adv.surface_roughness is not None and top_box_id is not None:
        rough = adv.surface_roughness
        # Roughness peaks are NOT sized into the top clearance by the global derive;
        # a peak must not reach the source or the top PML gap.
        top_clear = grid.domain_y_m - (cfg.pml_cells + PML_GAP_CELLS) * dx
        peak = ground_y + rough.amplitude_m
        if peak >= grid.tx_y_m or peak >= top_clear:
            raise ValueError(
                f"surface roughness amplitude {rough.amplitude_m:.4f} m lifts the "
                f"surface to y={peak:.4f} m, which reaches the source (tx_y="
                f"{grid.tx_y_m:.4f}) or the top PML gap (y<={top_clear:.4f}). "
                "Reduce amplitude or increase source_height_m."
            )
        lines.append(_surface_roughness_line(rough, grid, top_box_id, ground_y))
        lines.append("")

    # --- source + receiver ---
    lines.append(_source_line(ant, grid, wf))
    lines.append(_rx_line(grid))
    lines.append("")

    # --- buried objects (AFTER the fractal boxes so they override the soil) ---
    if sample.targets:
        for t in sample.targets:
            lines.append(_target_line(t, grid))
        lines.append("")

    # --- optional snapshots ---
    if adv is not None and adv.snapshots:
        for snap in adv.snapshots:
            lines.append(_snapshot_line(snap, grid))
        lines.append("")

    # --- geometry view (for --geometry-only inspection) ---
    lines.append(_geometry_view_line(grid, f"{title}_geo"))

    return "\n".join(lines) + "\n", labels


def serialize_scene(scene, grid, cfg, wf):
    """Pure serialization of validated v2 coordinates. No snapping or repairs."""
    def xyz(values):
        return " ".join(_g(v) for v in values)
    lines = [f"#title: {scene['title']}"]
    if cfg.num_threads is not None:
        lines.append(f"#num_threads: {cfg.num_threads}")
    lines += [f"#domain: {xyz([grid.domain_x_m, grid.domain_y_m, grid.domain_z_m])}",
              f"#dx_dy_dz: {xyz([grid.dx_m] * 3)}", f"#time_window: {grid.iterations}",
              f"#pml_cells: {' '.join(map(str, cfg.gprmax_pml_cells()))}", _waveform_line(wf, grid)]
    for layer in scene["layers"]:
        lines.append(_soil_peplinski_line(SampledLayer.model_validate(layer["requested"]), layer["soil_id"]))
        lines.append(_fractal_box_line(*layer["start_m"], *layer["end_m"], cfg.fractal_nbins,
                                      layer["soil_id"], layer["box_id"], layer["seed"]))
    rough = scene.get("roughness")
    if rough:
        lines.append(f"#add_surface_roughness: {xyz(rough['start_m'])} {xyz(rough['end_m'])} "
                     f"{_g(rough['fractal_dim'])} {xyz(rough['surface_weights_x_z'])} "
                     f"{_g(rough['height_min_m'])} {_g(rough['height_max_m'])} {rough['box_id']} {rough['seed']}")
    source = scene["source"]
    timing = grid.derivation["excitation"]
    suffix = ""
    if timing["stop_s"] is not None:
        suffix = f" {_g(timing['start_s'])} {_g(timing['stop_s'])}"
    resistance = f" {_g(source['resistance_ohm'])}" if source["kind"] != "hertzian_dipole" else ""
    lines.append(f"#{source['kind']}: {source['axis']} {xyz(source['position_m'])}{resistance} {_sanitize(wf.waveform_name)}{suffix}")
    lines.append(f"#rx: {xyz(scene['receiver']['position_m'])}")
    for target in scene["targets"]:
        radius = f" {_g(target['radius_m'])}" if target["kind"] == "cylinder" else ""
        lines.append(f"#{target['kind']}: {xyz(target['start_m'])} {xyz(target['end_m'])}{radius} pec n")
    for snapshot in scene["snapshots"]:
        lines.append(f"#snapshot: {xyz(snapshot['start_m'])} {xyz(snapshot['end_m'])} {xyz(snapshot['strides_m'])} {snapshot['iteration']} {snapshot['filename']}")
    # Native geometry hook verifies the full material map before field updates.
    # This view is a declared inspection output, separate from the solver grid.
    view = scene["geometry_view"]
    lines.append(f"#geometry_view: 0 0 0 {xyz(view['end_m'])} {xyz(view['strides_m'])} {view['filename']} n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class EmissionResult:
    output_dir: str
    in_dir: str
    n_written: int
    files: List[dict] = field(default_factory=list)   # {sample_id, filename, path, layers}
    errors: List[str] = field(default_factory=list)


def _resolve(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent.parent / path
    return path


def emit_dataset(
    output_dir: str,
    cfg: DatasetConfig,
    wf: ExtractedWaveform,
    ant: ExtractedAntenna,
    adv: Optional[ExtractedAdvancedParams] = None,
    in_subdir: str = "in_files",
    manifest_filename: str = "emitted_files.json",
    layers=None,
    target_ranges=None,
) -> EmissionResult:
    """Write one gprMax .in file per surviving sample onto the global grid.

    Reads global_derive.json + sampled_layers.json from `output_dir`, writes the
    .in files into `output_dir/in_subdir`, and records an emission manifest.
    """
    grid = read_global(output_dir)
    samples = read_samples(output_dir)

    out_dir = _resolve(output_dir)
    in_dir = out_dir / in_subdir
    in_dir.mkdir(parents=True, exist_ok=True)

    result = EmissionResult(output_dir=str(out_dir), in_dir=str(in_dir), n_written=0)
    contract = None
    if cfg.contract_version >= 2:
        from backend.dataset_sampling.contract import digest, solver_identity, POLICY_VERSION, validate_capabilities
        from backend.dataset_sampling.scene import resolve_scene, token
        from backend.resources import estimate_resources
        from backend.dataset_sampling.target_placement import distribution_summary
        validate_capabilities(cfg, target_ranges=target_ranges, waveform=wf, antenna=ant, advanced=adv)
        derived_data = json.loads((out_dir / "derived_layers.json").read_text())
        derived_by_id = {s["sample_id"]: s for s in derived_data["samples"]}
        summary = json.loads((out_dir / "sampled_layers.json").read_text()).get("sampling_summary") or {
            "accepted_draws": distribution_summary(samples),
            "notice": "Observed marginals after soil validation and placement; no uniformity claim"}
        contract = {"version": 2, "physics_policy": POLICY_VERSION, "dimensionality": cfg.dimensionality,
                    "coordinate_frame": cfg.coordinate_frame, "solver": solver_identity(),
                    "grid": grid.model_dump(), "pml_faces": list(cfg.gprmax_pml_cells()),
                    "clearance_cells": cfg.pml_cells + PML_GAP_CELLS,
                    "requested": {"dataset_config": cfg.model_dump(exclude={"output_dir", "num_threads"}),
                                  "waveform": wf.model_dump(), "antenna": ant.model_dump(),
                                  "layers": layers.model_dump() if layers else None,
                                  "target_ranges": target_ranges.model_dump() if target_ranges else None,
                                  "advanced": adv.model_dump() if adv else None},
                    "source_normalization": "native amplitude; Hertzian current moment = amplitude * cell spacing along selected axis",
                    "field_units": {"E": "V/m", "H": "A/m", "time": "s", "coordinates": "m"},
                    "terminal_layer": "bottom half-space; no independently realized last interface",
                    "resources": estimate_resources(grid, cfg, len(samples[0].layers) if samples else 0, adv),
                    "qualification": {"status": "unqualified", "training_eligible": False, "reuse_eligible": False},
                    "num_requested": cfg.num_samples, "placement_attempt_limit": 20,
                    "sampling_policy": "independent identity streams; fixed moisture band; rejection changes survivor distribution"}
        contract["digest"] = digest(contract)
        (out_dir / "dataset_contract.json").write_text(json.dumps(contract, indent=2))

    for sample in samples:
        basename = token(cfg.model_basename) if contract else _sanitize(cfg.model_basename)
        filename = f"{basename}_{sample.sample_id}.in"
        path = in_dir / filename
        try:
            scene = None
            if contract is not None:
                scene = resolve_scene(sample, grid, cfg, wf, ant, adv)
                for layer, derived in zip(scene["layers"], derived_by_id[sample.sample_id]["layers"], strict=True):
                    layer["material_provenance"] = derived["material_provenance"]
                scene["contract_digest"] = contract["digest"]
                scene["digest"] = digest(scene)
                text = serialize_scene(scene, grid, cfg, wf)
                labels = [LayerLabel(l["name"], l["thickness_m"], l["y_top_m"], l["y_bottom_m"]) for l in scene["layers"]]
                from backend.preflight import validate_deck_contract
                validate_deck_contract(text, contract, scene)
            else:
                text, labels = build_in_text(sample, grid, cfg, wf, ant, adv)
        except Exception as exc:
            result.errors.append(f"sample {sample.sample_id}: {exc}")
            if contract is not None:
                # A scene inconsistent with the frozen contract is not another
                # placement rejection. Invalidate the manifest for the batch.
                (out_dir / manifest_filename).write_text(json.dumps({
                    "contract": contract, "files": [], "n_written": 0,
                    "status": "invalid", "errors": result.errors}, indent=2))
                raise ValueError("Contract emission failed: " + result.errors[-1]) from exc
            continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        result.files.append({
            "sample_id": sample.sample_id,
            "filename": filename,
            "path": str(path),
            "layers": [
                {"name": l.name, "thickness_m": l.thickness_m,
                 "y_top_m": l.y_top_m, "y_bottom_m": l.y_bottom_m}
                for l in labels
            ],
            **({"contract_digest": contract["digest"], "input_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                "resolved_scene": scene} if contract else {}),
        })
        result.n_written += 1

    manifest = {
        "output_dir": str(out_dir),
        "in_dir": str(in_dir),
        "n_written": result.n_written,
        "grid": {
            "dx_m": grid.dx_m,
            "domain_x_m": grid.domain_x_m,
            "domain_y_m": grid.domain_y_m,
            "ground_y_m": grid.ground_y_m,
            "f_peak_hz": grid.f_peak_hz,
            "time_window_s": grid.time_window_s,
            "pml_cells": cfg.pml_cells,
        },
        "files": result.files,
        "errors": result.errors,
        **({"contract": contract, "contract_digest": contract["digest"],
            "num_requested": cfg.num_samples, "num_accepted": result.n_written,
            "qualification": contract["qualification"],
            "acceptance": {"distributions": summary, "dropped": json.loads((out_dir / "dropped_targets.json").read_text()).get("dropped_targets", []) if (out_dir / "dropped_targets.json").exists() else []}} if contract else {}),
    }
    with open(out_dir / manifest_filename, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return result
