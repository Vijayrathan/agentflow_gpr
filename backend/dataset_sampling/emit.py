"""
STAGE 8 — gprMax .in file emitter.

Turns the staged manifests into N gprMax input files, one per surviving sample,
all on the ONE global grid derived upstream. NOTHING here re-derives grid size,
physics, or frequency: the emitter only *transcribes* already-resolved values
into gprMax command syntax.

Inputs consumed
---------------
  * global_derive.json  (via read_global)  -> the fixed grid / domain / Tx-Rx /
    time window shared by every sample. In particular ``grid.f_peak_hz`` is the
    ALREADY-converted peak frequency (see global_derive.py:85-88); the emitter
    uses it verbatim and NEVER reads the raw collected centre frequency.
  * sampled_layers.json (via read_samples) -> per-sample concrete layers (+ an
    optional, already-placed buried target). This is the post-target-placement
    survivor set.
  * the collected waveform / antenna / advanced sections (graph state), for the
    #waveform, source, #rx and optional advanced objects.

Geometry conventions (see the approved plan / verified user_models examples)
--------------------------------------------------------------------------
  * 2D, thin z: #domain is (domain_x, domain_y, dx) so z spans exactly one cell.
  * x = horizontal (survey), y = vertical (positive up), z = thin/invariant.
  * The Hertzian dipole is polarised along the thin invariant axis (z) — exactly
    as in the known-good cylinder_Ascan_2D.in. The collected antenna_axis has no
    meaning in 2D and is overridden to 'z'.
  * Ground surface at grid.ground_y_m; soil fills [0, ground_y]; free_space above
    is implicit. Layers anchor at the surface, stack down, and the DEEPEST layer
    is extended to y=0 (continuous half-space, no air pocket).
  * Layer interfaces are snapped to whole Δy (= dx) cells so labels match the grid
    gprMax actually builds.
  * Each file is a single A-scan (one static Tx/Rx).

3D is out of scope: global_derive produces no z-domain.
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
    CylinderSchema,
    BoxSchema,
    SphereSchema,
    SurfaceRoughnessConfigSchema,
    SnapshotConfigSchema,
)
from backend.validation_tools_new import PML_GAP_CELLS
from dataset_sampling.global_derive import read_global
from dataset_sampling.layer_sampler import read_samples

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
    return f"{v:.10g}"


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

    Emits #hertzian_dipole (default) or #voltage_source. The optional [start end]
    timing pair is written only when BOTH are set (gprMax requires the pair).
    """
    kind = (ant.antenna_kind or "hertzian_dipole").lower()
    timing = ""
    if wf.source_start_time is not None and wf.source_end_time is not None:
        timing = f" {_g(wf.source_start_time)} {_g(wf.source_end_time)}"
    pos = f"{_g(grid.tx_x_m)} {_g(grid.tx_y_m)} 0"
    name = _sanitize(wf.waveform_name)
    if kind == "voltage_source":
        if ant.resistance is None:
            raise ValueError("voltage_source requires a resistance value")
        return f"#voltage_source: z {pos} {_g(ant.resistance)} {name}{timing}"
    return f"#hertzian_dipole: z {pos} {name}{timing}"


def _rx_line(grid: GlobalDerived) -> str:
    return f"#rx: {_g(grid.rx_x_m)} {_g(grid.rx_y_m)} 0"


def _target_cylinder_line(target: SampledTarget, grid: GlobalDerived) -> str:
    """Buried cylinder disc (axis along thin z). Dielectric smoothing OFF ('n') so
    a PEC target fully replaces the underlying fractal materials at its boundary."""
    y_center = grid.ground_y_m - target.depth_m
    return (
        f"#cylinder: {_g(target.x_center_m)} {_g(y_center)} 0 "
        f"{_g(target.x_center_m)} {_g(y_center)} {_g(grid.dx_m)} "
        f"{_g(target.radius_m)} {_sanitize(target.material)} n"
    )


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


def _cylinder_line(obj: CylinderSchema) -> str:
    smoothing = "y" if obj.dielectric_smoothing else "n"
    return (
        f"#cylinder: {_g(obj.x1)} {_g(obj.y1)} {_g(obj.z1)} "
        f"{_g(obj.x2)} {_g(obj.y2)} {_g(obj.z2)} {_g(obj.radius)} "
        f"{_sanitize(obj.material)} {smoothing}"
    )


def _box_line(obj: BoxSchema) -> str:
    smoothing = "y" if obj.dielectric_smoothing else "n"
    return (
        f"#box: {_g(obj.x1)} {_g(obj.y1)} {_g(obj.z1)} "
        f"{_g(obj.x2)} {_g(obj.y2)} {_g(obj.z2)} "
        f"{_sanitize(obj.material)} {smoothing}"
    )


def _sphere_line(obj: SphereSchema) -> str:
    smoothing = "y" if obj.dielectric_smoothing else "n"
    return (
        f"#sphere: {_g(obj.cx)} {_g(obj.cy)} {_g(obj.cz)} {_g(obj.radius)} "
        f"{_sanitize(obj.material)} {smoothing}"
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

    # --- buried target (AFTER the fractal boxes so it overrides the soil) ---
    if sample.target is not None:
        lines.append(_target_cylinder_line(sample.target, grid))
        lines.append("")

    # --- optional fixed advanced objects ---
    if adv is not None:
        obj_lines: List[str] = []
        for c in (adv.cylinders or []):
            obj_lines.append(_cylinder_line(c))
        for b in (adv.boxes or []):
            obj_lines.append(_box_line(b))
        for s in (adv.spheres or []):
            obj_lines.append(_sphere_line(s))
        if obj_lines:
            lines.extend(obj_lines)
            lines.append("")
        for snap in (adv.snapshots or []):
            lines.append(_snapshot_line(snap, grid))
        if adv.snapshots:
            lines.append("")

    # --- geometry view (for --geometry-only inspection) ---
    lines.append(_geometry_view_line(grid, f"{title}_geo"))

    return "\n".join(lines) + "\n", labels


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
) -> EmissionResult:
    """Write one gprMax .in file per surviving sample onto the global grid.

    Reads global_derive.json + sampled_layers.json from `output_dir`, writes the
    .in files into `output_dir/in_subdir`, and records an emission manifest.
    """
    if cfg.dimensionality == "3D":
        raise NotImplementedError(
            "The emitter targets the 2D global grid; global_derive produces no "
            "z-domain, so 3D emission is not supported."
        )

    grid = read_global(output_dir)
    samples = read_samples(output_dir)

    out_dir = _resolve(output_dir)
    in_dir = out_dir / in_subdir
    in_dir.mkdir(parents=True, exist_ok=True)

    result = EmissionResult(output_dir=str(out_dir), in_dir=str(in_dir), n_written=0)

    for sample in samples:
        filename = f"{_sanitize(cfg.model_basename)}_{sample.sample_id}.in"
        path = in_dir / filename
        try:
            text, labels = build_in_text(sample, grid, cfg, wf, ant, adv)
        except Exception as exc:  # emit what we can; log the rest
            result.errors.append(f"sample {sample.sample_id}: {exc}")
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
    }
    with open(out_dir / manifest_filename, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return result
