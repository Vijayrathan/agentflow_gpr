"""Versioned experiment policies shared by collection, planning and ingestion.

Version 1 identifies historical planar draws. Version 2 adds finite geometry,
independent random streams and resolved-scene provenance. A valid contract does
not by itself confer scientific qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

POLICY_VERSION = "peplinski-native-y-up-v2"
FRAME = "x-horizontal_y-up_z-crossline"
FIELD_OFFSETS = {"Ex": (.5, 0, 0), "Ey": (0, .5, 0), "Ez": (0, 0, .5),
                 "Hx": (0, .5, .5), "Hy": (.5, 0, .5), "Hz": (.5, .5, 0)}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_seed(seed: int, *identity) -> int:
    return int(digest([seed, *identity])[:8], 16)


def solver_identity() -> dict:
    import platform
    import numpy
    import scipy
    root = Path(__file__).resolve().parents[2] / "gprMax" / "gprMax"
    # Hash the pinned implementation, including its compiled-kernel sources.
    files = sorted(p for p in root.rglob("*") if p.suffix in (".py", ".pyx", ".pxd"))
    revision = digest({str(p.relative_to(root)): file_digest(p) for p in files})
    binaries = sorted(p for p in root.rglob("*") if p.suffix in (".so", ".pyd", ".dll"))
    from backend.dataset_sampling.numerics import native_version
    return {"name": "gprMax", "version": native_version(), "source_digest": revision,
            "binary_digest": digest({str(p.relative_to(root)): file_digest(p) for p in binaries}),
            "runtime": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__, "architecture": platform.machine()},
            "precision": "float32/complex64"}


def validate_capabilities(cfg, *, target_ranges=None, waveform=None, antenna=None, advanced=None):
    """Reject physical intent that the active writer cannot represent."""
    if cfg.contract_version >= 2 and cfg.cells_per_wavelength < 10:
        raise ValueError("Contract v2 requires at least 10 cells per wavelength")
    if target_ranges is not None:
        for t in [*target_ranges.cylinders, *target_ranges.boxes]:
            finite = t.length_min_m if t.kind == "cylinder" else t.crossline_size_min_m
            if cfg.dimensionality == "3D":
                if t.z_offset_min_m is None or finite is None:
                    raise ValueError(f"3D target '{t.name}' needs z_offset min/max and "
                                     + ("finite length min/max" if t.kind == "cylinder" else "crossline_size min/max"))
                if t.kind == "cylinder" and t.cylinder_axis is None:
                    raise ValueError(f"3D cylinder '{t.name}' requires cylinder_axis x/y/z")
            elif t.z_offset_min_m is not None or finite is not None or (t.kind == "cylinder" and t.cylinder_axis not in (None, "z")):
                raise ValueError("Finite crossline geometry requires 3D; it cannot be flattened into TMz")
    if waveform is not None and cfg.contract_version >= 2:
        if (waveform.waveform_kind or "ricker").lower() != "ricker":
            raise ValueError("Contract v2 currently supports Ricker excitation only; other families need their own spectral policy")
    if antenna is not None:
        if cfg.contract_version >= 2 and antenna.antenna_kind == "hertzian_dipole" and antenna.resistance is not None:
            raise ValueError("Hertzian dipoles do not accept resistance; select a supported resistive source to model it")
        if cfg.dimensionality == "2D" and cfg.contract_version >= 2 and antenna.antenna_axis != "z":
            raise ValueError("2D TMz requires z polarization; choose 3D to use x/y polarization")
        if antenna.rx_array is not None:
            raise ValueError("Receiver arrays/scans are unsupported; one fixed Tx/Rx pair per dataset")
        if antenna.rx_same_height is False and antenna.receiver_height_m is None:
            raise ValueError("rx_same_height=False requires receiver_height_m above ground")
        if antenna.rx_same_height is not False and antenna.receiver_height_m is not None:
            raise ValueError("receiver_height_m requires rx_same_height=False")
        if cfg.dimensionality == "2D" and antenna.tx_rx_crossline_offset_m != 0:
            raise ValueError("Crossline Tx/Rx separation requires 3D")
    if advanced is not None and advanced.surface_roughness is not None:
        if cfg.dimensionality == "3D":
            raise ValueError("Surface roughness is not supported in the initial 3D contract; use flat layers")
        if advanced.surface_roughness.add_water:
            raise ValueError("Surface water has no supported emission path")


def validate_release_access(cfg):
    """Collector/API gate; deterministic fixtures can exercise experimental 3D."""
    import os
    if cfg.dimensionality == "3D" and os.environ.get("GPR_ENABLE_EXPERIMENTAL_3D") != "1":
        raise ValueError("3D is experimental pending numerical qualification. A developer deployment must set "
                         "GPR_ENABLE_EXPERIMENTAL_3D=1 to collect/run 3D datasets; results remain unqualified.")


def component_positions(position, dx):
    return {field: [v + o * dx for v, o in zip(position, offsets)]
            for field, offsets in FIELD_OFFSETS.items()}
