"""
Post-simulation signal extraction from gprMax HDF5 output files.

Reads electromagnetic field components (Ex, Ey, Ez, Hx, Hy, Hz) from
receiver data in .out files and prepares batch updates for the
simulations table in PostgreSQL.
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
SIGNAL_KEYS = ("signal_ex", "signal_ey", "signal_ez",
               "signal_hx", "signal_hy", "signal_hz")


def validate_output(filepath, contract, scene, *, input_sha256=None, require_receipt=False):
    """Check actual HDF5 settings and vectors, never substitute planned values."""
    import json
    from backend.dataset_sampling.contract import file_digest
    from backend.preflight import verify_contract
    verify_contract(contract, scene)
    grid = contract["grid"]
    with h5py.File(filepath, "r") as f:
        dt, iterations = float(f.attrs["dt"]), int(f.attrs["Iterations"])
        counts = [int(v) for v in f.attrs["nx_ny_nz"]]
        spacing = [float(v) for v in f.attrs["dx_dy_dz"]]
        version = str(f.attrs["gprMax"])
        if counts != [grid["nx"], grid["ny"], grid["nz"]] or not np.allclose(spacing, [grid["dx_m"]] * 3, rtol=1e-13, atol=0):
            raise ValueError("Output grid differs from dataset contract")
        if version != contract["solver"]["version"]:
            raise ValueError("Output gprMax version differs from pinned contract")
        if not np.isfinite(dt) or dt <= 0 or not np.isclose(dt, grid["dt_s"], rtol=1e-13, atol=0) or iterations != grid["iterations"]:
            raise ValueError("Output time axis differs from dataset contract")
        if str(f.attrs["Title"]) != scene["title"]:
            raise ValueError("Output sample identity/title differs from resolved scene")
        if int(f.attrs["nrx"]) != 1 or int(f.attrs["nsrc"]) != 1 or list(f["rxs"].keys()) != ["rx1"]:
            raise ValueError("Output receiver/source count differs from single-pair acquisition")
        if any(np.any(f.attrs[k]) for k in ("srcsteps", "rxsteps")):
            raise ValueError("Output moved an acquisition position")
        rx = f["rxs/rx1"]
        if not np.allclose(rx.attrs["Position"], scene["receiver"]["position_m"], rtol=0, atol=1e-12):
            raise ValueError("Output receiver position mismatch")
        source_path = "tls/tl1" if scene["source"]["kind"] == "transmission_line" else "srcs/src1"
        source = f[source_path]
        if scene["source"]["kind"] == "transmission_line":
            if source.attrs["Resistance"] != scene["source"]["resistance_ohm"] or not np.isclose(source.attrs["dl"], grid["dx_m"], rtol=1e-13, atol=0):
                raise ValueError("Output transmission-line parameters mismatch")
        elif str(source.attrs["Type"]) != {"hertzian_dipole": "HertzianDipole", "voltage_source": "VoltageSource"}[scene["source"]["kind"]]:
            raise ValueError("Output source kind mismatch")
        if not np.allclose(source.attrs["Position"], scene["source"]["position_m"], rtol=0, atol=1e-12):
            raise ValueError("Output source position mismatch")
        for component in COMPONENTS:
            if component not in rx:
                raise ValueError(f"Missing receiver component {component}")
            data = np.asarray(rx[component])
            if data.shape != (iterations,) or not np.isfinite(data).all():
                raise ValueError(f"Invalid/nonfinite or mismatched {component} array")
        metadata = {"status": "integrity_valid", "dt_s": dt, "iterations": iterations,
                    "last_time_s": (iterations - 1) * dt, "cell_counts": counts,
                    "cell_spacings_m": spacing, "solver_version": version,
                    "receiver_position_m": [float(v) for v in rx.attrs["Position"]],
                    "coordinate_frame": contract["coordinate_frame"],
                    "components": list(COMPONENTS), "units": contract["field_units"],
                    "contract_digest": contract["digest"], "scene_digest": scene["digest"]}
    if require_receipt:
        receipt_path = Path(filepath).with_suffix(".execution.json")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("contract_digest") != contract["digest"] or receipt.get("scene_digest") != scene["digest"] or
            receipt.get("input_sha256") != input_sha256 or receipt.get("output_sha256") != file_digest(filepath) or
            receipt.get("preflight", {}).get("status") != "passed"):
            raise ValueError("Missing/stale output execution receipt or file hash")
        geometry_path = Path(filepath).with_suffix(".geometry.h5")
        if not geometry_path.is_file() or receipt["preflight"].get("geometry_sha256") != file_digest(geometry_path):
            raise ValueError("Missing/stale native geometry export or file hash")
        records = {s["path"]: s["sha256"] for s in receipt.get("snapshots", [])}
        expected = [str(Path(Path(filepath).stem + "_snaps") / (s["filename"] + ".vti")) for s in scene["snapshots"]]
        if set(records) != set(expected):
            raise ValueError("Snapshot artifact identities differ from the output specification")
        for name in expected:
            if records[name] != file_digest(Path(filepath).parent / name):
                raise ValueError("Missing/stale snapshot artifact hash")
    return metadata

# Matches the trailing _NNNN in filenames like gpr_dataset_0001.out
_SAMPLE_INDEX_RE = re.compile(r"_(\d+)$")


def _parse_sample_index(stem: str) -> int | None:
    """Extract sample_index from a filename stem like 'gpr_dataset_0001'."""
    m = _SAMPLE_INDEX_RE.search(stem)
    if m:
        return int(m.group(1))
    return None


def extract_signals_from_hdf5(filepath: str | Path) -> dict[str, Any]:
    """Extract EM field signals from a single gprMax .out (HDF5) file.

    Reads the first receiver (rx1) data.  Each component is returned as
    a plain Python list of floats suitable for PostgreSQL ARRAY(Float).

    Returns:
        Dict with keys signal_ex … signal_hz (list[float] | None),
        signal_length (int).

    Raises:
        FileNotFoundError, ValueError, OSError on bad input.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Output file not found: {filepath}")

    with h5py.File(str(filepath), "r") as f:
        nrx = f.attrs.get("nrx", 0)
        if nrx == 0:
            raise ValueError(f"No receivers in {filepath.name}")

        rx_path = "/rxs/rx1"
        if rx_path not in f:
            raise ValueError(f"Receiver group {rx_path} missing in {filepath.name}")

        rx_group = f[rx_path]
        signals: dict[str, Any] = {}
        length = 0

        for comp, key in zip(COMPONENTS, SIGNAL_KEYS):
            if comp in rx_group:
                arr = np.array(rx_group[comp], dtype=np.float64)
                if arr.ndim != 1 or not np.isfinite(arr).all() or length and len(arr) != length:
                    raise ValueError(f"Invalid field array {comp} in {filepath.name}")
                signals[key] = arr.tolist()
                length = max(length, len(arr))
            else:
                signals[key] = None

        signals["signal_length"] = length

    return signals


def read_ascan(filepath: str | Path) -> dict[str, Any]:
    """Read one gprMax .out (HDF5) file for A-scan display.

    Returns the first receiver's available field components plus the time
    axis info (dt, Iterations) needed to plot amplitude vs time.

    Returns:
        Dict with keys dt (float, seconds), iterations (int) and
        components ({"Ex": list[float], ...} — absent components omitted).

    Raises:
        FileNotFoundError, ValueError, OSError, KeyError on bad input.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Output file not found: {filepath}")

    with h5py.File(str(filepath), "r") as f:
        dt = float(f.attrs["dt"])
        iterations = int(f.attrs["Iterations"])
        rx_path = "/rxs/rx1"
        if rx_path not in f:
            raise ValueError(f"Receiver group {rx_path} missing in {filepath.name}")
        rx_group = f[rx_path]
        components = {
            comp: np.asarray(rx_group[comp], dtype=np.float64).tolist()
            for comp in COMPONENTS
            if comp in rx_group
        }

    return {"dt": dt, "iterations": iterations, "components": components}


def extract_and_prepare_batch(
    output_dir: str | Path,
    session_id: str,
    manifest=None,
    outputs=None,
) -> dict[str, Any]:
    """Extract signals from all .out files and prepare DB update payloads.

    Maps .out files to simulation DB rows by sample_index (parsed from
    the filename suffix, e.g. gpr_dataset_0001.out -> sample_index=1),
    scoped to the given session_id.

    Args:
        output_dir: Directory containing .out HDF5 files.
        session_id: Session UUID to look up simulation rows.

    Returns:
        Dict with keys:
            updates  – list[dict] ready for bulk_update_signals()
            succeeded – int
            failed    – int
            errors    – list[dict] with filename and error
    """
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from db.db import get_session, Simulation
    from sqlmodel import select

    output_dir = Path(output_dir)

    # Build sample_index -> row_id lookup (scoped to session)
    with get_session() as db:
        stmt = select(Simulation).where(
            Simulation.session_id == session_id  # type: ignore[arg-type]
        )
        rows = list(db.exec(stmt).all())

    idx_to_id: dict[int, str] = {}
    for row in rows:
        idx_to_id[row.sample_index] = str(row.id)

    updates: list[dict] = []
    succeeded = 0
    failed = 0
    errors: list[dict] = []

    by_name = {item["filename"]: item for item in (manifest or {}).get("files", [])}
    selected = {item["filename"]: Path(item["out_file"]) for item in outputs or []}
    out_files = list(selected.values()) if outputs is not None else sorted(output_dir.glob("*.out"))
    if not out_files:
        logger.warning("[SIGNAL] No .out files found in %s", output_dir)
        return {"updates": [], "succeeded": 0, "failed": 0, "errors": []}

    for out_file in out_files:
        emitted = by_name.get(out_file.stem + ".in")
        if manifest is not None and emitted is None:
            continue
        sample_idx = int(emitted["sample_id"]) if emitted else _parse_sample_index(out_file.stem)
        if sample_idx is None:
            logger.warning("[SIGNAL] Cannot parse sample_index from %s — skipping", out_file.name)
            continue

        row_id = idx_to_id.get(sample_idx)
        if row_id is None:
            logger.warning("[SIGNAL] No DB row for sample_index=%d (%s) — skipping", sample_idx, out_file.name)
            continue

        try:
            metadata = None
            if (manifest or {}).get("contract"):
                metadata = validate_output(out_file, manifest["contract"], emitted["resolved_scene"],
                                           input_sha256=emitted["input_sha256"], require_receipt=True)
            signals = extract_signals_from_hdf5(out_file)
            if metadata:
                signals["executed_metadata"] = metadata
                signals["qualification_status"] = "unqualified"
            signals["id"] = row_id
            signals["output_file_path"] = str(out_file)
            updates.append(signals)
            succeeded += 1
        except Exception as exc:
            logger.warning("[SIGNAL] Failed to extract %s: %s", out_file.name, exc)
            errors.append({"filename": out_file.name, "error": str(exc)})
            failed += 1

    logger.info(
        "[SIGNAL] Extraction done. %d succeeded, %d failed out of %d files",
        succeeded, failed, len(out_files),
    )

    return {
        "updates": updates,
        "succeeded": succeeded,
        "failed": failed,
        "errors": errors,
    }
