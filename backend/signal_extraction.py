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
                signals[key] = arr.tolist()
                length = max(length, len(arr))
            else:
                signals[key] = None

        signals["signal_length"] = length

    return signals


def extract_and_prepare_batch(
    output_dir: str | Path,
    session_id: str,
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

    out_files = sorted(output_dir.glob("*.out"))
    if not out_files:
        logger.warning("[SIGNAL] No .out files found in %s", output_dir)
        return {"updates": [], "succeeded": 0, "failed": 0, "errors": []}

    for out_file in out_files:
        sample_idx = _parse_sample_index(out_file.stem)
        if sample_idx is None:
            logger.warning("[SIGNAL] Cannot parse sample_index from %s — skipping", out_file.name)
            continue

        row_id = idx_to_id.get(sample_idx)
        if row_id is None:
            logger.warning("[SIGNAL] No DB row for sample_index=%d (%s) — skipping", sample_idx, out_file.name)
            continue

        try:
            signals = extract_signals_from_hdf5(out_file)
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
