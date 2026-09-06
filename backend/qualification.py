"""Reviewed scientific eligibility, separate from immutable experiment identity.

No default tolerances or automatic certification. A reviewer supplies criteria
and evidence explicitly covering the exact contract and accepted population.
Run `python -m backend.qualification --help` to record that review locally.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from backend.dataset_sampling.contract import digest
from backend.preflight import verify_contract

UNQUALIFIED = {"status": "unqualified", "training_eligible": False, "reuse_eligible": False}


def population_digest(manifest):
    return digest(sorted((e["sample_id"], e["input_sha256"], e["resolved_scene"]["digest"])
                         for e in manifest["files"]))


def assess(manifest, evidence, criteria):
    contract = manifest["contract"]
    verify_contract(contract)
    for entry in manifest["files"]:
        verify_contract(contract, entry["resolved_scene"])
    identity = {"contract_digest": contract["digest"], "population_digest": population_digest(manifest)}
    if not manifest["files"] or identity not in evidence.get("covered_experiments", []):
        raise ValueError("Evidence does not cover this exact contract and accepted population")
    if evidence.get("solver") != contract["solver"]:
        raise ValueError("Evidence solver implementation differs from the experiment")
    if identity not in criteria.get("approved_experiments", []) or not criteria.get("approved_by") or not criteria.get("intended_use"):
        raise ValueError("Reviewed criteria must name the experiment, reviewer and intended use")
    limits = criteria.get("maximum_errors", {})
    if not limits:
        raise ValueError("Reviewed numerical acceptance tolerances are required")
    results = {}
    for path, limit in limits.items():
        value = evidence
        for key in path.split("."):
            value = value[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"Missing/nonfinite qualification measurement: {path}")
        if not isinstance(limit, (float, int)) or not math.isfinite(limit) or limit < 0 or value < 0 or value > limit:
            raise ValueError(f"Qualification criterion failed: {path} = {value}, limit {limit}")
        results[path] = {"measured": value, "maximum": limit}
    return {**identity, "status": "qualified", "training_eligible": True, "reuse_eligible": True,
            "approved_by": criteria["approved_by"], "intended_use": criteria["intended_use"],
            "criteria_results": results}


def qualification_status(output_dir, manifest):
    try:
        record = json.loads((Path(output_dir) / "qualification.json").read_text())
        if record["digest"] != digest({k: v for k, v in record.items() if k != "digest"}):
            raise ValueError("Qualification attestation digest mismatch")
        result = assess(manifest, record["evidence"], record["criteria"])
        if record["execution_digest"] != execution_digest(output_dir, manifest, record["criteria"]):
            raise ValueError("Executed artifacts changed after scientific review")
        if result != record["assessment"]:
            raise ValueError("Qualification assessment changed")
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return dict(UNQUALIFIED)


def execution_digest(output_dir, manifest, criteria):
    from backend.signal_extraction import validate_output
    from backend.dataset_sampling.contract import file_digest
    allowed = criteria.get("allowed_backends", [])
    if not allowed or any(b not in ("cpu", "gpu") for b in allowed):
        raise ValueError("Reviewed qualification must declare allowed_backends")
    receipts = []
    for entry in manifest["files"]:
        if Path(entry["filename"]).name != entry["filename"] or file_digest(Path(output_dir) / "in_files" / entry["filename"]) != entry["input_sha256"]:
            raise ValueError("Input artifact changed after experiment construction")
        path = Path(output_dir) / "out_files" / (Path(entry["filename"]).stem + ".out")
        validate_output(path, manifest["contract"], entry["resolved_scene"],
                        input_sha256=entry["input_sha256"], require_receipt=True)
        receipt = json.loads(path.with_suffix(".execution.json").read_text())
        if receipt["backend"] not in allowed:
            raise ValueError("Executed backend lies outside qualification scope")
        receipts.append((entry["sample_id"], receipt["output_sha256"], receipt["preflight"]["geometry_sha256"], receipt["backend"]))
    return digest(sorted(receipts))


def record_qualification(output_dir, evidence, criteria):
    root = Path(output_dir)
    manifest = json.loads((root / "emitted_files.json").read_text())
    assessment = assess(manifest, evidence, criteria)
    record = {"version": 1, "assessment": assessment, "evidence": evidence, "criteria": criteria,
              "execution_digest": execution_digest(root, manifest, criteria)}
    record["digest"] = digest(record)
    path = root / "qualification.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False))
    return path


def sync_qualification_rows(output_dir, session_id):
    """Explicit reviewer CLI action for datasets already persisted in Postgres."""
    import uuid
    from sqlmodel import select
    from db.db import get_session, Simulation, ExtractionSession
    root = Path(output_dir)
    manifest = json.loads((root / "emitted_files.json").read_text())
    status = qualification_status(root, manifest)
    if not status["training_eligible"]:
        raise ValueError("Cannot synchronize an invalid/unqualified attestation")
    try:
        sid = uuid.UUID(session_id)
    except ValueError:
        sid = uuid.uuid5(uuid.NAMESPACE_URL, session_id)
    entries = {e["input_sha256"]: e for e in manifest["files"]}
    with get_session() as db:
        rows = list(db.exec(select(Simulation).where(Simulation.session_id == sid)).all())
        if len(rows) != len(entries) or any(r.contract_digest != manifest["contract"]["digest"] or r.input_sha256 not in entries for r in rows):
            raise ValueError("Database session does not match the reviewed experiment")
        for row in rows:
            row.qualification_status = status["status"]
            row.executed_metadata = {**(row.executed_metadata or {}), "qualification": status}
            db.add(row)
        session = db.get(ExtractionSession, sid)
        if session:
            session.model_config_data = {**(session.model_config_data or {}), "qualification": status}
            db.add(session)
        db.commit()
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence", required=True, help="Measured evidence JSON with covered_experiments")
    parser.add_argument("--criteria", required=True, help="Reviewer-authored criteria JSON; no defaults")
    parser.add_argument("--session-id", help="Also synchronize qualification metadata for this existing Postgres session")
    args = parser.parse_args()
    print(record_qualification(args.output_dir, json.loads(Path(args.evidence).read_text()),
                               json.loads(Path(args.criteria).read_text())))
    if args.session_id:
        print(f"Synchronized {sync_qualification_rows(args.output_dir, args.session_id)} database rows")
