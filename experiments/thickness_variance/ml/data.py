"""Read-only dataset admission and paired, locked splits.

Native HDF5 contains no input digest or completion receipt. Existing outputs
can be admitted with that historical linkage explicitly unverified. Strict
receipt admission remains available; neither mode manufactures historical receipts.
"""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import ROUND_FLOOR, ROUND_HALF_DOWN, Decimal
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .common import digest, read_json, sha256, write_json

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
MANIFESTS = (
    "emitted_files.json",
    "sampled_layers.json",
    "derived_layers.json",
    "global_derive.json",
)


def native_round(x):
    return int(Decimal(float(x)).quantize(Decimal(1), rounding=ROUND_HALF_DOWN))


def deck_metadata(path):
    commands = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            key, value = line.split(":", 1)
            commands[key.strip()].append(value.strip())

    def one(key):
        if len(commands[key]) != 1:
            raise ValueError(f"Expected exactly one {key}")
        return commands[key][0]

    spacing = np.asarray(list(map(float, one("#dx_dy_dz").split())))
    domain = np.asarray(list(map(float, one("#domain").split())))
    counts = np.asarray(
        [native_round(d / s) for d, s in zip(domain, spacing, strict=True)]
    )
    if len(spacing) != 3 or counts[2] != 1 or not np.all(counts[:2] > 1):
        raise ValueError("Expected 2D TMz geometry")
    dt = float(
        Decimal(
            float(1 / (299792458 * np.sqrt(np.sum(1 / spacing[:2] ** 2))))
        ).quantize(Decimal("1e-27"), rounding=ROUND_FLOOR)
    )
    iterations = math.ceil(float(one("#time_window")) / dt) + 1
    wave = one("#waveform").split()
    source = one("#hertzian_dipole").split()
    if (
        wave[:3] != ["ricker", "1", "750000000"]
        or source[0] != "z"
        or source[4] != wave[3]
    ):
        raise ValueError("Unexpected waveform, polarization or waveform reference")

    def position(values):
        return [
            native_round(float(v) / s) * s for v, s in zip(values, spacing, strict=True)
        ]

    return {
        "spacing": spacing.tolist(),
        "counts": counts.tolist(),
        "dt_s": dt,
        "iterations": iterations,
        "f_peak_hz": float(wave[2]),
        "source": position(source[1:4]),
        "receiver": position(one("#rx").split()),
        "title": one("#title"),
        "moistures": [
            list(map(float, s.split()[4:6])) for s in commands["#soil_peplinski"]
        ],
    }


def inspect_output(path, expected):
    with h5py.File(path, "r") as f:
        for attr, wanted in (
            ("Iterations", expected["iterations"]),
            ("nrx", 1),
            ("nsrc", 1),
            ("gprMax", "3.1.7"),
            ("Title", expected["title"]),
        ):
            actual = f.attrs.get(attr)
            if isinstance(actual, bytes):
                actual = actual.decode()
            if actual != wanted:
                raise ValueError(f"HDF5 {attr}: {actual!r} != {wanted!r}")
        if float(f.attrs["dt"]) != expected["dt_s"]:
            raise ValueError("HDF5 time step differs from native deck timing")
        for attr, wanted in (
            ("nx_ny_nz", expected["counts"]),
            ("dx_dy_dz", expected["spacing"]),
            ("rxsteps", [0, 0, 0]),
            ("srcsteps", [0, 0, 0]),
        ):
            if not np.array_equal(f.attrs[attr], wanted):
                raise ValueError(f"HDF5 {attr} mismatch")
        for key, wanted in (
            ("rxs/rx1", expected["receiver"]),
            ("srcs/src1", expected["source"]),
        ):
            if not np.allclose(f[key].attrs["Position"], wanted, rtol=0, atol=1e-14):
                raise ValueError(f"HDF5 {key} position mismatch")
        if f["srcs/src1"].attrs["Type"] != "HertzianDipole":
            raise ValueError("Source kind mismatch")
        for component in COMPONENTS:
            x = f[f"rxs/rx1/{component}"][:]
            if x.shape != (expected["iterations"],) or not np.isfinite(x).all():
                raise ValueError(f"Invalid {component} field")


def snapshot_inputs(cfg, output):
    """Freeze current input bytes by arm/relative path; never certify execution."""
    output = Path(output).resolve()
    snapshot = {
        "schema_version": 2,
        "scope": "Current input snapshot only; not historical execution provenance",
        "datasets": {},
    }
    for arm, folder in cfg["datasets"].items():
        p = Path(folder).resolve()
        if output == p or p in output.parents:
            raise ValueError("Input snapshot must be outside source dataset directories")
        manifest_paths = [p / name for name in MANIFESTS]
        before = {str(path): sha256(path) for path in manifest_paths}
        emission = read_json(p / "emitted_files.json")
        records = emission["files"]
        ids = [r["sample_id"] for r in records]
        names = [r["filename"] for r in records]
        if (
            emission["errors"]
            or len(ids) != cfg["expected_pairs"]
            or set(ids) != set(range(1, cfg["expected_pairs"] + 1))
            or len(set(names)) != len(names)
            or any(Path(n).name != n or not n.endswith(".in") for n in names)
        ):
            raise ValueError(f"{arm}: invalid emission identities or filenames")
        if set(names) != {path.name for path in (p / "in_files").glob("*.in")}:
            raise ValueError(f"{arm}: missing/extra deck files")
        paths = manifest_paths + [p / "in_files" / n for n in sorted(names)]
        hashes = {path.relative_to(p).as_posix(): sha256(path) for path in paths}
        if any(sha256(path) != h for path, h in before.items()) or any(
            sha256(p / relative) != h for relative, h in hashes.items()
        ):
            raise ValueError(f"{arm}: inputs changed while snapshotting")
        snapshot["datasets"][arm] = {
            "source_directory": str(p),
            "files": hashes,
        }
    # Never replace an established baseline with changed inputs, even on retry.
    if output.exists():
        if read_json(output) != snapshot:
            raise ValueError("Input snapshot already exists and differs; choose a new path")
    else:
        write_json(output, snapshot)
    return {
        "path": str(output),
        "input_hashes": sum(len(d["files"]) for d in snapshot["datasets"].values()),
        "scope": snapshot["scope"],
    }


def expected_hash(snapshot, dataset, relative, arm=None):
    if snapshot.get("schema_version") == 2:
        try:
            return snapshot["datasets"][arm]["files"][relative]
        except KeyError as error:
            raise ValueError(f"No input snapshot hash for {arm}/{relative}") from error
    suffix = f"/{dataset.name}/{relative}"
    matches = [v for k, v in snapshot.items() if ("/" + k).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"No unique audited input hash for {dataset.name}/{relative}. "
            "The configured baseline does not identify this dataset. "
            "Use snapshot-inputs for the configured datasets and set input_hashes "
            "to that snapshot; execution provenance remains a separate check."
        )
    return matches[0]


def audit(cfg):
    policy = cfg.get("provenance_policy", "require_receipts")
    if policy not in ("require_receipts", "existing_outputs"):
        raise ValueError("Unknown provenance_policy")
    snapshot = read_json(cfg["input_hashes"])
    receipts = {}
    if cfg["receipts"]:
        ledger = read_json(cfg["receipts"])
        if ledger.get("schema_version") != 1:
            raise ValueError("Execution receipt schema_version must be 1")
        for r in ledger["runs"]:
            key = (r["arm"], r["sample_id"])
            if key in receipts:
                raise ValueError(f"Duplicate execution receipt: {key}")
            receipts[key] = r
    report = {
        "inputs_ready": True,
        "outputs_ready": False,
        "issues": [],
        "warnings": [],
        "skipped_checks": [],
        "rows": [],
        "input_baseline_scope": snapshot.get("scope", "Historical input hash baseline"),
        "hashes": {str(Path(cfg["input_hashes"])): sha256(cfg["input_hashes"])},
        "arms": {},
    }
    if cfg["receipts"]:
        report["hashes"][cfg["receipts"]] = sha256(cfg["receipts"])
    common, by_arm = None, {}

    def issue(kind, arm, sid, message):
        report["issues"].append(
            {"kind": kind, "arm": arm, "sample_id": sid, "message": message}
        )
        if kind == "input":
            report["inputs_ready"] = False

    for arm, folder in cfg["datasets"].items():
        p = Path(folder)
        stats = {
            "expected": cfg["expected_pairs"],
            "present": 0,
            "valid_hdf5": 0,
            "verified_receipts": 0,
            "status": "checking",
            "inputs_checked": 0,
        }
        report["arms"][arm] = stats
        try:
            for name in MANIFESTS:
                h = sha256(p / name)
                if h != expected_hash(snapshot, p, name, arm):
                    raise ValueError(f"Manifest differs from configured input baseline: {name}")
                report["hashes"][str(p / name)] = h
            emission, sampled = (
                read_json(p / "emitted_files.json"),
                read_json(p / "sampled_layers.json"),
            )
            files = {r["sample_id"]: r for r in emission["files"]}
            samples = {r["sample_id"]: r for r in sampled["samples"]}
            expected_ids = set(range(1, cfg["expected_pairs"] + 1))
            if (
                set(files) != expected_ids
                or set(samples) != expected_ids
                or len(files) != len(emission["files"])
                or len(samples) != len(sampled["samples"])
                or emission["errors"]
            ):
                raise ValueError("Missing/duplicate identities or emission errors")
            if sampled["sampling"]["moisture_sampling"] != "uniform_per_sample":
                raise ValueError("Scalar moisture required")
            names = [r["filename"] for r in files.values()]
            if len(set(names)) != len(names) or any(
                Path(n).name != n or not n.endswith(".in") for n in names
            ):
                raise ValueError("Invalid/duplicate manifest filename")
            if set(names) != {f.name for f in (p / "in_files").glob("*.in")}:
                raise ValueError("Missing/extra deck files")
            expected_out = {Path(n).with_suffix(".out").name for n in names}
            extras = {f.name for f in (p / "out_files").glob("*.out")} - expected_out
            if extras:
                issue(
                    "output",
                    arm,
                    None,
                    f"Extra outputs outside manifest: {sorted(extras)}",
                )
        except (ValueError, KeyError, OSError) as error:
            issue("input", arm, None, str(error))
            stats.update(
                status="skipped_after_manifest_failure",
                present=None,
                valid_hdf5=None,
                verified_receipts=None,
            )
            continue
        by_arm[arm] = {}
        for sid in sorted(files):
            record, sample = files[sid], samples[sid]
            deck = p / "in_files" / record["filename"]
            output = p / "out_files" / deck.with_suffix(".out").name
            try:
                h = sha256(deck)
                if h != expected_hash(snapshot, p, f"in_files/{deck.name}", arm):
                    raise ValueError("Input deck changed relative to configured input baseline")
                report["hashes"][str(deck)] = h
                meta = deck_metadata(deck)
                moistures = [
                    [l["theta_v_min"], l["theta_v_max"]] for l in sample["layers"]
                ]
                if moistures != meta["moistures"] or any(a != b for a, b in moistures):
                    raise ValueError("Emitted moisture does not equal scalar label")
                theta = moistures[0][0]
                if not 0.1 <= theta <= 0.3:
                    raise ValueError("First-layer moisture outside population")
                acquisition = {
                    k: v for k, v in meta.items() if k not in ("title", "moistures")
                }
                if common is None:
                    common = acquisition
                if common != acquisition:
                    raise ValueError("A/B acquisition or time plan differs")
                row = {
                    "arm": arm,
                    "sample_id": sid,
                    "theta": theta,
                    "thickness_m": record["layers"][0]["thickness_m"],
                    "input": str(deck),
                    "output": str(output),
                    "input_sha256": h,
                    "hdf5_valid": False,
                    "provenance_verified": False,
                }
                report["rows"].append(row)
                by_arm[arm][sid] = moistures
                stats["inputs_checked"] += 1
            except (ValueError, KeyError, OSError) as error:
                issue("input", arm, sid, str(error))
                continue
            if not output.exists():
                issue("missing_output", arm, sid, "No native output")
                continue
            stats["present"] += 1
            try:
                before = sha256(output)
                inspect_output(output, meta)
                if sha256(output) != before:
                    raise ValueError(
                        "Output changed while reading (possibly active run)"
                    )
                row["output_sha256"] = before
                report["hashes"][str(output)] = before
                row["hdf5_valid"] = True
                stats["valid_hdf5"] += 1
                receipt = receipts.get((arm, sid))
                if receipt is None:
                    row["provenance_status"] = "unverified_existing_output"
                    if policy == "require_receipts":
                        issue(
                            "provenance",
                            arm,
                            sid,
                            "No execution-time input/output completion receipt",
                        )
                    continue
                if (
                    receipt.get("status") != "completed"
                    or receipt.get("input_sha256") != h
                    or receipt.get("output_sha256") != before
                    or receipt.get("solver_version") != "3.1.7"
                    or receipt.get("backend") not in ("cpu", "cuda")
                    or not receipt.get("run_id")
                    or not receipt.get("executor_identity")
                ):
                    raise ValueError(
                        "Execution receipt is incomplete or disagrees with artifacts"
                    )
                row["provenance_verified"] = True
                row["provenance_status"] = "verified_execution_receipt"
                row["backend"] = receipt["backend"]
                stats["verified_receipts"] += 1
            except (ValueError, KeyError, OSError) as error:
                issue("output", arm, sid, str(error))
        stats["status"] = (
            "scanned" if stats["inputs_checked"] == cfg["expected_pairs"]
            else "partially_scanned_after_input_failure"
        )
    complete = {
        arm: len(by_arm.get(arm, {})) == cfg["expected_pairs"]
        for arm in ("A", "B")
    }
    if not all(complete.values()):
        report["skipped_checks"].append("A/B moisture pairing: incomplete input inspection")
    elif by_arm["A"] != by_arm["B"]:
        issue("input", None, None, "A/B moisture pairing incomplete or inconsistent")
    theta = [r["theta"] for r in report["rows"] if r["arm"] == "A"]
    if not complete["A"]:
        report["skipped_checks"].append("Unique moisture identities: incomplete A inspection")
    elif len(set(theta)) != cfg["expected_pairs"]:
        issue(
            "input",
            None,
            None,
            "Expected unique moisture identities; duplicate scenes need grouped allocation",
        )
    for arm in ("A", "B"):
        if not complete[arm]:
            report["arms"][arm]["distinct_effective_thicknesses"] = None
            report["skipped_checks"].append(f"{arm} thickness role: incomplete input inspection")
            continue
        thicknesses = {r["thickness_m"] for r in report["rows"] if r["arm"] == arm}
        report["arms"][arm]["distinct_effective_thicknesses"] = len(thicknesses)
        if (arm == "A" and len(thicknesses) <= 1) or (
            arm == "B" and len(thicknesses) != 1
        ):
            issue(
                "input",
                arm,
                None,
                "Arm roles require varying A thickness and fixed B thickness",
            )
    backends = {r.get("backend") for r in report["rows"] if r["provenance_verified"]}
    if len(backends) > 1:
        issue(
            "output",
            None,
            None,
            "Mixed execution backends require separate parity evidence/protocol review",
        )
    report["metadata"] = common
    valid = sum(r["hdf5_valid"] for r in report["rows"])
    verified = sum(r["provenance_verified"] for r in report["rows"])
    limitation = (
        "Existing input/output bytes are hashed and native metadata and signals checked. "
        "Without execution receipts, historical input-to-output linkage, successful "
        "executor completion and solver backend are not independently verified. "
        "Results are conditional on the supplied files belonging to their stated inputs."
    )
    report["provenance"] = {
        "policy": policy,
        "valid_outputs": valid,
        "verified_receipts": verified,
        "unverified_outputs": valid - verified,
        "historical_execution_verified": verified == 2 * cfg["expected_pairs"],
        "limitation": limitation if valid > verified else "",
    }
    if policy == "existing_outputs" and valid > verified:
        report["warnings"].append(f"{valid - verified} outputs: {limitation}")
    report["outputs_ready"] = report["inputs_ready"] and not report["issues"]
    input_hashes = {
        p: h
        for p, h in report["hashes"].items()
        if not p.endswith(".out") and p != cfg["receipts"]
    }
    report["input_digest"] = digest(input_hashes)
    report["data_digest"] = digest(report["hashes"])
    report["scope"] = (
        "Input integrity and HDF5 consistency; historical execution status is in provenance. "
        "Not a convergence or CPU/GPU parity qualification."
    )
    write_json(Path(cfg["run_dir"]) / "audit.json", report)
    return report


def stratify(y):
    return np.clip(
        np.searchsorted(np.linspace(0.1, 0.3, 11), y, side="right") - 1, 0, 9
    )


def make_splits(ids, theta, seed=2026):
    ids, theta = np.asarray(ids), np.asarray(theta)
    order = np.argsort(ids)
    ids, theta = ids[order], theta[order]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate pair IDs")
    indices = np.arange(len(ids))
    train, rest = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=stratify(theta)
    )
    val, test = train_test_split(
        rest, test_size=0.5, random_state=seed, stratify=stratify(theta[rest])
    )
    result = {
        name: sorted(ids[ix].tolist())
        for name, ix in (("train", train), ("validation", val), ("test", test))
    }
    train_ids = np.asarray(result["train"])
    train_y = np.asarray([theta[np.flatnonzero(ids == sid)[0]] for sid in train_ids])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    result["folds"] = [
        {"train": train_ids[a].tolist(), "heldout": train_ids[b].tolist()}
        for a, b in folds.split(train_ids, stratify(train_y))
    ]
    return result


def split(cfg, report):
    if not report["inputs_ready"]:
        raise ValueError("Cannot split unaudited/inconsistent inputs")
    rows = sorted(
        (r for r in report["rows"] if r["arm"] == "A"), key=lambda r: r["sample_id"]
    )
    result = make_splits(
        [r["sample_id"] for r in rows], [r["theta"] for r in rows], cfg["split_seed"]
    )
    result["input_digest"] = report["input_digest"]
    path = Path(cfg["run_dir"]) / "splits.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Locked split changed; use a new protocol/run directory")
    write_json(path, result)
    return result


def load_signals(report, ids, arm):
    rows = {r["sample_id"]: r for r in report["rows"] if r["arm"] == arm}
    signals = []
    for sid in ids:
        row = rows[int(sid)]
        if not row["hdf5_valid"] or sha256(row["output"]) != row["output_sha256"]:
            raise ValueError(f"Missing/changed signal {arm}/{sid}")
        with h5py.File(row["output"], "r") as f:
            signals.append(f["rxs/rx1/Ez"][:].astype(np.float64))
        if sha256(row["output"]) != row["output_sha256"]:
            raise ValueError(f"Signal changed during feature loading: {arm}/{sid}")
    return (
        np.asarray(signals),
        np.asarray([rows[int(i)]["theta"] for i in ids]),
        np.asarray([rows[int(i)]["thickness_m"] for i in ids]),
    )
