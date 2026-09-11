"""Versioned configuration and atomic, hash-bound artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path

from . import PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        temporary = f.name
    os.replace(temporary, path)


def resolve(path):
    p = Path(path)
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def load_config(path=DEFAULT_CONFIG):
    cfg = read_json(path)
    required = set(read_json(DEFAULT_CONFIG)) - {"provenance_policy"}
    if not required <= set(cfg) or set(cfg) - required - {"provenance_policy"}:
        raise ValueError("Configuration keys must match config.json plus optional provenance_policy")
    cfg.setdefault("provenance_policy", "require_receipts")
    if cfg["provenance_policy"] not in ("require_receipts", "existing_outputs"):
        raise ValueError("provenance_policy must be require_receipts or existing_outputs")
    if set(cfg["datasets"]) != {"A", "B"}:
        raise ValueError("Exactly A and B are required")
    if cfg["expected_pairs"] != 1000 or cfg["split_seed"] != 2026:
        raise ValueError("Population or split changes require a new protocol")
    if cfg["fit_seeds"] != [11, 22, 33, 44, 55] or cfg["bootstrap_samples"] != 10000:
        raise ValueError("Seeds/bootstrap changes require a new protocol")
    if cfg["n_jobs"] < 1 or cfg["torch_threads"] < 1:
        raise ValueError("Thread counts must be positive")
    if cfg["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda; no silent fallback")
    for key in ("input_hashes", "run_dir"):
        cfg[key] = str(resolve(cfg[key]))
    cfg["datasets"] = {k: str(resolve(v)) for k, v in cfg["datasets"].items()}
    if cfg["receipts"]:
        cfg["receipts"] = str(resolve(cfg["receipts"]))
    run = Path(cfg["run_dir"])
    for dataset in cfg["datasets"].values():
        if run == Path(dataset) or Path(dataset) in run.parents:
            raise ValueError("Run directory must be outside dataset directories")
    return cfg


def environment(device="cpu"):
    versions = {}
    for package in (
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "h5py",
        "joblib",
        "matplotlib",
    ):
        versions[package] = importlib.metadata.version(package)
    hardware = {"logical_cpus": os.cpu_count(), "processor": platform.processor()}
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        hardware["torch_cuda_version"] = torch.version.cuda
        hardware["visible_gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        hardware["training_device_index"] = 0
    return {
        "protocol": PROTOCOL,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": hardware,
        "versions": versions,
    }


def source_digest():
    base = Path(__file__).parent
    return digest({p.name: sha256(p) for p in sorted(base.glob("*.py"))})


def bind_run(cfg, data_digest):
    run = Path(cfg["run_dir"])
    identity = {
        "config": cfg,
        "data_digest": data_digest,
        "source_digest": source_digest(),
        "environment": environment(cfg["device"]),
    }
    path = run / "identity.json"
    if path.exists() and read_json(path) != identity:
        raise ValueError(
            "Run inputs, code, configuration or environment changed; use a new run directory"
        )
    if not path.exists():
        write_json(path, identity)
    return run


def register(run, paths):
    index = run / "artifacts.json"
    hashes = read_json(index) if index.exists() else {}
    for path in paths:
        path = Path(path)
        hashes[str(path.relative_to(run))] = sha256(path)
    write_json(index, hashes)


def verify_artifacts(run, paths):
    hashes = read_json(run / "artifacts.json")
    for path in paths:
        key = str(Path(path).relative_to(run))
        if hashes.get(key) != sha256(path):
            raise ValueError(f"Missing or changed registered artifact: {key}")
