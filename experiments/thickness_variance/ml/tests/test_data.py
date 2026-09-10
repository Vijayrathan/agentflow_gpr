from pathlib import Path

import h5py
import numpy as np
import pytest

from experiments.thickness_variance.ml.common import read_json, sha256, write_json
from experiments.thickness_variance.ml.data import (
    audit,
    deck_metadata,
    make_splits,
)


@pytest.fixture
def dataset_fixture(tmp_path):
    hashes, receipts = {}, []
    cfg = {
        "datasets": {},
        "input_hashes": str(tmp_path / "input_hashes.json"),
        "receipts": str(tmp_path / "receipts.json"),
        "run_dir": str(tmp_path / "run"),
        "expected_pairs": 2,
    }
    for arm in ("A", "B"):
        p = tmp_path / arm
        cfg["datasets"][arm] = str(p)
        (p / "in_files").mkdir(parents=True)
        (p / "out_files").mkdir()
        files, samples = [], []
        for sid, theta in [(1, 0.15), (2, 0.25)]:
            deck = p / "in_files" / f"{arm}_{sid}.in"
            deck.write_text(
                f"#title: {arm}_{sid}\n#domain: .3 .3 .002\n#dx_dy_dz: .002 .002 .002\n"
                "#time_window: 1e-9\n#waveform: ricker 1 750000000 wave\n"
                f"#soil_peplinski: .4 .15 1.5 2.65 {theta} {theta} soil\n"
                "#hertzian_dipole: z .14 .21 0 wave\n#rx: .16 .21 0\n"
            )
            meta = deck_metadata(deck)
            out = p / "out_files" / f"{arm}_{sid}.out"
            with h5py.File(out, "w") as f:
                f.attrs.update(
                    Iterations=meta["iterations"],
                    nrx=1,
                    nsrc=1,
                    gprMax="3.1.7",
                    Title=meta["title"],
                    dt=meta["dt_s"],
                    nx_ny_nz=meta["counts"],
                    dx_dy_dz=meta["spacing"],
                    rxsteps=[0, 0, 0],
                    srcsteps=[0, 0, 0],
                )
                rx = f.create_group("rxs/rx1")
                rx.attrs["Position"] = meta["receiver"]
                src = f.create_group("srcs/src1")
                src.attrs.update(Position=meta["source"], Type="HertzianDipole")
                for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                    rx.create_dataset(c, data=np.zeros(meta["iterations"]))
            files.append(
                {
                    "sample_id": sid,
                    "filename": deck.name,
                    "layers": [
                        {"thickness_m": 0.21 + (sid * 0.001 if arm == "A" else 0)}
                    ],
                }
            )
            samples.append(
                {
                    "sample_id": sid,
                    "layers": [{"theta_v_min": theta, "theta_v_max": theta}],
                }
            )
            hashes[str(deck)] = sha256(deck)
            receipts.append(
                {
                    "arm": arm,
                    "sample_id": sid,
                    "status": "completed",
                    "input_sha256": sha256(deck),
                    "output_sha256": sha256(out),
                    "solver_version": "3.1.7",
                    "backend": "cpu",
                    "run_id": "fixture-only",
                    "executor_identity": "test fixture: not a scientific simulation",
                }
            )
        for name, content in {
            "emitted_files.json": {"files": files, "errors": []},
            "sampled_layers.json": {
                "samples": samples,
                "sampling": {"moisture_sampling": "uniform_per_sample"},
            },
            "derived_layers.json": {},
            "global_derive.json": {},
        }.items():
            write_json(p / name, content)
            hashes[str(p / name)] = sha256(p / name)
    write_json(cfg["input_hashes"], hashes)
    write_json(cfg["receipts"], {"schema_version": 1, "runs": receipts})
    return cfg


def test_receipted_output_admission(dataset_fixture):
    result = audit(dataset_fixture)
    assert result["inputs_ready"] and result["outputs_ready"]
    assert len(result["rows"]) == 4
    assert result["arms"]["A"]["verified_receipts"] == 2


def test_outputs_without_receipts_do_not_claim_provenance(dataset_fixture):
    dataset_fixture["receipts"] = None
    result = audit(dataset_fixture)
    assert result["inputs_ready"] and not result["outputs_ready"]
    assert result["arms"]["A"]["valid_hdf5"] == 2
    assert all(i["kind"] == "provenance" for i in result["issues"])


def test_changed_input_blocks_admission(dataset_fixture):
    deck = Path(dataset_fixture["datasets"]["A"]) / "in_files/A_1.in"
    deck.write_text(deck.read_text() + "\n")
    result = audit(dataset_fixture)
    assert not result["inputs_ready"] and not result["outputs_ready"]


@pytest.mark.parametrize(
    "fault",
    ["nonfinite", "length", "time", "identity", "position", "missing_component"],
)
def test_bad_output_blocks_admission(dataset_fixture, fault):
    p = Path(dataset_fixture["datasets"]["A"]) / "out_files/A_1.out"
    with h5py.File(p, "a") as f:
        if fault == "nonfinite":
            f["rxs/rx1/Ez"][10] = np.nan
        if fault == "length":
            del f["rxs/rx1/Ez"]
            f["rxs/rx1"].create_dataset("Ez", data=[1, 2])
        if fault == "time":
            f.attrs["dt"] = 1e-5
        if fault == "identity":
            f.attrs["Title"] = "another sample"
        if fault == "position":
            f["rxs/rx1"].attrs["Position"] = [0, 0, 0]
        if fault == "missing_component":
            del f["rxs/rx1/Hy"]
    result = audit(dataset_fixture)
    assert not result["outputs_ready"]
    assert result["arms"]["A"]["valid_hdf5"] == 1


def test_changed_finite_output_fails_receipt(dataset_fixture):
    p = Path(dataset_fixture["datasets"]["A"]) / "out_files/A_1.out"
    with h5py.File(p, "a") as f:
        f["rxs/rx1/Ez"][10] = 9
    result = audit(dataset_fixture)
    assert result["arms"]["A"]["valid_hdf5"] == 2
    assert not result["outputs_ready"]


def test_missing_output_blocks(dataset_fixture):
    (Path(dataset_fixture["datasets"]["B"]) / "out_files/B_1.out").unlink()
    result = audit(dataset_fixture)
    assert not result["outputs_ready"] and result["arms"]["B"]["present"] == 1


def refresh_fixture_hash(cfg, path):
    hashes = read_json(cfg["input_hashes"])
    hashes[str(path)] = sha256(path)
    write_json(cfg["input_hashes"], hashes)


def test_manifest_order_does_not_associate_wrong_signals(dataset_fixture):
    before = audit(dataset_fixture)
    for folder in dataset_fixture["datasets"].values():
        for name, key in (
            ("emitted_files.json", "files"),
            ("sampled_layers.json", "samples"),
        ):
            path = Path(folder) / name
            data = read_json(path)
            data[key].reverse()
            write_json(path, data)
            refresh_fixture_hash(dataset_fixture, path)
    after = audit(dataset_fixture)
    assert after["outputs_ready"]
    assert after["rows"] == before["rows"]


def test_duplicate_manifest_id_rejected(dataset_fixture):
    path = Path(dataset_fixture["datasets"]["A"]) / "emitted_files.json"
    data = read_json(path)
    data["files"][1]["sample_id"] = 1
    write_json(path, data)
    refresh_fixture_hash(dataset_fixture, path)
    assert not audit(dataset_fixture)["inputs_ready"]


def test_broken_moisture_pair_rejected_even_if_locally_consistent(dataset_fixture):
    folder = Path(dataset_fixture["datasets"]["B"])
    path = folder / "sampled_layers.json"
    data = read_json(path)
    data["samples"][0]["layers"][0].update(theta_v_min=0.16, theta_v_max=0.16)
    write_json(path, data)
    refresh_fixture_hash(dataset_fixture, path)
    deck = folder / "in_files/B_1.in"
    deck.write_text(deck.read_text().replace("0.15 0.15 soil", "0.16 0.16 soil"))
    refresh_fixture_hash(dataset_fixture, deck)
    result = audit(dataset_fixture)
    assert not result["inputs_ready"]
    assert any("pairing" in issue["message"] for issue in result["issues"])


def test_mixed_backends_not_silently_compared(dataset_fixture):
    ledger = read_json(dataset_fixture["receipts"])
    ledger["runs"][-1]["backend"] = "cuda"
    write_json(dataset_fixture["receipts"], ledger)
    result = audit(dataset_fixture)
    assert not result["outputs_ready"]
    assert any(
        "Mixed execution backends" in issue["message"] for issue in result["issues"]
    )


def test_swapped_or_duplicated_dataset_roles_block(dataset_fixture):
    dataset_fixture["datasets"]["B"] = dataset_fixture["datasets"]["A"]
    result = audit(dataset_fixture)
    assert not result["inputs_ready"]
    assert any("Arm roles" in issue["message"] for issue in result["issues"])


def test_split_exact_disjoint_reproducible_and_order_independent():
    ids = np.arange(1, 1001)
    theta = np.linspace(0.1001, 0.2999, 1000)
    result = make_splits(ids, theta)
    assert [len(result[k]) for k in ("train", "validation", "test")] == [700, 150, 150]
    assert result == make_splits(ids[::-1], theta[::-1])
    sets = [set(result[k]) for k in ("train", "validation", "test")]
    assert (
        len(set.union(*sets)) == 1000
        and not sets[0] & sets[1]
        and not sets[0] & sets[2]
        and not sets[1] & sets[2]
    )
    held = []
    for fold in result["folds"]:
        assert not set(fold["train"]) & set(fold["heldout"])
        assert set(fold["train"]) | set(fold["heldout"]) == sets[0]
        held.extend(fold["heldout"])
    assert sorted(held) == result["train"]
