"""Read-only audit of delivered A/B pilots; writes evidence beside this script."""
import collections
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "backend", ROOT / "gprMax"):
    sys.path.insert(0, str(p))
import numpy as np
from backend.schema import ExtractedLayers
from dataset_sampling.layer_sampler import sample_layers
from gprMax.grid import FDTDGrid
from gprMax.input_cmds_file import check_cmd_names
from gprMax.input_cmds_multiuse import process_multicmds
from gprMax.utilities import round_value

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--a", type=Path, default=ROOT / "dataset/vijay/moisture_A_pilot__e865f98b")
parser.add_argument("--b", type=Path, default=ROOT / "dataset/vijay/moisture_B_pilot__bb5f0247")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--a-min", type=float, default=0.18)
parser.add_argument("--a-max", type=float, default=0.24)
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "pilot_audit")
args = parser.parse_args()
assert args.n > 0 and 0 < args.a_min <= args.a_max
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
PATHS = {"A": args.a.resolve(), "B": args.b.resolve()}
ALLOWED = {"#title", "#num_threads", "#domain", "#dx_dy_dz", "#time_window",
           "#pml_cells", "#waveform", "#soil_peplinski", "#fractal_box",
           "#hertzian_dipole", "#rx", "#geometry_view"}
state, results, hashes, rows = {}, {}, {}, []


def load(path):
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


for arm, folder in PATHS.items():
    sampled = load(folder / "sampled_layers.json")
    derived = load(folder / "derived_layers.json")
    plan = load(folder / "global_derive.json")
    emission = load(folder / "emitted_files.json")
    assert sampled["num_samples"] == emission["n_written"] == args.n
    assert not emission["errors"] and not sampled["warnings"]
    samples = {s["sample_id"]: s for s in sampled["samples"]}
    files = {s["sample_id"]: s for s in emission["files"]}
    materials = {s["sample_id"]: s for s in derived["samples"]}
    assert len(sampled["samples"]) == len(emission["files"]) == len(derived["samples"]) == args.n
    assert set(samples) == set(files) == set(materials) == set(range(1, args.n + 1))
    assert {f.name for f in (folder / "in_files").glob("*.in")} == {f["filename"] for f in files.values()}
    provenance = sampled["sampling"]
    assert provenance["moisture_seed"] == 42
    assert provenance["moisture_sampling"] == "uniform_per_sample"
    assert provenance["moisture_stream_version"] == "uniform-moisture-v1"
    requested = ExtractedLayers.model_validate(provenance["requested_layers"])
    assert requested.soil_depth_m == 2.0 and requested.num_layers == 3
    assert requested.layers[0].thickness_m_min == (args.a_min if arm == "A" else 0.21)
    assert requested.layers[0].thickness_m_max == (args.a_max if arm == "A" else 0.21)
    assert (requested.layers[0].theta_v_min, requested.layers[0].theta_v_max) == (0.1, 0.3)
    repeated, _ = sample_layers(requested, args.n, seed=provenance["geometry_seed"],
                                moisture_sampling=provenance["moisture_sampling"], moisture_seed=42)
    assert [s.model_dump() for s in repeated] == sampled["samples"]
    records = {}
    for sid in sorted(samples):
        record, sample = files[sid], samples[sid]
        path = folder / "in_files" / record["filename"]
        assert Path(record["path"]).resolve() == path.resolve()
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        lines = [s.strip() for s in path.read_text().splitlines() if s.strip()]
        assert all(s.split(":", 1)[0] in ALLOWED for s in lines)
        single, multi, geometry = check_cmd_names([s + "\n" for s in lines])
        commands = collections.defaultdict(list)
        for text in lines:
            key, value = text.split(":", 1)
            commands[key].append(value.strip())
        assert len(commands["#soil_peplinski"]) == len(commands["#fractal_box"]) == 3
        assert len(commands["#waveform"]) == len(commands["#hertzian_dipole"]) == len(commands["#rx"]) == 1
        waveform_tokens = commands["#waveform"][0].split()
        assert len(waveform_tokens) == 4
        assert waveform_tokens[0] == "ricker" and list(map(float, waveform_tokens[1:3])) == [1, 750000000]
        source_tokens = commands["#hertzian_dipole"][0].split()
        assert len(source_tokens) == 5 and source_tokens[4] == waveform_tokens[3]
        assert commands["#pml_cells"] == ["10 10 0 10 10 0"]
        spacing = np.array(list(map(float, commands["#dx_dy_dz"][0].split())))
        assert len(set(spacing)) == 1
        domain = np.array(list(map(float, commands["#domain"][0].split())))
        counts = np.array([round_value(d / h) for d, h in zip(domain, spacing)])
        assert counts[2] == 1 and (counts[:2] > 20).all()
        dt = round_value(float(1 / (299792458 * np.sqrt(2 / spacing[0] ** 2))), decimalplaces=27)
        duration = float(commands["#time_window"][0])
        iterations = math.ceil(duration / dt) + 1
        boxes = [s.split() for s in commands["#fractal_box"]]
        edges = []
        for i, (box, layer, label) in enumerate(zip(boxes, sample["layers"], record["layers"], strict=True)):
            xyz = np.array(list(map(float, box[:6])))
            cell = np.array([round_value(xyz[j] / spacing[j % 3]) for j in range(6)])
            assert tuple(cell[[0, 2, 3, 5]]) == (0, 0, counts[0], 1)
            assert 0 <= cell[1] < cell[4] < counts[1]
            assert int(box[10]) == 50 and int(box[-1]) == sid * 1000 + i + 1
            assert box[11] == layer["name"]
            assert cell[4] - cell[1] >= 3
            if edges:
                assert cell[4] == edges[-1][0]
            edges.append((int(cell[1]), int(cell[4])))
            effective = (cell[4] - cell[1] - (10 if i == 2 else 0)) * spacing[0]
            assert abs(effective - label["thickness_m"]) < 1e-8
            assert (layer["sand_pct"], layer["clay_pct"], layer["silt_pct"],
                    layer["bulk_density_gcm3"], layer["particle_density_gcm3"]) == (40, 15, 45, 1.5, 2.65)
            theta = layer["theta_v_min"]
            assert theta == layer["theta_v_max"] and 0 < theta <= 0.30 < 1 - 1.5 / 2.65
            assert (0.10 <= theta <= 0.30) if i == 0 else theta == (0.05 if i == 1 else 0.08)
        assert edges[-1][0] == 0
        assert not sample["targets"]
        grid = FDTDGrid()
        grid.messages = False
        grid.mode = "2D TMz"
        grid.dx, grid.dy, grid.dz = spacing
        grid.nx, grid.ny, grid.nz = map(int, counts)
        grid.dt, grid.timewindow, grid.iterations = dt, duration, iterations
        grid.pmlthickness["z0"] = grid.pmlthickness["zmax"] = 0
        process_multicmds(multi, grid)
        assert len(grid.mixingmodels) == 3
        assert len(grid.hertziandipoles) == len(grid.rxs) == 1
        src, rx = grid.hertziandipoles[0], grid.rxs[0]
        assert src.polarisation == "z"
        for obj in (src, rx):
            assert 25 <= obj.xcoord <= grid.nx - 25
            assert edges[0][1] < obj.ycoord <= grid.ny - 25
            assert obj.zcoord == 0
        for soil, layer, saved in zip(grid.mixingmodels, sample["layers"], materials[sid]["layers"], strict=True):
            assert tuple(soil.mu) == (layer["theta_v_min"], layer["theta_v_max"])
            start = len(grid.materials)
            soil.calculate_debye_properties(50, grid, soil.ID + "_fbox")
            built = grid.materials[start:]
            vectors = np.array([[m.er, m.se, *m.deltaer, *m.tau] for m in built])
            assert np.isfinite(vectors).all() and (vectors >= 0).all()
            assert np.array_equal(vectors, np.broadcast_to(vectors[0], vectors.shape))
            assert all(tau > dt for m in built for tau in m.tau)
            assert built[0].calculate_er(750e6).real == saved["eps_r_dry"] == saved["eps_r_wet"]
            assert built[0].se == saved["sigma_dry"] == saved["sigma_wet"]
        header = {k: v for k, v in commands.items() if k not in ("#title", "#num_threads", "#geometry_view", "#fractal_box", "#soil_peplinski")}
        raw_header = dict(header)
        # Resolve the sole waveform reference before comparing physics across arms.
        # Alias differences are harmless; numeric pulse/source differences are not.
        header["#waveform"] = [" ".join(waveform_tokens[:3] + ["<waveform>"])]
        header["#hertzian_dipole"] = [" ".join(source_tokens[:4] + ["<waveform>"])]
        records[sid] = dict(header=header, moisture=[l["theta_v_min"] for l in sample["layers"]],
                            raw_header=raw_header, waveform_identifier=waveform_tokens[3],
                            material_commands=commands["#soil_peplinski"],
                            thickness_cells=edges[0][1] - edges[0][0], edges=edges,
                            theta=sample["layers"][0]["theta_v_min"],
                            requested_thickness=sample["layers"][0]["thickness_m"])
        rows.append(dict(arm=arm, sample_id=sid, theta=records[sid]["theta"],
                         requested_thickness_m=records[sid]["requested_thickness"],
                         effective_thickness_m=records[sid]["thickness_cells"] * float(spacing[0])))
    state[arm] = dict(records=records, plan=plan, derived=derived)
    theta = [r["theta"] for r in records.values()]
    thickness = [r["thickness_cells"] * float(spacing[0]) for r in records.values()]
    results[arm] = dict(n=args.n, unique_moisture=len(set(theta)), moisture_min=min(theta), moisture_max=max(theta),
                        waveform_identifiers=sorted({r["waveform_identifier"] for r in records.values()}),
                        unique_effective_thicknesses=len(set(thickness)), thickness_min_m=min(thickness),
                        thickness_max_m=max(thickness), shape=counts.tolist(), parsed_dt_s=dt, iterations=iterations,
                        moisture_decile_counts=np.histogram(theta, bins=np.linspace(0.1, 0.3, 11))[0].tolist(),
                        moisture_thickness_pearson=float(np.corrcoef(theta, thickness)[0, 1]) if arm == "A" else None,
                        existing_signal_outputs=len(list(folder.rglob("*.out"))))

assert state["A"]["plan"] == state["B"]["plan"]
assert state["A"]["derived"] == state["B"]["derived"]
same_geometry = []
for sid in range(1, args.n + 1):
    a, b = state["A"]["records"][sid], state["B"]["records"][sid]
    assert a["moisture"] == b["moisture"]
    assert a["header"] == b["header"] == state["A"]["records"][1]["header"]
    assert a["material_commands"] == b["material_commands"]
    if a["edges"] == b["edges"]:
        same_geometry.append(sid)
assert len({tuple(map(tuple, r["edges"])) for r in state["B"]["records"].values()}) == 1
summary = dict(status="PASS_INPUT_AUDIT", arms=results, exact_paired_moisture=True,
               exact_paired_material_commands=True,
               exact_common_numerical_and_acquisition_commands=all(state["A"]["records"][sid]["raw_header"] == state["B"]["records"][sid]["raw_header"] for sid in range(1, args.n + 1)),
               equivalent_common_numerical_and_acquisition_commands=True,
               matched_ids_with_identical_geometry=same_geometry,
               source_paths={arm: str(p) for arm, p in PATHS.items()},
               expected_a_thickness_range=[args.a_min, args.a_max],
               scope=f"All {2 * args.n} input files and native material/source parsing; no field solves in this stage")
(OUT / "input_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
(OUT / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
(OUT / "sample_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(summary, indent=2))
