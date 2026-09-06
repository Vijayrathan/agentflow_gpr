"""Reproducible native CPU qualification measurements (never self-certification).

Run: python -m backend.qualify_3d --output-dir /path/to/results
These small homogeneous benchmarks isolate solver, normalization and boundary
effects. They do not qualify arbitrary heterogeneous production populations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
from backend.simulate import run_batch_simulation, _cli_progress
from backend.dataset_sampling.contract import solver_identity, file_digest


def metrics(reference, actual, dt):
    denominator = np.linalg.norm(reference)
    scale = np.max(np.abs(reference))
    spectrum, other = np.fft.rfft(reference), np.fft.rfft(actual)
    frequencies = np.fft.rfftfreq(len(reference), dt)
    selected = (frequencies >= .481623e9) & (frequencies <= 1.636567e9) & (np.abs(spectrum) > np.max(np.abs(spectrum)) * .01)
    phases = np.angle(other[selected] * np.conj(spectrum[selected]))
    return {"normalized_l2_error": float(np.linalg.norm(actual - reference) / denominator) if denominator else None,
            "absolute_max_error": float(np.max(np.abs(actual - reference))),
            "relative_peak_amplitude_error": float(abs(np.max(np.abs(actual)) - scale) / scale) if scale else None,
            "absolute_peak_time_error_s": float(abs(np.argmax(np.abs(actual)) - np.argmax(np.abs(reference))) * dt) if scale else None,
            "analysis_band_weighted_phase_rms_rad": float(np.sqrt(np.average(phases**2, weights=np.abs(spectrum[selected])**2))) if selected.any() else None}


def make_deck(name, *, dx=.004, extent=(.28, .28, .28), pml=6, axis="z",
              source=None, receiver=None, window=6e-9, waveform="ricker", material=False, target=None):
    center = [value / 2 for value in extent]
    source = source or center
    receiver = receiver or [center[0] + .032, center[1] + .024, center[2] + .016]
    source, receiver = list(source), list(receiver)
    # Odd refinement admits the same physical E-component point on both Yee
    # lattices. Shift integer anchors so the source and selected E receiver
    # retain their coarse-grid physical positions relative to targets/interfaces.
    source["xyz".index(axis)] += (.004 - dx) / 2
    receiver["xyz".index(axis)] += (.004 - dx) / 2
    # Constant dipole current moment, compensating native dl=dx.
    amplitude = .004 / dx
    def coords(values):
        return " ".join(format(v, ".17g") for v in values)
    lines = [f"#title: {name}", "#num_threads: 1", f"#domain: {coords(extent)}",
             f"#dx_dy_dz: {coords([dx] * 3)}", f"#time_window: {window:.17g}",
             f"#pml_cells: {pml} {pml} {pml} {pml} {pml} {pml}",
             f"#waveform: {waveform} {amplitude:.17g} 1e9 pulse"]
    if material:
        # Homogeneous planar reference, not a sampled Peplinski medium.
        lines += ["#material: 4 0 1 0 dielectric",
                  f"#box: 0 0 0 {extent[0]} {center[1] - .032} {extent[2]} dielectric n"]
    if target:
        lines.append(target)
    lines += [f"#hertzian_dipole: {axis} {coords(source)} pulse", f"#rx: {coords(receiver)}"]
    return "\n".join(lines) + "\n"


def read(path):
    with h5py.File(path, "r") as f:
        return {"dt": float(f.attrs["dt"]), "n": int(f.attrs["Iterations"]),
                "spacing": tuple(f.attrs["dx_dy_dz"]), "counts": [int(v) for v in f.attrs["nx_ny_nz"]],
                "source_m": [float(v) for v in f["srcs/src1"].attrs["Position"]],
                "receiver_m": [float(v) for v in f["rxs/rx1"].attrs["Position"]],
                "relative_rx": f["rxs/rx1"].attrs["Position"] - f["srcs/src1"].attrs["Position"],
                "fields": {c: np.asarray(f["rxs/rx1"][c], dtype=float) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}}


def refinement_controls(coarse, fine, coarse_pml, fine_pml):
    """Verify physical invariants from executed coordinates, not deck strings."""
    result = {}
    for name in ("source", "receiver"):
        positions = []
        for entry in (coarse, fine):
            point = np.asarray(entry[name + "_m"]) + np.asarray([0, 0, entry["spacing"][2] / 2])
            positions.append(point)
        if not np.allclose(*positions, rtol=0, atol=1e-12):
            raise ValueError(f"Refinement changes the physical {name} Ez location")
        result[name + "_Ez_point_m"] = positions[0].tolist()
    widths = [np.asarray(entry["spacing"]) * pml for entry, pml in ((coarse, coarse_pml), (fine, fine_pml))]
    if not np.allclose(*widths, rtol=0, atol=1e-12):
        raise ValueError("Refinement changes physical PML thickness")
    result["pml_thickness_xyz_m"] = widths[0].tolist()
    return result


def run_suite(output_dir, analyze_only=False, refinement_only=False):
    output_dir = Path(output_dir).resolve()
    input_dir = output_dir / "in_files"
    input_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "base": {}, "fine": {"dx": .004 / 3, "pml": 18},
        "long": {"window": 9e-9}, "pml": {"pml": 10},
        "wide_x": {"extent": (.36, .28, .28)},
        "wide_y": {"extent": (.28, .36, .28)},
        "wide_z": {"extent": (.28, .28, .36)},
        "sym_x": {"axis": "x"},
        "sym_z": {"axis": "z", "receiver": [.156, .164, .172]},
        "reciprocal": {"source": [.172, .164, .156], "receiver": [.14, .14, .14]},
        "analytical": {"waveform": "gaussianprime"},
        "layered": {"material": True},
        "layered_fine": {"material": True, "dx": .004 / 3, "pml": 18},
        "finite_box": {"target": "#box: .112 .088 .112 .144 .108 .144 pec n"},
        "finite_box_fine": {"dx": .004 / 3, "pml": 18, "target": "#box: .112 .088 .112 .144 .108 .144 pec n"},
        "finite_cylinder": {"target": "#cylinder: .112 .100 .096 .112 .100 .160 .012 pec n"},
        "finite_cylinder_fine": {"dx": .004 / 3, "pml": 18, "target": "#cylinder: .112 .100 .096 .112 .100 .160 .012 pec n"},
    }
    for name, options in cases.items():
        path = input_dir / (name + ".in")
        if analyze_only or refinement_only and name != "fine" and not name.endswith("_fine"):
            if path.read_text() != make_deck(name, **options):
                raise ValueError("Benchmark input changed; rerun the solver before analyzing")
        else:
            path.write_text(make_deck(name, **options))
    if not analyze_only:
        result = run_batch_simulation(input_dir, output_dir / "out_files", gpu=False,
                                      gpu_ids=[], workers=1, progress=_cli_progress,
                                      filenames=[name + ".in" for name in cases if not refinement_only or name == "fine" or name.endswith("_fine")])
        if result["failed"]:
            (output_dir / "qualification_failed.json").write_text(json.dumps(result, indent=2))
            raise RuntimeError(f"{result['failed']} native benchmark(s) failed; see qualification_failed.json")
    data = {name: read(output_dir / "out_files" / (name + ".out")) for name in cases}
    for name, entry in data.items():
        if any(len(v) != entry["n"] or not np.isfinite(v).all() for v in entry["fields"].values()):
            raise ValueError(f"Incomplete or nonfinite native benchmark output: {name}")
    report = {"solver": solver_identity(), "status": "measured_unqualified",
              "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
              "benchmark_definition_sha256": file_digest(Path(__file__)),
              "scene_scope": "small free-space/homogeneous planar and finite PEC fixtures; not production heterogeneous soil",
              "component_time_convention": "Native receiver iteration indices; E/H retain native staggered updates without temporal collocation. The analytical comparison preserves the bundled reference's original sampling.",
              "normalization": "fixed 0.004 m native dipole length times unit amplitude (amplitude scales inversely with dx)",
              "tolerances": None, "measurements": {},
              "limitations": ["No experimental/lab calibration", "No CUDA parity measurement on this CPU host",
                              "No claimed convergence coverage for random Peplinski media or population extrema",
                              "Benchmark PEC box thickness and cylinder diameter span only 5 and 6 coarse cells; these are coarse numerical studies outside the production ten-cell target policy",
                              "Refinement uses Ez with matched physical source/Ez receiver locations and fixed dipole moment; other field components are not collocated across meshes",
                              "Only two meshes: convergence trend is measured, not an asymptotic convergence proof"]}
    report["metric_definitions"] = {
        "normalized_l2_error": "norm(actual-reference)/norm(reference); refinement uses interpolated fine fields as reference",
        "relative_peak_amplitude_error": "absolute difference in maximum absolute field, divided by reference maximum",
        "absolute_peak_time_error_s": "difference of maximum-absolute-field sample times; not an independent first-arrival estimator",
        "analysis_band_weighted_phase_rms_rad": "wrapped phase RMS weighted by reference spectral power, restricted to 0.481623-1.636567 GHz and reference amplitude above 1% of its spectral maximum; this is the Ricker useful band but only a comparison band for the analytical Gaussian-prime pulse",
        "absolute_max_error": "maximum pointwise absolute difference, in V/m for E and A/m for H",
        "zero_reference": "undefined normalized errors, phase and peak time are null; absolute error remains meaningful"}
    report["refinement_policy"] = "3:1 mesh ratio; constant physical E-source/Ez-receiver locations, dipole moment, geometry and PML thickness; homogeneous media"
    report["executed_refinement_controls"] = {}
    measured = report["measurements"]
    base = data["base"]
    for name in ("wide_x", "wide_y", "wide_z", "pml", "reciprocal"):
        measured[name] = metrics(base["fields"]["Ez"], data[name]["fields"]["Ez"][:base["n"]], base["dt"])
    for name in ("base", "layered", "finite_box", "finite_cylinder"):
        coarse, fine = data[name], data["fine" if name == "base" else name + "_fine"]
        report["executed_refinement_controls"][name] = refinement_controls(coarse, fine, 6, 18)
        interpolated = np.interp(np.arange(coarse["n"]) * coarse["dt"],
                                 np.arange(fine["n"]) * fine["dt"], fine["fields"]["Ez"])
        measured[name + "_refinement"] = metrics(interpolated, coarse["fields"]["Ez"], coarse["dt"])
        if name != "base":
            scattered_coarse = coarse["fields"]["Ez"] - base["fields"]["Ez"]
            scattered_fine = fine["fields"]["Ez"] - data["fine"]["fields"]["Ez"]
            interpolated_scattered = np.interp(np.arange(coarse["n"]) * coarse["dt"], np.arange(fine["n"]) * fine["dt"], scattered_fine)
            measured[name + "_scattered_refinement"] = metrics(interpolated_scattered, scattered_coarse, coarse["dt"])
    long = data["long"]["fields"]["Ez"]
    measured["longer_window"] = {**metrics(base["fields"]["Ez"], long[:base["n"]], base["dt"]),
                                 "omitted_tail_energy_fraction": float(np.sum(long[base["n"]:]**2) / np.sum(long**2))}
    measured["horizontal_axis_exchange"] = metrics(data["sym_x"]["fields"]["Ex"], data["sym_z"]["fields"]["Ez"], base["dt"])
    reference_path = Path(__file__).resolve().parents[1] / "gprMax/tests/analytical_solutions.py"
    report["analytical_reference_sha256"] = file_digest(reference_path)
    spec = importlib.util.spec_from_file_location("native_analytical_reference", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    analytical = data["analytical"]
    exact = reference.hertzian_dipole_fs(analytical["n"], analytical["dt"], analytical["spacing"], analytical["relative_rx"])
    measured["analytical_free_space"] = {component: metrics(exact[:, i], analytical["fields"][component], analytical["dt"])
                                         for i, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))}
    report["native_runs"] = {name: {"iterations": entry["n"], "dt_s": entry["dt"],
                                     "cell_counts": entry["counts"], "cell_spacings_m": entry["spacing"],
                                     "source_anchor_m": entry["source_m"], "receiver_anchor_m": entry["receiver_m"],
                                     "input_sha256": file_digest(input_dir / (name + ".in")),
                                     "output_sha256": file_digest(output_dir / "out_files" / (name + ".out")),
                                     "all_fields_finite": all(np.isfinite(v).all() for v in entry["fields"].values())} for name, entry in data.items()}
    path = output_dir / "qualification_measurements.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analyze-only", action="store_true", help="Reanalyze existing runs after verifying their input decks")
    parser.add_argument("--refinement-only", action="store_true", help="Rerun fine meshes only; all existing coarse input decks must match")
    args = parser.parse_args()
    print(run_suite(args.output_dir, args.analyze_only, args.refinement_only))
