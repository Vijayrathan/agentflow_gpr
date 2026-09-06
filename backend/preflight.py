"""Native model-build parity and output provenance, without modifying gprMax.

The build hook runs in an isolated simulation worker at the native geometry
export point, after material/edge construction and before any field updates.
An exception blocks solving. Hooks are always restored, including on failure.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from backend.dataset_sampling.contract import digest, file_digest
from backend.dataset_sampling.numerics import cell_index


def verify_contract(contract, scene=None):
    if contract.get("digest") != digest({k: v for k, v in contract.items() if k != "digest"}):
        raise ValueError("Dataset contract digest mismatch")
    if scene is not None:
        if scene.get("contract_digest") != contract["digest"] or scene.get("digest") != digest({k: v for k, v in scene.items() if k != "digest"}):
            raise ValueError("Resolved scene digest/contract mismatch")


def validate_deck_contract(text, contract, scene):
    from gprMax.input_cmds_file import check_cmd_names
    from backend.dataset_sampling.emit import serialize_scene
    from backend.schema import DatasetConfig, ExtractedWaveform, GlobalDerived
    verify_contract(contract, scene)
    # Native command-name/mandatory-command validation is safe (no script exec).
    check_cmd_names([line for line in text.splitlines(keepends=True) if line.startswith("#")])
    expected = serialize_scene(scene, GlobalDerived.model_validate(contract["grid"]),
                               DatasetConfig.model_validate(contract["requested"]["dataset_config"]),
                               ExtractedWaveform.model_validate(contract["requested"]["waveform"]))
    # Thread/output location are deployment settings; every physical command must
    # equal the canonical serialization, including precedence and identifiers.
    def physical(value):
        return [line.strip() for line in value.splitlines() if line.strip() and
                not line.startswith(("#output_dir:", "#num_threads:"))]
    if physical(text) != physical(expected):
        raise ValueError("Emitted input differs from resolved physical scene")


def check_native_grid(G, contract, scene):
    import numpy as np
    from backend.dataset_sampling.peplinski_derive import material_coefficients
    grid = contract["grid"]
    mode = "3D" if contract["dimensionality"] == "3D" else "2D TMz"
    if G.mode != mode or [G.nx, G.ny, G.nz] != [grid["nx"], grid["ny"], grid["nz"]]:
        raise ValueError("Native dimensionality/cell counts differ from contract")
    if not all(math.isclose(v, grid["dx_m"], rel_tol=1e-13) for v in (G.dx, G.dy, G.dz)):
        raise ValueError("Native spacing differs from contract")
    if not math.isclose(G.dt, grid["dt_s"], rel_tol=1e-13) or G.iterations != grid["iterations"]:
        raise ValueError("Native time axis differs from contract")
    faces = [G.pmlthickness[k] for k in ("x0", "y0", "z0", "xmax", "ymax", "zmax")]
    if faces != contract["pml_faces"]:
        raise ValueError("Native six-face PML profile differs from contract")
    sources = G.hertziandipoles + G.voltagesources + G.transmissionlines
    if len(sources) != 1 or len(G.rxs) != 1:
        raise ValueError("Native acquisition must contain exactly one source and receiver")
    expected_type = {"hertzian_dipole": "HertzianDipole", "voltage_source": "VoltageSource", "transmission_line": "TransmissionLine"}
    source = sources[0]
    if type(source).__name__ != expected_type[scene["source"]["kind"]] or source.polarisation != scene["source"]["axis"]:
        raise ValueError("Native source kind/polarization mismatch")
    timing = grid["derivation"]["excitation"]
    expected_stop = timing["stop_s"] if timing["stop_s"] is not None else grid["time_window_s"]
    if not math.isclose(source.start, timing["start_s"], abs_tol=1e-20) or not math.isclose(source.stop, expected_stop, rel_tol=1e-13):
        raise ValueError("Native source timing mismatch")
    if hasattr(source, "resistance") and source.resistance != scene["source"]["resistance_ohm"]:
        raise ValueError("Native source resistance mismatch")
    waveform = next(w for w in G.waveforms if w.ID == source.waveformID)
    requested_wf = contract["requested"]["waveform"]
    if waveform.type != (requested_wf["waveform_kind"] or "ricker") or waveform.amp != requested_wf["waveform_amplitude"] or not math.isclose(waveform.freq, timing["peak_hz"], rel_tol=1e-13):
        raise ValueError("Native waveform family, amplitude or peak frequency mismatch")
    if len(G.snapshots) != len(scene["snapshots"]):
        raise ValueError("Native snapshot count mismatch")
    for native, planned in zip(G.snapshots, scene["snapshots"]):
        if native.time != planned["iteration"] or native.basefilename != planned["filename"]:
            raise ValueError("Native snapshot identity/time mismatch")
        for suffix, key in (("s", "start_m"), ("f", "end_m")):
            if [getattr(native, axis + suffix) for axis in "xyz"] != [cell_index(v, grid["dx_m"]) for v in planned[key]]:
                raise ValueError("Native snapshot bounds mismatch")
        if [getattr(native, "d" + axis) for axis in "xyz"] != [cell_index(v, grid["dx_m"]) for v in planned["strides_m"]]:
            raise ValueError("Native snapshot strides mismatch")
    for obj, expected in ((source, scene["source"]), (G.rxs[0], scene["receiver"])):
        coords = [getattr(obj, axis + "coord") * grid["dx_m"] for axis in "xyz"]
        if not np.allclose(coords, expected["position_m"], rtol=0, atol=1e-12):
            raise ValueError("Native source/receiver coordinates mismatch")
    table_digests = []
    material_ids = []
    for layer in scene["layers"]:
        prefix = "|" + layer["box_id"] + "_"
        mats = [m for m in G.materials if m.ID.startswith(prefix)]
        actual = digest([material_coefficients(m) for m in mats])
        expected = layer["material_provenance"]["table_digest"]
        if actual != expected:
            raise ValueError(f"Native material table differs from derivation for {layer['name']}")
        table_digests.append(actual)
        ids = [m.numID for m in mats]
        material_ids.append(ids)
        lo, hi = (cell_index(layer[key][1], grid["dx_m"]) for key in ("start_m", "end_m"))
        rough = scene.get("roughness")
        if rough and layer["box_id"] == rough["box_id"]:
            hi = cell_index(rough["height_min_m"], grid["dx_m"])
        # Half-space continuation reaches x and z faces, and the bottom layer
        # reaches y=0. Targets are excluded from faces by placement policy.
        faces_to_check = [G.solid[0, lo:hi, :], G.solid[-1, lo:hi, :]]
        if contract["dimensionality"] == "3D":
            faces_to_check += [G.solid[:, lo:hi, 0], G.solid[:, lo:hi, -1]]
        if layer["terminal_halfspace"]:
            faces_to_check += [G.solid[:, 0, :]]
        if any(not np.isin(face, ids).all() for face in faces_to_check):
            raise ValueError("Native soil does not continue through declared half-space boundaries")
    histogram = Counter()
    geom_hash = hashlib.sha256()
    for plane in G.solid:
        unique, counts = np.unique(plane, return_counts=True)
        histogram.update({str(int(k)): int(v) for k, v in zip(unique, counts)})
        geom_hash.update(plane.tobytes(order="C"))
    target_labels = []
    total_target_voxels = 0
    for target in scene["targets"]:
        lo = [cell_index(v, grid["dx_m"]) for v in target["bbox_min_m"]]
        hi = [cell_index(v, grid["dx_m"]) for v in target["bbox_max_m"]]
        count = int(np.count_nonzero(G.solid[tuple(slice(a, b) for a, b in zip(lo, hi))] == 0))
        if count == 0:
            raise ValueError(f"Native target '{target['name']}' occupies no voxels")
        total_target_voxels += count
        target_labels.append({"name": target["name"], "target_index": target["target_index"], "occupied_pec_voxels": count,
                              "occupied_volume_m3": count * grid["dx_m"]**3,
                              "definition": "native solid material map, not analytic bounding-box volume"})
    if total_target_voxels != histogram.get("0", 0):
        raise ValueError("Native PEC voxels lie outside declared disjoint target bounds")
    return {"status": "passed", "mode": G.mode, "cell_counts": [G.nx, G.ny, G.nz],
            "backend": "gpu" if G.gpu is not None else "cpu",
            "cuda_device": {"id": G.gpu.deviceID, "name": G.gpu.name} if G.gpu is not None else None,
            "dt_s": G.dt, "iterations": G.iterations, "material_table_digests": table_digests,
            "solid_sha256": geom_hash.hexdigest(), "solid_material_histogram": dict(histogram),
            "material_ids_by_layer": material_ids, "targets": target_labels,
            "native_memory_estimate_bytes": int(G.memoryusage)}


@contextlib.contextmanager
def native_build_checks(contract, scene, output_path):
    import h5py
    import psutil
    import gprMax.model_build_run as native
    from gprMax.geometry_outputs import GeometryView
    verify_contract(contract, scene)
    original_write = GeometryView.write_vtk
    original_dispersion = native.dispersion_analysis
    result = {}

    def dispersion(G):
        data = original_dispersion(G)
        result["dispersion"] = {k: v.ID if hasattr(v, "ID") else v.item() if hasattr(v, "item") else v for k, v in data.items()}
        if data.get("error"):
            raise ValueError("Native numerical preflight failed: " + str(data["error"]))
        return data

    def write(view, G, pbar):
        result.update(check_native_grid(G, contract, scene))
        result["measured_build_rss_bytes"] = psutil.Process().memory_info().rss
        geometry_path = Path(output_path).with_suffix(".geometry.h5")
        with h5py.File(geometry_path, "w") as geometry:
            geometry.create_dataset("solid", data=G.solid, compression="gzip", shuffle=True)
            geometry.attrs["dx_dy_dz"] = (G.dx, G.dy, G.dz)
            geometry.attrs["coordinate_frame"] = contract["coordinate_frame"]
            geometry.attrs["contract_digest"] = contract["digest"]
            geometry.attrs["scene_digest"] = scene["digest"]
            geometry.attrs["materials"] = json.dumps({str(m.numID): m.ID for m in G.materials})
        result["geometry_file"] = str(geometry_path)
        result["geometry_sha256"] = file_digest(geometry_path)
        return original_write(view, G, pbar)

    GeometryView.write_vtk = write
    native.dispersion_analysis = dispersion
    try:
        yield result
        if result.get("status") != "passed":
            raise ValueError("Native geometry preflight did not execute")
    finally:
        GeometryView.write_vtk = original_write
        native.dispersion_analysis = original_dispersion
