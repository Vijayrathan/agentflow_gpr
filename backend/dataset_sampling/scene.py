"""Resolve and validate the geometry gprMax will receive, before serialization.

Resolved command geometry is distinct from occupied voxels; the latter is
measured during native preflight. Cylinder radii remain analytic native inputs,
while their axis endpoints use the native grid rounding rule.
"""
from __future__ import annotations

import math
from backend.dataset_sampling.contract import component_positions, stream_seed, validate_capabilities
from backend.dataset_sampling.numerics import cell_index
from backend.dataset_sampling.target_shapes import half_extents_3d


def token(value):
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", value or "unnamed")


def snap(values, grid, cfg):
    result = [cell_index(v, grid.dx_m) * grid.dx_m for v in values]
    if cfg.quantization_policy == "exact" and any(abs(a - b) > 1e-12 for a, b in zip(values, result)):
        raise ValueError("Geometry requires endpoint/coordinate quantization under exact policy")
    return result


def resolve_target(target, grid, cfg):
    dx = grid.dx_m
    dim3 = cfg.dimensionality == "3D"
    if dim3 and (target.z_offset_m is None or
                 (target.kind == "box" and target.crossline_size_m is None) or
                 (target.kind == "cylinder" and (target.length_m is None or target.cylinder_axis is None))):
        raise ValueError(f"Target '{target.name}' lacks finite 3D geometry")
    center = [grid.domain_x_m / 2 + target.x_offset_m,
              grid.ground_y_m - target.depth_m,
              grid.domain_z_m / 2 + target.z_offset_m if dim3 else dx / 2]
    extents = list(half_extents_3d(target))
    if not dim3:
        extents[2] = dx / 2
    requested_min = [v - h for v, h in zip(center, extents)]
    requested_max = [v + h for v, h in zip(center, extents)]
    record = {"name": target.name, "kind": target.kind, "material": target.material,
              "requested": target.model_dump(), "dielectric_smoothing": False}
    if target.kind == "box":
        start, end = snap(requested_min, grid, cfg), snap(requested_max, grid, cfg)
        feature = min(b - a for a, b in zip(start, end)) if dim3 else min(end[i] - start[i] for i in (0, 1))
        bbox_min, bbox_max = start, end
    else:
        axis = "xyz".index(target.cylinder_axis or "z")
        a, b = list(center), list(center)
        a[axis], b[axis] = requested_min[axis], requested_max[axis]
        start, end = snap(a, grid, cfg), snap(b, grid, cfg)
        record.update(radius_m=target.radius_m, cylinder_axis=target.cylinder_axis or "z")
        feature = min(2 * target.radius_m, end[axis] - start[axis]) if dim3 else 2 * target.radius_m
        # Conservative voxel enclosure of a native staircased circular section.
        bbox_min = [math.floor((v - target.radius_m) / dx + 1e-10) * dx for v in start]
        bbox_max = [math.ceil((v + target.radius_m) / dx - 1e-10) * dx for v in end]
        bbox_min[axis], bbox_max[axis] = start[axis], end[axis]
        requested_min, requested_max = a, b
    record.update(start_m=start, end_m=end, bbox_min_m=bbox_min, bbox_max_m=bbox_max,
                  start_cells=[cell_index(v, dx) for v in start],
                  end_cells=[cell_index(v, dx) for v in end], minimum_feature_m=feature,
                  quantization_delta_start_m=[a - b for a, b in zip(start, requested_min)],
                  quantization_delta_end_m=[a - b for a, b in zip(end, requested_max)],
                  label_definition="resolved native command; occupied voxels measured in preflight")
    return record


def validate_resolved_target(target, grid, cfg, surface_min=None):
    errors = []
    margin = (cfg.pml_cells + 15) * grid.dx_m
    dims = [grid.domain_x_m, grid.domain_y_m, grid.domain_z_m]
    axes = range(3) if cfg.dimensionality == "3D" else range(2)
    for axis in axes:
        if target["bbox_min_m"][axis] < margin - 1e-10 or dims[axis] - target["bbox_max_m"][axis] < margin - 1e-10:
            errors.append(f"target '{target['name']}' violates PML+15 clearance on {'xyz'[axis]}")
    if target["minimum_feature_m"] < 10 * grid.dx_m - 1e-12:
        errors.append(f"target '{target['name']}' has fewer than 10 cells across an intrinsic feature")
    surface = grid.ground_y_m if surface_min is None else surface_min
    if target["bbox_max_m"][1] > surface + 1e-10:
        errors.append(f"target '{target['name']}' is not fully buried beneath the lowest surface")
    return errors


def bounds_overlap(a, b, mode):
    axes = range(3) if mode == "3D" else range(2)
    # The declared disjoint-bounds policy conservatively excludes even touching
    # bounds. It cannot inadvertently merge two PEC objects through staircasing.
    return all(a["bbox_min_m"][i] <= b["bbox_max_m"][i] + 1e-12 and
               b["bbox_min_m"][i] <= a["bbox_max_m"][i] + 1e-12 for i in axes)


def resolve_outputs(adv, grid, title):
    outputs = []
    names = set()
    for request in adv.snapshots or [] if adv else []:
        name = token(request.filename)
        if name in names or not request.filename or name != request.filename:
            raise ValueError("Snapshot filenames must be unique simple identifiers")
        names.add(name)
        if not (0 < request.time_s <= grid.time_window_s):
            raise ValueError("Snapshot time must be positive and inside the final recording window")
        lower = [request.x1, request.y1, request.z1]
        upper = [v if v is not None else d for v, d in zip(
            (request.x2, request.y2, request.z2), (grid.domain_x_m, grid.domain_y_m, grid.domain_z_m))]
        steps = [v if v is not None else grid.dx_m for v in (request.dx, request.dy, request.dz)]
        dims = [grid.domain_x_m, grid.domain_y_m, grid.domain_z_m]
        for lo, hi, step, dim in zip(lower, upper, steps, dims):
            if not (0 <= lo < hi <= dim + 1e-12) or step < grid.dx_m:
                raise ValueError("Snapshot region/stride lies outside the resolved grid")
            if any(abs(v / grid.dx_m - cell_index(v, grid.dx_m)) > 1e-8 for v in (lo, hi, step)):
                raise ValueError("Snapshot bounds and strides must be integer cells")
        outputs.append({"start_m": lower, "end_m": upper, "strides_m": steps,
                        "time_s": request.time_s,
                        "iteration": cell_index(request.time_s, grid.dt_s) + 1,
                        "effective_time_s": cell_index(request.time_s, grid.dt_s) * grid.dt_s,
                        "fields": ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], "filename": name})
    return outputs


def resolve_roughness(rough, grid, cfg, surface_layer, sample_id):
    if rough is None:
        return None
    if cfg.dimensionality != "2D":
        raise ValueError("3D surface roughness is unsupported")
    low, high = snap([grid.ground_y_m - rough.amplitude_m,
                      grid.ground_y_m + rough.amplitude_m], grid, cfg)
    if low >= high:
        raise ValueError("Surface roughness collapses on the common grid")
    if low < surface_layer["y_bottom_m"] + 3 * grid.dx_m - 1e-12:
        raise ValueError("Roughness trough leaves fewer than three cells in the top layer")
    if high >= min(grid.tx_y_m, grid.rx_y_m) - 1e-12:
        raise ValueError("Roughness peak reaches the source or receiver")
    if high > grid.domain_y_m - (cfg.pml_cells + 15) * grid.dx_m + 1e-12:
        raise ValueError("Roughness peak violates top PML+15 clearance")
    return {"requested": rough.model_dump(), "height_min_m": low, "height_max_m": high,
            "start_m": [0, grid.ground_y_m, 0],
            "end_m": [grid.domain_x_m, grid.ground_y_m, grid.domain_z_m],
            "box_id": surface_layer["box_id"], "fractal_dim": rough.fractal_dim,
            "surface_weights_x_z": [rough.weight_x, rough.weight_y],
            "seed": rough.seed if rough.seed is not None else stream_seed(cfg.seed, sample_id, "surface")}


def resolve_scene(sample, grid, cfg, wf, ant, adv=None):
    validate_capabilities(cfg, waveform=wf, antenna=ant, advanced=adv)
    dx = grid.dx_m
    if cfg.contract_version != grid.contract_version or cfg.dimensionality != grid.dimensionality:
        raise ValueError("Grid and configuration contracts disagree; regenerate the common plan")
    title = f"{token(cfg.model_basename)}_{sample.sample_id}"
    top = cell_index(grid.ground_y_m, dx)
    layers = []
    for i, layer in enumerate(sample.layers):
        last = i == len(sample.layers) - 1
        cells = cell_index(layer.thickness_m, dx)
        bottom = 0 if last else top - cells
        if bottom < 0 or top - bottom < 3:
            raise ValueError(f"sample {sample.sample_id} layer {i} collapses or has fewer than three cells")
        if cfg.quantization_policy == "exact" and not last and abs(cells * dx - layer.thickness_m) > 1e-12:
            raise ValueError("Finite layer thickness requires quantization under exact policy")
        layers.append({"name": layer.name, "requested": layer.model_dump(),
                       "soil_id": f"soil_{i + 1}", "box_id": f"layer_{i + 1}_volume",
                       "start_m": [0, bottom * dx, 0], "end_m": [grid.domain_x_m, top * dx, grid.domain_z_m],
                       "thickness_m": (top - bottom) * dx, "y_top_m": top * dx,
                       "y_bottom_m": bottom * dx,
                       "terminal_halfspace": last, "sampled_thickness_m": layer.thickness_m,
                       "quantization_delta_m": None if last else cells * dx - layer.thickness_m,
                       "seed": stream_seed(cfg.seed, sample.sample_id, "volume", i)})
        top = bottom
    if not layers:
        raise ValueError("A resolved soil scene requires at least one layer")
    targets = [resolve_target(t, grid, cfg) for t in sample.targets]
    rough = resolve_roughness(adv.surface_roughness if adv else None, grid, cfg, layers[0], sample.sample_id)
    surface_min = rough["height_min_m"] if rough else grid.ground_y_m
    for i, target in enumerate(targets):
        target["target_index"] = i
        errors = validate_resolved_target(target, grid, cfg, surface_min)
        if errors:
            raise ValueError("; ".join(errors))
        if any(bounds_overlap(target, prior, cfg.dimensionality) for prior in targets[:i]):
            raise ValueError(f"Target '{target['name']}' violates disjoint_bounds overlap policy")
    tx = [grid.tx_x_m, grid.tx_y_m, grid.tx_z_m]
    rx = [grid.rx_x_m, grid.rx_y_m, grid.rx_z_m]
    return {"sample_id": sample.sample_id, "title": title, "layers": layers, "targets": targets,
            "source": {"kind": ant.antenna_kind, "axis": ant.antenna_axis, "position_m": tx,
                       "component_position_m": component_positions(tx, dx)["E" + ant.antenna_axis],
                       "resistance_ohm": ant.resistance},
            "receiver": {"id": "rx1", "position_m": rx,
                         "component_positions_m": component_positions(rx, dx)},
            "roughness": rough, "snapshots": resolve_outputs(adv, grid, title),
            "geometry_view": {"end_m": [grid.domain_x_m, grid.domain_y_m, grid.domain_z_m],
                              "strides_m": [max(1, math.ceil(n / 160)) * dx for n in (grid.nx, grid.ny, grid.nz)],
                              "filename": title + "_geo"},
            "provenance": sample.provenance, "status": "resolved_valid"}
