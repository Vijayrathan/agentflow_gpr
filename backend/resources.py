"""Conservative per-model admission; scheduling never changes the experiment."""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import replace


def reserve_batch(function):
    """Serialize admissions across API threads and local CLI/server processes.

    Per-batch workers still run concurrently. A second batch waits until all
    reservations are released before measuring available RAM/VRAM/disk again.
    OS locks release automatically if a driver dies.
    """
    from functools import wraps
    @wraps(function)
    def reserved(*args, **kwargs):
        import fcntl
        import tempfile
        from pathlib import Path
        path = Path(os.environ.get("GPR_RESOURCE_LOCK_PATH", str(Path(tempfile.gettempdir()) / "gprmax-dataset-resource.lock")))
        with path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return reserved


def estimate_resources(grid, cfg, layer_count, adv=None):
    nx, ny, nz = (round(v / grid.dx_m) for v in (grid.domain_x_m, grid.domain_y_m, grid.domain_z_m))
    cells, nodes = nx * ny * nz, (nx + 1) * (ny + 1) * (nz + 1)
    # Native float32 fields, uint32 edge IDs, 1 Debye pole complex64 auxiliaries,
    # uint8 rigid E/H flags and uint32 solid IDs. FFT construction is host-only;
    # reserve multiple complex128 work arrays rather than the old 146 B/cell.
    fields = 6 * 4 * nodes
    geometry = 6 * 4 * nodes + (12 + 6 + 4) * cells
    dispersive = 3 * 8 * nodes
    p = cfg.pml_cells
    pml_cells = 2 * p * (ny * nz + nx * nz + (nx * ny if nz > 1 else 0))
    pml = pml_cells * 16 * 4
    materials = 2 + layer_count * cfg.fractal_nbins + 8 * layer_count
    coefficients = materials * 10 * 4
    traces = 6 * 4 * (grid.iterations or math.ceil(grid.time_window_s / grid.dt_s) + 1)
    snapshots = 0
    for s in adv.snapshots or [] if adv else []:
        lo = (s.x1, s.y1, s.z1)
        hi = tuple(v if v is not None else d for v, d in zip((s.x2, s.y2, s.z2), (grid.domain_x_m, grid.domain_y_m, grid.domain_z_m)))
        strides = (s.dx or grid.dx_m, s.dy or grid.dx_m, s.dz or grid.dx_m)
        snapshots += 6 * 4 * math.prod(math.ceil((b - a) / step) + 1 for a, b, step in zip(lo, hi, strides))
    base = fields + geometry + dispersive + pml + coefficients + traces
    return {"policy": "native-array-estimate-v2-with-FFT-and-runtime-reserves",
            "cells": cells, "host_peak_bytes": math.ceil(1.25 * (base + 192 * cells + snapshots)) + 256 * 1024**2,
            "device_peak_bytes": math.ceil(1.25 * (fields + 6 * 4 * nodes + dispersive + pml + coefficients + traces + snapshots)) + 128 * 1024**2,
            "coefficient_bytes": coefficients, "output_bytes": 8 * cells + snapshots + traces + 1024**2,
            "scratch_bytes": 12 * cells + snapshots + traces + 1024**2,
            "measured": False}


def admit(plan, estimate, output_dir, model_count):
    import psutil
    reserve = int(os.environ.get("GPR_HOST_RESERVE_BYTES", str(2 * 1024**3)))
    available = max(0, psutil.virtual_memory().available - reserve)
    host_limit = int(os.environ.get("GPR_HOST_BUDGET_BYTES", str(available)))
    host_slots = min(available, host_limit) // estimate["host_peak_bytes"]
    if host_slots < 1:
        raise ValueError(f"3D admission: one model needs an estimated {estimate['host_peak_bytes']/1024**3:.2f} GiB host RAM; "
                         f"{min(available, host_limit)/1024**3:.2f} GiB available after reserve. Scene unchanged.")
    free_disk = shutil.disk_usage(output_dir).free
    required = estimate["output_bytes"] * model_count + estimate["scratch_bytes"] * min(plan.workers, host_slots)
    if required > free_disk:
        raise ValueError(f"3D admission: estimated output/scratch {required} bytes exceeds free disk {free_disk}")
    workers = min(plan.workers, host_slots)
    if plan.gpu:
        # Probe in an isolated process: no CUDA context in the API thread.
        import subprocess, sys, json
        probe = subprocess.run([sys.executable, "-m", "backend.resources", "--cuda-capacity"],
                               capture_output=True, text=True, check=True)
        devices = json.loads(probe.stdout)
        selected = plan.gpu_ids or [0]
        eligible = []
        for device in selected:
            info = devices[str(device)]
            if info["free"] < estimate["device_peak_bytes"] or info["constant"] < estimate["coefficient_bytes"]:
                raise ValueError(f"3D admission: model exceeds CUDA device {device} memory/coefficient capacity")
            eligible.append(device)
        # At most one 3D model per device in the initial policy. The scheduler
        # reserves actual device identities until completion, not file modulo.
        workers = min(workers, len(eligible))
    return replace(plan, workers=workers,
                   omp_threads=max(1, (os.cpu_count() or 1) // workers) if workers > 1 else plan.omp_threads)


if __name__ == "__main__":
    import json
    import pycuda.driver as cuda
    cuda.init()
    capacities = {}
    for i in range(cuda.Device.count()):
        device = cuda.Device(i)
        context = device.make_context()
        try:
            free, _ = cuda.mem_get_info()
            capacities[str(i)] = {"free": free, "constant": device.total_constant_memory}
        finally:
            context.pop()
            context.detach()
    print(json.dumps(capacities))
