"""
simulate.py - Batch gprMax forward-model runner.

Runs every .in file in an input directory sequentially through the gprMax
Python API, writing .out (HDF5) files to the output directory. This is the
deterministic execution stage of the pipeline: no LLM involvement, plain
Python only.

Two entry points:
- `run_batch_simulation(...)` — called by `api.py` when the user presses
  "Run forward model" in the UI (with a per-file `progress` callback that
  the server streams to the frontend over the session WebSocket).
- CLI: `python backend/simulate.py --input-dir <dir> [...]`.

The output directory defaults to out_files/ as a sibling of the input
directory, e.g. dataset/<name>/in_files -> dataset/<name>/out_files (each
dataset lives in its own directory named after its model basename).

Usage:
    python backend/simulate.py \\
        --input-dir dataset/in_files \\
        [--output-dir dataset/out_files] \\
        [--n 1] [--gpu] [--gpu-ids 0 1] [--workers 4] \\
        [--skip-existing] [--verbose]
    python backend/simulate.py --check-gpu     # preflight: list CUDA devices
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import contextlib
import gc
import io
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# Bootstrap: gprMax lives as a source checkout at the repo root.
GPRMAX_ROOT = Path(__file__).resolve().parent.parent / "gprMax"
if str(GPRMAX_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(GPRMAX_ROOT.parent))
if str(GPRMAX_ROOT) not in sys.path:
    sys.path.insert(0, str(GPRMAX_ROOT))

logger = logging.getLogger(__name__)

# Called before ("start") and after ("done") each file with:
#   {event, index, total, filename, status?, elapsed_s?, out_file?, error?}
# status is "ok" | "failed" | "skipped" (skips emit a single "done" event).
ProgressCallback = Callable[[dict[str, Any]], None]


def _gprmax_api():
    """Import the solver lazily: api.py imports this module at startup, but
    gprMax (compiled Cython extensions) is only needed when a run starts."""
    from gprMax.gprMax import api

    return api


# ---------------------------------------------------------------------------
# Execution backend: CPU (OpenMP) or GPU (CUDA), N models at a time
# ---------------------------------------------------------------------------
# gprMax's only GPU backend is NVIDIA CUDA through pycuda — there is no Metal
# or OpenCL path — so GPU mode is a property of the DEPLOYMENT, not of the
# code: the GPU host sets GPR_GPU=1, laptops and CI leave it unset and get the
# OpenMP CPU solver. Explicit arguments always beat the environment.
#
#   GPR_GPU=1          use the CUDA solver (requires pycuda: `uv sync --extra gpu`)
#   GPR_GPU_IDS=0,1    device IDs to spread models over (default: device 0)
#   GPR_SIM_WORKERS=4  models solved concurrently (default: 2 per GPU, 1 on CPU)
#
# Concurrency is per-MODEL, not inside one model: each worker is a separate
# process that runs one .in file end to end. That is what actually keeps a GPU
# busy for this workload — a 2D GPR grid (single-cell z) is far too small to
# saturate a modern device, and one sample's serial CPU-side model build
# overlaps another's GPU solve. Every concurrent model holds its own field
# arrays in device memory, so raise GPR_SIM_WORKERS only as far as VRAM allows.
_DEFAULT_WORKERS_PER_GPU = 2


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved answer to "GPU or CPU, and how many models at once?"."""

    gpu: bool = False
    gpu_ids: list[int] = field(default_factory=list)
    workers: int = 1
    omp_threads: int | None = None  # per-worker OMP_NUM_THREADS (None = default)

    def gpu_arg_for(self, position: int):
        """The `gpu=` value to hand gprMax's `api()` for the position-th model.

        None means the CPU solver. Otherwise a device list, round-robined so
        that with several GPUs concurrent workers land on different devices
        (gprMax flattens the nesting and keeps the first matching device)."""
        if not self.gpu:
            return None
        if not self.gpu_ids:
            return [[]]  # gprMax default: device 0
        return [[self.gpu_ids[position % len(self.gpu_ids)]]]

    def describe(self) -> str:
        if not self.gpu:
            return f"CPU (OpenMP), {self.workers} model(s) at a time"
        devices = ",".join(str(i) for i in self.gpu_ids) or "0"
        return f"GPU (CUDA device {devices}), {self.workers} model(s) at a time"


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("[SIMULATE] ignoring non-integer %s=%r", name, raw)
        return None


def _env_ids(name: str) -> list[int] | None:
    """Parse "0,1" / "0 1" into device IDs. Malformed values are ignored
    rather than raised: a bad env var must not make the run unstartable."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return [int(part) for part in raw.replace(",", " ").split()]
    except ValueError:
        logger.warning("[SIMULATE] ignoring malformed %s=%r", name, raw)
        return None


def resolve_execution(
    gpu: bool | None = None,
    gpu_ids: list[int] | None = None,
    workers: int | None = None,
) -> ExecutionPlan:
    """Merge explicit arguments with the environment (arguments win)."""
    if gpu_ids is None:
        gpu_ids = _env_ids("GPR_GPU_IDS")
    if gpu_ids:
        gpu = True  # naming devices implies wanting them
    if gpu is None:
        gpu = bool(_env_flag("GPR_GPU"))
    gpu_ids = list(gpu_ids or [])

    if workers is None:
        workers = _env_int("GPR_SIM_WORKERS")
    if workers is None:
        # On CPU the solver already uses every core, so a second model would
        # only contend. On GPU one model leaves the device (and the cores)
        # idle for long stretches, so overlap is the whole point.
        workers = _DEFAULT_WORKERS_PER_GPU * max(1, len(gpu_ids)) if gpu else 1
    workers = max(1, workers)

    omp_threads = None
    if workers > 1:
        # Split the cores so W concurrent models don't each claim all of them
        # (gprMax reads OMP_NUM_THREADS while building the model).
        omp_threads = max(1, (os.cpu_count() or workers) // workers)
    return ExecutionPlan(
        gpu=gpu, gpu_ids=gpu_ids, workers=workers, omp_threads=omp_threads
    )


def describe_gpus(gpu_ids: list[int] | None = None) -> list[str]:
    """Preflight: what CUDA devices does this host actually expose? Raises
    ImportError (no pycuda) or GeneralError (no device) with gprMax's own
    message, which is what the operator needs to see."""
    from gprMax.utilities import detect_check_gpus

    _, alltext = detect_check_gpus(list(gpu_ids or []))
    return alltext


def inject_output_dir(content: str, output_dir: Path) -> str:
    """Replace or insert #output_dir command pointing to abs output_dir."""
    abs_out = str(output_dir.resolve())
    new_line = f"#output_dir: {abs_out}"
    lines = content.splitlines(keepends=True)

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("#output_dir"):
            lines[i] = new_line + "\n"
            return "".join(lines)

    # Not found — prepend before first non-empty line
    lines.insert(0, new_line + "\n")
    return "".join(lines)


def run_simulation(tmp_in: Path, n: int, gpu_arg, verbose: bool) -> None:
    """Call gprMax api, optionally suppressing its console output."""
    api = _gprmax_api()
    if verbose:
        api(str(tmp_in), n=n, gpu=gpu_arg)
    else:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            api(str(tmp_in), n=n, gpu=gpu_arg)


def _reset_gprmax_state() -> None:
    """Undo the process-wide state a FAILED model leaves behind.

    Two leaks, both fatal to every later file in the same process:

    1. gprMax keeps the built grid in a module-level global (`run_model`'s
       `global G`) and only runs `del G` at the END of a clean run. After an
       exception it survives, and the next api() call takes the "input file
       (not re-processed, i.e. geometry fixed)" branch — solving the DEAD
       model's geometry under the new file's name.
    2. A GPU model that raises inside `solve_gpu` never reaches its
       `ctx.pop()`. pycuda then aborts the whole process ("context stack was
       not empty upon module cleanup") at interpreter exit.

    Both are cleaned by name so nothing is imported that isn't already loaded.
    """
    mbr = sys.modules.get("gprMax.model_build_run")
    if mbr is not None and hasattr(mbr, "G"):
        with contextlib.suppress(Exception):
            del mbr.G
    drv = sys.modules.get("pycuda.driver")
    if drv is not None:
        with contextlib.suppress(Exception):
            while drv.Context.get_current() is not None:
                drv.Context.pop()


def _execute_one(task: dict[str, Any]) -> dict[str, Any]:
    """Run ONE .in file end to end and report how it went.

    Module-level and picklable on purpose: the parallel path ships it to a
    worker process, the serial path calls it inline (so a monkeypatched
    `run_simulation` still applies). Never raises — a failed model is a
    result, not an exception, or one bad deck would end the batch."""
    in_file = Path(task["in_file"])
    output_dir = Path(task["output_dir"])
    # Per-file tmp dir: concurrent models must never sweep up each other's
    # in-flight deck or geometry artifacts during the cleanup below.
    tmp_dir = Path(task["tmp_dir"])
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # gprMax names the .out after the .in stem; running from a tmp copy with
    # the same stem keeps the output name while letting us inject #output_dir.
    tmp_in = tmp_dir / in_file.name

    t0 = time.perf_counter()
    try:
        content = in_file.read_text()
        output_path = output_dir / f"{in_file.stem}.out"
        contract = task.get("contract")
        if contract:
            from backend.dataset_sampling.contract import file_digest
            from backend.preflight import validate_deck_contract, native_build_checks
            from backend.signal_extraction import validate_output
            if file_digest(in_file) != task["entry"]["input_sha256"]:
                raise ValueError("Input changed after manifest admission")
            scene = task["entry"]["resolved_scene"]
            validate_deck_contract(content, contract, scene)
            output_path.unlink(missing_ok=True)
            output_path.with_suffix(".execution.json").unlink(missing_ok=True)
        tmp_in.write_text(inject_output_dir(content, output_dir))
        with native_build_checks(contract, scene, output_path) if contract else contextlib.nullcontext() as preflight:
            run_simulation(tmp_in, n=task["n"], gpu_arg=task["gpu_arg"], verbose=task["verbose"])
        if contract:
            if preflight["backend"] != ("gpu" if task["gpu_arg"] else "cpu"):
                raise ValueError("Native execution backend differs from the admitted backend")
            snapshots = []
            for snapshot in scene["snapshots"]:
                relative = Path(in_file.stem + "_snaps") / (snapshot["filename"] + ".vti")
                target_path = output_dir / relative
                target_path.parent.mkdir(exist_ok=True)
                (tmp_dir / relative).replace(target_path)
                snapshots.append({"path": str(relative), "sha256": file_digest(target_path)})
            with contextlib.suppress(OSError):
                (tmp_dir / (in_file.stem + "_snaps")).rmdir()
            metadata = validate_output(output_path, contract, scene)
            receipt = {"contract_digest": contract["digest"], "scene_digest": scene["digest"],
                       "input_sha256": task["entry"]["input_sha256"], "output_sha256": file_digest(output_path),
                       "preflight": preflight, "actual": metadata, "qualification_status": "unqualified",
                       "backend": "gpu" if task["gpu_arg"] else "cpu", "snapshots": snapshots}
            output_path.with_suffix(".execution.json").write_text(json.dumps(receipt, indent=2))
        result = {
            "status": "ok",
            "out_file": str(output_dir / f"{in_file.stem}.out"),
        }
    except Exception:
        result = {"status": "failed", "error": traceback.format_exc()}
        # Must happen before the next file runs in this process.
        _reset_gprmax_state()
    finally:
        tmp_in.unlink(missing_ok=True)
        # gprMax writes sibling artifacts next to the input file (e.g.
        # #geometry_view .vti) — move them to the real output dir.
        for stray in tmp_dir.iterdir():
            try:
                stray.replace(output_dir / stray.name)
            except OSError:
                pass
        with contextlib.suppress(OSError):
            tmp_dir.rmdir()
        gc.collect()

    result["filename"] = in_file.name
    result["index"] = task["index"]
    result["elapsed_s"] = time.perf_counter() - t0
    return result


def _init_worker(omp_threads: int | None) -> None:
    """Worker-process setup: cap this model's OpenMP threads so W concurrent
    workers share the cores instead of each claiming all of them."""
    if omp_threads:
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)


def _make_pool(plan: ExecutionPlan) -> futures.Executor:
    """Process pool for the parallel path.

    'spawn' is mandatory, not a preference: forking a process that holds (or
    will build) a CUDA context is undefined behaviour. Tests substitute a
    thread pool here to exercise the driver without a real solver."""
    return futures.ProcessPoolExecutor(
        max_workers=plan.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(plan.omp_threads,),
    )


from backend.resources import reserve_batch


@reserve_batch
def run_batch_simulation(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    n: int = 1,
    gpu: bool | None = None,
    gpu_ids: list[int] | None = None,
    workers: int | None = None,
    skip_existing: bool = False,
    verbose: bool = False,
    on_error: Optional[Callable[[str, str, str], None]] = None,
    stop_on_first_error: bool = False,
    progress: Optional[ProgressCallback] = None,
    filenames: Optional[list[str]] = None,
    manifest: Optional[dict] = None,
) -> dict:
    """Run gprMax simulations on .in files in input_dir.

    Returns a dict with keys: succeeded, failed, skipped, total, output_dir,
    outputs, errors, mode, workers, gpu_ids. `outputs` maps each completed
    file to its .out path: [{"filename": ..., "out_file": ...}, ...] (skipped
    files with an existing .out are included — the output exists either way).
    It is keyed by filename, not order, so callers stay correct when models
    finish out of order.

    GPU batches containing an explicit #transmission_line command in a deck
    that needs solving raise ValueError before starting any models or workers.

    Args:
        gpu/gpu_ids/workers: solver backend and concurrency. Leave as None to
            take them from the environment (see ExecutionPlan above) — that is
            what api.py does, so the GPU host and a laptop run the same code.
        on_error: Optional callback invoked on each simulation failure with
            (filename, traceback_str, in_file_content).
        progress: Optional per-file callback (see ProgressCallback above).
            Exceptions raised by it are swallowed — reporting must never
            kill the batch.
        filenames: Optional restriction to these .in names. A dataset's
            in_files dir can hold stale decks (re-emission, basename reuse),
            so a batch must run its own manifest's files — not everything
            on disk.
    """
    plan = resolve_execution(gpu=gpu, gpu_ids=gpu_ids, workers=workers)

    input_dir = Path(input_dir).resolve()
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
    else:
        output_dir = (input_dir.parent / "out_files").resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Generated v2 CLI runs inherit manifest scoping too. Never glob stale decks.
    manifest_path = input_dir.parent / "emitted_files.json"
    if manifest is None and manifest_path.exists():
        candidate = json.loads(manifest_path.read_text())
        if candidate.get("contract"):
            manifest = candidate
    contract = (manifest or {}).get("contract")
    entries = {entry["filename"]: entry for entry in (manifest or {}).get("files", [])}
    if contract:
        from backend.dataset_sampling.contract import file_digest, solver_identity
        from backend.preflight import verify_contract
        from backend.resources import admit
        verify_contract(contract)
        if len(entries) != len(manifest["files"]) or len({e["sample_id"] for e in entries.values()}) != len(entries):
            raise ValueError("Manifest filenames and sample identities must be unique")
        if n != 1:
            raise ValueError("Contract execution is one fixed acquisition per physical sample (n=1)")
        if solver_identity() != contract["solver"]:
            raise ValueError("Pinned solver implementation changed; regenerate and requalify")
        filenames = filenames if filenames is not None else list(entries)
        if len(set(filenames)) != len(filenames) or any(name not in entries for name in filenames):
            raise ValueError("Execution filenames must be unique current manifest entries")
        for name in filenames:
            if Path(name).name != name or not (input_dir / name).is_file() or file_digest(input_dir / name) != entries[name]["input_sha256"]:
                raise ValueError(f"Missing/stale input file or hash: {name}")
            verify_contract(contract, entries[name]["resolved_scene"])
            if entries[name]["sample_id"] != entries[name]["resolved_scene"]["sample_id"] or Path(name).stem != entries[name]["resolved_scene"]["title"]:
                raise ValueError("Manifest filename/sample identity differs from resolved scene")
        output_dir.mkdir(parents=True, exist_ok=True)

    in_files = sorted(input_dir.glob("*.in"))
    if filenames is not None:
        wanted = {Path(name).name for name in filenames}
        in_files = [p for p in in_files if p.name in wanted]
        missing = wanted - {p.name for p in in_files}
        if missing:
            raise FileNotFoundError(f"Manifest input files missing: {sorted(missing)}")
    if not in_files:
        raise FileNotFoundError(f"No .in files found in {input_dir}")

    # Inspect the actual decks so this covers generated datasets, uploads,
    # and CLI runs without needing the collection-stage antenna config.
    # Check the whole pending batch before starting even its first model.
    if plan.gpu:
        for in_file in in_files:
            if skip_existing and (output_dir / f"{in_file.stem}.out").exists():
                continue
            try:
                with in_file.open() as deck:
                    has_transmission_line = any(
                        line.partition(":")[0].strip().lower() == "#transmission_line"
                        for line in deck
                    )
            except (OSError, UnicodeError):
                # Preserve per-file failure reporting for unreadable decks.
                continue
            if has_transmission_line:
                raise ValueError(
                    f"{in_file.name}: transmission_line sources require CPU solving; "
                    "the gprMax CUDA solver does not support them. "
                    "Disable GPU execution (gpu=False, gpu_ids=[] in Python; "
                    "GPR_GPU=0 with GPR_GPU_IDS unset and no --gpu/--gpu-ids flags "
                    "for CLI/server runs)."
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    if contract:
        plan = admit(plan, contract["resources"], output_dir, len(in_files))
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    total = len(in_files)
    succeeded = 0
    failed = 0
    skipped = 0
    outputs: list[dict[str, str]] = []
    errors: list[dict] = []

    def _report(payload: dict[str, Any]) -> None:
        if progress is None:
            return
        try:
            progress(payload)
        except Exception:
            logger.exception("simulation progress callback failed")

    completed = 0   # files finished (ok/failed/skipped) — monotonic progress
    cursor = 0      # position in in_files of the next file to consider
    halted = False  # stop_on_first_error tripped
    available_devices = list(plan.gpu_ids or [0]) if contract and plan.gpu else []

    logger.info(
        f"[SIMULATE] {total} input file(s) in {input_dir} -> {output_dir} "
        f"| {plan.describe()}"
    )

    def _next_task() -> dict[str, Any] | None:
        """The next file that must actually be solved. Files whose .out
        already exists are reported as skipped in place, so the event stream
        keeps batch order even though solving may not."""
        nonlocal cursor, skipped, completed
        while cursor < len(in_files):
            in_file = in_files[cursor]
            idx = cursor + 1
            cursor += 1
            expected_out = output_dir / f"{in_file.stem}.out"
            if contract and skip_existing and expected_out.exists():
                from backend.signal_extraction import validate_output
                validate_output(expected_out, contract, entries[in_file.name]["resolved_scene"],
                                input_sha256=entries[in_file.name]["input_sha256"], require_receipt=True)
            if skip_existing and expected_out.exists():
                logger.info(f"[SIMULATE] [{idx}/{total}] SKIP {in_file.name}")
                skipped += 1
                completed += 1
                outputs.append(
                    {"filename": in_file.name, "out_file": str(expected_out)}
                )
                _report({
                    "event": "done", "index": idx, "total": total,
                    "completed": completed, "filename": in_file.name,
                    "status": "skipped", "out_file": str(expected_out),
                })
                continue
            device = available_devices.pop(0) if available_devices else None
            return {
                "in_file": str(in_file),
                "output_dir": str(output_dir),
                "tmp_dir": str(tmp_dir / f"{idx:06d}_{in_file.stem}"),
                "n": n,
                # position-based so concurrent workers spread over the devices
                "gpu_arg": [[device]] if device is not None else plan.gpu_arg_for(idx - 1),
                "reserved_device": device,
                "verbose": verbose,
                "index": idx,
                "filename": in_file.name,
                **({"contract": contract, "entry": entries[in_file.name]} if contract else {}),
            }
        return None

    def _start(task: dict[str, Any]) -> None:
        _report({
            "event": "start", "index": task["index"], "total": total,
            "completed": completed, "filename": task["filename"],
        })

    def _finish(task: dict[str, Any], res: dict[str, Any]) -> None:
        nonlocal succeeded, failed, completed, halted
        completed += 1
        if task.get("reserved_device") is not None:
            available_devices.append(task["reserved_device"])
        idx, name = task["index"], task["filename"]
        elapsed = res.get("elapsed_s", 0.0)
        if res.get("status") == "ok":
            succeeded += 1
            logger.info(f"[SIMULATE] [{idx}/{total}] OK {name} ({elapsed:.1f}s)")
            outputs.append({"filename": name, "out_file": res["out_file"]})
            _report({
                "event": "done", "index": idx, "total": total,
                "completed": completed, "filename": name, "status": "ok",
                "elapsed_s": elapsed, "out_file": res["out_file"],
            })
            return
        tb = res.get("error") or "unknown solver failure"
        failed += 1
        logger.warning(
            f"[SIMULATE] [{idx}/{total}] FAILED {name} ({elapsed:.1f}s)\n{tb}"
        )
        errors.append({"filename": name, "error": tb})
        if on_error:
            try:
                content = Path(task["in_file"]).read_text()
            except OSError:
                content = ""
            on_error(name, tb, content)
        _report({
            "event": "done", "index": idx, "total": total,
            "completed": completed, "filename": name, "status": "failed",
            "elapsed_s": elapsed, "error": tb.strip().splitlines()[-1],
        })
        if stop_on_first_error:
            halted = True

    # GPU runs ALWAYS go through the pool, even at workers=1: api.py calls this
    # from a thread of the uvicorn process, and a CUDA context must never be
    # built there — a solver failure can leave one on the stack, which makes
    # pycuda abort the whole server at interpreter exit.
    if plan.workers == 1 and not plan.gpu and not contract:
        while not halted and (task := _next_task()) is not None:
            _start(task)
            _finish(task, _execute_one(task))
    else:
        # Bounded submission: at most `workers` models are in flight, so a
        # "start" event still means "this file is running now" and the pool
        # never queues the whole batch ahead of the first failure.
        with _make_pool(plan) as pool:
            inflight: dict[futures.Future, dict[str, Any]] = {}
            while True:
                while not halted and len(inflight) < plan.workers:
                    task = _next_task()
                    if task is None:
                        break
                    inflight[pool.submit(_execute_one, task)] = task
                    _start(task)
                if not inflight:
                    break
                done, _ = futures.wait(
                    inflight, return_when=futures.FIRST_COMPLETED
                )
                for fut in done:
                    task = inflight.pop(fut)
                    try:
                        res = fut.result()
                    except Exception:
                        # The worker process itself died (OOM, CUDA fault):
                        # charge it to this file and keep the batch going.
                        res = {"status": "failed", "error": traceback.format_exc()}
                    _finish(task, res)

    with contextlib.suppress(OSError):
        tmp_dir.rmdir()

    logger.info(
        f"[SIMULATE] Done. {succeeded} succeeded | {failed} failed | {skipped} skipped (total: {total})"
    )

    return {
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "output_dir": str(output_dir),
        "outputs": outputs,
        "errors": errors,
        "mode": "gpu" if plan.gpu else "cpu",
        "workers": plan.workers,
        "gpu_ids": plan.gpu_ids,
    }


def _cli_progress(event: dict[str, Any]) -> None:
    # One self-contained line per event: with several models in flight a
    # start/done pair is no longer adjacent, so nothing may be continued.
    idx, total, name = event["index"], event["total"], event["filename"]
    if event["event"] == "start":
        print(f"[{idx}/{total}] RUN   {name} ...", flush=True)
        return
    done = event.get("completed", idx)
    status = event["status"]
    if status == "skipped":
        print(f"[{done}/{total}] SKIP  {name} (output exists)", flush=True)
    elif status == "ok":
        print(f"[{done}/{total}] OK    {name} ({event['elapsed_s']:.1f}s)", flush=True)
    else:
        print(f"[{done}/{total}] FAIL  {name} ({event['elapsed_s']:.1f}s)", flush=True)
        print(event.get("error", ""), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run gprMax simulations on a directory of .in files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default=None, help="Directory containing .in files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write .out files (default: out_files/ sibling of --input-dir)",
    )
    parser.add_argument("-n", type=int, default=1, help="Number of traces per file (1 = A-scan)")
    parser.add_argument("--gpu", action="store_true", default=None, help="Enable GPU acceleration (default: $GPR_GPU)")
    parser.add_argument("--gpu-ids", type=int, nargs="+", metavar="ID", help="Specific GPU device IDs (implies --gpu; default: $GPR_GPU_IDS)")
    parser.add_argument("--workers", type=int, default=None, metavar="N", help="Models to solve concurrently (default: $GPR_SIM_WORKERS, else 2 per GPU / 1 on CPU)")
    parser.add_argument("--check-gpu", action="store_true", default=False, help="List the CUDA devices gprMax can see, then exit")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip files whose .out already exists")
    parser.add_argument("--verbose", action="store_true", default=False, help="Show gprMax output per simulation")
    args = parser.parse_args()

    if args.check_gpu:
        # Preflight for the GPU host: surface pycuda/driver problems here
        # rather than in the middle of a 200-model batch.
        try:
            for line in describe_gpus(args.gpu_ids):
                print(f"GPU {line}")
        except Exception as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        plan = resolve_execution(gpu=args.gpu, gpu_ids=args.gpu_ids,
                                 workers=args.workers)
        # Detecting a device says nothing about whether runs will USE it —
        # spell out the difference or "GPU 0 - ..." followed by "CPU" reads
        # like a contradiction.
        print(f"Configured: {plan.describe()}")
        if not plan.gpu:
            print(
                "A GPU is available but NOT enabled — set GPR_GPU=1 "
                "(optionally GPR_SIM_WORKERS=N) in the environment the server "
                "starts with, then restart it."
            )
        return

    if not args.input_dir:
        parser.error("--input-dir is required (unless --check-gpu)")

    try:
        result = run_batch_simulation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            n=args.n,
            gpu=args.gpu,
            gpu_ids=args.gpu_ids,
            workers=args.workers,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
            progress=_cli_progress,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("-" * 60)
    print(
        f"Done. {result['succeeded']} succeeded | {result['failed']} failed | "
        f"{result['skipped']} skipped  (total: {result['total']})"
    )
    print(f"Solver: {result['mode'].upper()} | {result['workers']} worker(s)")
    print(f"Output directory: {result['output_dir']}")

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
