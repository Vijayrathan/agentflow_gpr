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
        [--n 1] [--gpu] [--gpu-ids 0 1] [--skip-existing] [--verbose]
"""

from __future__ import annotations

import argparse
import gc
import io
import logging
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Any, Callable, Optional

# Bootstrap: gprMax lives as a source checkout at the repo root.
GPRMAX_ROOT = Path(__file__).resolve().parent.parent / "gprMax"
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


def run_batch_simulation(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    n: int = 1,
    gpu: bool = False,
    gpu_ids: list[int] | None = None,
    skip_existing: bool = False,
    verbose: bool = False,
    on_error: Optional[Callable[[str, str, str], None]] = None,
    stop_on_first_error: bool = False,
    progress: Optional[ProgressCallback] = None,
    filenames: Optional[list[str]] = None,
) -> dict:
    """Run gprMax simulations on .in files in input_dir.

    Returns a dict with keys: succeeded, failed, skipped, total, output_dir,
    outputs, errors. `outputs` maps each completed file to its .out path:
    [{"filename": ..., "out_file": ...}, ...] (skipped files with an existing
    .out are included — the output exists either way).

    Args:
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
    if gpu_ids:
        gpu = True

    input_dir = Path(input_dir).resolve()
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
    else:
        output_dir = (input_dir.parent / "out_files").resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # gprMax names the .out after the .in stem; running from a tmp copy with
    # the same stem keeps the output name while letting us inject #output_dir.
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    in_files = sorted(input_dir.glob("*.in"))
    if filenames is not None:
        wanted = {Path(name).name for name in filenames}
        in_files = [p for p in in_files if p.name in wanted]
    if not in_files:
        raise FileNotFoundError(f"No .in files found in {input_dir}")

    gpu_arg = ([gpu_ids] if gpu_ids else [[]]) if gpu else None

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

    logger.info(f"[SIMULATE] {total} input file(s) in {input_dir} -> {output_dir}")

    for idx, in_file in enumerate(in_files, start=1):
        expected_out = output_dir / f"{in_file.stem}.out"

        if skip_existing and expected_out.exists():
            logger.info(f"[SIMULATE] [{idx}/{total}] SKIP {in_file.name}")
            skipped += 1
            outputs.append({"filename": in_file.name, "out_file": str(expected_out)})
            _report({
                "event": "done", "index": idx, "total": total,
                "filename": in_file.name, "status": "skipped",
                "out_file": str(expected_out),
            })
            continue

        tmp_in = tmp_dir / in_file.name
        original_content = in_file.read_text()
        tmp_in.write_text(inject_output_dir(original_content, output_dir))

        _report({
            "event": "start", "index": idx, "total": total,
            "filename": in_file.name,
        })
        t0 = time.perf_counter()
        try:
            run_simulation(tmp_in, n=n, gpu_arg=gpu_arg, verbose=verbose)
            elapsed = time.perf_counter() - t0
            logger.info(f"[SIMULATE] [{idx}/{total}] OK {in_file.name} ({elapsed:.1f}s)")
            succeeded += 1
            outputs.append({"filename": in_file.name, "out_file": str(expected_out)})
            _report({
                "event": "done", "index": idx, "total": total,
                "filename": in_file.name, "status": "ok",
                "elapsed_s": elapsed, "out_file": str(expected_out),
            })
        except Exception:
            elapsed = time.perf_counter() - t0
            tb = traceback.format_exc()
            logger.warning(
                f"[SIMULATE] [{idx}/{total}] FAILED {in_file.name} ({elapsed:.1f}s)\n{tb}"
            )
            errors.append({"filename": in_file.name, "error": tb})
            if on_error:
                on_error(in_file.name, tb, original_content)
            failed += 1
            _report({
                "event": "done", "index": idx, "total": total,
                "filename": in_file.name, "status": "failed",
                "elapsed_s": elapsed, "error": tb.strip().splitlines()[-1],
            })
            if stop_on_first_error:
                break
        finally:
            tmp_in.unlink(missing_ok=True)
            # gprMax writes sibling artifacts next to the input file (e.g.
            # #geometry_view .vti) — move them to the real output dir.
            for stray in tmp_dir.iterdir():
                try:
                    stray.replace(output_dir / stray.name)
                except OSError:
                    pass
            gc.collect()

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

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
    }


def _cli_progress(event: dict[str, Any]) -> None:
    idx, total, name = event["index"], event["total"], event["filename"]
    if event["event"] == "start":
        print(f"[{idx}/{total}] RUN   {name} ...", end=" ", flush=True)
        return
    status = event["status"]
    if status == "skipped":
        print(f"[{idx}/{total}] SKIP  {name} (output exists)")
    elif status == "ok":
        print(f"OK ({event['elapsed_s']:.1f}s)")
    else:
        print(f"FAILED ({event['elapsed_s']:.1f}s)")
        print(event.get("error", ""))


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run gprMax simulations on a directory of .in files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing .in files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write .out files (default: out_files/ sibling of --input-dir)",
    )
    parser.add_argument("-n", type=int, default=1, help="Number of traces per file (1 = A-scan)")
    parser.add_argument("--gpu", action="store_true", default=False, help="Enable GPU acceleration")
    parser.add_argument("--gpu-ids", type=int, nargs="+", metavar="ID", help="Specific GPU device IDs (implies --gpu)")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip files whose .out already exists")
    parser.add_argument("--verbose", action="store_true", default=False, help="Show gprMax output per simulation")
    args = parser.parse_args()

    try:
        result = run_batch_simulation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            n=args.n,
            gpu=args.gpu,
            gpu_ids=args.gpu_ids,
            skip_existing=args.skip_existing,
            verbose=args.verbose,
            progress=_cli_progress,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("-" * 60)
    print(
        f"Done. {result['succeeded']} succeeded | {result['failed']} failed | "
        f"{result['skipped']} skipped  (total: {result['total']})"
    )
    print(f"Output directory: {result['output_dir']}")

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
