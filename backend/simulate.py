"""
simulate.py - Batch gprMax simulation runner.

Processes all .in files in an input directory sequentially using the gprMax
Python API, writing .out (HDF5) files to the output directory.

The output directory defaults to out_files/ as a sibling of the input
directory, e.g. datasets/my_dataset/files -> datasets/my_dataset/out_files.

Usage:
    python simulate.py \\
        --input-dir datasets/my_dataset/files \\
        [--output-dir datasets/my_dataset/out_files] \\
        [--n 1] [--gpu] [--gpu-ids 0 1] [--skip-existing] [--verbose]
"""

from __future__ import annotations

import argparse
import gc
import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

# Bootstrap: add gprMax repo to sys.path so `from gprMax.gprMax import api` resolves
GPRMAX_ROOT = Path(__file__).parent.parent / "gprMax"
if str(GPRMAX_ROOT) not in sys.path:
    sys.path.insert(0, str(GPRMAX_ROOT))

from gprMax.gprMax import api


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


def build_gpu_arg(args) -> list | None:
    """Build the gpu argument expected by gprMax api."""
    if not args.gpu:
        return None
    if args.gpu_ids:
        return [args.gpu_ids]   # e.g. [[0, 1]]
    return [[]]                 # default device, equivalent to -gpu with no id


def run_simulation(tmp_in: Path, n: int, gpu_arg, verbose: bool) -> None:
    """Call gprMax api, optionally suppressing output."""
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
    on_error: 'Callable[[str, str, str], None] | None' = None,
    stop_on_first_error: bool = False,
) -> dict:
    """Run gprMax simulations on all .in files in input_dir.

    Callable from Python (used by the dataset generation pipeline).
    Returns a dict with keys: succeeded, failed, skipped, total, output_dir, errors.

    Args:
        on_error: Optional callback invoked on each simulation failure with
            (filename, traceback_str, in_file_content).
    """
    import logging
    logger = logging.getLogger(__name__)

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
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    in_files = sorted(input_dir.glob("*.in"))
    if not in_files:
        raise FileNotFoundError(f"No .in files found in {input_dir}")

    # Build gpu arg
    gpu_arg = None
    if gpu:
        gpu_arg = [gpu_ids] if gpu_ids else [[]]

    total = len(in_files)
    succeeded = 0
    failed = 0
    skipped = 0
    errors: list[dict] = []

    logger.info(f"[SIMULATE] {total} input file(s) in {input_dir} -> {output_dir}")

    for idx, in_file in enumerate(in_files, start=1):
        stem = in_file.stem
        expected_out = output_dir / f"{stem}.out"

        if skip_existing and expected_out.exists():
            logger.info(f"[SIMULATE] [{idx}/{total}] SKIP {in_file.name}")
            skipped += 1
            continue

        tmp_in = tmp_dir / in_file.name
        original_content = in_file.read_text()
        modified_content = inject_output_dir(original_content, output_dir)
        tmp_in.write_text(modified_content)

        t0 = time.perf_counter()
        try:
            run_simulation(tmp_in, n=n, gpu_arg=gpu_arg, verbose=verbose)
            elapsed = time.perf_counter() - t0
            logger.info(f"[SIMULATE] [{idx}/{total}] OK {in_file.name} ({elapsed:.1f}s)")
            succeeded += 1
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
            if stop_on_first_error:
                break
        finally:
            tmp_in.unlink(missing_ok=True)
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
        "errors": errors,
    }


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

    if args.gpu_ids:
        args.gpu = True

    input_dir = Path(args.input_dir).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (input_dir.parent / "out_files").resolve()

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    in_files = sorted(input_dir.glob("*.in"))
    if not in_files:
        print(f"[ERROR] No .in files found in {input_dir}")
        sys.exit(1)

    gpu_arg = build_gpu_arg(args)
    total = len(in_files)
    succeeded = 0
    failed = 0
    skipped = 0

    print(f"Found {total} input file(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"GPU: {'enabled' if args.gpu else 'disabled'} | n={args.n} | skip-existing={args.skip_existing}")
    print("-" * 60)

    for idx, in_file in enumerate(in_files, start=1):
        stem = in_file.stem
        expected_out = output_dir / f"{stem}.out"

        if args.skip_existing and expected_out.exists():
            print(f"[{idx}/{total}] SKIP  {in_file.name} (output exists)")
            skipped += 1
            continue

        # Write modified .in file to tmp dir (same stem = same output name)
        tmp_in = tmp_dir / in_file.name
        original_content = in_file.read_text()
        modified_content = inject_output_dir(original_content, output_dir)
        tmp_in.write_text(modified_content)

        print(f"[{idx}/{total}] RUN   {in_file.name} ...", end=" ", flush=True)
        t0 = time.perf_counter()

        try:
            run_simulation(tmp_in, n=args.n, gpu_arg=gpu_arg, verbose=args.verbose)
            elapsed = time.perf_counter() - t0
            print(f"OK ({elapsed:.1f}s)")
            succeeded += 1
        except Exception:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s)")
            print(traceback.format_exc())
            failed += 1
        finally:
            tmp_in.unlink(missing_ok=True)
            gc.collect()

    # Clean up tmp dir if empty
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print("-" * 60)
    print(f"Done. {succeeded} succeeded | {failed} failed | {skipped} skipped  (total: {total})")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()