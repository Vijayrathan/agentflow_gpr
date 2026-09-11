# Implementation checkpoint — 2026-09-10

## Remote input admission correction

The user runs the experiment on `~/agentflow_gpr` on a Linux server, using the
existing activated Python environment. Their native dataset directories are
`moisture_A__db8ef04d` and `moisture_B__2ba27927`. The initial remote audit failed
before output inspection because the historical input-hash lookup required the
old local session directory names. Four dependent checks then produced misleading
additional failures. The zero output counts were uninspected counters, not
evidence of absent remote outputs.

`snapshot-inputs --output PATH` now freezes current manifests/decks by arm and
relative filename. Schema-v2 snapshots support directory relocation/renaming,
preserve content-change detection, and refuse to replace a differing baseline.
This records present input integrity only; it neither recreates the historical
scientific audit nor certifies which inputs were executed. Execution-receipt
admission remains separate. Audit now reports null counts for skipped output
inspection, labels partial inspections, and lists skipped dependent checks.

`config.remote.json` selects the user's two remote datasets, a new snapshot path,
and `runs/remote_v2`. README commands use the existing environment. The remote
datasets themselves have not been accessed here; the user must sync the updated
code and run snapshot/audit there. No production models have been fit.

**45 data-admission, feature, model and evaluation tests passed in 6.70 seconds**
with the existing `.venv/bin/python`; `git diff --check` also passed. Six new
regressions cover skipped checks, portable snapshots, changed inputs, immutable
baselines, incomplete inputs, and the snapshot/audit CLI sequence. The plot-based
workflow suite was not rerun because Matplotlib is absent from this local venv.
The historical full-suite results and local dataset counts below predate this
remote correction.

The standalone ML workflow is implemented. **No production ML model has been
trained or evaluated**, because the full output/provenance admission gate is not
yet satisfied. No platform code, dataset input, physical scene or solver source
was changed.

## Software verification

- **44 tests passed in 9.23 seconds** using Python 3.12.12, NumPy 2.4.2,
  SciPy 1.17.1, scikit-learn 1.8.0, PyTorch 2.10.0, h5py 3.16.0,
  joblib 1.5.3, Matplotlib 3.10.8 and pytest 9.1.1.
- Ruff checks passed on the experiment package.
- Tests cover signed lobes, prominence/separation, times/widths/areas, missing
  slots, plateau/tie behavior, overlapping bases, statistics, unclipped
  training-fitted scaling, supervised selection, exact RF/NN architecture,
  model reload parity, NN reproducibility, metrics/units and paired bootstrap.
- Dataset fixtures cover changed inputs/outputs, broken pairings, duplicate
  identities, shuffled manifest order, incorrect array lengths, nonfinite or
  missing fields, timing/acquisition mismatches, missing receipts, mixed backends,
  and duplicated dataset roles.
- A synthetic end-to-end workflow test covers feature caching, fixed RF/peak
  training, SVR CV tuning, mean baseline, validation-only reporting, restart,
  explicit test unlock, paired test evaluation, final plots and training freeze.
  Another test observes preprocessing fits to verify that every CV fit receives
  only that fold's training labels. Fixture populations are deliberately small;
  these are software tests, not production scientific results.
- CLI help and the real `train --recipe primary` admission block were exercised.

Test command:

```bash
uv run --quiet --no-project --python .venv/bin/python --with matplotlib==3.10.8 --with numpy==2.4.2 python -m pytest experiments/thickness_variance/ml/tests -q
```

The temporary dependency overlay provides Matplotlib without changing the platform
environment, project metadata or lockfile. Full isolated setup is in `README.md`.

## Actual dataset inspection

| Check | A | B |
|---|---:|---:|
| Audited input identities | 1,000 | 1,000 |
| Effective first-layer thickness levels | 62 | 1 |
| Existing native output files | 311 | 0 |
| Passed HDF5 structure/timing/acquisition/finite-array checks | 311 | 0 |
| Verified execution receipts supplied | 0 | 0 |

All 2,008 historical input/manifests still match their audit hashes. The shared
time step, native waveform/source settings and scalar moisture pairing pass.
There are 1,689 missing outputs and 311 outputs without supplied execution-time
completion provenance. Native HDF5 alone cannot close the latter gap. Neither
hashing existing outputs now nor assigning a receipt retrospectively proves
which input bytes were executed.

The gate correctly reports `inputs_ready=true`, `outputs_ready=false`. Its full
per-file evidence is [runs/main/audit.json](runs/main/audit.json).

## Prepared experimental artifacts

- Locked paired 700/150/150 IDs and five matched CV folds:
  [runs/main/splits.json](runs/main/splits.json).
- Ten available **training-only** A-signal peak diagnostics:
  [development diagnostic index](runs/main/development_diagnostics/index.json).
- Example full-trace/later-lobe plots:
  [A/10](runs/main/development_diagnostics/A_10.png),
  [A/358](runs/main/development_diagnostics/A_358.png).

All ten inspected training traces produced five candidates under the approved
thresholds. Slot six is empty with its presence indicator zero. Early strong
lobes and a later retained lobe are visible; this observation assigns no soil
interface identity and establishes no predictive accuracy. Prominence-base
intervals sometimes span much of the trace; these are the prescribed magnitude
integration bounds, not isolated reflection-energy windows. Thresholds, feature
definitions and signal samples were not changed after this inspection.

These incomplete diagnostics create no trainable feature cache, fit no selector,
use no test signals for plots and provide no scientific qualification. Runtime
artifacts are local and ignored by Git.

## Remaining prerequisites and work

1. Complete both original simulation populations, preserving audited decks.
2. Supply trusted execution-time input/output/completion receipts; recover actual
   execution evidence or rerun where that evidence is unavailable.
3. Run full feature extraction and inspect training diagnostics, then execute all
   recipes and controls. The complete hyperparameter grid has not been run here.
4. Review validation reports, freeze fitting, unlock tests, and generate final
   results through the documented CLI.

CUDA execution and full-population model training remain untested on this host.
No accuracy benefit, numerical convergence, CPU/GPU equivalence, new-seed
generalization or real-soil calibration is claimed.
