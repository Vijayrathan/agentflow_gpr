# Thickness-variation moisture experiment

Standalone implementation of the approved RF/peak-first A/B study. The platform,
solver, physical scenes and dataset files are never changed by these commands.
All run products live under `run_dir` (default `ml/runs/main`, ignored by Git).

## Execution

Run from the repository root using the existing project environment. On the
inspected local host only Matplotlib was missing. Install it into that same
environment if needed (no new environment or dependency overlay):

```bash
uv pip install --python .venv/bin/python matplotlib==3.10.8
.venv/bin/python -m experiments.thickness_variance.ml --help
```

In the commands below, `python` means the activated existing project interpreter;
alternatively use `.venv/bin/python` explicitly. `requirements.txt` records the
tested versions. Runs record their actual environment and reject changes midway.

### Remote datasets with different session names

The supplied `config.remote.json` selects `moisture_A__db8ef04d` and
`moisture_B__2ba27927`, with a new run directory `runs/remote_v2`. First snapshot
their manifests and emitted decks, then audit their contents and native outputs:

```bash
python -m experiments.thickness_variance.ml --config experiments/thickness_variance/ml/config.remote.json snapshot-inputs --output experiments/thickness_variance/ml/runs/remote_inputs.json
python -m experiments.thickness_variance.ml --config experiments/thickness_variance/ml/config.remote.json audit
```

Use the same `--config experiments/thickness_variance/ml/config.remote.json`
before every subsequent subcommand. The snapshot records current input integrity,
not successful historical execution. Without execution receipts, the audit can
inspect all native outputs and report their actual validity, but training remains
blocked by provenance. The snapshot command creates no receipts, reads no output
signals, and refuses to overwrite a differing snapshot. A retry of unchanged
inputs is safe.

Schema-v2 snapshots use arm labels and paths relative to each dataset, so moving
or renaming the enclosing directory does not break hash lookup. The old flat
historical hash format remains supported, with its original directory-name binding.
Manifest failures now produce `status=skipped_after_manifest_failure` and null
output counts; dependent pairing/thickness checks appear under `skipped_checks`.
Null means uninspected, not missing. Audit prints the first ten actual issues.

The default configuration is `config.json`. Relative dataset/hash/run/receipt paths resolve
against the repository root, not the working directory or config location.
`--config /path/to/config.json` is a global option before the subcommand. Copy
the config for another host. Use a dataset-specific snapshot for different input
populations. Do not edit source manifests to relocate data.
`n_jobs`, `torch_threads`, and `device` control local compute. Defaults are one
thread per worker and CPU. CUDA only applies to NN fitting, with explicit failure
if unavailable; NN inference and serialized models use CPU. The RF uses CPU.

```bash
python -m experiments.thickness_variance.ml audit
python -m experiments.thickness_variance.ml split
python -m experiments.thickness_variance.ml features --available-training
# Once all outputs and execution provenance pass admission:
python -m experiments.thickness_variance.ml features
python -m experiments.thickness_variance.ml train --recipe primary
python -m experiments.thickness_variance.ml train
python -m experiments.thickness_variance.ml report
# Read REPORT.md, validation_summary.json, and the training peak plots first:
python -m experiments.thickness_variance.ml evaluate --review-validation
python -m experiments.thickness_variance.ml report
```

`audit` exits 2 when admission is blocked and leaves a complete issue report.
`split` can succeed with missing outputs if input integrity/pairing passes.
`features --available-training` creates explicitly incomplete **training-only**
diagnostic plots and no trainable feature cache. It never reads test signals for
plots. This mode does not qualify outputs or manufacture execution provenance.

`train --recipe NAME` supports bounded work and restart. Default `train` completes
every prescribed recipe and control. CV checkpoints are per fold/candidate;
final-fit checkpoints are per arm/seed. Work interrupted before a checkpoint is
rerun. No failure is replaced with a different model or silently skipped.
The full search is substantially more expensive than fitting just the primary
model; a large GPU does not accelerate sklearn random forests.

All recipes/controls must finish before tests unlock. The validation-review flag
records that review and freezes further fitting in the run directory. Test
features are created only after this transition. Re-running evaluation is allowed
for unchanged artifacts. Any changed input, code, environment or configuration
requires a new run directory. A scientific recipe change also requires a protocol
revision; using a new directory does not erase prior test exposure.

## Data integrity and execution receipts

The historical `dataset_1000_audit/input_hashes.json` binds all 2,008 audited
input/manifests. Ingestion checks those hashes, unique manifest identities, exact
A/B scalar moisture pairing, deck waveform references, common native timing and
acquisition, output identities, six finite fields, and their lengths. Physical
positions use the bundled solver's half-down cell rounding; CFL verification uses
two active axes and the solver's directed decimal rounding. No scene is derived
or repaired. Symbolic waveform aliases may differ.

Native gprMax HDF5 does **not** record the input digest or establish that a run
completed under the current deck. Matching filenames, titles and finite signals
alone cannot prove provenance. For final training, configure `receipts` to point
to a trusted executor's execution-time ledger:

```json
{
  "schema_version": 1,
  "runs": [
    {
      "arm": "A",
      "sample_id": 1,
      "status": "completed",
      "input_sha256": "<digest recorded for the executed input>",
      "output_sha256": "<digest recorded after successful completion>",
      "solver_version": "3.1.7",
      "backend": "cpu",
      "run_id": "<execution identifier>",
      "executor_identity": "<host/job/solver-build identity>"
    }
  ]
}
```

The executor must record input bytes at execution, successful solver completion,
and the completed output digest. `backend` is `cpu` or `cuda`. This package
verifies receipts but deliberately does not create them retrospectively from
unverified `.out` files. Recover original execution evidence or rerun those decks
with a recording executor when that evidence is absent. Receipts are trusted
provenance assertions, not cryptographic attestations by the ML package. Mixed
backends are blocked in this protocol; parity qualification is separate work.

Passing admission establishes integrity, metadata consistency and provenance.
It is not proof of waveform convergence, CPU/GPU parity or lab calibration.

## Scientific procedure

- Pair IDs are split 700/150/150, seed 2026, stratifying ten equal-width moisture
  bins over [0.10, 0.30]. Five training-only CV folds share membership across arms.
  Unique moisture identities in the audited population prevent cross-ID duplicate
  scenes; unexpected duplicate labels block allocation for explicit grouped review.
- Input is full signed rx1/Ez, with native time zero and recording length. There
  is no AGC, clipping, smoothing, resampling, reflection alignment or per-trace
  amplitude normalization. Thickness, label, IDs and arm are never predictor columns.
- Each arm fits its own transforms and estimator on its own training examples.
  Both models evaluate on both held-out geometries. CV fits preprocessing once
  per arm/fold, reusing it across hyperparameter candidates without exposing any
  held-out labels to the transformation fit.
- The fixed primary RF has 1,000 trees, depth 3, squared-error splits, bootstrap,
  all five features per split, minimum leaf size 1. Tuned `peaks_rf` is a separate
  secondary recipe, not a replacement for `primary`.
- NN reference: 64 ReLU, 64 ReLU, one linear output; Adam 0.001; 100 epochs;
  batch 16; MSE plus 0.01 times sum of squared weight matrices (not biases).
  Targets remain moisture fractions; no clipping of predictions or early stopping.
- RF/GBR/SVR/NN raw benchmarks and RF/NN statistics, peaks, F-test and PCA
  comparisons use exactly the approved grids in `models.py`. One shared candidate
  per recipe minimizes the mean of 20 CV RMSEs (5 folds × 2 train arms × 2 eval
  geometries). Grid order breaks ties. No validation/test refitting occurs.
- Final stochastic fits use seeds 11, 22, 33, 44, 55. SVR and the training-mean
  baseline fit once. Dummy-feature RF trains on one constant-zero feature.
  Permuted-label RF permutes training labels, **including for supervised peak
  selection**, with the same seed-specific permutation in both arms.

## Exact peak feature schema

Detection operates on `abs(Ez)` with `find_peaks(prominence=0.005*max(abs(Ez)),
distance=ceil(0.5/(f_peak*dt)), width=(None,None), rel_height=0.5, wlen=None)`.
Distance is at least one sample. SciPy applies distance filtering before prominence.
Its flat-peak midpoint convention is retained. End samples cannot be local peaks
without two neighbors. No Hilbert envelope is substituted for the absolute trace.

Rank detected candidates by descending prominence, then ascending time index.
Keep up to six and reorder those six chronologically. Each slot contains:

1. original signed amplitude;
2. arrival time in ns;
3. prominence in native field units;
4. half-prominence width in ns using interpolated intersections;
5. trapezoidal integral of absolute field between integer prominence bases,
   inclusive, in native field-units × ns;
6. a 0/1 presence indicator.

Append five adjacent retained-peak time gaps and the total candidate count before
the six-peak cap: **42 columns**. Missing slots and missing gaps are zero-filled;
presence indicators distinguish missing values. Record boundary-touching bases,
flat-adjacent counts, zero/no-peak signals, and SciPy warnings. Overlapping base
areas are independent descriptors, never summed as a claimed energy. These are
waveform lobes, not labeled reflections or inferred soil interfaces.

A training-only RF selector (1,000 trees/depth 3) ranks the 42 features against
moisture, keeping five with feature-name tie breaking. Zero-importance ties are
allowed and reported. A/B selectors may choose different columns. Fit and refit
selection only inside training partitions/folds.

Annotated plots show signed traces, all detected candidates, retained slots,
prominence segments, half-prominence widths, and shaded base intervals. The later
panel begins two source periods after the largest absolute sample (capped at 80%
of the window), ends two periods after the last retained peak (with a minimum
four-period span, capped at the recording end), and rescales only its display.
The full recording remains visible in the first panel. No plotted rescaling enters ML.
Diagnostics choose training IDs spanning moisture/thickness quantiles. In incomplete
mode the quantiles use the available valid-HDF5 training subset.

Other feature sets: nine descriptive statistics reduced to five by the same RF
selector; top 100 continuous-target F-test samples; two centered, unwhitened PCA
components. Extracted columns use training-fitted min/max scaling; raw traces use
one global training-fitted min/max. Training-constant columns always map to zero,
and held-out values are not clipped.

## Evaluation and outputs

Primary ΔRMSE is **B-trained minus A-trained**, on the 150 A test geometries.
Report the same comparison on B test geometries. RMSE is computed per fitted seed
and differences averaged; predictions are not averaged into an unplanned ensemble.
The 10,000-draw bootstrap resamples test IDs jointly for both models and every
seed, giving a conditional test-sample interval. Training-seed SD/min/max are
reported separately. A CI entirely above zero favors A, entirely below zero
favors B, and crossing zero means no detectable difference for this endpoint.

Artifacts include `identity.json`, `audit.json`, `splits.json`, hash-registered
feature NPZ/CSV tables, training peak plots, CV checkpoints, fitted joblib
pipelines, feature rankings, NN loss histories, validation summaries, the test
unlock record, per-test-sample `predictions.csv`, metrics and timings, paired
bootstrap results, selection stability and `REPORT.md`. Joblib artifacts are
trusted local outputs only; do not load untrusted model pickles.

`REPORT.md` can be generated before test unlock to review validation results.
The final report adds test prediction/error plots and the prespecified primary
conclusion. `scores.csv` retains seed-level metrics and inference times. VWC
errors × 100 are **percentage points**, not relative percent errors. Unknown R²
for constant labels is recorded as null.

## Verification

```bash
uv run --no-project --python 3.12 --with-requirements experiments/thickness_variance/ml/requirements.txt python -m pytest experiments/thickness_variance/ml/tests -q
```

Tests use small synthetic fixtures, including explicitly fabricated HDF5/receipts,
to exercise software contracts. They are not scientific simulation evidence.
The real-data audit/development plots are separate artifacts. See
`IMPLEMENTATION_STATUS.md` for the actual verification record and data readiness.
