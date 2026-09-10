# ML design for the thickness-variation experiment

## 1. Objective and agreed scope

Test whether training on **A’s varied first-layer thickness** improves first-layer moisture prediction compared with training on **B’s fixed thickness**.

Follow the paper’s sequence: signal preparation → raw-signal model comparison → feature extraction and selection → model comparison → held-out evaluation. Its methods provide the reference; missing implementation details receive explicit, reproducible defaults. [Paper, §§II-C–III and Tables IV–VI](https://soilx.wpi.edu/pdfs/advancing_precision_agriculture_gpr_ml.pdf)

Source reviewed: :codex-file-citation{path="/private/tmp/thickness_ml_reference.pdf" purpose="source"}

Agreed decisions:

- Synthetic data only; real-radar calibration and mixed-data training are deferred.
- Split the existing paired populations **70%/15%/15%**.
- Omit AGC.
- Prespecify **RF with peak features** as the primary comparison.
- Predict first-layer volumetric moisture directly from its Peplinski label.
- Use thickness only for evaluation and analysis, never as a model input.
- Report improvement, no detectable benefit, or degradation according to the evidence.

Implement the experiment under `experiments/thickness_variance/ml/`, with no platform, physics, or dataset changes.

## 2. Data preparation and paired splits

**Input:** the two audited datasets’ manifests, labels, and completed native HDF5 outputs.

Join records through emission-manifest identities and `sample_id`. Verify matching A/B moisture labels, input integrity, output provenance, receiver/source metadata, grid, time step, recording length, and all six finite field arrays. Zero-valued symmetry components are valid. Use **rx1/Ez** for ML.

Retain the native **10,819-point** signal and time axis when confirmed across the completed population. Do not force the paper’s signal length onto these simulations.

At inspection, A contained 311 outputs and B contained none locally. Full model training waits for all 2,000 outputs to pass ingestion checks; incomplete runs can support development checks only. Missing or invalid cases produce an explicit failure report rather than silently changing the population.

**Split policy:**

- Sort the 1,000 pair IDs and stratify by ten equal-width moisture bins over 0.10–0.30.
- With split seed **2026**, allocate 700 training, 150 validation, and 150 test IDs. Apply identical membership to both arms.
- Keep paired and physically identical scenes within the same split.
- Save the split manifest before feature fitting or model comparison.
- Model A trains exclusively on A’s 700 signals; model B exclusively on B’s corresponding 700.
- Both models evaluate on the same validation and test collections: the held-out A signals and held-out B signals.

This choice tests unseen samples from the existing populations. It does **not** establish generalization to a new sampling seed, an unseen thickness range, or a different acquisition setup.

## 3. Signal handling and feature representations

Preserve complete signed traces and their native timing. Apply no AGC, per-trace normalization, reflection alignment, gain, smoothing, resampling, or amplitude clipping. Use a zero time shift because these synthetic traces share a source clock; do not introduce a velocity-dependent correction.

Extract features from the unmodified signals. Fit subsequent scaling, feature selection, and PCA exclusively on the relevant training arm, including inside every cross-validation fold. Pipelines will enforce that separation. [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)

Compare these representations:

| Representation | Definition |
|---|---|
| Raw | Complete Ez trace; one global training-derived min–max transformation shared across its time samples. |
| Descriptive statistics | Minimum, maximum, mean, population standard deviation, quartiles, skewness, and excess kurtosis; RF importance selects five. |
| Peak features | Reproducible peak candidate bank described below; RF importance selects five. |
| Regression F-test | Select the 100 highest-scoring time samples using `f_regression`, adapting the paper’s ANOVA approach to a continuous target. |
| PCA | Two components, fitted to centered training traces, without whitening; record explained variance. |

For extracted representations, fit a separate min–max scaler per retained feature. Do not clip validation/test values outside the training range. Constant training features map to zero.

**Peak extraction defaults**, explicitly our implementation choices because the paper does not supply its complete extractor:

- Detect local maxima of `abs(Ez)`.
- Minimum prominence: **0.5% of the trace’s maximum absolute amplitude**.
- Minimum separation: **0.5/source peak frequency**, converted to a whole number of samples.
- Retain the six most prominent peaks, ordered by arrival time.
- Record each peak’s signed amplitude, time in ns, prominence, half-prominence width, and integral of `abs(Ez)` between its prominence bases.
- Add total detected peak count, consecutive retained-peak spacings, and presence indicators.
- Missing slots and unavailable spacings receive zero with their presence indicators retained.
- Fit a 1,000-tree, depth-3 RF selector on training data; retain the five highest-importance candidates, breaking ties by feature name.

Save candidate definitions, selected feature names, and training-only diagnostic plots. Do not call the reconstructed five-feature set identical to the authors’ private implementation.

## 4. Models and training procedure

The primary model is fixed before examining validation or test performance:

**Ez → peak candidate bank → training-fitted RF selection of five features → RF regression → first-layer moisture**

Its RF uses **1,000 trees, maximum depth 3, squared-error splits, bootstrap sampling, all available features per split, and minimum leaf size 1**.

The paper contains conflicting RF descriptions. Use its later detailed configuration as the reference baseline, documenting that choice rather than claiming the inconsistency is resolved.

The NN reference is:

**Input → Dense(64), ReLU → Dense(64), ReLU → Dense(1), linear**

Use Adam, learning rate 0.001, batch size 16, 100 epochs, and MSE plus a 0.01 L2 penalty on weight matrices. One output replaces the paper’s six outputs. Keep targets in volumetric-moisture units and do not clip predictions.

**Comparison sequence:**

1. Benchmark RF, gradient boosting regression, RBF-SVR, and NN on raw signals.
2. Compare RF and NN across all four extracted representations.
3. Preserve the fixed primary RF/peak result separately from tuned secondary results.

Use five-fold cross-validation within the 700 training IDs, with matched fold membership across arms. Fit each arm independently; score both models on both geometries of each held-out fold.

For secondary tuning, use these bounded grids:

| Model | Search |
|---|---|
| RF | Trees {100, 1,000}; depth {3, 8, unlimited}. |
| Gradient boosting | Trees {100, 300}; depth {2, 3}; learning rate {0.03, 0.1}. |
| SVR | C {1, 10, 100}; epsilon {0.001, 0.01}; gamma=`scale`. |
| NN | Hidden layers fixed at 64–64; learning rate {0.0003, 0.001}; L2 {0, 0.001, 0.01}. |

Choose one shared hyperparameter setting per model/representation by mean cross-validation RMSE across the four train-arm/evaluation-geometry combinations. Break ties by listed grid order. Each arm still fits its own transformations and estimator.

Use validation results to summarize secondary recipes before unlocking tests. Do not retrain on validation data: final models retain equal 700-sample training populations.

Use scikit-learn for classical models and transformations, and PyTorch for the NN. Record exact dependency versions, seeds, hardware, and execution settings. Repeat stochastic final fits with matched seeds **11, 22, 33, 44, 55**; deterministic SVR needs one fit.

## 5. Evaluation, deliverables, and acceptance

The primary endpoint is:

**ΔRMSE = RMSE(B-trained model) − RMSE(A-trained model)**  
evaluated on the **150 variable-thickness test signals**, using the fixed RF/peak recipe.

Positive values favor A. Also report both models on the 150 fixed-thickness test signals, so a benefit under variation is considered alongside performance at B’s training geometry.

Report:

- RMSE, MAE, signed bias, and R² in moisture units; express errors additionally as volumetric-water-content percentage points.
- Per-run results and their mean and spread.
- A 95% paired bootstrap interval for ΔRMSE, using 10,000 resamples of test pair IDs and keeping repeated-run predictions together. This interval measures held-out sample uncertainty conditional on the fitted runs; training-seed variability is reported separately.
- Error versus effective thickness and moisture, predicted-versus-true plots, training/validation gaps, and feature-selection stability.
- Feature extraction, fitting, and inference time separately.
- Training-mean and dummy-feature controls, plus a training-label permutation check for suspicious leakage.

A favorable result requires evidence on the prespecified endpoint; secondary winners do not replace a null primary result. Any conclusion remains specific to these synthetic populations and this training procedure.

Provide a configuration-driven CLI with `audit`, `split`, `features`, `train`, `evaluate`, and `report` commands. Save data hashes, split membership, preprocessing and model artifacts, selected features, cross-validation results, per-sample predictions, metrics, and a paper-to-implementation decision record.

Acceptance checks cover shuffled file order, broken A/B pairing, stale or partial outputs, train/test overlap, held-out-data changes leaving fitted preprocessing unchanged, peak edge cases, model save/reload parity, reproducibility, and correct error units. Passing these checks establishes a reproducible experiment—not a predetermined advantage for A.
