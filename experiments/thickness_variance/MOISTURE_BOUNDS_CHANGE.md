# Equal moisture bounds on thickness_variance

Date: 2026-09-07. Scope: accept explicitly equal moisture bounds in the platform
on the user-authorized `thickness_variance` branch.

Follow-up: [seeded moisture sampling](SEEDED_MOISTURE_SAMPLING.md) implements
per-sample scalar assignment after this stage. The record below describes the
earlier equal-bound acceptance change.

## Stage 1 — inspect the generation path

The extracted-layer schema already accepted equal bounds. The sampled-layer
schema, layer sampler and per-sample validator rejected them. The agent prompt
and an extracted-schema comment incorrectly claimed that a moisture sub-band
was drawn for each sample; the actual sampler preserves the input endpoints.

## Stage 2 — implementation

All three ordering checks now accept `theta_v_min <= theta_v_max`. Reversed
bounds remain invalid. Existing porosity and calibration-cap checks remain in
place. The agent prompt explains that equal bounds specify spatially uniform
moisture and must not be widened automatically.

`DatasetConfig.fractal_nbins` now requires at least two bins. Previously it
accepted one, which native Peplinski bin construction cannot use even when the
two moisture endpoints are equal. The default remains 50.

The requested endpoints still pass unchanged to every sample. This change does
not assign different moisture values across samples, collapse nonzero ranges,
alter the gprMax solver, or establish a shared numerical plan across A/B jobs.

## Stage 3 — verification

Added `backend/dataset_sampling/tests/test_uniform_moisture.py`:

- Uniform moisture 0.10, 0.20 and 0.30 with both 2 and 50 bins survives
  extraction, sampling, manifest round-trip, material derivation and emission.
- The emitted material declaration is parsed by bundled gprMax and every
  native Debye bin has identical electrical properties for equal bounds.
- A nonzero 0.10–0.20 range remains a range and produces different edge-bin
  responses; it is not silently converted to a scalar.
- Reversed bounds, moisture above the calibration cap, moisture exceeding
  porosity and a one-bin configuration are rejected.

The initial test run exposed an incomplete native-parser test fixture and the
missing minimum-bin guard. The fixture now supplies the parser's full command
key set, and the configuration guard was added before verification continued.

Results:

```text
.venv/bin/python -m pytest backend/dataset_sampling/tests/test_uniform_moisture.py -q
12 passed in 0.41s

.venv/bin/python -m pytest backend/tests backend/dataset_sampling/tests -q
280 passed in 2.23s

git diff --check
passed
```

These checks exercise the material build, not a new time-domain field solve or
GPU validation. The earlier uniform-moisture signal experiment remains the
evidence for waveform equivalence in its tested CPU/2D configuration.

## Remaining dataset preparation

The target is first-layer moisture only. A 100-sample-per-arm pilot followed by
1,000 per arm was proposed, not yet fixed as the experimental sample count.
The first layer needs a matched per-sample scalar-moisture schedule across A
and B, independent of thickness. A single equal-bound pair in an N-file job
still gives N copies of the same moisture label. The geometry compensation
rule and common numerical plan across the two arms also need to be settled
before final scene specifications and modelling datasets are generated.
