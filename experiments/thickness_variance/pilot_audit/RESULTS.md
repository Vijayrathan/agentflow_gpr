# Delivered A/B pilot audit

Date: 2026-09-07. Decision: **PASS for scaling input-file generation to 1,000
samples per arm**, with both arms regenerated as complete datasets on their
new shared numerical plan. This is not a signal-quality or ML validation.

Audited folders:

- `dataset/vijay/moisture_A_pilot__e865f98b`
- `dataset/vijay/moisture_B_pilot__bb5f0247`

## Stage 1 — inventory and provenance

Each arm has exactly 100 manifest entries, sample IDs 1–100, derived-label
entries and input files. No files are missing or extra; no sampling/emission
errors were saved. Manifest file paths resolve to the expected files.
Both use `uniform_per_sample`, moisture seed 42 and stream version
`uniform-moisture-v1`. Replaying the recorded sampling inputs reproduces all
saved sample records exactly. SHA-256 hashes of the 200 input files and eight
JSON manifests are in `input_hashes.json`.

## Stage 2 — physical inputs and native material parsing

Every input was inspected, passed through gprMax command-name checking and
its native multi-command parser, and checked against its saved labels.
No embedded Python, include files, unexpected objects, roughness or source
types are present. All 600 layer declarations use equal moisture bounds;
all 30,000 native material bins have finite, nonnegative coefficients and
identical physical properties within their respective layers. Native in-band
permittivity and conductivity agree exactly with the saved derived labels.
Moisture fits the cap and porosity. The native Debye relaxation time exceeds
the 2D CFL time step calculated from the emitted spacing.

All samples retain sand/clay/silt 40/15/45%, bulk density 1.50 g/cm³ and
particle density 2.65 g/cm³. Middle/lower moisture remains 0.05/0.08.
The waveform is Ricker with a 750 MHz peak and the source is z-polarized in
the 2D TMz plane. Source and receiver clear the soil and active PML faces.

Geometry coordinates were resolved to cells using the bundled solver's
rounding function. Each layer spans the full lateral and invariant dimensions;
layers are ordered, contiguous and resolved by more than three cells. The
terminal layer reaches the floor, and its labelled thickness excludes the
bottom absorber. Effective thickness labels agree with the emitted geometry
to within 1e-8 m (the generic geometry serializer uses limited decimal precision).

This stage did not build the full native voxel/Yee coefficient arrays or run
time-domain updates. It checks the input/material declarations and their
resolved coordinates, with those limits kept separate from execution evidence.

## Stage 3 — A/B comparison and coverage

| Quantity | A | B |
| --- | --- | --- |
| Samples | 100 | 100 |
| Unique first-layer moisture labels | 100 | Same 100 |
| Actual moisture range | 0.10004244–0.29880592 | Same |
| Distinct effective top-layer thicknesses | 20 | 1 |
| Effective top-layer thickness | 0.18159099–0.23815211 m | 0.21136000 m |
| Signal output files present | 0 | 0 |

All 100 pairs have exactly matching material commands and moisture labels.
Both global-plan manifests are identical, as are domain, spacing, recording
window, waveform, source, receiver and PML commands across all 200 files.
The emitted grid resolves to 419 × 869 × 1 cells, with approximately
2.976901 mm spacing and a 75.822525 ns requested window. Native 2D timing
rules imply approximately 7.021481 ps and 10,800 iterations.

The platform's informational CFL/iteration warning uses a three-active-axis
calculation and therefore reports a different iteration estimate. The emitted
deck does not specify that estimate; gprMax computes its own 2D time step.
This is a diagnostic discrepancy, not an A/B geometry or timing difference.

Moisture counts in ten equal-width bins over [0.10, 0.30] are
10, 11, 13, 11, 9, 7, 8, 10, 9, 12. The observed Pearson correlation between
moisture and effective thickness in A is -0.02178. This is a descriptive check;
the independent sampling streams, not this statistic, establish the design.

Four pairs (IDs 9, 32, 55, 58) have identical A/B geometries because their
A thickness rounds to B's thickness. This is expected overlap, not an error.
Keep paired identities together when assigning splits; do not put an identical
scene in one arm's training set and the counterpart in a shared test set.

## Stage 4 — 1,000-sample planning check

A read-only, in-memory sampling/derivation check at N=1,000 passed the global
gate for both arms. It created no production input files or signals. Their
new plans still match exactly, and the first 100 moisture labels remain the
same. The maximum sampled moisture rises to 0.29926100, causing the automatically
derived spacing to change from 2.976901 mm to 2.974268 mm. The derived domain,
acquisition coordinates and time window also change slightly.

**Generate all 1,000 files afresh in each arm, using separate new dataset
names. Do not append 900 newly planned files to the existing 100-file pilot.**
Retain the same mode, seed, layer order, ranges and settings in both arms.
After generation, verify the saved files again; identical planning in this
check does not protect against later collector/configuration changes.

## Scope of the decision

No input-validity or A/B-pairing blocker was found. Generating the larger input
datasets can proceed. Neither pilot contains `.out` files, so waveform
stability, useful reflected-signal strength, preprocessing quality, GPU
execution and moisture-estimation performance remain untested for these decks.
The earlier uniform-moisture equivalence study remains evidence for its own
tested configuration, not a substitute for running these pilots.

Evidence: `input_audit.json`, `sample_audit.json`, `input_hashes.json`,
`scale_1000_check.json`. The pilot audit is reproducible with
`python experiments/thickness_variance/audit_pilot.py`.
