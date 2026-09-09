# Full A/B dataset input audit

Decision: **PASS input validation; no blocking input issue found before
simulation.** This decision does not claim successful solver execution or
signal quality. No simulations were started during the audit.

Inputs:

- A: `dataset/vijay/moisture_A__9b87ee4d`
- B: `dataset/vijay/moisture_B__5bc60fb5`

## Stage 1 — inventory, provenance and reproducibility

Each folder contains exactly 1,000 input files, 1,000 sampled records and 1,000
derived/emitted label records. IDs 1–1,000 are complete and unique. Manifest
paths match the files on disk; there are no extra input files, recorded sampling
warnings or emission errors. No `.out` files are present.

Both use independent scalar moisture sampling, seed 42, stream version
`uniform-moisture-v1`, three layers and a 2 m requested soil depth. A's first
layer requests 0.12–0.30 m; B requests 0.21 m. Replaying the recorded sampling
inputs reproduces every saved sample record exactly.

## Stage 2 — all-file and native material checks

All 2,000 files passed bundled gprMax command-name and material/source parsing.
The 6,000 layer declarations preserve each sampled moisture label as equal
native bounds. Every one of the 300,000 constructed material bins has finite,
nonnegative Debye coefficients; all bins in a given layer have identical
electrical properties. Their native 750 MHz permittivity and conductivity
agree exactly with the saved derived labels. Moisture fits the supported cap
and porosity; relaxation time exceeds the calculated native 2D time step.

Texture and densities are fixed as specified. Middle/lower moisture remains
0.05/0.08. There are no targets, roughness or embedded executable input blocks.
Layer coordinates resolve to contiguous, ordered volumes spanning the lateral
domain, with more than three cells per layer. The last layer reaches the floor;
its saved thickness excludes the bottom absorber. Source/receiver positions
clear the soil and active PML faces, and polarization is z for the thin-z 2D
TMz geometry.

The geometry labels agree with the cell-resolved emitted coordinates to within
1e-8 m. Requested-to-effective finite-layer thickness differences stay within
half a grid cell (1.487134 mm). The largest observed difference is 1.485677 mm
in A and 1.220903 mm in B.

This stage parses native material/source declarations and resolves coordinates;
it does not construct full native voxel/Yee arrays or run field updates.

## Stage 3 — paired comparison and numerical plan

| Property | A | B |
| --- | --- | --- |
| Samples | 1,000 | 1,000 |
| Unique first-layer moisture labels | 1,000 | Same 1,000 |
| Moisture range | 0.10004244–0.29926100 | Same |
| Distinct effective first-layer thicknesses | 62 | 1 |
| Effective thickness | 0.11897070–0.30040103 m | 0.21117300 m |

All paired moisture vectors and native material commands match exactly by ID.
The two complete global-plan manifests match exactly and also match the earlier
N=1,000 planning check. Emitted numerical values agree with that plan at the
serializer's precision. Source height meets the half-air-wavelength rule, the
useful frequency band lies within 0.3–1.3 GHz, and the configured wavelength
resolution check passes.

Both use 419 × 869 × 1 cells, approximately 2.974268 mm spacing, 10-cell active
PML faces, a 75.88730643 ns requested recording window and identical source/
receiver coordinates. Applying the native 2D timing rules to the emitted
spacing gives approximately 7.015269 ps and 10,819 iterations. These are
input-derived values, not measurements from a completed simulation.

### Harmless waveform-name discrepancy

A uses `ricker_750mh`; B uses `ricker_750mhz_peak`. The earlier pilot audit first
stopped on its hard-coded expected name. Inspection confirmed that every A
source correctly references A's identifier and every B source references B's.
Both definitions are Ricker pulses with amplitude 1 and peak frequency 750 MHz.
The audit now resolves this reference before comparing excitation physics.
No input was changed, and no numerical difference was waived. Raw source/
waveform command strings differ, while their resolved physical settings match.

### Coverage observations

Moisture counts in ten equal-width bins over [0.10, 0.30] are
107, 96, 120, 88, 100, 78, 113, 95, 94, 109. A's observed moisture/thickness
Pearson correlation is -0.02406; this is descriptive, not a proof of statistical
independence. The independent sampling streams define the intended independence.

There are 18 paired IDs whose A thickness rounds to B's thickness, making their
physical scenes identical. They are recorded in `input_audit.json`. Such overlap
is expected; it must not become leakage between training and shared evaluation
sets. The lower-layer interface moves with the top-layer thickness in A while
the middle layer retains its own fixed thickness, as previously specified.

## Stage 4 — scope and integrity

The audit read the submitted datasets and wrote evidence only in the experiments
directory. Hashes cover all 2,000 input files and eight JSON manifests. A final
integrity comparison is saved in `final_integrity_check.json`.

Input validation is complete. Native model-build/field execution, CPU/GPU
behaviour, waveform stability, signal informativeness and downstream modelling
quality remain to be checked using generated outputs. A successful input audit
does not substitute for those execution checks.

Evidence: `input_audit.json`, `sample_audit.json`, `input_hashes.json`,
`planning_consistency.json`, and `final_integrity_check.json`.

Reproduction:

```bash
.venv/bin/python experiments/thickness_variance/audit_pilot.py --a dataset/vijay/moisture_A__9b87ee4d --b dataset/vijay/moisture_B__5bc60fb5 --n 1000 --a-min 0.12 --a-max 0.30 --out experiments/thickness_variance/dataset_1000_audit
```
