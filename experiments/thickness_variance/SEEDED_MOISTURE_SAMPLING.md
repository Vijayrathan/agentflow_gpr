# Matched moisture sampling for A/B datasets

Date: 2026-09-07. Branch: `thickness_variance`.

## Stage 1 — agreed behaviour

The user specified a dataset population range, N moisture draws and a shared
seed for the varied- and fixed-thickness arms. Each realized sample must be
uniform within its layer, with the same moisture label as its A/B counterpart.
The modelling target remains first-layer moisture.

## Stage 2 — implementation

Dataset configuration now accepts:

```text
moisture_sampling=uniform_per_sample
moisture_seed=42
```

The existing per-layer `theta_v_min` and `theta_v_max` fields define that layer's
population range in this mode. This avoids a second, conflicting set of range
fields. For example, first-layer bounds 0.10 and 0.30 produce samples between
10% and 30% volumetric water content. Both native bounds in each file receive
the same sampled value. Equal bounds in the lower layers keep their moisture
fixed across the dataset. The range must be positive and within the supported
moisture and porosity limits; zero is excluded because native Peplinski
coefficient construction divides by moisture.

The default remains `preserve_band` for existing heterogeneous-soil workflows.
The agent collects and explains the new mode and seed, and the active sampling
node forwards both to the deterministic sampler. Editing these configuration
fields participates in the existing configuration-staleness/resampling path.

Each scalar uses Python's seeded random generator with an explicit string key:
`uniform-moisture-v1:<moisture_seed>:<sample_id>:<zero-based-layer-index>`.
There is no dependency on the dataset name, thickness settings, target draws,
geometry seed, or previous RNG consumption. Soil-validity retries keep the
chosen moisture fixed. An infeasible sample fails rather than replacing its
label with another draw. Increasing N preserves the existing sequence prefix.

The sampling manifest records the mode, moisture seed, stream version,
requested layer ranges, geometry seed and all realized per-sample bounds.
Moisture bounds are emitted with 17 significant digits to preserve their
floating-point values through the native parser. Similarity reuse filters
separate different moisture modes and, in scalar mode, different moisture
seeds. Old search-index entries lacking these fields require re-indexing to
match the new filters; no database or index operations were performed here.

## Stage 3 — verification

- A 100-sample varied-thickness arm and a 100-sample fixed-thickness arm had
  exactly matching moisture labels by sample ID. The test changed the geometry
  seed and consumed extra geometry random numbers before each soil draw to
  verify that pairing is independent of that stream.
- First-layer values stayed in the requested 0.10–0.30 range, with equal bounds
  inside each sample. Lower-layer moisture stayed exactly 0.05.
- Repeated seeds reproduced labels; changing the moisture seed changed them;
  extending N preserved the original prefix.
- The active pipeline node saved the requested ranges and seed. Every emitted
  moisture value in its five-sample fixture exactly matched the manifest label.
- Invalid population ranges were rejected. Existing nonzero-band behaviour
  and equal-bound native material checks continued to pass.
- Reuse filters distinguished spatial-band and scalar-sampling populations,
  as well as different scalar-sampling seeds.

```text
.venv/bin/python -m pytest backend/dataset_sampling/tests/test_uniform_moisture.py backend/tests/test_sim_similarity.py -q
36 passed in 0.80s

.venv/bin/python -m pytest backend/tests backend/dataset_sampling/tests -q
287 passed in 2.43s

git diff --check
passed
```

These are generation and material checks, not new field simulations or model
training results. They extend the earlier equal-bound acceptance change.

## Conditions for the A/B experiment

Use the same mode, moisture seed, per-layer moisture ranges, layer ordering and
N in both arms. Match samples by `sample_id`, not directory listing order. If
later target placement rejects samples, compare common IDs and account for
rejections; do not renumber surviving files to manufacture pairs.

Seeded moisture pairing does not itself freeze the numerical domain, grid,
antenna positions or time axis across independently generated jobs. The shared
numerical plan and thickness-compensation rule remain to be settled before
final A/B scene specifications are used for modelling data generation.
