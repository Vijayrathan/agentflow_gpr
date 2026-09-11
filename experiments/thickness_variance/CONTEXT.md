# Thickness-variance experiment: continuation context

## Latest checkpoint — 2026-09-10

Remote continuation: the user has native A/B datasets on a Linux server in
`~/agentflow_gpr/dataset/vijay/moisture_A__db8ef04d` and
`moisture_B__2ba27927`, plus root JSONL exports. Use the native directories.
Their first audit failed on old session-name-bound input hashes before inspecting
outputs. The ML CLI now supports `snapshot-inputs` and `config.remote.json` to
freeze those current inputs under portable arm/relative-path keys. This is not
execution provenance. Remote output validity/receipts remain to be checked by the
user; the local counts below do not describe remote availability. See
[remote correction notes](ml/IMPLEMENTATION_STATUS.md). Use the existing Python
environment only; no temporary dependency overlays or new venvs.

The approved standalone ML design is now implemented in [ml/README.md](ml/README.md).
See [implementation status](ml/IMPLEMENTATION_STATUS.md) for 44 passing tests,
the input/output audit and training-only peak diagnostics. Production training
is blocked by incomplete outputs and missing supplied execution receipts.
The user selected paired 70/15/15 splits of the existing population, no AGC,
synthetic-only evaluation, and fixed RF/peak features as the primary comparison.
Those decisions supersede the earlier proposals for separate evaluation seeds
and paper-style AGC. The remainder below preserves the previous handoff.

Saved on 2026-09-08 from the user's previous-chat handoff. Read the complete
[previous-chat summary](PREVIOUS_CHAT_SUMMARY.md) for the detailed decisions,
evidence, numeric results, limitations, and artifact paths. This index records
reported historical status; saving it did not rerun audits or verify current
simulation output status.

## Research question and agreed design

Does training with varied first-layer thickness improve estimation of **first-layer
volumetric moisture**, compared with training at fixed thickness?

- Full arm A: 1,000 samples, requested first-layer thickness 0.12–0.30 m.
- Full arm B: 1,000 samples, requested first-layer thickness 0.21 m.
- Match moisture labels by `sample_id`: `uniform_per_sample`, moisture seed 42,
  first-layer population interval 0.10–0.30. Each sample emits its scalar label
  as both native Peplinski moisture bounds, with 17 significant digits.
- Three layers, fixed texture/densities, middle-layer thickness 0.32 m, lower
  moisture values 0.05 and 0.08. The terminal third layer fills the remaining
  depth; both deeper interfaces move when the first-layer thickness changes.
- 2D TMz, y-up, z-polarized Hertzian dipole, Ricker 750 MHz peak, 0.08 m Tx/Rx
  separation, requested 0.45 m height; no targets, roughness, or added noise.
- Use effective cell-resolved thickness labels for analysis. A realizes 62
  thicknesses; B realizes one. Original `INPUT_A.txt`, `INPUT_B.txt`, and scene
  templates describe the narrower 100-sample pilot, not the full dataset.

## Evidence and its limits

- Separate uniform-moisture CPU/2D fixture: revision 2 completed 44/44 cases;
  all 36 equal-bound candidates matched uniform references bit-for-bit across
  all six recorded components. Reference-only scalar-type normalization fixed
  a rounding mismatch; original failure evidence was preserved.
- Equal bounds and independent seeded per-sample moisture sampling were
  implemented with platform-change authorization on `thickness_variance`.
  The historical full-suite result was 287 passing tests.
- Full input audit passed for 2,000 decks, 6,000 layer declarations, and 300,000
  native material bins, with exact A/B moisture pairing and identical global
  numerical plans. It did not run full field simulations of those decks.
- At the prior inspection, full-dataset signals were absent. H100 access was
  stated available, but remote access and GPU execution were not validated.
- No finalized preprocessing, model, split, training, evaluation, or accuracy
  result exists in the handoff. Improved accuracy, novelty, explicit learning
  of thickness, GPU equivalence, and 3D validity have not been established.

## Dataset and evidence locations

- Full A: `../../dataset/vijay/moisture_A__9b87ee4d`
- Full B: `../../dataset/vijay/moisture_B__5bc60fb5`
- [Full input audit](dataset_1000_audit/RESULTS.md)
- [Uniform-moisture solver validation](../uniform_moisture_validation/revision_02/RESULTS.md)
- [Equal-bound change](MOISTURE_BOUNDS_CHANGE.md)
- [Seeded sampling change](SEEDED_MOISTURE_SAMPLING.md)
- [Pilot audit](pilot_audit/RESULTS.md)

## Continuation constraints and pending work

The latest substantive next step was a close, paper-guided ML pipeline design,
paused before the new review pass. The user's present request is to retain this
context; it does not ask to resume that work automatically.

Follow [Advancing Precision Agriculture using GPR and ML](https://soilx.wpi.edu/pdfs/advancing_precision_agriculture_gpr_ml.pdf)
closely while developing our own implementation. Protected author resources
are not a dependency. Resolve the reported 100-versus-1,000-tree ambiguity and
verify preprocessing/model details from the paper before selecting a design.

Use equal training sizes, paired moisture populations, and the same evaluation
population. Include B's fixed geometry and held-out thickness conditions. Keep
paired/duplicate scenes together across splits, fit preprocessing on training
data only, and use separate evaluation moisture seeds. Eighteen A/B pairs have
identical rounded geometry; their IDs are recorded in the full summary.

Maintain one numerical/acquisition/time plan across training and evaluation:
independently generated populations can change material extrema and therefore
the automatic grid. Design the shared plan explicitly before generating tests.
Associate signals through manifests and sample identities; reject stale,
partial, or failed outputs. Output validation, training, repeated comparisons,
and statistical uncertainty analysis remain pending.

Keep experiment work standalone except for previously authorized platform
generation support. Read root `AGENT.md` fully and
`docs/3D_IMPLEMENTATION_STATUS.md` before future physics, validation, or emission
changes. All sampling, physics, planning, validation, and simulation remain
deterministic Python using native Peplinski routines.
