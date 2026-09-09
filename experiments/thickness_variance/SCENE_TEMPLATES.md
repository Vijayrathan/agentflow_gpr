# A/B pilot scene templates

The complete collector prompts are [INPUT_A.txt](INPUT_A.txt) and
[INPUT_B.txt](INPUT_B.txt). Use separate generation sessions on the updated
`thickness_variance` branch. The files follow the reference conversation's
dataset, layers, buried-target, waveform and antenna sections and explicitly
skip the optional features.

## Selected pilot scene

| Setting | A | B |
| --- | --- | --- |
| Number of samples | 100 | 100 |
| First-layer requested thickness | 0.18–0.24 m | 0.21 m |
| Middle-layer requested thickness | 0.32 m | 0.32 m |
| Terminal layer | Remaining depth | Remaining depth |
| Requested soil depth | 2 m | 2 m |
| First-layer moisture population | 0.10–0.30 | 0.10–0.30 |
| Moisture seed | 42 | 42 |
| Middle/lower moisture | 0.05 / 0.08 | 0.05 / 0.08 |

All layers use fixed sand 40%, clay 15%, derived silt 45%, bulk density
1.50 g/cm³ and particle density 2.65 g/cm³. The lower-layer moisture values
give fixed interfaces while first-layer moisture remains the only varying
material parameter. There are no buried targets or roughness. The excitation
is a 750 MHz peak-frequency Ricker pulse, with a z-polarized Hertzian dipole
in the platform's 2D TMz plane, 0.45 m requested height and 0.08 m Tx/Rx offset.

This uses the platform's current geometry rule: as layer 1 thickens in A,
the layer 2/3 boundary moves downward and the terminal layer's computational
extent shrinks. Layer 2 keeps its own thickness. This is the geometry intervention
to describe in any subsequent modelling comparison; deeper interfaces are not
all held at fixed absolute depths. B has fixed interfaces across its samples.

The template uses `soil_depth_m`, not the ambiguous reference alias `depth_z`.
The terminal layer omits independent thickness fields. Texture and density
ranges are fixed to isolate thickness and moisture. Targets are explicitly
skipped instead of copying the example metal pipe into the experiment.

## Generation check, 2026-09-07

`prepare_scene_templates.py` constructed and validated both exact configurations,
sampled and derived all 200 scenes, and assembled their input text in memory.
No field solves were started and no production `.in` dataset was saved.

- All 100 corresponding moisture-label vectors matched exactly by sample ID.
- Both complete derived numerical plans matched exactly.
- Domain, spacing, time-window, PML, waveform, source and receiver declarations
  matched across all 200 assembled decks.
- Emitted moisture bounds equalled the sampled labels, including their
  floating-point precision.
- Waveform/antenna and global validation reported no errors. The validator
  emitted its informational CFL/iteration-count message for each arm.

Common derived spacing is approximately 2.9769 mm, with an approximately
1.2473 m × 2.5869 m × one-cell domain and a 75.8225 ns recording window.
The requested 0.21 m B thickness snaps to approximately 0.21136 m on this grid;
use emitted effective thickness labels for geometry analysis. Full outputs
are preserved in [SCENE_TEMPLATE_CHECKS.json](SCENE_TEMPLATE_CHECKS.json).

These checks establish input compatibility and A/B matching for this pilot,
not numerical convergence, GPU qualification or ML performance. The common
plan is identical here because the matched material population is identical
and the requested 2 m soil depth dominates the layer-stack sizing constraint
in both arms. Arbitrary later scene changes do not inherit that guarantee.

To expand the pilot, change N in both prompts to the same number and regenerate
both arms. Recheck the numerical plans because a larger moisture population
can change its extrema and therefore the automatically derived grid. Create
validation/test data with separate moisture seeds and keep their IDs out of
training; these two pilot prompts alone do not define the held-out evaluation.
