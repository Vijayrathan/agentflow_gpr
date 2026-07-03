# agent.md — GPR Synthetic Dataset Pipeline

Guidance for any coding assistant working in this repo. Read fully before editing.
This encodes **physics invariants, design decisions, and rejected alternatives** — things you
cannot infer from the code. Do not revive rejected approaches.

---

## Purpose

Generate `N` labeled gprMax `.in` files (a synthetic dataset) to train ML on subsurface soil
characterization: predict sand/clay %, volumetric water content (θv), layer thickness, and
buried-object properties from GPR signals. Thesis project; real lab data exists for sim-to-real.

---

## Hard architectural boundary (non-negotiable)

- **Agentic layer = collection/extraction only.** LLM agents extract config and re-elicit on
  _human-decision_ failures (infeasible texture, out-of-band frequency).
- **Deterministic core = everything else.** Sampling, ε-solving, derivation, validation, emission,
  gprMax execution, error handling are plain code.
- The orchestrator is **langgraph nodes**, not an agent.
- **Never** route gprMax numerical errors through an LLM "rectifier" — most are pre-flight
  catchable by Tier-3 validators, and an LLM could paper over a real physics violation.
- **Never** make physics/derivation/sampling into agents.

---

## The parameter selection chain (strict order — from Khosravi §III)

```
soil electrical (ε, σ)  →  waveform + center freq  →  Wang band (fmin/fmax)
  →  Peplinski validity gate (0.3–1.3 GHz)  →  wavelength budget (λmin, λmax)
  →  {Δx, domain, antenna height, depth, buffer}  →  Δt (CFL)  →  time window
```

- `domain_x`, `domain_y`, `source_height_m`, `max_cell_m`, `Δt`, and `time_window` are **DERIVED**,
  never collected as free user values.
- **Δt is the OUTPUT of the chain, never an input.** ε → λ → Δx → Δt is one-way.
- Soil-before-waveform is justified **only** by the Peplinski gate (soil model defines the legal
  frequency window). fmin/fmax depend **only** on the waveform, not on permittivity.

---

## Permittivity: gprMax-native Peplinski ONLY

- gprMax natively supports **only Peplinski**. Do **not** reimplement CRIM / Dobson / Mironov or any
  manual mixer. The old four-model `physics_modelling.py` solver is deleted; do not resurrect it.
- Only **five** soil params map to `#soil_peplinski`: sand fraction, clay fraction, bulk density,
  sand-particle density, and the **θv RANGE**. Silt, porosity, organic fraction, salinity,
  porewater σ, temperature are **auxiliary/label/validity only** — not gprMax inputs.

### Computing a-priori ε for grid sizing (the one surviving "solver" need)

Use gprMax's own routine — the SAME ε it will simulate:

1. `PeplinskiSoil(name, sand_frac, clay_frac, rho_b, rho_s, (theta_v_min, theta_v_max))`
   (sand/clay = %/100; moisture is the band).
2. Pass a **stub Grid `G`** — `calculate_debye_properties` only reads `len(G.materials)` and appends;
   it does **not** read `G.dt`.
3. `calculate_debye_properties(nbins, G, name)` to populate bins.
4. `Material.calculate_er(f_center).real` on the bins.
   **Do NOT read raw `m.er`** — that's the infinite-frequency value; it understates ε and would make
   the grid too coarse. `calculate_er` folds Debye relaxation + conductivity back in.

- **Bin midpoint shift:** materials are made at bin midpoints, so the wettest sits **half a bin above
  `theta_v_max`**. Use **`mumaterials[-1]`** for the wettest, and bind the porosity cap accordingly.
- Wettest bin → largest ε → smallest λmin → drives **global Δx**.
- Driest bin → smallest ε → largest λmax → drives **global domain**.
- σ is **not** needed for sizing (only real ε enters the wavelength budget).
- τ-vs-Δt (watertau > Δt) is a **validation gate after Δt exists**, not a computation input; with
  watertau = 9.231 ps it essentially always passes.

---

## Ricker peak-vs-center caveat (CRITICAL failure mode)

`#waveform ricker` takes the **PEAK** frequency. Wang 2015 band edges (from peak `fp`):

- `fmin = 0.481623 · fp` `fmax = 1.636567 · fp` `f_center = 1.059095 · fp`

Rules:

- If the collected center IS the peak → write it directly.
- If the collected center is the band-CENTER → convert to peak via **÷ 1.059095** before writing.
- **Silent-invalidation trap:** writing an 825 MHz _band-center_ as the peak gives
  `[397, 1350] MHz` → breaches the 1300 MHz Peplinski ceiling. The `center_freq_is_peak` boolean in
  `DatasetConfig` gates this.
- **Verified test vector:** 825 MHz center → fp ≈ 778.96 → fmin ≈ 375.17, fmax ≈ 1274.83,
  BW ≈ 899.66 MHz; must pass the Peplinski gate.

### Two different "highest frequencies" — do not conflate

- **Validity gate**: Wang −6 dB edges (fmin/fmax) vs 0.3–1.3 GHz.
- **Resolution check** (`Δx ≤ λmin/10`): uses the highest **significant** frequency (~−40 dB,
  ~2–3× center), via `high_freq_factor` (default 3.0) or per-waveform `BW_MULTIPLIER`.
  **Never default the multiplier to 1.0** (disables the safety).

---

## Coordinate convention (verified against known-good `.in`)

- `#domain: x y z` → **x = horizontal**, **y = VERTICAL**, **z = thin invariant axis (1 cell)**.
- Layout is **soil-on-bottom, air-on-top, antenna at surface**.
- Emission is positional `(x, y, z)`: vertical y in slot 2, single-cell z in slot 3; every object
  spans the full z cell.
- Keep vertical = y (gprMax 2D convention). Do **not** switch to z-vertical.
- Layout math (y vertical): `pad = (pml_cells + buffer_cells)·Δx`; `ground_y = pad + depth`;
  `tx_y = ground_y + source_height`; target at depth d → `y_center = ground_y − d`.
  Bottom→top: `pad | soil(depth) | air(source_height) | pad`.

---

## Global-grid & sampling invariants

- **ONE grid for the whole dataset** (Δx=Δy=Δz cubic, sized from worst-case ε corners) → identical
  time axis and boundary geometry across all samples → signals directly comparable. The grid must
  **never** follow a target per-sample.
- **θv is always a per-sample BAND `(min, max)`**, passed whole to `#soil_peplinski`. **Never a
  scalar** — gprMax needs a range to build the fractal material series. The band varies per sample.
  Draw densities first, then cap `θv_max` at porosity `n = 1 − rho_b/rho_s` (or reject).
- **Per-sample thickness is a label** and varies inside a fixed-depth box. Global depth is sized to
  the worst-case (deepest) stack. Vary the physics you want predicted inside a frame held constant.
- **CFL (cubic):** `Δt = Δ / (c·√n_dim)`, n_dim = 2 (2D) or 3 (3D).
- Smallest target feature ≥ 10 cells may tighten Δx below λmin/10.

---

## Variable targets + static antenna

- Target **geometry** (position, depth, radius) is **per-sample**; its **grid/domain requirements
  aggregate to global worst-case corners** (smallest feature tightens Δx; largest extent enlarges
  domain; deepest bottom enlarges depth).
- **Tx/Rx are STATIC** — derived once in global derive, validated once.
- **Targets are PEC** → they do **not** feed ε corners (only size/extent). Revisit only if target
  material becomes sampled.
- **Redraw on placement failure:** radius range `[r_floor, original_radius]` (shrink/reposition
  only, never grow). `r_floor = max(5·Δx, radius_min_m)`. If no valid center exists even at
  `r_floor`, drop immediately. Assert `original_radius >= r_floor` at runtime so a future Δx change
  fails loudly. Cap attempts at `MAX_TARGET_ATTEMPTS`.
- **Dropped samples reduce N** (not backfilled); log `(sample_id, reason)`. Document the delivered
  count in the manifest.
- **Ghost corner** (a dropped sample may have set a grid corner) is **accepted, not re-derived** —
  re-deriving is circular, and the error is always in the safe direction (grid oversized, never
  undersized). Document in out-of-scope.

---

## Pipeline stage order

1. Layer ranges (pull center frequency forward so step 2 can gate).
2. Draw N samples (layers + target geometry); fire grid-independent validation.
3. Extract waveform + antenna (single, non-sampled).
4. Cross-stage validation (Peplinski band gate, antenna config).
5. Per-sample physics + aggregate corners (ε/σ at f_center → wet/dry-edge ε; target bbox, smallest
   feature = 2·radius, extent = 2·radius, bottom depth = depth+radius). Aggregate the global corners.
6. Global derive once (λ budget → **freeze Δx first**, then clearance/depth math, domain, source
   height, static Tx/Rx coords, snap to integer cells, Δt, time window).
7. Global validation once (grid/numerics; cascade: domain-fit → placement/stratigraphy →
   feasibility).
8. Per-sample placement validation (in-domain, ≥ pml+15 cells from boundary, ≥ 10 cells across,
   fully buried `depth ≥ radius`); redraw then drop.
9. Emit N `.in` files (pure string assembly; **no derivation in the writer**).
10. Run gprMax; store.

---

## Validation tiers

- **Tier 0 Schema** (construction): single-field/within-stage invariants (ranges min≤max,
  sand+clay≤100, θv-envelope ≤ loosest porosity, num_layers==len, resistance bound, band min<max).
- **Tier 1 Collect** (per agent): texture closure, Peplinski calibration envelope, band gate on
  **derived Wang edges**, antenna config.
- **Tier 2 Per-sample**: texture sum, bulk<particle, θv_max ≤ porosity, Peplinski bounds.
- **Tier 3 Global** (needs Δx/Δt/domain): grid λ/10, alignment, CFL, τ-vs-Δt, PML-vs-domain, memory,
  placement, travel time, stratigraphy. **Cascade order:** domain-fit first, then placement, then
  feasibility.
- **Tier 4 Emission**: material name rules, mandatory commands (`#domain`, `#dx_dy_dz`,
  `#time_window`), snapshot ≤ window.

Severity when reviewing: **CRITICAL** (fails/invalid) / **WARNING** (degraded/artifact) /
**SUGGESTION** (best practice).

---

## REJECTED alternatives — DO NOT reintroduce

- ❌ Manual dielectric mixers (CRIM / Dobson / Mironov). Peplinski-only, gprMax-native.
- ❌ Scalar θv anywhere. Always a band.
- ❌ Per-sample grids. One global grid.
- ❌ Solving ε before sampling. Sample first, then solve.
- ❌ Reading raw `m.er` for grid sizing. Use `Material.calculate_er(f).real`.
- ❌ Peplinski gate on the center frequency. Gate on the derived Wang band edges.
- ❌ Deriving Δt as an input, or reordering the chain to feed Δt upstream.
- ❌ BW multiplier defaulting to 1.0.
- ❌ temperature_c affecting the simulation (gprMax hardcodes water at 1.3 GHz; label-sanity only).
- ❌ Treating `time_window` as `source_end_time` (separate concepts).
- ❌ The "50–100 Ω recommended resistance" warning (ungrounded; removed).

## Key constants

| Quantity                  | Value                               | Source                     |
| ------------------------- | ----------------------------------- | -------------------------- |
| C0                        | 299792458 m/s                       | —                          |
| Wang lower / fp           | 0.481623                            | Wang 2015                  |
| Wang upper / fp           | 1.636567                            | Wang 2015                  |
| Wang central / fp         | 1.059095                            | Wang 2015                  |
| Peplinski validity        | 0.3–1.3 GHz                         | Peplinski 1995             |
| Peplinski θv max          | 0.30                                | calibration                |
| Peplinski texture         | sand 15–50, silt 35–65, clay 5–20 % | 1995 Table I (our policy)  |
| watertau                  | 9.231e-12 s                         | gprMax Material            |
| watereri / waterer        | 4.9 / 80.1                          | gprMax Material            |
| erealw hardcoded freq     | 1.3e9 Hz                            | calculate_debye_properties |
| Max resistance            | 376.73 Ω (exclusive)                | free-space impedance       |
| PML gap                   | 15 cells                            | gprMax guidance            |
| pml_cells default         | 10                                  | gprMax default             |
| buffer_cells              | 10                                  | our default                |
| cells_per_wavelength      | 10                                  | Yee rule                   |
| high_freq_factor (ricker) | 2.5 table / 3.0 scalar              | resolution check           |
| r_floor                   | max(5·Δx, radius_min_m)             | 10-cell rule               |
| MAX_TARGET_ATTEMPTS       | 20 (placeholder)                    | our config                 |
| fractal_nbins             | 50 default                          | gprMax fractal_box         |
| surface dimension         | 1.5·λmax                            | Khosravi                   |
| antenna height            | ≥ λmax_air/2                        | Khosravi                   |
| CFL (cubic)               | Δt = Δ/(c·√n_dim), n_dim=2 or 3     | —                          |
| Reserved material names   | pec, free_space, grass, water       | gprMax                     |

---

## Open decisions (do not silently resolve — flag to the human)

1. Store-once vs store-twice (params before gprMax + signals after).
2. Dropped-sample handling: drop-and-reduce-N (chosen) vs backfill.
3. `MAX_TARGET_ATTEMPTS` real value.
4. Sampler θv policy (center+width vs two ordered values) and ML label (midpoint/range/edge).
5. Lab instrument antenna/frequency, per-scan ground truth, survey geometry (gates Chapter 3).

---

## Working rules for the assistant

- SI units throughout. Positional `(x, y, z)` with y vertical, z single-cell.
- Every derived value must trace to the chain above; show the calculation, don't hardcode.
- Flag any config that breaks CFL, cells-per-wavelength, PML buffer, or places a source/target in
  PML — even if not asked.
- Mandatory commands present: `#domain`, `#dx_dy_dz`, `#time_window`.
- No derivation logic in the emission/writer stage.
