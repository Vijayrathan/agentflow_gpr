# GPR Synthetic Dataset Pipeline — Complete Conversation Summary

Comprehensive record of every decision, consideration, source, constant, and open question
from the design conversation. Nothing filtered for relevance.

---

## 0. PROJECT OVERVIEW

**Goal:** Generate N labeled gprMax `.in` input files (a synthetic dataset) for ML training on
subsurface soil characterization — predicting sand/clay %, volumetric water content (θv), layer
thickness, and buried-object properties from GPR signals.

**Thesis context:** This is a thesis. Real lab GPR data is available. Four-chapter arc (see §11).

**Top-level architecture:** A small **agentic** conversational front-end that builds one validated
`DatasetConfig`, then a **deterministic** batch generator (sample → solve → derive → validate →
emit → run gprMax). The hard boundary between agentic and deterministic is a central design
principle, repeatedly defended.

**Artifacts produced in this conversation:**
1. `gpr_dataset_schema.py` — Pydantic schema + per-sample/global derive scaffolding.
2. `validation_rules.py` — tiered, Peplinski-only validators.
3. `claude_code_migration_prompt.md` — full-migration prompt (if base pipeline not migrated).
4. `claude_code_target_geometry_prompt.md` — incremental prompt (base already migrated).

---

## 1. SOURCES / PAPERS / FILES REFERENCED

- **gprMax** (Warren, Giannopoulos, Giannakis 2016, Computer Phys. Commun. 209:163-170). The FDTD
  simulator. Native Peplinski soil; computes ε/σ and Debye poles itself.
- **gprMax `PeplinskiSoil` class** + **`Material` class** (source pasted in full). Key methods:
  `calculate_debye_properties(nbins, G, fractalboxname)` and `Material.calculate_er(freq)`.
- **FDTD Medium Dimension Selection Guidelines** (Khosravi Largani, Zekavat, Namdari — IEEE GRSL,
  WPI). Section III is the parameter-selection reference. Key result: surface dimension ≈ **1.5·λmax**.
- **Wang 2015, "Frequencies of the Ricker wavelet"** (Geophysics 80(2):A31–A37). Provides the
  Ricker band-edge equations via the Lambert W function.
- **Peplinski, Ulaby, Dobson 1995, "Dielectric properties of soils in the 0.3–1.3 GHz range"**
  (IEEE TGRS 33(3):803–807). The mixing model gprMax implements; validity window 0.3–1.3 GHz.
- **Dobson 1985** — Eq. (22) for solid permittivity `εs = (1.01 + 0.44·ρs)² − 0.062`, used
  internally by gprMax's Peplinski. Also a 1.4–18 GHz model branch (NOT used; gprMax native is
  the 0.3–1.3 GHz branch).
- **Project files also present (context only):** Giannakis ML fast-forward solver for GPR/FWI,
  deep-learning FWI, DeclutterGAN (clutter removal), wavelet scattering, Topp, Mironov 2004.
- **Example known-good `.in` file** (metal cylinder in dielectric half-space) — used to VERIFY the
  coordinate convention (see §9).

---

## 2. PARAMETER GROUPING (ORIGINAL TASK)

Started from a ~50-parameter list. Grouped into 6 logical groups for agent allocation, ordered by
dependency. Final agent grouping (after the Khosravi paper reshaped it):

1. **Soil / subsurface properties** (collect first): sand/silt/clay %, θv band, bulk density,
   particle density, porosity, organic_fraction, salinity_classes, porewater_sigma_Sm,
   temperature_c, fractal_nbins, surface_roughness, enforce_validity, name.
2. **Waveform / illumination**: waveform_kind, center_freq_hz, amplitude, name; source_start/end_time.
3. **Antenna**: antenna_kind, antenna_axis, resistance, tx_rx_offset_m, source_height_m,
   rx_same_height, rx_array.
4. **Buried targets / geometry**: cylinders, boxes, spheres (later made per-sample variable).
5. **Domain/discretization/boundaries** (DERIVED, not collected): max_cell_m, cells_per_wavelength,
   domain_x, domain_y, pml_cells, num_samples.
6. **Run control / output**: model, title, output_dir, num_threads, snapshots.

**CRITICAL grounding flag:** Only **five** soil parameters map directly to `#soil_peplinski`:
sand fraction, clay fraction, bulk density, sand-particle density, and the **volumetric water
fraction RANGE**. **Silt, porosity, organic fraction, salinity, porewater σ, and temperature are
NOT direct gprMax inputs** — they are auxiliary/validity/label quantities.

---

## 3. THE KHOSRAVI PAPER (Section III) — SELECTION CHAIN

The paper's parameter selection order (became our derivation ordering):
1. Soil electrical (ε, σ) chosen first: pairs **(4,1), (8,80), (38,80), (38,20)**.
2. Waveform = **Ricker**, center freq **fc = 825 MHz**.
3. Derive **fmin = 375.168 MHz, fmax = 1274.831 MHz** (from Wang 2015), bandwidth **899.663 MHz**.
4. **Peplinski validity gate**: band chosen to stay inside **300–1300 MHz**.
5. Bandwidth → range resolution **~17 cm** → medium depth set to **20 cm** (> resolution).
6. Compute **λmax** (from fmin, ~80 cm free-space) and **λmin** (from fmax, ~23.5 cm free-space).
7. From the wavelength budget:
   - **pixel size Δx = λmin/10** (Yee rule) → 0.001 m
   - **antenna height > λmax/2** → 41 cm (intermediate-field region)
   - **antenna-to-boundary ≥ 20 pixels** buffer
   - **surface x = y = 1.5·λmax** (the headline result; 5·λmax ideal but too costly; ~120–140 cm)
8. **Time window** = 10⁻⁸ s.
Paper specifics: **Hertzian dipole, polarization Y, centered in x–y, single homogeneous layer,
NO buried targets**. The structure is a **fork**: soil + waveform → wavelength budget (λmax,λmin)
→ {depth, Δx, antenna height, surface dim, buffer} → time window.

**Key consequence:** `domain_x`, `domain_y`, `source_height_m`, `max_cell_m`, time window must be
**DERIVED**, not collected as free user values.

---

## 4. RICKER FREQUENCY DERIVATION (Wang 2015) — EXACT

The band is defined at **half-max of the amplitude spectrum (−6 dB)**. Using the Lambert W function:
- **W₀(−1/2e) ≈ −0.231961**, **W₋₁(−1/2e) ≈ −2.67835**
- Lower edge: **f_l1 = 0.481623·fp**
- Upper edge: **f_l2 = 1.636567·fp**
- Central frequency (geometric band center): **f_c = 1.059095·fp**
- Half-bandwidth: **f_b = 0.577472·fp**

where **fp = peak frequency**.

**Critical inference about the Khosravi paper:** their "fc = 825 MHz" is the **central** frequency,
NOT the peak. Back out peak: **fp = 825 / 1.059095 ≈ 778.96 MHz**. Then:
- fmin = 0.481623 × 778.96 ≈ **375.17 MHz** ✓
- fmax = 1.636567 × 778.96 ≈ **1274.83 MHz** ✓
- bandwidth ≈ **899.66 MHz** ✓

**CRITICAL CAVEAT (peak vs center):** gprMax's `#waveform ricker` takes the **PEAK** frequency.
- If the collected center IS the peak → write it directly. Band [375,1275] MHz ✓ inside Peplinski.
- If the collected center is the band-CENTER → convert to peak via **÷1.059095** before writing.
- **Failure mode:** naively writing 825 MHz as the peak → band [0.481623×825, 1.636567×825] =
  **[397, 1350] MHz** → fmax breaches the 1300 MHz Peplinski ceiling. This silently invalidates.

Schema flag for this: **`center_freq_is_peak`** boolean in DatasetConfig.

Caveat on scope: Wang's edges are for the ideal continuous Ricker spectrum; gprMax's discrete
windowed pulse deviates at tails. Wang's fmin/fmax are for the **validity gate**; the **resolution**
check uses a different, higher "significant frequency" (see §5).

---

## 5. TWO DIFFERENT "HIGHEST FREQUENCIES" — DO NOT CONFLATE

- **Validity gate** uses the Wang −6 dB band edges (fmin/fmax) against 0.3–1.3 GHz.
- **Resolution check** (Δx ≤ λmin/10) uses the **highest SIGNIFICANT frequency** at ~−40 dB,
  which is ~2–3× the center frequency. Encoded as `high_freq_factor` (default 3.0) or a per-waveform
  table (BW_MULTIPLIER): ricker 2.5, gaussiandotdot 2.5, gaussianprime 2.5, gaussiandoubleprime 3.0,
  gaussiandot/gaussiandotnorm/gaussian 2.0, sine/contsine 1.2. **Never default to 1.0** (disables
  the safety).

---

## 6. WHY SOIL-BEFORE-WAVEFORM (corrected understanding)

- fmin/fmax depend **ONLY** on waveform (peak freq + fixed Lambert-W constants). **NO permittivity.**
- Soil-before-waveform is justified **only** by the Peplinski validity gate (choosing the soil model
  defines the legal frequency window), NOT by band math.
- ε matters **downstream**: λ = c/(f·√εr). The wavelength budget needs soil ε AND the waveform band
  **together**. Soil and waveform are largely **independent collect-stages** joined by the gate.

---

## 7. SOURCE TIMING & AMPLITUDE (pass-through params)

- **source_start_time / source_end_time**: optional source on/off timing — `[f4 f5]` for
  hertzian/magnetic dipole, `[f5 f6]` for voltage/transmission_line. First = delay, second = removal.
  They gate WHEN the source is on/off, NOT the pulse shape or spectrum. Real dependency is the **time
  window** (resolved late), not the waveform band. The paper omits them (source active full window).
  Rule: start-without-end is an error; end-without-start defaults start to 0.
- **waveform_amplitude**: `f1` in `#waveform` — scaling of max amplitude. Units **Amps** (hertzian)
  or **Volts** (voltage/transmission_line). **Required positional arg**; NOT documented as default 1,
  though all examples use 1. gprMax materials are linear → amplitude scales output linearly →
  immaterial to signal SHAPE for a single source. Matters only for multiple weighted sources or
  absolute-level matching. Treated as a collected pass-through.

---

## 8. PERMITTIVITY (ε/σ) — gprMax NATIVE, NEVER REIMPLEMENT

### Decision: Peplinski-only
gprMax natively supports **only Peplinski**. The four-model manual solver (CRIM/Dobson/Mironov/
Peplinski) in the old `physics_modelling.py` is **redundant and removed**. ~2/3 of that file is
deletable.

### How gprMax computes it (from the actual `PeplinskiSoil` / `Material` source)
- `calculate_debye_properties(nbins, G, name)`: takes the moisture **range** `(mu[0], mu[1])`,
  slices into `nbins`, emits **one Debye material per bin**. Each bin has: `er` (infinite-freq
  permittivity, `eri`), `deltaer`, `tau = watertau`, `sig`. **`fractal_nbins` = nbins.**
- Internal constants: **watertau = 9.231e-12 s**, watereri = 4.9, waterer = 80.1.
- `erealw` is **hardcoded at f = 1.3e9** inside the routine — a model-construction constant, NOT the
  simulation frequency.
- The routine hardcodes the **0.3–1.3 GHz branch** (`sigf = 0.0467 + 0.2204·ρb − 0.411·S +
  0.6614·C`, and the `1.15·er − 0.68` linear correction). The 1.4–18 GHz `sigf` is commented out.
  → This is one more reason the frequency gate is NOT optional.
- **Bin midpoint shift:** `mubins = linspace(mu[0], mu[1], nbins)`, then
  `mumaterials = mubins + (mubins[1]−mubins[0])/2`. **Materials are made at bin midpoints**, so the
  wettest material sits **half a bin ABOVE mu[1]**. Use **`mumaterials[-1]`** (not mu[1]) for the
  wettest, and bind the porosity cap with that in mind.

### A-priori ε for grid sizing (the one surviving "solver" need)
There is **no separate ε estimator**. "A priori" is temporal only — the SAME ε gprMax simulates,
computed ahead of the grid via gprMax's own routine:
1. Instantiate `PeplinskiSoil(name, sand_frac, clay_frac, rho_b, rho_s, (theta_v_min, theta_v_max))`.
   (sand/clay are %/100; moisture is the band.)
2. Pass a **stub Grid `G`** — `calculate_debye_properties` only reads `len(G.materials)` and appends;
   it does NOT read `G.dt`.
3. Call `calculate_debye_properties(nbins, G, name)` to populate bins.
4. Call **`Material.calculate_er(f_center)`** on the bins, take `.real`. **Do NOT read raw `m.er`**
   (that's the infinite-frequency value, understates ε, would make the grid too coarse).
   `calculate_er` folds the Debye relaxation + conductivity term back in.
- **Wettest bin (`mumaterials[-1]`)** → largest ε → smallest λmin → drives **global Δx**.
- **Driest bin** → smallest ε → largest λmax → drives **global domain**.
- σ NOT needed for sizing (only real ε enters the wavelength budget).
- Optional conservative variant: evaluate the wettest bin at `f_high` instead of `f_center`.

### Ordering (NO reordering needed — dependency is one-way)
ε → λ → Δx → **Δt**. Δt is the **OUTPUT** of the chain, never an input. `calculate_er` needs no Δt.
The **τ-vs-Δt** check (watertau > Δt) is a **validation gate AFTER Δt exists**, not a computation
input — and with watertau = 9.231 ps it essentially always passes. `calculate_debye_properties`
needs `G` as an argument (API coupling), but the ε math doesn't use `G.dt`.

---

## 9. COORDINATE CONVENTION (VERIFIED against known-good `.in`)

The example `.in`:
```
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#hertzian_dipole: z 0.100 0.170 0 my_ricker
#rx: 0.140 0.170 0
#box: 0 0 0 0.240 0.170 0.002 half_space
#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.010 pec
```
**Verified facts:**
- **x = 1st coord** (horizontal, 0.240). **y = 2nd coord** (VERTICAL). **z = 3rd coord** (THIN
  invariant axis = 0.002 = exactly 1 cell).
- Soil `#box` fills y ∈ [0, 0.170]; **air** above to 0.210; **antenna at y = 0.170** (soil surface).
  So layout is **soil-on-bottom, air-on-top, antenna at surface**.
- Cylinder endpoints differ only in 3rd coord (0 → 0.002): the **cylinder axis runs along thin z**,
  spanning the full one-cell depth. In-plane it's a **disc of diameter 2·radius**.

**Decisions:**
- **Keep vertical = y** (matches gprMax 2D convention). Do NOT switch to z-vertical.
- **Rename validator params axis-neutral**: `tx_z_m → tx_vertical_m`, `domain_z_m →
  domain_vertical_m`, `ground_z_m → ground_vertical_m`; "z face" message → "vertical face". z keeps
  its real (thin) name. Caller passes y into the vertical params once.
- **Rename `ground_z_m → ground_y_m`** in the derive (it holds a y-coordinate).
- **Emission is positional `(x, y, z)`**: vertical y in slot 2, single-cell z in slot 3. Thin axis
  `domain_z = Δz` (1 cell); every object spans full z.

**Layout math (y vertical):** `pad = (pml_cells + buffer_cells)·Δx`;
`ground_y = pad + depth_z`; `tx_y = ground_y + source_height`; target at depth d →
`y_center = ground_y − d`. **Bottom→top: `pad | soil(depth_z) | air(source_height) | pad`.**

---

## 10. THE SCHEMA (`gpr_dataset_schema.py`)

### Constants
`C0 = 299792458`; Wang ratios 0.481623 / 1.636567 / 1.059095; `PEPLINSKI_FMIN_HZ = 0.3e9`,
`PEPLINSKI_FMAX_HZ = 1.3e9`; `MAX_TL_RESISTANCE_OHM = 376.73`.

### Stages
- **STAGE 0 `DatasetConfig`**: `num_samples` (dataset size, NOT time samples), model_basename,
  output_dir, num_threads, `pml_cells = 10`, `buffer_cells = 10`, `cells_per_wavelength = 10`,
  dimensionality ("2D"/"3D"), `high_freq_factor = 3.0`, **`center_freq_is_peak`** (peak-vs-center
  decision), `fractal_nbins = 50`.
- **STAGE 1 `ExtractedLayerParams`**: per-layer; `thickness_m_min/max` REQUIRED; sand/clay %
  min/max; `theta_v_min/max` = per-layer ENVELOPE (gprMax takes a range, never scalar); bulk &
  particle density min/max. Validators: ranges min ≤ max; **sand_min + clay_min ≤ 100** (texture
  closure); **θv_max ≤ loosest porosity** `n_max = 1 − bulk_min/particle_max`.
- **STAGE 1 `ExtractedLayers`**: num_layers, layers[]; validator **num_layers == len(layers)**.
- **STAGE 2 `ExtractedWaveform`**: kind, amplitude (default 1.0), center_freq_hz, name;
  source_start/end_time. **NO λ/Δx here** (needs soil ε).
- **STAGE 3 `ExtractedAntenna`**: kind, axis (default "x"), tx_rx_offset_m, resistance,
  rx_same_height, **source_height_m** (added; derived ≥ λmax_air/2 if None), rx_array. Validator:
  resistance required + `0 < R < 376.73` for **transmission_line OR voltage_source**.
- **STAGE 4 `ExtractedAdvancedParams`**: surface_roughness, snapshots, cylinders, boxes, spheres.
- **STAGE 5 `SampledLayer`**: name, thickness_m, sand_pct, clay_pct, **theta_v_min/theta_v_max**
  (per-sample band drawn inside the envelope), bulk_density, particle_density. Validator
  **theta_v_min < theta_v_max** (real band; degenerate band defeats fractal moisture distribution).
  `SampledSample`: sample_id, layers[].
- **STAGE 6 `derive_per_sample`** → `DerivedLayer`: silt label (`100 − sand − clay`, label only);
  porosity guard (`bulk < particle`, `θv_max ≤ porosity`); pct→fraction; ε at **both band edges**
  (`eps_r_dry` at θv_min, `eps_r_wet` at θv_max — monotonic in moisture); σ at edges;
  `theta_v_label` = band midpoint.
- **STAGE 7 `GlobalDerived` / `derive_global`**: aggregates `eps_r_max_global` (max wet-edge ε),
  `eps_r_min_global` (min dry-edge ε, floored at 1.0). Computes fp (peak vs center), Wang band +
  Peplinski gate, f_high, λmin/λmax, Δx (= λmin/cpw, tightened by smallest_feature/10), domain,
  depth, source_height, **static Tx/Rx absolute coords**, Δt (CFL), time window. Worst-case corners:
  highest εr → finest Δx; lowest εr → largest domain.
- **STAGE 8 `PipelineBundle`**: dataset + layers + waveform + antenna + advanced + grid.

---

## 11. KEY PHYSICS / ML DECISIONS

- **θv is a per-sample BAND** `(min, max)`, passed whole to `#soil_peplinski`. NEVER a scalar
  (gprMax requires a range to build the fractal material series). The band VARIES per sample.
  Sampler policy: draw center+width OR two ordered values; **draw densities first, then cap θv_max
  at porosity** (or reject). θv label for ML = band midpoint (changeable to range/edge).
- **Global grid/domain/depth/time-window** for ML comparability: ONE grid for the whole dataset,
  identical time axis and boundary geometry across all samples → CIRs directly comparable.
- **Per-sample THICKNESS is fine and wanted** (it's a label) — varies INSIDE the fixed-depth box.
  Rule: **vary the physics you want predicted inside a frame held constant**. Global depth = sized
  to the worst-case (deepest) possible stack. Cost: shallow samples simulate empty space (compute
  tax, not accuracy loss).
- **Cubic cells (Δx = Δy = Δz)**: not forced by gprMax (it supports anisotropic via `#dx_dy_dz`),
  but chosen because the discretization criterion is direction-agnostic, numerical dispersion stays
  isotropic, and CFL simplifies to `Δt = Δ/(c·√n_dim)` (√2 for 2D, √3 for 3D). Globalness matters
  more than equality.
- **Smallest target feature ≥ 10 cells** can tighten Δx below λmin/10.
- **Memory** scales O(N³), **time** O(N⁴) with grid points (Khosravi/Taflove) — the compute budget
  driver.

---

## 12. VALIDATION (`validation_rules.py`) — 4 TIERS

Defined by what info is available when the check runs (NOT pre/post sampling):
- **TIER 0 SCHEMA** (construction): single-field/within-stage invariants (ranges, min<max,
  sand+clay≤100, θv-envelope ≤ loosest porosity, num_layers==len, resistance bound, band min<max).
  NOT repeated in the tool file.
- **TIER 1 COLLECT** (per agent): texture envelope feasibility (3-fraction closure to 100),
  Peplinski calibration envelope, **waveform + Peplinski band gate on DERIVED edges**, antenna config.
- **TIER 2 PER-SAMPLE** (in extraction, per draw): texture sum=100, bulk<particle, θv_max ≤
  porosity, Peplinski texture/moisture bounds.
- **TIER 3 GLOBAL** (once, needs Δx/Δt/domain): grid (λ/10), domain alignment, CFL/iterations,
  τ-vs-Δt, PML-vs-domain, memory, resolution adequacy, antenna placement, source-height-vs-domain,
  two-way travel time, layer thickness/stack, target resolution/PML-distance/in-domain, rx_array step.
- **TIER 4 EMISSION** (writer): material names (whitespace/unique/reserved), essential commands
  (`#domain`/`#dx_dy_dz`/`#time_window`), snapshot ≤ window.

### Flaws fixed from the old validation files
1. **Peplinski gate was on CENTER freq → must be on DERIVED Wang band** (CRITICAL).
2. **temperature_c no longer affects the simulation** (gprMax hardcodes water at 1.3 GHz) — kept
   only as label-sanity.
3. **time_window ≠ source_end_time** (separate concepts).
4. **BW multiplier inconsistent** (1.0/1.5/2.0 across tools); unified, never defaults to no-op.
5. **resistance** applies to transmission_line too; the **50–100 Ω "recommended" warning is
   ungrounded** — removed.
- **Cut entirely:** CRIM/Dobson/Mironov constraints, `clamp_texture_to_model`, `estimate_porosity`
  (texture-fallback), custom-material coupling to manual mixers.

### Validation constants
`PEPLINSKI_FREQ_HZ = (0.3e9, 1.3e9)`, `PEPLINSKI_THETA_V_MAX = 0.30`,
`PEPLINSKI_TEXTURE_PCT = {sand:(15,50), silt:(35,65), clay:(5,20)}` (1995 Table I calibration —
gprMax does NOT enforce these; it's our validity policy), `PEPLINSKI_WATER_TAU_S = 9.231e-12`,
`RESERVED_MATERIAL_NAMES = {pec, free_space, grass, water}`, `BUILTIN = {pec, free_space}`,
`PML_GAP_CELLS = 15`, `RESISTANCE_MAX_OHM = 376.73`.
**Cascade ordering in Tier 3:** domain-fit (alignment, PML-vs-domain) FIRST, then placement/
stratigraphy, then feasibility (memory, iterations) — to avoid reporting downstream errors that
stem from one upstream one.

---

## 13. FINAL PIPELINE STAGE ORDER (with variable targets + static antenna)

1. **Layer ranges** (pull center frequency forward so step 2 can gate).
2. **Draw N samples (layers + target geometry)**: each sample draws soil values AND target
   position/depth/radius. Fire grid-INDEPENDENT validation: soil caps (texture closure, θv ≤
   porosity, Peplinski envelope) + target sanity (radius > 0, box ordering, coords well-formed).
3. **Extract waveform + antenna** (single, non-sampled values).
4. **Cross-stage validation** ("validation till now"): Peplinski band gate (needs waveform),
   antenna config.
5. **Per-sample physics + aggregate corners** (one pass): ε/σ via gprMax routine at f_center →
   wet/dry-edge ε; per-sample target bbox, smallest feature (`2·radius`), in-plane extent
   (`2·radius`), target bottom depth (`depth + radius`). Aggregate: `eps_r_max/min_global`,
   `smallest_feature_global` (min), `largest_extent_global` (max), `deepest_target_bottom_global`
   (max), `max_stack_global`.
6. **Global derive** (once, from corners): λmin/λmax; Δx (= λmin/cpw, then `min(Δx,
   smallest_feature_global/10)`); domain_x (≥ largest_extent + 2·clearance); depth_z (≥
   deepest_target_bottom clearance term, max_stack, range_res); source_height (≥ λmax_air/2);
   **static Tx/Rx absolute coords** (`x_mid=domain_x/2`, `tx_x=x_mid−offset/2`, `rx_x=x_mid+offset/2`,
   `tx_y=ground_y+source_height`); snap domains to integer cell multiples; Δt (CFL); time window.
7. **Global validation** (once, grid/numerics): cascade-gated — fundamentals/domain-fit, then
   placement/stratigraphy (incl. static Tx/Rx vs PML + source-height-vs-domain), then feasibility.
8. **NEW per-sample placement validation** (back over N samples, needs domain): each target's bbox
   in-domain, ≥ (pml+15) cells from boundary, ≥ 10 cells across, fully-buried (`depth ≥ radius`).
   **Redraw-with-MAX_TARGET_ATTEMPTS** (shrink/reposition) then **drop + log** on exhaustion.
9. **Emit N `.in` files** (pure string assembly; no derivation in writer).
10. **Run gprMax over all samples; store once** (+ open question on store-before/after-gprMax).

**Three validation moments total:** at draw (soil + grid-independent target), after waveform/antenna
(cross-stage band+antenna), after derive (grid/numerics + per-sample placement).

---

## 14. VARIABLE TARGETS + STATIC ANTENNA — DETAILED DECISIONS

- **Target GEOMETRY (position, depth, radius) is per-sample** (variety for ML, sim-to-real).
- **Target-derived GRID/DOMAIN requirements stay GLOBAL**: aggregate worst-case corners
  (smallest feature → tightens Δx; largest extent → enlarges domain; deepest bottom → enlarges
  depth). The grid must NOT follow the target per-sample (would destroy comparability).
- **Tx/Rx are STATIC** (antenna fixed across dataset) → derived once in global derive, validated
  once. (If antenna placement were later randomized per sample, Tx/Rx move to the per-sample path.)
- **Object types:** cylinders only for now; box/sphere stubbed (`NotImplementedError`).
- **Dropped samples:** drop and reduce N; log (sample_id, reason). Final dataset may have < N.
- **Target material → εr corners:** if PEC → does NOT feed εr (only size/extent). If dielectric →
  "treat as one more material in the aggregation" (high-εr tightens Δx, low-εr/void enlarges domain).
  Current decision: targets PEC, material does NOT feed εr corners. (Revisit if target material
  becomes sampled.)
- **Target-range collection moved EARLY** (separate mini-stage right after `layers`, before the
  draw), because it's now a sampled quantity. Mirrors the waveform-center-frequency-forward
  precedent. Non-sampled advanced params (snapshots, roughness, rx_array) stay last.
- **Redraw mechanics:** radius range `[r_floor, original_radius]` (shrink/reposition only, never
  grow). `r_floor = max(5·Δx, radius_min_m)` (grid-faithful 10-cell floor, NOT
  smallest_feature_global/2). **Feasibility short-circuit**: if even at r_floor no valid center
  exists, drop immediately (no wasted attempts). Shrink-to-recover handles the "too large for
  clearance" case.

### The "ghost corner" question (resolved)
Dropping samples AFTER the grid is derived means a corner can be set by a sample later dropped →
grid slightly oversized. **Accepted, not re-derived**, because: (a) re-deriving post-drop creates a
**circularity** (the drop decision needs the grid; the grid needs survivors → fixed-point loop,
may not converge); (b) the error is **always in the safe direction** (oversized, never undersized —
no survivor ever fails to fit); (c) the corner-setter (big/deep target) is the one most likely
dropped, but direction is benign. **Documented as a conscious safe-direction approximation.**

---

## 15. AGENT / ORCHESTRATION ARCHITECTURE

- **Most of the pipeline is DETERMINISTIC, not agentic.** LLM agents only for **extraction/
  collection** and **re-eliciting on human-decision failures**. Sampling, ε-solving, derivation,
  validation, emission, error-handling are deterministic code.
- **Do NOT make physics/derivation/sampling agents.** Do NOT route gprMax numerical errors through
  an LLM "rectifier" (it could paper over a physics violation; most gprMax errors are pre-flight
  catchable by Tier-3 validators).
- **Agent count:** start with ONE collection agent (split a soil specialist later only if its
  instructions crowd the prompt). The Peplinski gate couples soil+waveform, so any decomposition
  needs a join point anyway. The orchestrator should be CODE (deterministic state machine), not an
  agent.
- **Feedback fork by failure type:** human-decision failures (infeasible texture, out-of-band
  frequency) → back to collection agent; deterministic failures (grid too big, alignment) → fixed/
  hard-failed in code, never the LLM.
- **LangGraph decision:** good fit for the **collection front-end only** (dynamic control flow,
  conditional re-elicit edges, human-interrupt, checkpointer for resumability/observability). NOT a
  fit for the deterministic core (fixed linear, no branching) — wrapping deterministic stages as
  graph nodes adds boilerplate and invites LLM creep. Recommended topology: **LangGraph collection
  subgraph → single deterministic `generate_dataset(config)` node**. The redraw loop and per-sample
  loop are LOCAL loops inside deterministic stages (a `for` loop and a `map`), NOT graph control
  flow. Do NOT rewrite already-working deterministic code into nodes.

### Current codebase state (per the user's block diagram)
4 extraction agents → API → SQL DB; "dielectric solvers"; post-validation; sampling; dataset
generation; gprMax; feedback loop; vector DB. **Problems flagged:** (a) ordering inverted (solves
before sampling — must sample first); (b) global-derive stage missing from the diagram; (c) 3 agents
wrap deterministic work (Generation, Rectifier, Feedback); (d) vector DB unmotivated; (e) two-write
pattern vs store-once. **The user later stated the staged pipeline IS already migrated** (layer
ranges → draw → waveform/antenna → cross-validation → per-sample ε/σ + global derive → global
validation), so the incremental prompt targets just the target/Tx-Rx/redraw additions.

---

## 16. THESIS DIRECTION

Four-chapter arc, **risk decreasing backwards**:
- **Ch 1 — Generation framework** (the instrument; already largely built). Includes the agent layer
  as a section (physics-grounded agentic workflow; evaluate validator-constrained agent vs naive
  prompting). Systems contribution.
- **Ch 2 — Inversion + uncertainty** (synthetic data; low risk). Posterior/distributional output
  (SBI / neural posterior estimation) over point regression; identifiability analysis (which params
  collapse into ε-equivalence); band-supervised θv labels.
- **Ch 3 — Sim-to-real on lab data** (HEADLINE; real risk). Train synthetic, evaluate on lab data;
  expect naive transfer to degrade; contribution = characterizing why + mitigation ladder (domain
  randomization at generation; antenna-leakage removal / CIR extraction as the preprocessing bridge;
  optional learned translator). Even a negative-leaning result is thesis-grade with rigorous gap
  analysis.
- **Ch 4 — Active learning** (stretch, cut if needed): posterior uncertainty drives which FDTD runs
  happen; simulation efficiency. Exploits the global-grid comparability.

### AI-novelty options discussed
Active learning over the parameter space (strongest, exploits global grid); uncertainty/SBI;
neural surrogate/diffusion forward model (NOTE: Giannakis already did an ML fast-forward solver —
in the project files — so bare idea is taken; open edges = diffusion surrogates, full-stratigraphy
conditioning, surrogate-in-the-active-learning-loop); the agent layer itself; sim-to-real.
Recommended thread: **#1 + #2 together**.

### Lab data audit (GATING — must confirm before scaling compute)
1. **Ground truth per scan?** (measured moisture/texture/depths → inversion validation; else pivot
   to detection/localization or signal-statistics).
2. **Instrument antenna + center frequency?** If outside **0.3–1.3 GHz**, the Peplinski-gated
   generator **cannot legitimately simulate** that soil at that frequency (would need Dobson
   1.4–18 GHz, NOT gprMax-native, reviving the deleted manual solver). Also Hertzian-dipole synthetic
   vs real shielded antenna ringing = the core sim-to-real gap.
3. **Survey geometry?** (A-scan/B-scan, antenna height, Tx-Rx offset) → synthetic generation must
   mirror it (cheap alignment at generation time).

---

## 17. CLAUDE CODE / KNOWLEDGE TRANSFER

- This chat's memory does NOT cross into Claude Code; the bridge is a **`CLAUDE.md`** at repo root
  (read before every session; plain Markdown; keep **under ~200 lines**; use `.claude/rules/` if it
  grows). Claude Code also auto-maintains learned project notes — don't waste lines on what it can
  infer from code (file layout, that you use Pydantic). **Encode what it CAN'T infer: physics
  constraints, design decisions, and especially REJECTED alternatives** (so it doesn't revive the
  manual solver, per-sample grids, scalar θv). Use `/memory` to see what it already knows and prune.
- **Two migration prompts produced:** full migration (if base not migrated) and incremental
  (target geometry only, base already migrated). Both carry the invariants self-contained (Claude
  Code lacks this chat). **Verified test vector:** center (band-center) 825 MHz → peak ≈ 778.96 MHz
  → fmin ≈ 375.17, fmax ≈ 1274.83, bandwidth ≈ 899.66 MHz; must pass the Peplinski gate.

---

## 18. THE PLAN REVIEW + AGENT CORRECTIONS (latest state)

The implementation plan (variable targets + static Tx/Rx) was reviewed. Confirmed fixed: coordinate
naming (ground_y_m, axis-neutral validators), redraw "too large" handling (r_floor + shrink-to-
recover + feasibility short-circuit), explicit y-extent binding case (deepest_target_bottom),
grid-global test. **Still-loose items flagged:** (1) y-clearance formula must use a SHARED `clearance`
constant (buffer-vs-gap), (2) Δx must be FROZEN before any clearance/depth math, (3) ghost corner
should be DOCUMENTED in out-of-scope, (4) the grid-global test should ALSO assert the grid DIFFERS
from a no-target grid (proves target corners actually fed Δx/domain).

### Two agent corrections analyzed (latest)
- **Correction 1 — use `clearance` (pml+15) not `pad` (pml+10) for antenna margins:** ACCEPT. This
  fixes the buffer-vs-gap mismatch (pad gives pml+10 but the validator demands pml+15, so the static
  antenna failed placement every time). Verify `pad` and `clearance` are used CONSISTENTLY (no
  margin validated against `clearance` is sized with `pad`).
- **Correction 2 — redraw radius range `[r_floor, original_radius]` (shrink/reposition only):**
  ACCEPT the logic. The "original_radius ≥ r_floor always holds" claim was VERIFIED:
  `Δx ≤ smallest_feature_global/10 = min_radius/5` → `5·Δx ≤ min_radius ≤ every original_radius`,
  and the `radius_min_m` branch holds by construction (draws respect their lower bound); additional
  Δx tightening only makes `5·Δx` smaller (safe). **Action: convert the "always holds" comment into
  a RUNTIME ASSERTION** (`assert original_radius >= r_floor`) so a future Δx-derivation change fails
  loudly instead of producing an inverted interval. Then PROCEED (do not revert).

---

## 19. OPEN DECISIONS (still unresolved / to confirm)

1. **Store-once vs store-twice:** params/labels before gprMax (protects ground truth if a run fails)
   + signals after = two writes, contradicting "store once after dataset." Decide deliberately.
2. **Dropped-sample handling:** drop-and-reduce-N (chosen) vs backfill-to-hold-N. If any downstream
   experiment assumes exactly N, document the actual delivered count prominently in the manifest.
3. **MAX_TARGET_ATTEMPTS** value (placeholder 20) — set the real cap.
4. **Sampler θv policy:** center+width vs two ordered values; and the moisture ML label
   (midpoint vs range vs edge).
5. **Lab instrument antenna/frequency, per-scan ground truth, survey geometry** — gates Chapter 3
   and possibly forces generator changes (frequency band, antenna model, geometry mirroring).
6. **Vector DB purpose** — motivate (active-learning diversity?) or cut.
7. **Whether to adopt LangGraph** for the collection layer — depends on whether current orchestration
   causes real pain (lost state, retries, human-interrupt) vs aesthetics.

---

## 20. KEY CONSTANTS / VALUES (quick reference)

| Quantity | Value | Source |
|---|---|---|
| Speed of light C0 | 299792458 m/s | — |
| Wang lower edge / fp | 0.481623 | Wang 2015 |
| Wang upper edge / fp | 1.636567 | Wang 2015 |
| Wang central / fp | 1.059095 | Wang 2015 |
| W₀(−1/2e) | −0.231961 | Wang 2015 |
| W₋₁(−1/2e) | −2.67835 | Wang 2015 |
| Peplinski validity | 0.3–1.3 GHz | Peplinski 1995 / gprMax |
| Peplinski θv max | 0.30 | calibration |
| Peplinski texture | sand 15–50, silt 35–65, clay 5–20 % | 1995 Table I |
| watertau | 9.231e-12 s | gprMax Material |
| watereri / waterer | 4.9 / 80.1 | gprMax Material |
| erealw hardcoded freq | 1.3e9 Hz | calculate_debye_properties |
| Max resistance | 376.73 Ω (exclusive) | gprMax / free-space impedance |
| PML gap cells | 15 | gprMax guidance |
| Default pml_cells | 10 | gprMax default |
| buffer_cells | 10 | our default |
| cells_per_wavelength | 10 | Yee rule |
| high_freq_factor (ricker) | 2.5 (table) / 3.0 (scalar default) | resolution check |
| r_floor (redraw) | max(5·Δx, radius_min_m) | 10-cell rule on fixed grid |
| MAX_TARGET_ATTEMPTS | 20 (placeholder) | our config |
| fractal_nbins | 50 (default) | gprMax fractal_box |
| surface dimension | 1.5·λmax | Khosravi |
| antenna height | ≥ λmax_air/2 | Khosravi |
| Verified test vector | 825 MHz center → fp 778.96, fmin 375.17, fmax 1274.83, BW 899.66 | Wang chain |
| Reserved material names | pec, free_space, grass, water | gprMax |
| CFL (cubic) | Δt = Δ/(c·√n_dim) | n_dim = 2 (2D) or 3 (3D) |

---

*End of summary.*