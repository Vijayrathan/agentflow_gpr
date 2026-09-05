# Enabling physics-consistent 3D dataset generation

Research date: 4 September 2026. Scope: the current repository and its bundled gprMax 3.1.7. This is an implementation proposal, not a claim that the platform now supports 3D.

## 1. Recommendation and evidence

**Keep gprMax as the electromagnetic solver and extend the platform around it. Implement a versioned 3D dataset contract before enabling 3D in the agent.** The work spans collection, sampling, material derivation, geometry, numerical planning, validation, emission, execution, labels, reuse, and visualization. Removing the emitter's `NotImplementedError` alone would produce inconsistent or incorrectly described datasets.

The bundled solver already selects full 3D FDTD when all three grid dimensions contain more than one cell. Its one-cell cases select a corresponding 2D TM mode. This behavior is explicit in `gprMax/gprMax/input_cmds_singleuse.py:144`. The underlying software and its FDTD modeling approach are documented in [Warren, Giannopoulos and Giannakis, 2016](https://repository.uwl.ac.uk/id/eprint/5367/).

I also ran the bundled CPU solver directly, independently of the platform emitter. A 0.20 m cube with 4 mm cubic cells, an x-polarized Hertzian source, six PML faces, and a 6 ns window completed successfully. It used a 50 × 50 × 50 grid, a 7.703332806185882 ps time step, and 780 iterations. All six receiver field arrays contained 780 finite values. Appendix A records the input. **This proves that the installed solver can execute 3D; it does not establish soil-model accuracy, production domain adequacy, GPU compatibility, or convergence.**

The recommended first release supports:

- A genuinely three-dimensional, layered soil volume using the same native Peplinski material path.
- Finite PEC boxes and finite axis-aligned cylinders, with positions in all three coordinates.
- One fixed transmitter and receiver configuration per dataset, with correctly honored x, y, or z source polarization.
- The existing explicitly supported source kinds, subject to the bundled solver's execution restrictions.
- One cubic grid, domain, time axis, coordinate convention, and acquisition configuration shared by every accepted sample.
- Geometry inspection, reproducible manifests, and physical/numerical acceptance checks before results become reusable training data.

Arbitrarily rotated targets, realistic antenna assemblies, additional target materials, receiver arrays, and B-/C-scan acquisition can follow as separate extensions. None is a prerequisite for a correct 3D single-receiver dataset. If an option is not supported in the first release, collection must reject it explicitly.

## 2. What must remain invariant, and why

The project's rules combine physical constraints, numerical requirements, and dataset policies. Keeping those categories distinct prevents an empirical sizing recommendation from being mistaken for a universal physical law.

| Requirement | Nature and reason | Required 3D behavior |
| --- | --- | --- |
| Maxwell-consistent propagation and material response | Physical model implemented by the native solver | Use the full 3D solver and its constitutive updates; do not assemble independent 2D solves and label the result 3D. |
| Admissible soil composition and moisture | Physical/material constraint | Texture fractions must close; densities must be positive and ordered; every instantiated moisture bin must fit available pore space. |
| Native Peplinski derivation | Project material-model contract | Derive and emit through the same pinned native implementation. Do not introduce a second hand-written soil mixing model. |
| Frequency validity | Empirical validity of the chosen material model | Preserve the declared Peplinski band gate and document what part of the pulse spectrum it covers. Geometry becoming 3D does not expand the model's calibration range. |
| CFL and dispersive-update stability | Numerical requirements | Derive the actual 3D time step from the final emitted grid; check it against all relevant relaxation times. |
| Wavelength and feature resolution | Numerical accuracy, with project thresholds | Preserve the common resolution budget; include every finite 3D feature. Demonstrate convergence rather than treating ten cells as proof of accuracy. |
| One domain and cubic grid per dataset | Dataset-design policy | Freeze the three dimensions and all cell spacings once. Never fit a different domain or grid to each sample. |
| One time axis and acquisition plan | Dataset-design policy | Keep actual time step, iteration count, source, receiver layout, and excitation identical across accepted samples unless acquisition variation is explicitly a different dataset contract. |
| Fixed targets stay fixed | Sampling contract | Preserve every fixed field, including fields inside otherwise dynamic objects. Only the declared grid-quantization policy may change physical representation, and it must be recorded. |
| Bounded placement and whole-sample rejection | Sampling contract | Preserve the existing bounded retry/drop policy. Do not silently backfill, resize the domain, remove just one required target, or expand requested ranges. |
| Labels describe the simulated scene | Scientific data integrity | Store effective voxelized geometry and actual solver settings alongside requested and drawn values. |
| LLM extracts intent; deterministic code performs physics | Agent architecture | The agent can explain derived choices, but cannot invent material constants, grid sizes, time steps, or numerical “repairs.” |

### Why the domain must remain fixed across samples

Changing the domain also changes boundary locations, propagation paths to those boundaries, and potentially the amount or realization of heterogeneous soil. Changing the grid changes staircasing, numerical dispersion, and source discretization. Changing recording duration changes which returns are captured. Those changes can become unintended explanations for differences in training signals.

A fixed domain does not make all samples physically identical; it holds the numerical experiment constant while declared soil and target properties vary. It also gives consistent coordinates and tensor shapes. This is a strong dataset policy, not a requirement that Maxwell's equations impose on unrelated simulations.

The 3D implementation should aggregate conservative bounds over the sampled population **and every change still permitted after that aggregation**. Those bounds include material response, smallest intrinsic target feature, target footprints in x and z, deepest target bottom, pinned positions, roughness limits, and the fixed acquisition envelope. Choose the common grid and domain from those bounds, then freeze them. Retaining a conservative bound contributed by a subsequently rejected sample is safe; shrinking the dataset domain after rejection is unnecessary and should remain prohibited.

The guarantee applies to the generated population and its permitted placement operations. It must not be described as covering every possible future draw from the original parameter ranges unless those full ranges were explicitly bounded too. New samples that exceed the contract require a newly planned dataset.

### Coordinate convention

Keep the repository's existing convention: **y is vertical; x and z are horizontal**. Positive depth below ground is a separate semantic quantity, converted into decreasing y. The new spatial extent is crossline z. The current `depth_z_m` name denotes vertical soil depth despite its name; do not reuse it as domain z. Introduce a clear name such as `soil_depth_m` with a versioned compatibility mapping.

Several gprMax examples and papers use z as vertical. Their antenna coordinates, roughness directions, and polarization must be transformed when adopted here. A change in coordinate names without transforming the geometry would alter the experiment.

### 2D and 3D are different physical experiments

The current thin-z TMz model assumes invariance along z. A circular target cross-section represents an invariant cylinder, not a finite buried object. A full 3D scene permits finite target ends, crossline scattering, and three-dimensional field spreading. Consequently, a 3D trace should not be required to match the amplitude of its 2D counterpart. Keep both modes available, versioned, and separate in dataset reuse and scientific comparisons.

## 3. Repository audit: where changes are needed

Locations refer to the inspected working tree. Paths are repository-relative for navigation; line numbers identify useful starting points and may move as implementation proceeds.

| Component | Current behavior | Required change |
| --- | --- | --- |
| `backend/agentflow_single_agent.py:296` | Saving dataset configuration forces `dimensionality = "2D"`. | Accept and persist an explicit supported mode. Include mode and all new geometry fields in downstream invalidation. |
| `backend/single_agent_prompts.py` | Dimensionality is server-fixed; prompts describe 2D targets. Antenna collection offers axes that emission does not honor. | Present mode-specific schemas and explanations; reject unsupported geometry and source choices without substitution. |
| `backend/schema.py:212` | Configuration already permits 2D/3D and constructs six-face PML tuples. Active target schemas remain 2D. | Complete the schema through sampled, derived, placed, emitted, and stored representations. A configuration enum alone is not 3D support. |
| `backend/dataset_sampling/layer_sampler.py` | Deterministic draws; a moisture band is passed through. | Preserve soil semantics. Add reproducible crossline/finite-extent draws without silently changing legacy random sequences. |
| `backend/dataset_sampling/peplinski_derive.py:74` | Native material generation; first/last bins evaluated at the actual waveform peak. Aggregates x/vertical target bounds. | Retain native derivation; validate the full material table and add crossline/finite-geometry bounds. Separate label-frequency evaluation from numerical spectral bounds. |
| `backend/dataset_sampling/global_derive.py:43` | A conditional 3D CFL expression exists, but no 3D domain or z antenna positions are produced. | Derive all three dimensions, integer cell counts, complete Tx/Rx coordinates, and a recording window covering the 3D experiment. |
| `backend/dataset_sampling/target_shapes.py:106` | Two half-extents; target bounding boxes span only one z cell. | Use shape-aware 3D bounds, intrinsic minimum features, finite cylinder lengths, and full burial tests. |
| `backend/dataset_sampling/target_placement.py:128` | Retries shrink/reposition in x/y. Repositioning can leave the user's requested position ranges. | Intersect physical feasibility with every requested range in 3D; preserve fixed fields; validate the actual discretized result. |
| `backend/dataset_sampling/global_validation.py:124` | Validation still supplies `domain_z = dx`. | Pass the resolved domain and acquisition geometry; enforce six-face checks and common-contract consistency. |
| `backend/validation_tools_new.py:243` | CFL/Debye helpers use three axes even when other code derives 2D timing; some boundary checks infer 2D from thickness. | Use one authoritative mode-aware numerical plan; remove thickness heuristics that can silently treat a nominal 3D scene as 2D. |
| `backend/dataset_sampling/emit.py:406` | Explicit 3D rejection; thin-z geometry, z source polarization, z=0 source/receiver coordinates. Geometry snapping occurs during writing. | Emit from a validated resolved scene; use finite z bounds and selected source axes; keep the writer free of new geometric decisions. |
| `backend/simulate.py:147` | Native execution already works with gprMax inputs; GPU default is two concurrent models per device. | Add resource- and source-aware admission, per-device capacity accounting, and manifest-scoped execution. |
| `backend/signal_extraction.py:36` | Reads all six fields from `rx1`; a separate display reader reads actual time metadata. | Preserve actual numerical and receiver metadata in stored results. Generalize receiver indexing only when multiple receivers are enabled. |
| `db/db.py:105` | Simulation rows have scalar domain x/y and legacy geometry JSON fields. | Migrate domain z, mode, coordinate frame, finite target geometry, acquisition, actual timing, and contract version. |
| `backend/api.py`, simulation-row builder | Persists collected axis and nominal sample geometry. | Persist canonical emitted geometry and source settings; distinguish requested, sampled, resolved, and executed states. |
| `backend/sim_similarity.py:139` | Version-1 features are planar. Dimensionality is already a hard filter. | Preserve that filter; add crossline geometry, finite lengths, resolved polarization, acquisition and physics-version compatibility. |
| `backend/viz_projection.py:312`, `frontend/app/data.jsx` | Scene projection and display use a 2D cross-section. | Add actual 3D inspection or orthogonal slices with explicit slice positions and matching resolved coordinates. |
| `AGENT.md`, `README.md`, tests | Document and enforce the existing 2D contract. | Add a separate explicit 3D contract and its acceptance criteria while retaining 2D coverage. |

The preceding source-type fix already provides distinct Hertzian, voltage, and transmission-line emission and rejects unknown kinds. Preserve that behavior. This research does not propose returning to any generic Hertzian fallback.

## 4. Proposed stage sequence and the reason for each stage

The conversational collection order can remain familiar. The deterministic computation must follow dependencies. All collected inputs, including advanced geometry settings, must be complete before a final dataset contract is generated.

| Stage | Work and derived result | Why it belongs here |
| --- | --- | --- |
| 1. Declare dataset mode and frame | Mode, coordinate convention, target/source capabilities, dataset size, seeds, policy version. | These determine what subsequent fields mean and which questions are valid. |
| 2. Collect physical ranges and acquisition | Layers, target extents/positions, waveform semantics, source kind/axis, Tx/Rx layout, roughness and requested outputs. | Collect intent before deriving numerical quantities. Reject incompatible combinations early. |
| 3. Validate envelopes and draw population | Validate range feasibility; draw materials and geometry; retain deterministic identities and provenance. | Native material response and target feature limits depend on actual draws. |
| 4. Resolve excitation and native materials | Actual peak frequency, declared useful band, design spectrum, native Debye bins, evaluated permittivity/conductivity labels. | Numerical resolution depends on the excitation and instantiated materials, not just the entered center frequency. |
| 5. Aggregate numerical and geometric bounds | Worst resolved wavelength, intrinsic feature limits, x/z footprint bounds, depth bounds, roughness/acquisition envelope. | A single plan must cover every sample and all permitted later placement changes. |
| 6. Resolve common grid and scene coordinates | Final cubic cell size, three integer cell counts, physical domain, ground reference, Tx/Rx cell positions, candidate discretized geometry. | Domain padding and geometry quantization depend on cell size. They cannot be finalized before resolution. |
| 7. Derive time axis | Mode-correct CFL time step, relaxation checks, source duration/delay, 3D travel-time bound, common iteration count. | Timing follows the final grid and physical scene envelope. |
| 8. Validate and finalize placement | Bounded deterministic placement within requested ranges; checks on resolved geometry; whole-sample drop if necessary. | Every accepted object must fit the already frozen experiment. Placement cannot resize it. |
| 9. Emit and preflight | Pure serialization; native parser/model-build checks; geometry and material-table parity checks. | Checks must cover what gprMax actually builds, not only what an earlier schema intended. |
| 10. Admit and execute | Hardware/source compatibility, peak memory budget, controlled workers, declared input manifest. | Resource choices can schedule the experiment but cannot change it. |
| 11. Validate outputs and publish dataset | Finite fields, actual timing/layout consistency, complete provenance, scientific benchmark status, persistent labels. | Successful process exit alone is insufficient evidence that the training sample is valid. |

Snapshots are validated against the final domain and time axis, then serialized in stage 9. They must not drive or bypass the numerical derivation. A late change to mode, material, waveform, target ranges, source layout, or roughness invalidates affected downstream artifacts; the agent must regenerate the contract rather than reuse stale manifests.

The existing conceptual ordering—physical samples and excitation before wavelengths, wavelengths before final grid, final grid before timing—remains intact. If a plan is infeasible, reject it or restart the common planning process with explicitly changed policy. Never repair individual emitted samples by giving them different numerical settings.

## 5. Geometry, collection, and sampling design

### 5.1 Required schema additions

Use explicit semantic fields rather than overloading existing 2D names.

| Record | Required additions or clarifications |
| --- | --- |
| Dataset configuration | Explicit mode; coordinate-frame identifier; physics/schema version; reproducible seed policy. Keep cell size and physical domain derived. |
| Box ranges and samples | Crossline center offset and finite crossline size, in addition to x offset, center depth, width and height. Distinguish center depth from top/bottom depth. |
| Cylinder ranges and samples | Finite axial length and axis selection, plus radius and full center position. Axis-aligned cylinders can be the first supported set. |
| Shape abstraction | Three-dimensional bounds, intrinsic minimum feature, bottom/top depth, canonical emitted geometry, material, and fixed/dynamic status per field. |
| Antenna/acquisition | Explicit source axis; complete resolved Tx/Rx coordinates; horizontal separation components where supported; receiver height semantics; acquisition identifier. |
| Global aggregate | x and z footprint bounds, pinned-position bounds, deepest target bottom, minimum finite feature, material/spectral envelope, roughness envelope. |
| Global derived plan | `domain_x_m`, `domain_y_m`, `domain_z_m`; `nx`, `ny`, `nz`; all spacings; ground reference; source/receiver cell coordinates; actual timing and iteration count. |
| Resolved sample | Original draw, effective cell geometry, material identifiers/table provenance, placement attempts, quantization differences, final acceptance status. |

A 2D cylinder cannot be migrated to 3D by assigning it an arbitrary finite length. That length is new physical information and must be supplied or accepted as an explicit dataset-level default. Likewise, a 2D box has no known crossline size. Existing 2D data remains 2D.

For optional future rotations, an axis-aligned bounding box is sufficient for conservative boundary clearance, but not for minimum-feature resolution or exact collision detection. A long thin object does not become well resolved because rotation enlarges its bounding box. Keep intrinsic thickness separate from spatial extent. The native `#box` describes an axis-aligned box; arbitrary rotations need an explicitly supported construction path and voxelization checks.

### 5.2 Reproducible random draws

Maintain the current rule that the LLM supplies ranges and deterministic code draws samples. Document whether a moisture band is fixed across samples or sampled as a sub-band: the current active sampler carries the supplied band through unchanged despite comments describing sub-band sampling. Adding 3D should not silently change that distribution.

For the new contract, derive independent reproducible random streams from dataset seed, sample identity, layer/target identity, and operation. Adding a z field should not accidentally change soil composition because one shared random stream consumed an extra draw. Preserve the legacy seed behavior for old contract versions. Record seeds for volumetric soil and surface roughness as well as scalar parameter draws.

Placement retries alter the distribution of surviving targets. Record requested ranges, initial draws, accepted values, retry counts, and rejection reasons. Report survivor counts and distributions; never describe a constrained/rejected population as an unchanged uniform distribution. When splitting training and evaluation data, keep related traces and variants of the same physical realization together to avoid leakage.

### 5.3 Placement must satisfy both user intent and physical feasibility

For each axis and depth, intersect the requested range with the feasible region computed from the object's full extent, burial requirement, and PML clearance. Empty intersection means infeasible placement. For example, a target requested at crossline offsets 0.20–0.25 m cannot be relocated to z-center offset 0 merely because the central position fits.

Every fixed field stays fixed. An object with variable radius and fixed depth is not permission to vary depth. Retain the current limit of 20 placement attempts and whole-sample rejection. Dynamic shrinkage must stay within user bounds, must not go below the common feature-resolution floor, and must not enlarge another extent beyond the global plan.

Define target overlap policy explicitly. Distinct, non-overlapping PEC objects are a sensible initial contract. Overlap is not inherently a violation of Maxwell's equations, but it can merge targets into a different geometry and invalidate object-count or shape labels. If overlaps or embedded components are later permitted, specify material precedence and label semantics.

### 5.4 Resolve geometry before final validation

The current writer snaps ground/layer coordinates and extends the deepest layer to y=0 after earlier validation. The database can retain nominal thicknesses while the emitted geometry has different effective thicknesses. Three-dimensional volume labels amplify this problem.

Introduce a deterministic resolved-scene representation before emission:

1. Establish integer domain cell counts and a common integer ground reference.
2. Resolve layer interfaces, target geometry, Tx/Rx locations, and output regions with the native solver's rounding conventions.
3. Validate these effective coordinates and extents, including half-cell/component staggering where relevant to a source or receiver.
4. Record both physical requests and effective representations, with quantization errors and units.
5. Serialize enough numeric precision to reconstruct the intended cell counts. Reparse the emitted values and compare them with the plan.

Exact arbitrary physical coordinates are not always representable on a Yee grid. A fixed requested position therefore means no stochastic relocation; it also requires either an explicitly accepted quantization tolerance or rejection when exact placement is required. Never claim sub-cell geometry has been represented exactly by silently rounding it.

For labels such as occupied target volume, soil volume fractions, or actual interface shape, obtain the effective values from the native built geometry/material map. A cylinder's bounding box is not its occupied volume, and a resolved command alone does not prove how curved boundaries were staircased. Record the definition of each label: requested analytic geometry, effective voxel geometry, native material parameter, or a frequency-specific derived quantity.

Make the terminal-layer convention explicit. If the deepest material is a half-space extended into the bottom boundary, its effective extent differs from a sampled finite-layer thickness. Preserve that convention with honest labels, or introduce finite layers plus an explicit background half-space in a new contract. Do not train on a purported independently realized bottom interface that was never emitted.

## 6. Material, spectrum, grid, and domain physics

### 6.1 Keep native material generation and strengthen its validation

The existing `_GridStub` correctly permits native Peplinski material generation before a time step exists. Continue using it or an equivalent narrow adapter. The material coefficients are available before spatial planning; the stability check involving time step follows afterward.

The bundled native code at `gprMax/gprMax/materials.py:269` creates Debye materials from the moisture band. Its actual material moisture values are shifted by half a bin, including a final value above the supplied upper band endpoint. Therefore:

- Require at least two bins; the current `fractal_nbins > 0` schema permits one, but native generation indexes a second bin.
- Validate the actual highest instantiated moisture value against porosity and the project's moisture calibration cap. Checking only the requested band maximum is insufficient near saturation.
- Validate finite, positive densities; finite texture/moisture inputs; nonnegative fractions; closure; and an admissible nonzero moisture band before native evaluation.
- Inspect every generated material for finite coefficients, admissible permittivity/relaxation parameters, and passive response. PEC's intentionally idealized infinite conductivity is a separate built-in case, not an ordinary finite-soil validation failure.
- Persist the native model version, bin count, band, evaluation frequency, and material-table digest. Compare the preflight native table with the table used in derivation.

The current edge-bin evaluation uses `Material.calculate_er(f).real`, correctly avoiding the high-frequency `Material.er` field as a substitute for in-band permittivity. Keep that distinction. The current sigma labels are native `Material.se` values; they are not a complete frequency-dependent attenuation curve. Derive any new spectral or attenuation labels explicitly from the native complex response and name them accordingly.

The full material table provides conservative available-material bounds. It does not imply that every bin occupies an equal fraction of a particular generated volume, or that every bin appears in that realization. If training labels include realized averages, distributions or correlation statistics, derive them from the generated material map rather than substituting the bin endpoints or the input band midpoint.

The native implementation identifies its 0.3–1.3 GHz conductivity branch in code. Its correspondence to the original empirical model should remain versioned, not silently replaced during 3D work. Native use ensures implementation consistency; it does not by itself establish that every proposed soil composition lies inside the empirical calibration data.

### 6.2 Distinguish three frequency concepts

The platform must consistently distinguish the entered frequency and its meaning, the actual emitted waveform peak, and the high-frequency design limit used for numerical resolution. Current derivation and validation use different high-frequency factors, which can produce different judgments about the same grid.

Preserve the current Ricker peak/band-center conversion and its band-edge gate, using one authoritative helper. Persist the converted peak and the frequency interpretation flag. If other waveform families are allowed, their spectral treatment must be derived from that waveform; Ricker-specific band factors cannot be applied indiscriminately.

A Ricker pulse has spectral tails beyond its selected useful band. Consequently, the useful-band Peplinski gate does not prove that every nonzero spectral component is inside the calibration range. The high-frequency resolution limit can also extend beyond that useful band. Document the spectral cutoff criterion and the treatment of out-of-band energy; do not describe a numerically finer grid as extending the empirical material model's validity.

For numerical bounds, evaluate native material response across the declared design frequencies and all instantiated bins, or establish a tested conservative bound. Evaluating only dry/wet bins at the peak is not a general proof of the worst wavelength over frequency. Preserve peak-frequency labels separately. For strongly lossy cases, assess phase wavelength and attenuation length from the complex native response; using only real permittivity is an approximation that needs validation. This is electromagnetic analysis of the existing material response, not a replacement mixing model.

### 6.3 Extend the common resolution budget

Use the most demanding requirement across all samples. Preserve cubic cells initially, even though cubic spacing is a project simplification rather than a universal FDTD requirement.

| Geometry/material feature | 3D resolution consideration |
| --- | --- |
| Propagating fields | Shortest relevant wavelength across the declared numerical spectrum and native materials. |
| Box | Smallest of all three physical side lengths. |
| Finite cylinder | Diameter and axial length; both must be resolved. |
| Optional sphere | Diameter, plus convergence of the staircased boundary. |
| Layer | Actual resolved thickness for every sample, not only the maximum requested thickness. |
| Roughness | Height variation and meaningful lateral scales in both horizontal directions. |
| Future detailed antenna | Smallest required feed/structural feature and the antenna model's supported mesh. |

Keep the current wavelength/feature cell-count rules as minimum acceptance policy. They do not establish adequate accuracy for every geometry or loss level. Native modeling guidance recommends approximately ten cells per shortest wavelength and separation between PML and sources/targets; these are starting criteria to be checked by convergence. [gprMax modeling guidance](https://docs.gprmax.com/en/latest/gprmodelling.html)

### 6.4 Derive all three domain dimensions once

Resolve x and z independently from the common lateral design floor, the full target/acquisition footprints, and face clearances. A square lateral domain is a reasonable initial option, but neither the solver nor physics requires x and z to be equal. If one axis needs more space, enlarging both is conservative but costs memory.

The vertical envelope includes the soil/background convention, deepest possible accepted target, bottom clearance, rough-surface extremes, source/receiver heights, and top clearance. The domain origin and ground reference remain fixed across samples. Sampled layer thicknesses change interfaces below the shared ground; they must not shift antennas or the domain.

Round domain sizes upward to integer cells. Then resolve the common source/receiver locations to cell coordinates and recheck margins. In 3D, require more than one cell in every direction and sufficient interior cells after both PML faces and required clearance. Do not permit a small z extent to silently trigger TMz.

The project derives a lateral floor of 1.5 times its maximum wavelength and a default source height of half that wavelength. The author-hosted study behind this policy investigated lateral-domain sensitivity for particular soil/source arrangements; it found a useful accuracy/cost compromise, not a universal sufficient boundary condition for arbitrary 3D targets and antennas. Retain the policy as an initial floor, then qualify its use against representative 3D scenes. Also make clear whether the reported lateral span includes PML or denotes usable interior, so comparison with the study is meaningful. [Khosravi Largani, Zekavat and Namdari, *FDTD Medium Dimension Selection Guidelines for GPR Synthetic Data Generation*](https://soilx.wpi.edu/wp-content/uploads/2024/09/Letter_Noushin_FDTD-Medium-Dimension-Selection-Guidelines.pdf)

### 6.5 PML and half-space continuation

Use all six PML faces in 3D, with the native order x0, y0, z0, xmax, ymax, zmax. The existing configuration helper already distinguishes this from thin-z 2D. PML occupies cells inside the domain; it is not extra space outside the declared dimensions. [gprMax input specification](https://docs.gprmax.com/en/latest/input.html#pml-cells)

Preserve the current distinction between `(pml + buffer)` padding and `(pml + 15)` source/target clearance. The latter measures distance from the outside domain face, including the PML thickness; it leaves the extra gap inside the physical interior. Generalize the checks to all six faces and to the full physical antenna extent if detailed antennas are introduced.

Soil intended to continue as a lateral half-space must reach the lateral boundaries, including the new z faces. Finite buried targets must remain clear of them. Stopping the soil before a face creates an unintended soil-air interface; extruding a finite target into a boundary changes it toward an invariant/truncated object. Verify both through actual native geometry inspection.

## 7. Time step and recording duration

### 7.1 Derive the actual 3D time step

Use the same mode and emitted spatial values that gprMax uses. At equal cubic spacing, the 3D CFL step is about 81.65% of the 2D TM step, so the same recording duration requires about 22.5% more updates. Do not reuse a 2D `dt` because the material parameters are unchanged. Native mode selection, rounding, optional stability-factor handling, and iteration calculation are implemented together in `gprMax/gprMax/input_cmds_singleuse.py:144–204`.

Make all validators consume that common result instead of independently recomputing inconsistent approximations. Keep time step downstream of the final spatial plan; the agent must not ask for it as an independent physics input. Compare the planned value with native preflight and actual HDF5 output, using a defined numerical tolerance and exact iteration-count checks.

The bundled soil relaxation time is 9.231 ps. Check every relevant relaxation time against actual `dt`, including future water or additional dispersive materials. The current error text recommending that a user “coarsen grid to raise dt” when relaxation time is too short has the direction wrong: raising `dt` worsens that condition. Reject such a plan or restart global planning with finer resolution. An optional native stability factor could reduce `dt` downstream, but enabling it would need a documented common policy, changed iteration/memory estimates, and tests; it must never be applied to only some samples.

### 7.2 Extend the recording window to the 3D region of interest

The current vertical two-way estimate does not explicitly account for crossline offsets, full bistatic paths, or user-defined source delays. For the 3D planner:

1. Define the physical region of interest: the interfaces and target-placement envelope whose returns the dataset promises to record.
2. Bound travel from the fixed transmitter through that region to every enabled receiver. Include lateral/crossline separation and the slowest relevant response across the population.
3. Include the waveform's actual delay and useful duration, plus a documented late-return margin.
4. Select one iteration count from the common time step; record the actual last sample time, which need not equal the requested continuous-time window exactly.
5. Qualify the estimate with longer-window runs and a defined tail/truncation criterion for representative difficult scenes.

A straight-path geometric bound with a conservative material speed is an initial planning estimate, not proof that all refracted, dispersive, resonant, or multiply scattered energy has ended. In lossy/dispersive soil, peak-frequency phase speed is not automatically a bound on pulse/group delay. Validate the chosen estimate over the declared spectrum and scene class. The goal is to capture the scientifically relevant response, not to promise that every possible late reflection is zero.

The native smoke check illustrated why this matters: a preliminary 1 ns window ran but produced a warning that the source pulse was truncated and numerical dispersion analysis was skipped. The recorded 6 ns check removed that warning. Production validation must catch this condition before accepting a dataset, rather than counting solver completion as success.

## 8. Sources, receiver components, volumetric soil, and roughness

### 8.1 Correct source polarization and source-kind behavior

For existing 2D TMz datasets, z polarization is physically appropriate to the selected mode. Fix their collection/metadata contract so the requested and emitted axes agree. For full 3D, emit the selected x, y, or z axis exactly and persist it as the resolved source axis. In this coordinate system y polarization is vertical; z is a horizontal crossline orientation.

| Source kind | 3D handling |
| --- | --- |
| Hertzian dipole | Emit the chosen axis, full position, waveform, and timing. Treat it as an idealized source, not a calibrated commercial antenna. |
| Voltage source | Preserve its own command and required source parameters; use the declared axis and position. |
| Transmission line | Preserve its command, resistance, axis, and timing. The bundled parser rejects GPU execution for this kind, so route explicitly to CPU when execution policy allows, or reject the incompatible request. |
| Any unsupported kind | Reject with an explanation. Do not substitute a different source. |

The CPU restriction is explicit in `gprMax/gprMax/input_cmds_multiuse.py:318`. Adding 3D does not remove it. Source impedance rules should come from the selected native source and the established platform contract, not from an invented universal antenna impedance.

Keep physical source normalization explicit. Native Hertzian injection uses a dipole length equal to the cell spacing along its axis (`input_cmds_multiuse.py:194`, `sources.py:156`). Therefore, keeping the same waveform amplitude while refining the grid does not automatically preserve the same physical current moment. Convergence tests must preserve or correctly account for the intended source moment; dataset manifests must record the normalization convention. Do not compare differently normalized 2D and 3D traces as if their amplitude difference were a solver error.

The current `rx_same_height=False` branch still assigns Tx height to Rx. Implement an actual receiver-height parameter/derivation or reject that unsupported setting. Similarly, the schema containing an `rx_array` is not proof of emission support: the current writer emits one `#rx`.

### 8.2 Preserve vector field meaning

The current extractor already reads Ex, Ey, Ez, Hx, Hy, and Hz from the first receiver. Retain all available components for 3D and label their units and coordinate frame. Do not assume Ez is the only meaningful signal. Individual zero components can be valid because of symmetry or orientation; they are not automatically failed simulations.

If the product exposes a receiver-oriented measurement, define its relationship to the field components. An ideal projected electric-field signal is not automatically the voltage of a modeled physical antenna. Multiple receivers require a receiver dimension in storage and APIs; avoid silently discarding all groups except `rx1` if arrays are enabled. The official output format describes receiver field arrays and their units. [gprMax output specification](https://docs.gprmax.com/en/latest/output.html)

### 8.3 Build a volume rather than repeating a slice

Each layered `#fractal_box` must extend through the resolved z domain and generate a three-dimensional heterogeneous field. Repeating one random 2D slice along z imposes artificial invariance and does not provide the intended 3D heterogeneous population. The native advanced example demonstrates Peplinski material bins distributed across a volume with directional fractal weights and a rough surface. Its numerical values are illustrative; do not copy its waveform into this project's stricter frequency contract. [gprMax heterogeneous-soil example](https://docs.gprmax.com/en/latest/examples_advanced.html)

Keep the existing native fractal parameter meanings. In particular, do not automatically add one to the current fractal parameter merely because the simulation gains a spatial dimension. If spatial correlation is important to the intended dataset, measure and document it in physical units; directional weights alone should not be advertised as specified correlation lengths.

For convergence experiments, the same seed with different grid dimensions does not generally produce the same physical random medium. Hold a physical realization fixed through a controlled representation/resampling procedure, or start with homogeneous benchmark scenes. Otherwise, apparent mesh or boundary effects can actually be changes in soil realization.

### 8.4 Treat surface roughness as a two-horizontal-direction surface

The ground is a y-valued height field over x and z. Map the roughness weights to those two directions; the existing generic `weight_y` parameter in the thin-z writer must not accidentally become the vertical weighting of a surface generator.

Check peaks against Tx/Rx and top clearance, and troughs against burial and layer-interface constraints. A target beneath the nominal flat surface can become exposed by a trough. A first implementation may conservatively require burial beneath the lowest permitted surface everywhere; a more permissive implementation needs validated local surface heights. Ensure interfaces remain ordered and no negative-thickness or unintended air/soil gaps appear.

If roughness is deferred from the first 3D release, reject it in 3D rather than retaining a 2D profile silently. The same applies to collected water/vegetation settings that lack an active emission path. Published realistic GPR modeling work treats heterogeneous soil and rough surfaces as substantive parts of the scene, rather than cosmetic geometry. [Giannakis, Giannopoulos and Warren, 2016](https://www.research.ed.ac.uk/en/publications/a-realistic-fdtd-numerical-modeling-framework-of-ground-penetrati/)

## 9. Required validation matrix

Validation must be a named, recorded part of the pipeline. Failures should identify the offending input or derived quantity and the rule it violates. Warnings that affect scientific eligibility must not disappear between stages.

| Gate | Checks required for 3D | Failure handling |
| --- | --- | --- |
| Configuration | Supported explicit mode; consistent frame; positive sample count; valid numeric ranges; finite inputs; no unsupported options. | Reject collection/configuration. |
| Soil envelopes | Texture feasibility and closure; positive ordered densities; moisture-band ordering and feasibility; declared empirical-domain policy. | Reject infeasible envelopes; explicitly label or reject allowed extrapolation. |
| Concrete soil draw | Actual composition, porosity, positive thickness and native material validity for every layer. | Bounded redraw according to sampler policy, with reason recorded. |
| Native material table | At least two bins; actual bin moisture bounds; finite/passive coefficients; expected material count; parity with emitted model. | Reject the affected sample or shared material policy. |
| Waveform | Supported family; finite amplitude/frequency; one peak/band interpretation; useful-band gate; explicit numerical spectrum; valid source timing. | Reject inconsistent excitation before final planning. |
| Grid | Positive cubic spacing; global wavelength and intrinsic-feature budget; actual per-sample layer resolution. | Reject the common plan; do not silently coarsen. |
| Dimensionality | `nx`, `ny`, `nz` all greater than one in 3D; native mode reports 3D; no thin-axis fallback. | Reject the entire mismatched contract. |
| Domain/PML | Integer dimensions; valid six-face PML profile; sufficient interior in each axis; sources/receivers/finite targets clear all applicable faces. | Reject plan or bounded placement, according to ownership. |
| Source | Exact kind/axis/timing/position; valid resistance where applicable; CPU/GPU compatibility; declared source normalization. | Reject incompatible source/backend; never substitute source kind or polarization. |
| Targets | Positive finite dimensions; PEC-only contract; full 3D bounds; requested-range membership; fixed-field preservation; grid feature floor; burial; declared overlap policy. | At most the permitted retries; then drop the whole sample. |
| Stratigraphy | Ordered resolved interfaces; no collapsed layers; no unintended gaps; explicit terminal half-space semantics. | Reject invalid resolved scene. |
| Roughness | Correct surface axes; seeded realization; valid peak/trough envelope; burial and interface ordering; source/receiver clearance. | Reject unsupported or infeasible roughness. |
| Timing | Native-compatible CFL and rounding; relaxation-time conditions; source fits window; path envelope and final iteration count. | Reject common time plan; forbid per-sample shortening. |
| Requested outputs | Snapshot/geometry bounds, positive valid strides, unique identifiers, snapshot times inside actual window. | Reject invalid output settings before model build. |
| Emitted input | Required commands, resolvable material/waveform names, correct geometry precedence, exact mode, no unsupported commands. | Native preflight failure blocks execution. |
| Resource admission | Host RAM, device VRAM, material coefficient limits, scratch/output capacity and concurrent-worker reservation. | Queue, reduce concurrency, use a permitted backend, or reject with cost details. Never change the scene automatically. |
| Output integrity | Correct file/sample identity; expected receivers/components; equal lengths; finite values; actual cell counts, timing and acquisition match the contract. | Mark result failed/ineligible; exclude from reuse/training exports. |
| Scientific eligibility | Benchmark/convergence coverage for the declared scene family; recorded tolerances and exceptions. | Keep unqualified outputs distinguishable from validated production datasets. |
| Batch invariance | Same mode, grid, domain, ground reference, waveform, source/receiver plan, `dt`, iteration count and contract version across survivors. | Block dataset completion on mismatch. |

The existing standalone validators for material names, essential commands and snapshot timing are not a substitute for wiring those checks into emission/preflight. Likewise, checking the maximum requested layer thickness does not prove that every drawn layer remains resolved.

Separate three statuses: configuration valid, solver execution successful, and scientifically qualified for the dataset's intended use. They answer different questions. Every output need not be an expensive convergence experiment, but its scene class and parameter envelope must lie within a documented qualification scope.

## 10. Execution, memory, and storage planning

### 10.1 Replace the current coarse capacity check

The current memory validator estimates 146 bytes per cell plus 50 MB against a default 32 GiB host limit. That is insufficient as a 3D admission policy. The bundled solver allocates geometry/material IDs, staggered electric/magnetic fields, dispersive auxiliary arrays, PML state, and optional snapshots; fractal construction also has temporary allocations. See `gprMax/gprMax/grid.py` and `gprMax/gprMax/model_build_run.py`.

Estimate peak host memory separately from peak device memory. Include native precision, number of dispersive poles, full material-table size, PML faces, geometry-build/fractal work, outputs, and runtime reserve. Device coefficient-table limits can be distinct from total VRAM. Account for other active workloads, including the API and retrieval/model services, rather than assuming all machine RAM is available.

The default of two concurrent models per GPU is unsuitable as an unconditional 3D default. Start with one, then admit additional workers only if each device and host have enough reserved capacity. CPU execution also needs memory admission; removing the GPU does not remove the volumetric cost.

The documented MPI/GPU route distributes independent models as tasks. It does not establish that one oversized model can be split across the combined memory of several GPUs. Budget each model against the device that will execute it. [gprMax GPU execution guide](https://docs.gprmax.com/en/latest/gpu.html)

### 10.2 Illustrative scaling, not an approved production grid

For a fixed 1.2 × 0.8 × 1.2 m box and 20 ns requested recording duration:

| Cubic cell size | Grid cells | Total cells | Current coarse memory estimate | Approximate native CFL step | Nominal iterations |
| --- | --- | --- | --- | --- | --- |
| 5 mm | 240 × 160 × 240 | 9.216 million | 1.30 GiB | 9.629 ps | 2,079 |
| 2.5 mm | 480 × 320 × 480 | 73.728 million | 10.07 GiB | 4.815 ps | 4,156 |
| 2 mm | 600 × 400 × 600 | 144 million | 19.63 GiB | 3.852 ps | 5,194 |

These estimates reproduce the current validator's arithmetic, including its 50 MB allowance; they are not measured capacity requirements. Actual numerical/material checks still apply. In particular, **the 5 mm row's CFL step exceeds the bundled Peplinski relaxation time**, so that row would fail the existing soil stability condition at the unmodified CFL step. A table of affordable cell counts must not be confused with a table of valid soil simulations.

For fixed physical extent and recording duration, halving cubic spacing multiplies cell count by eight and approximately doubles the updates per cell, producing about sixteen times the field-update work. Extending a former one-cell z direction to `Nz` cells adds approximately `Nz` times the bulk cell count, plus the change from 2D to 3D timing and additional boundary state. Actual runtime also depends on preprocessing, memory bandwidth, materials and hardware; benchmark representative cases before estimating throughput.

### 10.3 Schedule without changing the experiment

Resource recovery may reduce concurrency, queue jobs, resume incomplete samples, or use another explicitly permitted execution backend. It may not automatically remove layers/targets, coarsen the grid, shorten the recording window, reduce the domain, replace the source, or run 2D in place of 3D.

Execute only files belonging to the current accepted manifest. Directory globbing can pick up stale input/output files from an earlier generation or mode; match contract identifiers, sample identities and file hashes during execution and ingestion. Preserve the existing isolated worker processes and per-file temporary-directory behavior.

Full-volume snapshots can exceed receiver trace storage by orders of magnitude. Make snapshot stride, fields and times explicit, estimate disk/scratch usage, and validate them against the resolved grid. Coarser visualization output is acceptable when declared; silently making the solver grid coarser is not. Retain native output artifacts or a documented lossless scientific storage representation, with database rows serving as indexed metadata where scale warrants it.

## 11. Persistence, reuse, and visualization

### 11.1 Introduce a canonical dataset contract

Persist a versioned manifest containing at least:

- Mode, coordinate frame, physics-policy version and exact gprMax version/revision.
- Common cell spacings/counts, physical domain, ground reference, PML configuration and boundary-clearance policy.
- Entered frequency semantics, actual waveform peak, useful/design bands, source parameters and normalization.
- Native material-model identity, bin policy, label-evaluation frequency and material provenance.
- Complete fixed Tx/Rx/acquisition layout, actual time step, iteration count and final time sample.
- Sampling/placement/roughness seeds, permitted ranges, rejection policy and output specification.
- A contract digest linking the input manifest, resolved scene, database record and solver output.

Material values and target geometry may vary by sample, so they belong in per-sample resolved records linked to this shared contract. Keep requested values, original draws, effective geometry and executed metadata separately inspectable. After execution, validate the HDF5 settings against the contract rather than copying the planned values and assuming success.

### 11.2 Migrate safely

Add domain z and the new fields through explicit database/API migrations. Keep existing 2D records readable. Backfill a thin z extent only when it can be established from the original grid or emitted file; do not invent missing finite target lengths or positions. Historical source metadata that may disagree with emitted inputs needs provenance-aware verification before reuse.

The similarity subsystem already filters on dimensionality. Preserve that existing protection. Version its planar feature representation and add z offsets, finite lengths, orientation, actual source axis, acquisition layout and detailed roughness settings. Include the peak-vs-band-center interpretation, which is currently absent from the feature payload despite affecting physics.

Distinguish “similar configuration to inspect” from “equivalent result eligible for substitution.” Numerical-contract/solver compatibility and the actual requested experiment need exact checks for reuse; a high vector similarity score alone is not sufficient. Never offer a 2D result as an equivalent replacement for a 3D request.

### 11.3 Display the same resolved scene that was simulated

A first release can provide three orthogonal slices: x–y, z–y and x–z, with explicit slice positions. A full interactive 3D view can follow. Both must use the canonical resolved geometry, including source/receiver axes, ground, interfaces, targets and PML/interior bounds.

Do not project every target into a single cross-section as if it intersects that slice. Mark previews as provisional until the common grid and placement are resolved. Provide native geometry exports for detailed inspection; gprMax's output documentation describes the geometry formats and ParaView workflow. [gprMax geometry outputs](https://docs.gprmax.com/en/latest/output.html)

Signal charts need receiver/component selection, explicit time units, and actual solver timing. If only one receiver is supported, show that scope clearly. A 3D field simulation can still produce a single A-scan; a 3D-looking geometry viewer does not imply C-scan acquisition.

## 12. What the agent should be able to do after enablement

The agent remains a collector and orchestrator backed by deterministic tools. Its expanded capabilities should be:

1. Explain the difference between the supported 2D mode and full 3D, and collect the selected mode without silently overriding it.
2. Collect finite target dimensions, crossline ranges, source polarization, supported acquisition settings and roughness intent using mode-specific schemas.
3. Validate units and feasibility early, and identify missing physical information such as cylinder length or receiver height.
4. Explain each derived choice from recorded provenance: which material/spectral condition sets resolution, which target or acquisition extent sets a domain dimension, and which path/duration condition sets the time window.
5. Show one common domain, grid, time axis and estimated cost before the user proceeds to the existing forward-simulation action.
6. Report concrete validation failures, rejected samples and realized distributions rather than promising that every requested sample was emitted.
7. Trigger native preflight, supported execution, output validation, geometry inspection and component-aware trace display.
8. Detect stale downstream state after edits and regenerate it; distinguish similar past datasets from compatible reusable ones.

The agent should not invent arbitrary antenna support, choose unrequested target material models, silently alter source kind/polarization, change the grid per sample, or describe an unqualified numerical result as experimentally validated. Responses should be generated from structured derivation/validation records, not reconstructed guesses about why a value was chosen.

Update `AGENT.md`, stage prompts, tool descriptions, error messages, README and API schemas together. Merely adding `"3D"` to an enum while the prompt or store still forces 2D would leave the agent contract internally inconsistent.

## 13. Implementation order and release boundaries

| Phase | Concrete work | Exit condition |
| --- | --- | --- |
| A. Establish contracts | Versioned mode/frame schema; 2D compatibility fixtures; canonical scene/manifest design; source-axis semantics; migration design. | Every new field has unambiguous physical meaning and a persistence path. |
| B. Build deterministic geometry and material planning | Finite target schemas/sampling; native-bin checks; spectral-budget unification; 3D bounds; grid/domain resolution; quantization before validation. | Deterministic fixtures generate consistent three-dimensional resolved scenes. |
| C. Complete numerical and placement validation | Unified CFL/timing; all-face clearance; actual-layer checks; range-preserving placement; roughness policy; batch invariants. | Invalid scenes fail deterministically; valid scenes stay within the frozen contract. |
| D. Emit and preflight 3D | Mode-aware sources/receivers/geometry/snapshots; pure writer; native model-build and material/geometry parity. | Native gprMax reports 3D and builds exactly the intended supported scene. |
| E. Execute and persist | Source/backend compatibility, resource admission, manifest-scoped runs, HDF5 validation, migrations and reuse restrictions. | Results are complete, correctly labeled and tied to the executed contract. |
| F. Qualify and expose | Numerical benchmarks, convergence envelope, UI slices/components, agent explanations, documentation. | The declared first-release scene family meets recorded acceptance criteria. |
| Later extensions | Arbitrary rotations; new target materials; realistic antennas; receiver arrays/scans; broader soil/surface models. | Each extension adds its own physics, representation, resource and validation coverage. |

Do not expose a generally available 3D switch before phases A–F are complete for its declared capability set. Internal fixtures and developer-only execution can exercise intermediate work without presenting partial support as finished.

Realistic antennas are a separate modeling project. Native library models have specific geometry and supported resolutions; some use 0.5, 1 or 2 mm meshes and a different vertical-coordinate convention. Their full dimensions, feed placement, orientation, calibration and grid restrictions must enter planning rather than being treated as a new source-name alias. [gprMax antenna-model library](https://docs.gprmax.com/en/latest/user_libs_antennas.html)

If B-/C-scans are added, define a common acquisition path/grid per dataset. Separate physical sample identity from trace/scan-position identity; keep the same soil/target realization while the acquisition moves. Steps must preserve intended cell geometry, and all acquisition positions must fit the same domain and time-window envelope. Resampling the soil for each trace would create a sequence of different scenes rather than a scan of one scene.

## 14. Tests and numerical qualification required before release

These are proposed acceptance tests, not tests already passed by the platform.

### Deterministic contract and integration tests

- Mode survives collection, save/load, regeneration, emission, execution metadata, export and reuse filtering.
- A true 3D plan cannot produce any one-cell dimension or a native 2D mode. Existing 2D fixtures retain their intended TMz behavior.
- Every supported source/axis combination creates the selected native source. Unknown kinds fail; transmission-line/GPU incompatibility is caught before an expensive run.
- Off-axis targets and targets close to each of the six faces exercise complete bounds. Include finite cylinder ends, thin box sides and partial fixed-field cases.
- Boundary and half-cell cases produce the same effective coordinates in planning, native model build, labels and visualization.
- Moisture near porosity/calibration limits exercises the shifted native bins; one-bin settings fail cleanly; invalid or nonfinite material responses cannot proceed.
- Source delays, pulse truncation, sparse snapshot times and iteration rounding cannot create mismatched time axes.
- All accepted files and outputs share the same contract; adding a later out-of-contract sample cannot mutate an existing dataset.
- Regeneration cannot run or ingest stale files; retries/resume preserve sample identity; failed outputs cannot enter training/reuse exports.
- New 3D data is never matched as a compatible substitute for 2D data; old records remain readable without invented 3D information.

### Scientific tests

| Test | What it establishes | Important control |
| --- | --- | --- |
| Free-space dipole against an analytical solution | Propagation, vector components and source normalization are implemented consistently. | Match source moment, coordinates and sampling conventions. |
| Homogeneous and planar layered scenes | Arrival times and interface reflection behavior agree with independent reference expectations. | Use a reference appropriate to the source and incidence; a near-field dipole is not automatically a normal-incidence plane wave. |
| Horizontal rotation/symmetry | Swapping x and z consistently in an otherwise equivalent isotropic scene transforms the response correctly. | Rotate sources, receivers, objects and components together; keep y vertical. |
| Reciprocity in an applicable passive reciprocal setup | Tx/Rx interchange has the expected response relationship. | Match source/receiver definitions and normalization; do not compare different physical observables. |
| Mesh refinement | Waveform timing, shape and amplitude approach a stable result. | Keep the physical geometry/material realization and source moment consistent. |
| Separate x/z/vertical domain enlargement | Boundary proximity does not materially change the region-of-interest response. | Keep central geometry and heterogeneous realization fixed while enlarging the exterior. |
| PML thickness/clearance variation | Absorbing-boundary errors are below the selected tolerance. | Do not confuse a changed physical domain with a PML-only experiment. |
| Longer recording windows | The chosen window captures required returns without source or response truncation. | Compare identical early samples and scientifically meaningful tail metrics. |
| Finite-target scattering and orientation cases | Finite ends, off-plane location and target representation behave sensibly. | Use suitable analytical, independently converged or experimentally characterized references. |
| CPU/GPU parity for supported cases | Backend choice preserves results within a measured tolerance. | Same input, precision, source and output quantities; no transmission-line GPU case. |
| Representative extreme population | The common policy covers difficult soil/target combinations in the declared envelope. | Include high permittivity/loss, smallest features, deepest/off-axis targets and roughness extrema. |

The native project supplies an analytical free-space Hertzian-dipole comparison and reference-solution implementation, which is a useful starting benchmark. Its reported example error is specific to that setup; it should not be copied as a universal tolerance for this platform. [gprMax analytical comparisons](https://docs.gprmax.com/en/latest/comparisons_analytical.html)

Choose and record tolerances for arrival-time error, normalized waveform error, amplitude error, boundary sensitivity and late-window energy according to the intended scientific task. A single universal percentage cannot certify every scene or derived label. Do not require each field component to be nonzero, or an open lossy domain to conserve field energy internally as if it were a closed lossless system. Check for unexplained growth after excitation with the source, losses and outward flux accounted for.

Keep inexpensive contract tests in routine CI. Run representative native solves in integration testing and the more expensive refinement/domain studies as a versioned qualification suite. Requalify when solver, material model, source normalization, numerical policy or supported scene envelope changes.

## 15. Decisions to settle during implementation

The following choices need an explicit contract, but do not prevent beginning the schema and deterministic-planning work:

| Decision | Recommended initial choice |
| --- | --- |
| Vertical coordinate | Retain y; rename ambiguous derived depth fields through versioned compatibility. |
| Target scope | Finite PEC boxes and axis-aligned finite cylinders; reject unsupported rotations/materials. |
| Acquisition | One fixed Tx/Rx configuration per dataset; collect missing height/separation semantics. |
| Polarization | Honor x/y/z in 3D; explicitly enforce z for the existing 2D TMz path. |
| Grid | Retain one cubic grid; refine or reject the global plan when required. |
| Roughness | Implement the complete 3D surface checks or explicitly disable it for the initial 3D release. |
| Terminal soil layer | Preserve a documented half-space convention with effective labels; use a new contract if introducing a finite last interface. |
| Sampling corrections | Version range-preserving placement and any random-stream changes; report acceptance effects. |
| Numerical eligibility | Publish the tested scene envelope and metric tolerances, not just “solver succeeded.” |
| Out-of-band material use | Explicit spectral validity policy; do not imply that the existing useful-band gate certifies all pulse tails. |
| Hardware | Source-aware CPU/GPU selection and measured peak-memory admission; begin with one 3D job per device. |

The essential release condition is that **the collected experiment, resolved scene, emitted gprMax model, actual solver output, and training labels all describe the same three-dimensional physical setup**, with a numerical policy demonstrated to be adequate for the promised dataset envelope.

## Appendix A. Native 3D execution check performed during this research

The check used the bundled gprMax 3.1.7 CPU solver directly. It did not exercise the platform's collector, sampler, global planner, emitter, database or UI.

```text
#title: 3D research smoke check, not an accuracy benchmark
#num_threads: 1
#domain: 0.20 0.20 0.20
#dx_dy_dz: 0.004 0.004 0.004
#time_window: 6e-9
#pml_cells: 5 5 5 5 5 5
#waveform: ricker 1 0.8e9 pulse
#hertzian_dipole: x 0.10 0.10 0.10 pulse
#rx: 0.112 0.108 0.10
```

Run from the bundled `gprMax` directory with the project's Python environment and `python -m gprMax <input-file>`. No soil or buried target was included. This is deliberately a small solver-capability fixture, not a recommendation for production domain/PML settings.

| Observed result | Value |
| --- | --- |
| Native mode | 3D |
| Grid | 50 × 50 × 50 cells |
| Native time step | 7.703332806185882e-12 s |
| Iterations | 780 |
| Source | Hertzian, x polarized, at (0.10, 0.10, 0.10) m |
| Boundary build | Six PML faces |
| Output | Ex, Ey, Ez, Hx, Hy, Hz; all arrays length 780 and all values finite |
| CPU solve | Completed successfully |
| Native reported memory | Approximately 102 MB used for this small run |

The first attempt was blocked by sandboxed hardware detection; the same solver was subsequently run with the required permission. No host-detection or solver code was changed. The preliminary shorter window was extended after the native solver identified pulse truncation. GPU execution, native heterogeneous-soil 3D execution, analytical error, and production-scale performance were not measured in this research.

## Appendix B. Research scope and source interpretation

This report combines direct inspection of the active agent/pipeline/schema/validation/emission/execution/storage/UI code, inspection of the bundled native solver, official gprMax documentation, primary author/institution-hosted research, and the small CPU execution check above. Current documentation was used for context; where behavior matters for this repository, the bundled implementation is the compatibility reference.

The lateral-sizing source was read as the author-hosted manuscript linked in section 6.4. The realistic-modeling paper's institutional abstract/publication record supports its stated modeling scope; this report does not claim to reproduce its experiments. Recommendations about canonical geometry, dataset contracts, migrations, placement restrictions, agent behavior and release sequencing are conclusions from the repository audit, not claims that an external paper prescribes this architecture.

This research adds documentation only. It does not enable 3D, modify the physics implementation, or certify the existing dataset generator. The earlier source-kind changes in the working tree remain separate from this report.
