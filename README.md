# GPR Synthetic Dataset Pipeline

This project turns a conversation about soil layers, buried objects, and radar settings into a labeled dataset of ground-penetrating radar (GPR) simulations. It generates gprMax input files, runs the electromagnetic forward model on request, and stores the resulting signals alongside the parameters that produced them. The intended use is training and evaluating models that infer soil texture, moisture, layer structure, and buried-object properties from GPR measurements.

The central design choice is **one computational grid, domain, waveform, and transmitter/receiver arrangement per generated dataset**. Soil properties and object geometry vary inside that shared measurement setup. This makes signals directly comparable without changing their time axis or computational boundaries for each sample.

The active implementation uses one conversational collection agent, a supporting knowledge agent, and a deterministic Python pipeline. The README describes the implemented behavior; differences between the design notes and the current code are called out below.

## What the application can do

- Collect dataset settings and ranges for multiple soil layers through chat, explain parameters, and revise earlier choices without restarting collection.
- Generate heterogeneous Peplinski soil realizations with optional PEC cylinders and rectangular boxes, including both fixed and varying objects.
- Derive material properties, a shared spatial grid, the domain, antenna coordinates, and the simulation duration; validate them before emission.
- Show a live subsurface preview, switch between range overviews and individual samples, inspect generated input files, and download dataset artifacts.
- Run gprMax on CPU or NVIDIA CUDA, stream progress and per-file failures, and display receiver signals from the resulting HDF5 files.
- Import ZIP archives of existing `.in` files after command-syntax checks.
- Recommend reusing a sufficiently similar completed dataset, with a choice to adopt it or simulate anyway.
- Keep separate chats and datasets per user ID and restore saved conversations after reconnecting or restarting the backend.

Dataset generation ends with input files and labels. **Forward simulation is a separate action**, started with **Run forward model** or the simulation CLI. The repository provides dataset infrastructure, not an end-to-end trained soil-inversion model.

## How the agent works

`SingleAgentSession` owns a conversation thread and six structured sections: `dataset_config`, `layers`, `target_ranges`, `waveform`, `antenna`, and `advanced_params`. The collection model is currently configured as `gpt-4.1-mini` in `backend/agentflow_single_agent.py`.

At each collection stage, the orchestrator supplies focused instructions and the section's JSON schema. The agent asks for missing information, uses defaults when the user accepts them, and saves a complete section through `save_section`. That tool validates the payload with Pydantic before storing it. A separate completeness check prevents a schema-valid but incomplete section from advancing the workflow. Optional target and advanced sections are skipped by explicitly saving an empty section after the user declines them.

The same agent retains context throughout collection and remediation. It can read an earlier section with `get_section`, change agreed values, and replace that section in full. The orchestrator compares saved values with the inputs used for sampling. Changes to layers, targets, or dataset configuration trigger a new draw; changes to waveform, antenna, or advanced settings rerun the dependent derivation and validation steps without requiring a fresh soil draw.

For knowledge questions, the collection agent delegates to `knowledge-agent`. Its `rag_search` tool retrieves research and documentation passages from Qdrant using BGE-M3 semantic and keyword retrieval, then reranks the candidates. The knowledge agent is instructed to cite retrieved evidence; if retrieval returns no relevant results, it can answer from general domain knowledge and identify that fallback. Knowledge answers help the user choose parameters; they do not determine whether a physics gate passes.

**The LLM collects and explains; ordinary code samples, derives, validates, emits, and simulates.** The CLI expresses the workflow as LangGraph nodes, and the WebSocket API drives the same deterministic nodes through its session state machine. Neither lets the model invent a grid or repair a solver failure by rewriting physics. Sampling and validation failures that require a user decision return to the same conversation with concrete errors; the corrected inputs are then checked again.

After generation, discussion remains available. The agent is instructed to obtain confirmation before saving an edit because regeneration replaces the dataset's simulation rows and clears previous solver outputs. Incomplete edits leave the existing dataset in place. A new simulation belongs in a separate chat. API conversations use PostgreSQL-backed LangGraph checkpoints; the standalone collection CLI uses in-memory conversation checkpoints.

## Pipeline stages and why they are ordered this way

The order follows information dependencies: validate a choice as soon as its inputs exist, and derive expensive numerical settings only after the physical configuration is known.

| Stage | What happens | Why it belongs here |
| --- | --- | --- |
| 1. Dataset configuration | Collect sample count, naming, grid-resolution policy, PML thickness, buffer size, material-bin count, and whether frequency means peak or band center. | These settings control every later stage, but do not require a soil calculation. The server fixes the output location, 2D mode, and deployment-controlled threading. |
| 2. Layer ranges | Collect layers from the surface downward: thickness, sand and clay percentages, moisture band, and bulk/particle density ranges. | Soil defines the material model and its feasibility constraints. Texture and density must exist before concrete realizations can be drawn. |
| 3. Target ranges | Collect zero or more cylinders and boxes, with size, center depth, and horizontal offset ranges. | Objects affect the eventual grid and domain, so they must be drawn before global sizing. Coordinates are relative to the surface and domain center because absolute bounds do not yet exist. |
| 4. Layer and target sampling | Draw concrete realizations, reject infeasible soil draws, and save the accepted samples. | Actual sampled compositions determine material properties. Drawing first avoids deriving a separate grid for every sample or sizing from a composition that was never used. Grid-independent checks can already run. |
| 5. Waveform | Collect pulse kind, amplitude, frequency, name, and optional source timing. | The soil model establishes the frequency-validity policy; the waveform supplies the spectral band needed for wavelengths and resolution. Band edges depend on the waveform, not soil permittivity. |
| 6. Antenna and cross-stage gate | Collect source type, polarization, Tx/Rx separation, and optional settings; validate waveform bandwidth and antenna configuration. | These checks need both excitation and source settings. A bad shared frequency band invalidates the entire dataset, so it is rejected before numerical sizing. |
| 7. Advanced options | Collect optional surface roughness and field snapshots. Recheck whether earlier edits made the samples stale. | Optional detail follows the required acquisition configuration. This is the last collection stage before the deterministic derivation chain. |
| 8. Per-sample material derivation | Evaluate gprMax-native Peplinski materials for every sampled layer; aggregate dielectric and target-geometry extremes. | The single grid needs a summary of the most demanding realizations across the dataset. |
| 9. Global derivation | Resolve frequency conventions, wavelengths, cell size, clearances, depth, domain, antenna coordinates, time step, and time window once. | Each quantity depends on earlier results. In particular, cell size must be final before converting boundary clearances from cells to meters. |
| 10. Global validation | Check grid/domain fundamentals, then placement and layers, then computational feasibility. | Placement requires meaningful domain dimensions, and cost estimates require valid cell counts. Cascading gates avoid reporting secondary failures caused by one upstream problem. |
| 11. Target placement | Validate varying objects against the frozen grid; shrink/reposition invalid draws or drop the sample. | Absolute placement becomes possible only after the domain is known. Keeping the grid fixed prevents object placement from changing the measurement setup. Fixed targets were checked at the global gate. |
| 12. Emission and storage | Write one `.in` file per surviving sample, record emitted geometry labels, and store generated samples in PostgreSQL through the API. | The writer consumes resolved physics and shared settings. It performs geometry transcription and cell snapping, not a new material or grid solve. |
| 13. Forward simulation | On request, run the emitted decks, store output paths, and extract receiver signals. | Generation can be reviewed before spending solver time. Solver errors remain deterministic execution failures and are reported per file. |

## Physics choices and derived quantities

### Soil composition and heterogeneity

gprMax uses finite-difference time-domain (FDTD) simulation: it advances electric and magnetic fields through a spatial grid over successive time steps. Changes in soil electrical properties and buried conductors alter propagation and produce reflections. Relative permittivity affects propagation speed and wavelength; conductivity contributes to loss.

This pipeline uses **gprMax's native Peplinski soil model** throughout. It passes sand fraction, clay fraction, bulk density, particle density, and a moisture band to `#soil_peplinski`. Using the solver's own material construction avoids disagreement between an independent dielectric approximation used for sizing and the materials actually simulated.

The initial sampler draws thickness, sand, clay, and both densities uniformly from their ranges. It derives silt as the percentage remaining after sand and clay. It derives porosity from the two densities to check whether the specified water content can physically fit in the pore space. Infeasible draws are rejected rather than silently changing the user's moisture band. Each layer gets up to 200 attempts before sampling returns a remediation error.

**Current moisture behavior:** each layer's full `theta_v_min`–`theta_v_max` band is passed unchanged into every sample. Although some prompts and design comments describe drawing a different sub-band per sample, the active sampler does not do that. Moisture still varies spatially within a realization: `#fractal_box` distributes the Peplinski material bins through the layer. The default is 50 bins, fractal dimension 1.5, and equal directional weights. These are discretized heterogeneous materials, not one scalar water content for the whole layer.

Silt is stored as a derived label; porosity is calculated for validation. Neither is an additional Peplinski command input. Temperature, salinity, organic content, and porewater conductivity are not independently modeled by the active extraction pipeline. Density inputs use **g/cm³**, texture uses **percent**, moisture uses **volume fraction**, and geometry, frequency, and time use meters, hertz, and seconds.

### Frequency convention and bandwidth

A Ricker pulse's peak frequency and band-center frequency are different. `center_freq_is_peak` records which meaning the user supplied. The deterministic code converts a band center to the peak expected by gprMax and derives the lower and upper Wang band edges. The writer uses that resolved peak directly.

The Peplinski gate checks that the **entire derived −6 dB band** lies within the project's 0.3–1.3 GHz validity window. Checking only the entered frequency would miss pulses whose upper band edge exceeds the model's range. For example, an **825 MHz band center** becomes approximately **779 MHz peak**, with edges near **375 and 1275 MHz**. Treating 825 MHz as the peak would put the upper edge near 1350 MHz and fail the gate.

Grid sizing uses a separate, higher frequency to account for significant spectral content beyond the −6 dB band. The derivation defaults to three times the entered frequency through `high_freq_factor`; the independent resolution validator also applies a waveform-specific significant-frequency factor. The validity band and the grid-resolution frequency serve different purposes and must not be interchanged. The current band calculation is Ricker-based even when another waveform name is selected; this limitation is listed below.

### Material properties before a grid exists

For each sampled layer, `peplinski_derive.py` creates a `PeplinskiSoil` and asks `calculate_debye_properties` to build its material bins on a minimal temporary grid object. This operation needs a material list, not a finalized spatial grid or time step, so it avoids a circular dependency.

The code evaluates `Material.calculate_er(f).real` on the driest and wettest bins. That includes the Debye material's frequency response. Reading the raw stored `er` instead would use its infinite-frequency value, potentially underestimating permittivity and allowing cells that are too large. The active code evaluates these labels at the **resolved peak frequency**, recorded as `frequency_hz` in the derived manifest.

Each layer receives dry/wet relative-permittivity labels and the corresponding effective-conductivity labels from `Material.se`. Conductivity is recorded in S/m and remains part of gprMax's material behavior, but it does not set this pipeline's wavelength budget. The wettest actual bin is used for sizing because gprMax's bin construction places it half a bin above the requested moisture maximum; the remaining porosity-check limitation is documented below.

### Why the domain is fixed across samples

Changing the grid for each sample would change numerical dispersion, time-step spacing, signal length, and the distance to absorbing boundaries alongside the soil properties being studied. Changing antenna positions would also change the acquisition geometry. Those differences would complicate comparison and could introduce unintended cues into an ML dataset.

The pipeline therefore aggregates the most demanding properties before deriving anything global:

| Aggregate | How it influences the shared setup |
| --- | --- |
| Largest wet-bin permittivity across all drawn layers | Supplies the slowest medium and shortest wavelength for spatial resolution and travel-time budgeting. |
| Smallest dry-bin permittivity, also considering air | Supplies the longest wavelength. In the normal air-over-soil layout, air determines this limit. |
| Smallest drawn target feature | Can tighten the grid beyond the wavelength requirement so the object spans at least ten cells. |
| Largest drawn horizontal target extent | Can widen the domain to fit the object and its boundary clearance. |
| Deepest drawn target bottom | Can deepen the soil region enough to preserve bottom-boundary clearance. |
| Largest fixed-object footprint about the domain center | Widens the domain symmetrically to accommodate objects pinned to either side. |
| Sum of the maximum requested layer thicknesses | Reserves enough depth for the deepest permitted layer stack. |

The dielectric and varying-target extremes come from the **drawn dataset**, not an exhaustive search over every possible combination in the input ranges. The layer-stack bound uses the range maxima. Independent extremes can come from different samples; combining them is deliberately conservative for the realizations being emitted.

`global_derive.py` then resolves the shared quantities in this order:

1. **Wavelength limits and cell size.** Use the high significant frequency and largest soil permittivity for the shortest wavelength. Allocate ten cells per wavelength by default. Tighten further if the smallest target needs it. A cylinder's feature is its diameter; a box's feature is its smaller in-plane side. The single-cell thickness in z never counts as a target feature.
2. **Boundary distances.** After cell size is frozen, convert PML and buffer cells into physical distances. Perfectly matched layers (PML) absorb outgoing waves to reduce artificial reflections from the computational boundary. Domain padding uses the configured buffer, while source/object clearance requires an additional 15 cells beyond the PML. With defaults, padding is 20 cells and required clearance is 25 cells; these are distinct distances.
3. **Soil depth.** Fit the maximum layer stack, a bandwidth-based range-resolution floor, and the deepest target with any missing bottom clearance. The range-resolution floor is a sizing policy, not proof that all interfaces will be distinguishable in a simulated signal.
4. **Antenna height and domain width.** Default the source height to half the longest air wavelength. Start the horizontal span at one and a half longest wavelengths, then enlarge it for Tx/Rx separation and target footprints. An explicit source-height override is accepted by the current schema but must pass the same minimum-height and clearance checks.
5. **Domain height and coordinates.** Place soil below the ground reference and air above it, with room for the antenna and top clearance. Round domain dimensions upward to whole cells, then place Tx and Rx symmetrically around the horizontal midpoint at the same height.
6. **Time step.** Derive the 2D Courant–Friedrichs–Lewy (CFL) stability step from the final cell size. Smaller cells require shorter steps so field propagation remains numerically stable. The time step is an output of spatial resolution, never an input used to select soil or grid settings. gprMax sets its actual step when building the model.
7. **Time window.** Budget a round trip to the deepest reflector using the slowest soil speed, include source height conservatively at that speed, and add a pulse-duration margin. Source start/end times control excitation separately; they are not the simulation duration.

These results are written once to `global_derive.json` and reused by every emitted file. The common geometry is **x horizontal, y vertical, z one cell thick**. Despite its historical name, `depth_z_m` stores soil depth along y. The 2D PML command has zero thickness on both z faces; applying the in-plane PML there would consume the whole thin dimension.

The grid stays fixed after target redraws or dropped samples. A sample that later disappears may have set an extreme; retaining its more conservative grid avoids a circular resize-and-revalidate process. A regenerated dataset may receive a new global grid when its inputs change.

### Layer and target geometry in the emitted model

Layers begin at the surface and stack downward. The emitter snaps the ground and layer interfaces to cell boundaries, and extends the **deepest layer to the bottom of the domain**. This provides a continuous soil background through the lower padding instead of leaving an artificial air pocket beneath a shorter sampled stack. Consequently, the last layer's emitted thickness generally differs from its sampled thickness. Use the geometry labels in `emitted_files.json` when the target label must match the simulated layer extent; database `layers` currently retains the original sampled values.

Each object is specified by a signed horizontal offset from the domain center and the depth of its center below ground. A cylinder appears as a circular cross-section with its axis along z; a box is a rectangle spanning the same thin z cell. Both are **perfect electric conductors (PEC)**. Their geometry affects grid sizing, but they do not contribute a sampled dielectric permittivity. Targets are emitted after the fractal soil, with dielectric smoothing disabled, so the conductor replaces the soil at its boundary.

Objects with every range endpoint equal are fixed. Their placement is validated once, and a failure returns to the user rather than moving their chosen coordinates. Other objects are dynamic. Invalid placements can be redrawn up to 20 times, shrinking sizes while preserving the user's minimum size and the ten-cell resolution floor. If the smallest allowed object cannot fit, the sample is dropped immediately. Failure of any object drops the **whole sample**, preserving the requested object set in every surviving file. Drops reduce the delivered count; there is no backfill.

Dynamic repositioning currently draws a new center from the whole physically valid domain envelope, which can go outside the originally requested offset/depth ranges. Redraws and rejection also mean final accepted distributions are not necessarily uniform. The updated sampled manifest is the record of the geometry actually passed to emission.

## Validation coverage

Validation is split by when the required information becomes available: schema checks are Tier 0, collection checks Tier 1, concrete soil draws Tier 2, global checks Tier 3, and emission checks Tier 4. Errors stop the relevant gate or reject a draw; warnings are retained without automatically changing inputs.

| Location | Checks currently performed | Result |
| --- | --- | --- |
| Section schemas and completeness | Positive sample count and grid-policy values; nonnegative PML/buffer counts; ordered ranges; feasible minimum sand/clay sum; moisture envelope fitting the loosest density-derived porosity; matching layer count; required section content. | Invalid saves are rejected; incomplete sections do not advance. |
| Target schemas | Ordered size/position ranges, positive radius/box dimensions, supported shapes, and PEC-only material. | Reject invalid target configurations before sampling. |
| Antenna schema and collection gate | Supported source type; required positive resistance below 376.73 ohms for voltage/transmission-line configurations; valid axis; finite/ranged resistance; source start/end ordering and missing end time. | Reject invalid configurations. |
| Each soil draw | Texture closure; real moisture band; bulk density below particle density; physical porosity; moisture maximum no greater than that draw's porosity; Peplinski moisture ceiling of 0.30. | Reject and redraw; exhausted attempts return to remediation. |
| Calibration warnings | Sand outside 15–50%, silt outside 35–65%, or clay outside 5–20%. | Warn that composition lies outside the project's Peplinski calibration envelope; texture excursions alone do not reject a draw. |
| Waveform gate | Recognized waveform name, positive frequency, finite amplitude, and derived band edges inside 0.3–1.3 GHz. | Stop before material/grid derivation; the band gate is also reasserted during global derivation. |
| Global phase 1: grid/domain | Wavelength resolution; Debye relaxation time versus computed step; integer-cell domain alignment; opposing PML faces leaving an interior region. | Stop before placement checks if fundamentals fail. |
| Global phase 2: placement/layers | Static Tx/Rx clearance; source height and top clearance; fixed-target placement; maximum layer stack fitting the shared depth; travel-time window; receiver-array steps at least one cell when positive. | Stop on errors. Layer thickness below three cells produces a warning. |
| Global phase 3: feasibility | Estimated memory against a default 32 GiB budget; positive grid/time inputs and estimated iteration count; very fine target-driven grids. | Memory-budget excess is an error. More than 50,000 estimated iterations, or a grid over three times finer than the wavelength budget, produces an advisory. These are estimates, not device-memory measurements. |
| Dynamic target placement | Entire bounding box inside the domain, 15-cell clearance beyond PML, at least ten cells across, and full burial below the ground reference. | Resolution/clearance warnings from the shared helper become placement failures; redraw or drop the whole sample. |
| Emission | Required domain/grid/time commands are constructed; layer identifiers are sanitized and disambiguated; inverted, zero-height, or below-floor layer boxes are rejected; roughness peaks must stay below the source and top clearance. | Skip failed samples and record emission errors. |
| Uploaded decks | UTF-8 text, gprMax command names and syntax, mandatory commands, duplicate single-use commands, duplicate archive basenames, and limits of 2,000 accepted decks and 5 MiB per deck. Embedded Python and include-file directives are rejected. | Accept valid `.in` members and report rejected members individually. This is syntax validation, not the generated-dataset physics pipeline. |

The current preflight is **not exhaustive**. These implementation boundaries matter when interpreting a passing result:

- Soil range schemas do not uniformly enforce positive thickness/density, finite values, or every lower/upper physical bound. Per-draw checks add constraints but do not constitute complete input-domain validation.
- The porosity check caps the requested moisture maximum, without reserving the extra half-bin margin used by gprMax's wettest material. Material sizing uses the actual wettest bin, but a near-saturation configuration is not fully protected by the porosity gate.
- Global thin-layer checks use maximum requested thicknesses, not every realized or minimum thickness. Emission prevents invalid layer boxes but does not enforce three cells for every drawn layer.
- CFL iteration and Debye helpers currently calculate a three-axis step, while global derivation and the emitted solver geometry use 2D. Their reported step/iteration diagnostics therefore differ from the solver, and the relaxation-time check is not an exact check of the emitted 2D step. Actual output timing comes from HDF5 metadata.
- Standalone validators for material names, mandatory commands, and snapshot times exist in `validation_tools_new.py`, but the generated-file writer does not call those helpers. It builds mandatory commands and handles identifiers itself; snapshot-window validation is not currently wired into that emission path.
- The source-timing gate accepts an end time without a start time, but the writer emits explicit timing only when both are supplied. Specify both to preserve an intended source cutoff.
- Target checks use bounding boxes and the flat ground reference. They do not check object-object overlap, confinement to a particular layer, or burial relative to every rough-surface trough. Optional snapshot bounds/resolution and source timing against the total simulation window are also not fully checked.

## Artifacts, labels, and reproducibility

API-generated datasets live under `dataset/<sanitized-user>/<basename>__<session-hash>/`. CLI generation uses `dataset/<basename>/`. The pipeline writes these artifacts, with placement and solver files appearing only when those stages run:

| Artifact | Contents |
| --- | --- |
| `sampled_layers.json` | Accepted soil values, derived silt labels, moisture bands, current targets, warnings, and surviving sample IDs; placement updates this file. |
| `derived_layers.json` | Per-layer dry/wet permittivity and conductivity labels, their evaluation frequency and bin count, and the global material/target extremes. It can still include samples subsequently dropped. |
| `global_derive.json` | Shared spectral quantities, wavelength limits, cell size, domain, nominal ground and antenna positions, time-step estimate, and time window. |
| `dropped_targets.json` | Sample IDs and reasons for target-placement drops, when target placement runs. |
| `emitted_files.json` | Successfully written files, delivered count, common grid summary, snapped layer geometry labels, and emission errors. |
| `in_files/` | One gprMax `.in` deck per successfully emitted sample. |
| `out_files/` | Solver `.out` HDF5 files and associated generated outputs after a run. |

Join records by **sample ID**, not list position: dropped samples can leave gaps, and concurrent simulations finish out of order. Check emitted counts and drop records rather than assuming the requested count was delivered. Preserve both sampled labels and emitted geometry labels when constructing training targets.

The pipeline uses seed 42 for initial parameter sampling, seed 1234 for target placement, and deterministic per-sample/per-layer seeds for fractal soil. Surface roughness has its own optional seed. Reproducibility also depends on retaining the configuration, manifests, and solver environment; unchanged seeds alone do not identify a dataset after its inputs change.

PostgreSQL stores collected ranges in `ExtractionSession`, one generated-sample record in `Simulation`, and resumable application state in `ChatSession`. After simulation, available `Ex`, `Ey`, `Ez`, `Hx`, `Hy`, and `Hz` arrays from the **first receiver** are attached to simulation records. The output viewer reads the actual time step and iteration count from each HDF5 file. Imported decks can be run and viewed, but the upload path does not create the generated dataset's structured simulation-label rows.

## Simulation execution and reuse

The batch runner defaults to CPU/OpenMP. NVIDIA CUDA is available through PyCUDA on a suitably configured host; there is no Metal backend. Parallelism runs independent input files in separate processes. Each concurrent model holds its own field arrays, so worker count affects memory use as well as throughput. Individual failures are reported, gprMax state is cleared after failed models, and the batch continues where possible.

Before a generated dataset is simulated, the application can search for a completed compatible dataset in Qdrant's separate `sim_sessions` collection. This search uses normalized numerical configuration features, categorical filters, and deterministic weighted range-overlap scoring. It does not use language-model embeddings or ask the agent to judge physical equivalence. The default recommendation threshold is 0.95; similarity is a reuse heuristic, not proof that two configurations produce identical signals.

Adoption requires the user's choice and a verified source dataset with completed outputs. It copies source files and labels into the current session and records provenance. The user's requested ranges remain available for later regeneration, so adopted artifacts describe the source realization. A Qdrant failure disables the recommendation for that attempt and allows normal simulation to proceed.

## Current supported scope

The generated model is a **2D, z-polarized, static single-Tx/single-Rx A-scan** with Peplinski soil and optional PEC cylinders/boxes. Several fields exposed by schemas or prompts are broader than what the emitter currently implements:

| Setting | Current behavior |
| --- | --- |
| 3D, spheres, dielectric targets, sampled target material | Not supported by the active generated-dataset path. |
| Source type | Hertzian dipole, voltage source, and transmission line are emitted as selected. Case and whitespace are normalized (e.g. `Hertzian Dipole` becomes `hertzian_dipole`). Unsupported, empty, or null types are rejected; omitting the field uses the Hertzian default. Transmission lines require CPU solving: GPU batches with an explicit `#transmission_line` command in a pending deck are rejected before any models or workers start. |
| Polarization and receiver height | Emission forces z polarization; derivation places Rx at Tx height even if `rx_same_height` is false. Stored requested metadata can therefore differ from the deck. |
| Receiver array | Can be collected and step-checked, but the emitter writes one `#rx`; it does not emit the array. |
| Waveform families | Several names are accepted, but band conversion/gating always uses Ricker coefficients. Other families do not yet have a matching spectral derivation. |
| Surface water | `add_water` and `water_depth_m` can be collected but are not emitted. Surface roughness itself is emitted. |
| Moving-antenna B-scans | The generated files contain no antenna stepping. A dataset of different soil samples is not a spatial B-scan. |

The fixed grid controls numerical consistency; it does not establish agreement with a laboratory antenna or real soil measurements. Antenna calibration, noise, attenuation-based detectability, and sim-to-real validation require additional work. The web application currently uses typed user IDs for organization without authentication, assumes one backend worker, and does not automatically recover an interrupted solver run after a server crash.

## Run locally

Use Python 3.12, `uv`, Docker Compose, and a C/C++ toolchain with OpenMP support for the bundled gprMax extensions. The frontend loads React and Babel from CDNs and needs no Node build step.

From the repository root, install the Python dependencies and start PostgreSQL and Qdrant:

```bash
uv sync
docker compose up -d
```

Create a root `.env` containing `OPENAI_API_KEY`. Optional connection overrides are `DATABASE_URL` and `QDRANT_URL`; the defaults match `docker-compose.yml`. Export connection overrides in the shell used for migrations as well. `HF_TOKEN` can be supplied for Hugging Face downloads used by retrieval.

```bash
uv run alembic -c db/alembic.ini upgrade head
```

The root dependency install does not compile the bundled solver. Build its extensions in place with the same environment; compiler setup is described in `gprMax/README.rst`:

```bash
cd gprMax
uv run python setup.py build_ext --inplace
cd ..
```

Start the backend and, in another terminal, serve the frontend:

```bash
uv run uvicorn backend.api:app --reload --host 127.0.0.1 --port 8000
```

```bash
python3 -m http.server 8080 --directory frontend
```

Open [the application](http://127.0.0.1:8080/html-design.html), enter a user ID, create a chat, and describe the dataset. Keep one backend worker. PostgreSQL stores application records and conversation checkpoints; Compose volumes preserve database contents across container recreation.

For document-backed knowledge answers, place source documents in `knowledge_source/` and index them:

```bash
uv run python backend/rag.py training
```

Here, `training` means document ingestion/indexing, not training the collection model. Retrieval uses the `gpr_research` collection and may download embedding/reranking models on first use. Missing indexed documents yield no retrieved evidence; retrieval startup failures are logged without blocking the deterministic pipeline.

Standalone collection and simulation are also available:

```bash
uv run python backend/agentflow_single_agent.py
uv run python backend/simulate.py --input-dir dataset/soil_sample/in_files
```

The first command generates decks through the CLI conversation. The second runs an existing input directory; replace the path with the dataset directory you generated. API-managed label persistence and chat features belong to the web workflow. Standalone simulation and indexing commands read their settings from the process environment, so export the relevant overrides before running them.

| Environment variable | Purpose / default |
| --- | --- |
| `OPENAI_API_KEY` | Required for the conversational agent. |
| `DATABASE_URL` | PostgreSQL connection; defaults to the local Compose database. |
| `QDRANT_URL` | Retrieval and reuse service; defaults to `http://localhost:6333`. |
| `GPR_GPU` | Enable CUDA; off by default. |
| `GPR_GPU_IDS` | Comma-separated CUDA device IDs; specifying IDs enables GPU mode. |
| `GPR_SIM_WORKERS` | Concurrent models; defaults to one on CPU or two per GPU. |
| `OMP_NUM_THREADS` | Host OpenMP thread budget, divided across concurrent workers. |
| `SIM_REUSE_ENABLED` | Enable reuse recommendations; on by default. |
| `SIM_REUSE_THRESHOLD` | Minimum recommendation score; defaults to `0.95`. |
| `SIM_REUSE_TOPK` | Candidate count for reuse search; defaults to `10`. |

On a CUDA host, install PyCUDA separately with `uv pip install pycuda` after installing the CUDA toolkit and compiler headers. It is intentionally excluded from the project's dependencies so CPU-only installations do not require CUDA. Check device availability with `uv run python backend/simulate.py --check-gpu`, then enable GPU execution in the backend's environment. CLI simulation flags can override environment settings.

## Source map and checks

| Location | Responsibility |
| --- | --- |
| `backend/agentflow_single_agent.py`, `backend/single_agent_prompts.py` | Collection agent, section tools, prompts, CLI graph, and deterministic-node entry points. |
| `backend/api.py`, `backend/checkpointer.py` | WebSocket/HTTP workflow, persistence, regeneration, uploads, downloads, and solver actions. |
| `backend/schema.py`, `backend/validation_tools_new.py` | Structured inputs, derived records, and shared validation rules. |
| `backend/dataset_sampling/` | Sampling, Peplinski derivation, global sizing/validation, target placement, and emission. |
| `backend/rag.py`, `backend/sim_similarity.py` | Document retrieval and independent numerical dataset-reuse search. |
| `backend/simulate.py`, `backend/signal_extraction.py`, `backend/deck_validation.py` | Solver execution, HDF5 signal extraction, and upload syntax checks. |
| `backend/viz_projection.py`, `frontend/` | Live scene projection, chat, dataset inspection, and signal viewing. |
| `db/`, `gprMax/` | Database models/migrations and the bundled electromagnetic solver. |
| `AGENT.md` | Physics design invariants and rationale; some intended behaviors differ from the implementation documented here. |

Run the existing backend and geometry regression suites from the repository root:

```bash
uv run pytest backend/tests backend/dataset_sampling/tests -v
```

The tests cover section storage, workflow routing and regeneration, persistence, uploads, visualization, dataset scoping, target geometry/PML behavior, output handling, and reuse. Many integration tests stub external services or the solver; passing them does not replace a real gprMax run and physical validation of a dataset.
