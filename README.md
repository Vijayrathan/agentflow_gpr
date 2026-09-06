# GPR Synthetic Dataset Pipeline

This project turns a conversation about soil layers, buried objects, and radar settings into a labeled dataset of ground-penetrating radar (GPR) simulations. It generates gprMax input files, runs the electromagnetic forward model on request, and stores the resulting signals alongside the parameters that produced them. The intended use is training and evaluating models that infer soil texture, moisture, layer structure, and buried-object properties from GPR measurements.

The central design choice is **one computational grid, domain, waveform, and transmitter/receiver arrangement per generated dataset**. Soil properties and object geometry vary inside that shared measurement setup. This makes signals directly comparable without changing their time axis or computational boundaries for each sample.

**3D is implemented as an experimental capability, gated by `GPR_ENABLE_EXPERIMENTAL_3D=1`.** General release remains blocked on reviewed scientific tolerances and population coverage. See [implementation and qualification status](docs/3D_IMPLEMENTATION_STATUS.md). Existing 2D datasets remain readable.

The active implementation uses one conversational collection agent, a supporting knowledge agent, and a deterministic Python pipeline. The README describes the implemented behavior; differences between the design notes and the current code are called out below.

## What the application can do

- Collect dataset settings and ranges for multiple soil layers through chat, explain parameters, and revise earlier choices without restarting collection.
- Generate heterogeneous Peplinski soil realizations with optional PEC cylinders and rectangular boxes, including both fixed and varying objects.
- Derive material properties, a shared spatial grid, the domain, antenna coordinates, and the simulation duration; validate them before emission.
- Show a live subsurface preview, switch between range overviews and individual samples, inspect generated input files, and download dataset artifacts.
- Run gprMax on CPU or NVIDIA CUDA, stream progress and per-file failures, and display receiver signals from the resulting HDF5 files.
- Import ZIP archives of existing `.in` files after command-syntax checks.
- Find similar configurations and offer substitution only for scientifically qualified results with exact experiment and population compatibility.
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
| 1. Dataset configuration | Collect sample count, naming, grid-resolution policy, PML thickness, buffer size, material-bin count, and whether frequency means peak or band center. | These settings control every later stage, but do not require a soil calculation. The server fixes output location, contract version 2, and deployment-controlled threading. It preserves the selected mode; experimental 3D requires the deployment flag. |
| 2. Layer ranges | Collect layers from the surface downward: thickness, sand and clay percentages, moisture band, and bulk/particle density ranges. | Soil defines the material model and its feasibility constraints. Texture and density must exist before concrete realizations can be drawn. |
| 3. Target ranges | Collect zero or more cylinders and boxes, with size, center depth, and horizontal offset ranges; 3D also requires crossline offsets, finite cylinder length/axis or box crossline size. | Objects affect the eventual grid and domain, so they must be drawn before global sizing. Coordinates are relative to the surface and domain center because absolute bounds do not yet exist. |
| 4. Layer and target sampling | Draw concrete realizations, reject infeasible soil draws, and save the accepted samples. | Actual sampled compositions determine material properties. Drawing first avoids deriving a separate grid for every sample or sizing from a composition that was never used. Grid-independent checks can already run. |
| 5. Waveform | Collect pulse kind, amplitude, frequency, name, and optional source timing. | The soil model establishes the frequency-validity policy; the waveform supplies the spectral band needed for wavelengths and resolution. Band edges depend on the waveform, not soil permittivity. |
| 6. Antenna and cross-stage gate | Collect source type, polarization, Tx/Rx separation, and optional settings; validate waveform bandwidth and antenna configuration. | These checks need both excitation and source settings. A bad shared frequency band invalidates the entire dataset, so it is rejected before numerical sizing. |
| 7. Advanced options | Collect optional surface roughness and field snapshots. Recheck whether earlier edits made the samples stale. | Optional detail follows the required acquisition configuration. This is the last collection stage before the deterministic derivation chain. |
| 8. Per-sample material derivation | Evaluate gprMax-native Peplinski materials for every sampled layer; aggregate dielectric and target-geometry extremes. | The single grid needs a summary of the most demanding realizations across the dataset. |
| 9. Global derivation | Resolve frequency conventions, wavelengths, cell size, clearances, depth, domain, antenna coordinates, time step, and time window once. | Each quantity depends on earlier results. In particular, cell size must be final before converting boundary clearances from cells to meters. |
| 10. Global validation | Check grid/domain fundamentals, then placement and layers, then computational feasibility. | Placement requires meaningful domain dimensions, and cost estimates require valid cell counts. Cascading gates avoid reporting secondary failures caused by one upstream problem. |
| 11. Target placement | Validate varying objects against the frozen grid; shrink/reposition invalid draws or drop the sample. | Absolute placement becomes possible only after the domain is known. Keeping the grid fixed prevents object placement from changing the measurement setup. Fixed targets were checked at the global gate. |
| 12. Emission and storage | Write one `.in` file per surviving sample, record emitted geometry labels, and store generated samples in PostgreSQL through the API. | The writer consumes resolved physics and shared settings. Geometry is snapped and validated in a resolved-scene stage before pure serialization; native preflight then checks the built model before field updates. |
| 13. Forward simulation | On request, run the emitted decks, store output paths, and extract receiver signals. | Generation can be reviewed before spending solver time. Solver errors remain deterministic execution failures and are reported per file. |

## Physical and numerical contract

New collection sessions use contract version 2. Version 1 remains the historical planar implementation for reading/reproducing old artifacts; its assumptions must not be applied to a new 3D scene. A valid configuration, successful native field execution, and scientific qualification are three distinct states.

### Material and excitation

The deterministic order is sampling → native material response at the chosen excitation → one spectral/grid budget → domain/acquisition → native CFL step → common recording window. No numerical error is sent to an LLM to repair the scene.

Only gprMax-native Peplinski supplies dielectric properties. Each layer's entire moisture band is preserved across samples; it is neither a scalar nor a randomly narrowed sub-band. Densities, texture and thickness are drawn first, with up to 200 attempts. Every actual native material bin must be finite and passive, with positive ordered densities and moisture within porosity and the 0.30 calibration cap. At least two bins are required. The wettest instantiated bin is half a bin above the supplied maximum; the validator includes this shift. Texture excursions outside the chosen calibration envelope remain recorded warnings, and never imply empirical qualification.

`Material.calculate_er(f)` supplies the complex response. Dry/wet permittivity labels are evaluated at the resolved peak frequency; `Material.se` is the native conductivity coefficient in S/m, not a claim that it equals total frequency-dependent loss. Resolution uses every native bin over the declared design band, including conductivity and relaxation in a conservative bound on the phase index. Sampled phase wavelengths and attenuation lengths are also recorded. The derived coefficient table is hashed and compared with the table built by the native solver.

Version 2 supports Ricker excitation only. The entered frequency can mean peak or Wang band center; a band center is divided by 1.059095 before emission. The Wang useful-band edges (0.481623 and 1.636567 times the peak) must lie in 0.3–1.3 GHz. Resolution separately uses `high_freq_factor × actual peak` (default 3, minimum 2.5). Those spectral tails use the native model's extrapolation; useful-band validity does not certify the entire tail empirically. An explicit start requires an end; an end alone starts at zero. The interval must include the native pulse duration, and the recording window covers both excitation and returns.

### Geometry and the shared grid

Coordinates are always `(x, y, z)`, with **y vertical** and x/z horizontal. In 2D TMz, z is one cell and its two PML thicknesses are zero. In 3D, all three dimensions contain more than one cell, with six active PML faces. `soil_depth_m` is the vertical soil sizing depth; `domain_z_m` is the physical crossline extent. Historical `depth_z_m` remains a vertical-depth compatibility alias.

One cubic grid, domain, ground reference, acquisition and time axis serve every accepted sample. The shortest spectral wavelength, smallest intrinsic target feature and thinnest drawn layer constrain spacing. Targets need at least ten cells per intrinsic feature; finite layers need at least three after quantization. The grid reserves a cell for endpoint quantization. Source/receiver field locations account for Yee staggering, and sources/targets/receivers clear every applicable face by PML + 15 cells. Configured larger buffers can enlarge the domain further.

The lateral floor is 1.5 longest wavelengths, including PML. It is enlarged independently in x and z for target and acquisition extents. The source-height floor is half the longest air wavelength; an explicit override cannot violate it. Source and receiver positions, including crossline separation and a distinct receiver height when requested, are resolved once. Requested and effective positions remain separate.

`numerics.py` uses the bundled solver's rounding convention, with two active CFL terms for TMz and three for 3D. The actual step must be below every material relaxation time; failure requires a revised common grid, never a per-sample time adjustment. The recording estimate bounds Tx → interior ROI corners → receiver-component paths, includes source delay/removal and explicit return margins, then resolves one integer iteration count. The final time is `(iterations - 1) × dt`. This conservative timing policy requires scene-specific scientific qualification.

Layers cover full x/z volumes and continue through lateral and bottom boundaries. The final layer is a terminal half-space; its sampled thickness is not an independently realized bottom interface. Both nominal and effective layer geometry are stored. Native fractal generation produces a volume, not repeated copies of a 2D slice. Fractal weights are not advertised as physical correlation lengths.

PEC boxes have x width, y height and finite z crossline size. Cylinders have radius, finite axial length and explicit x/y/z axis. Positions use signed offsets from the domain center in x/z and center depth below nominal ground. Arbitrary rotations, spheres and other target materials are unsupported. Quantization is either declared nearest-cell or exact; exact mode rejects required snapping. The resolved scene records endpoint changes and conservative bounds; native preflight records occupied voxels separately.

Every fixed range field stays fixed, including fields of otherwise varying targets. Placement can shrink sizes only within the original bounds and the ten-cell floor, and reposition only inside the intersection of requested and physically valid ranges. The overlap policy conservatively forbids touching or overlapping target bounds. After at most 20 attempts, failure drops the whole sample without backfill. Static-target failure stops the shared gate. Original draws, rejection reasons, attempts and accepted marginal distributions are retained. The common grid stays fixed even when a dropped sample supplied an extreme.

### Supported scope and release gate

| Option | Version 2 behavior |
| --- | --- |
| 2D TMz | One-cell z, z-polarized single-pair A-scan. |
| 3D | Experimental layered Peplinski volume with finite PEC boxes/cylinders and x/y/z source polarization. |
| Source kinds | Hertzian dipole, voltage source, transmission line, honored exactly. Resistive sources require the existing resistance bounds. Transmission lines are CPU-only. |
| Receiver layout | One static receiver; signed x/z separation and explicit receiver height are honored. Arrays and scans are rejected. |
| Roughness | 2D roughness has resolved height limits, seeded native construction, trough burial/interface and peak acquisition checks. 3D roughness is explicitly rejected. |
| Unsupported options | Other waveform families, surface water, vegetation, arbitrary rotations, new target materials and realistic antenna assemblies are rejected or absent from collection. |
| Inspection | Three orthogonal 3D slices with explicit positions, finite-object intersections, PML and acquisition coordinates. Native voxel geometry is downloadable. |

A full 3D field simulation still produces a single receiver A-scan. Field charts select among Ex/Ey/Ez/Hx/Hy/Hz and use actual HDF5 timing, with E in V/m, H in A/m and time displayed in ns.

### Execution, persistence and reuse

Admission estimates host memory, per-device VRAM, material coefficient capacity, output and scratch space separately. It includes native arrays, FFT work and runtime reserves; estimates are not measured capacity guarantees. Build receipts record observed RSS. Available host RAM is reduced by a configurable reserve. Scheduling may reduce workers or queue a batch, but never coarsens or truncates the experiment. At most one version-2 model runs per GPU; local batches share an OS resource lock and device identities stay reserved until completion. CPU version-2 execution also uses isolated worker processes.

Execution uses current manifest entries and input hashes, with `n=1`. Native preflight verifies mode, grid, timing, six PML faces, source kind/axis/timing, receiver location, snapshot settings, material-table parity, half-space continuation and target voxel occupancy before field updates. Missing/stale decks fail admission. HDF5 ingestion checks actual grid, solver version, sample title, acquisition, timing and all six finite equal-length components. Zero-valued components are valid. Raw output, native geometry and requested snapshots are bound by hashes in an execution receipt; stale or failed results cannot become reusable output.

| Artifact | Contents |
| --- | --- |
| `dataset_contract.json` | Immutable common mode/frame, solver identity, policies, spectral/grid/acquisition/time settings, requested ranges, output plan, seeds and resource estimates. |
| `sampled_layers.json` | Accepted draws, original target draws, soil/placement rejection provenance and accepted distributions. |
| `derived_layers.json` | Native material tables, hashes, labels, spectral evaluations and aggregate extrema; ghost-corner samples may remain. |
| `global_derive.json` | Common numerical plan and derivation reasons. |
| `dropped_targets.json` | Whole-sample drops and reasons; no backfill. |
| `emitted_files.json` | Current accepted identities, hashes, full resolved scenes, requested/delivered counts and warnings. |
| `out_files/*.out`, `*.geometry.h5`, `*.execution.json` | Native field arrays, lossless voxel material maps and verified execution metadata. Requested snapshots are retained in per-sample folders. |
| `qualification.json` | Separate reviewed attestation binding evidence, tolerances, intended use, permitted backends and executed artifacts. It does not change the experiment digest. |

PostgreSQL stores requested values, resolved scenes and actual execution metadata separately, with domain z, version/frame and digests. Apply the new nullable migration before starting the updated backend. Historical records are left readable without invented finite dimensions or overwritten source metadata. Their provenance is unverified until established from artifacts.

The versioned Qdrant collection `sim_sessions_v2` includes mode, frequency interpretation, crossline geometry, orientation and acquisition details. Similarity scores identify candidates; substitution requires exact contract and accepted-population compatibility plus verified scientifically qualified outputs. Unqualified runs are not indexed for automatic reuse. Qualification has no default tolerance or automatic certification: see [the implementation record](docs/3D_IMPLEMENTATION_STATUS.md) for measured results, limitations and reviewer workflow.

The app uses typed user IDs for organization without authentication, assumes one backend worker, and does not automatically recover an interrupted run after a server crash. CLI `--skip-existing` can resume only artifacts whose current receipts validate.

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
| `GPR_SIM_WORKERS` | Requested concurrency; version 2 caps at admitted host capacity and one model per GPU. Legacy GPU default is two. |
| `GPR_ENABLE_EXPERIMENTAL_3D` | Set to `1` only on a developer deployment to collect/run experimental 3D. Off by default. |
| `GPR_HOST_RESERVE_BYTES`, `GPR_HOST_BUDGET_BYTES` | Host reserve (default 2 GiB) and optional additional execution budget. |
| `GPR_RESOURCE_LOCK_PATH` | Shared local batch-admission lock; all API/CLI processes on a host must use the same path. |
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
| `docs/PIPELINE_CONTRACT_V2.md` | Versioned physics invariants and implementation guidance. |

Run the existing backend and geometry regression suites from the repository root:

```bash
uv run pytest backend/tests backend/dataset_sampling/tests -v
```

The tests cover section storage, workflow routing and regeneration, persistence, uploads, visualization, dataset scoping, target geometry/PML behavior, output handling, and reuse. Many integration tests stub external services or the solver; passing them does not replace a real gprMax run and physical validation of a dataset.
