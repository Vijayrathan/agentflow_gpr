# Live Subsurface Visualization — Frontend ↔ Agent Integration

## Context

The single-agent pipeline (backend/api.py WebSocket ↔ `SingleAgentSession`) already collects all six parameter sections and generates the `.in` dataset, but the frontend canvas (`SubsurfaceView` in viz.jsx) still renders only a static demo model — nothing the agent collects ever reaches it. This change streams the evolving model to the canvas: it starts **blank**, builds up **live** as sections are saved (ranges → midpoint layers + uncertainty bands), and gains concrete per-sample realizations once the deterministic pipeline runs.

**User decisions (locked):**

- Collection phase renders **midpoint values + uncertainty bands** (thickness min/max spread at layer boundaries).
- The canvas gets **two small tabs**: **Overview** (ranges/midpoint view, always available) and **Samples** (per-sample view with a sample **dropdown**, enabled only after `layer_sampling` produces samples).
- The app **always starts blank** (`makeInitialModel()` → empty scene).
- **Out of scope** (don't wire, don't break): inspector, radargram, ML model selector, `choice_required`.

Constraint (CLAUDE.md/AGENT.md): all physics stays deterministic Python. Preview ε uses gprMax-native Peplinski (`Material.calculate_er(f).real`, never raw `er`) — no LLM involvement.

## Key facts verified in code

- `ChatPane` destructuring ([chatbot.jsx:58-63](frontend/app/chatbot.jsx#L58-L63)) ignores the `model`/`setModel` props App already passes ([app.jsx:376-389](frontend/app/app.jsx#L376-L389)).
- **Stale-closure trap**: `ws.onmessage` is bound once at mount ([chatbot.jsx:107-115](frontend/app/chatbot.jsx#L107-L115)) and captures the first render's `handleServerEvent`. Any new callback prop must be read through a ref.
- `layer_sampling` runs **mid-collection** (after `target_ranges`, before waveform/antenna — api.py `_advance_after_collect` ~line 329), so range data and sample data coexist in one scene payload; the tabs design handles this naturally.
- `makeInitialModel()` is also the "utility" preset (data.jsx ~658, app.jsx ~138) — blanking it needs a `makeUtilityModel()` split.
- `target_placement` rewrites `sampled_layers.json` and can **drop samples** → sample_ids may be non-contiguous; the dropdown must show `sample_id`, not index.
- Per-sample ε already exists on disk after `peplinski_derive` (`derived_layers.json`: `eps_r_dry`/`eps_r_wet` per layer) — join it, don't recompute.
- Cylinder `depth` is to **center** in both backend (emit.py:152) and viz (viz.jsx:536) — no conversion.
- Backend sends **no σ** (AGENT.md: only real in-band εr is derived); viz labels show εr only.

---

## Step 1 — New backend module: `backend/viz_projection.py`

Pure, key-free projection functions (no FastAPI imports).

```python
PREVIEW_FREQ_HZ = 0.9e9   # mid Peplinski band fallback until waveform is saved
SAMPLE_CAP = 200          # max realizations shipped to frontend

def build_scene(store, flags, output_dir, stage=None) -> Optional[dict]
    # store: SingleAgentSession.store (raw section dicts)
    # flags: {"sampled","derived","grid","placed","emitted"}
    # output_dir: resolved dataset dir (None => skip manifest reads)
```

- **Preview ε**: `_preview_layer_eps(layer_range, freq_hz, nbins)` wraps `dataset_sampling.peplinski_derive.derive_layer_eps` at midpoint sand/clay/densities with the layer's full θv band → `(eps_dry, eps_wet, eps_mid)`. Frequency: `peak_frequency(...)` from `backend/validation_tools_new.py` (~line 87) if waveform saved, else `PREVIEW_FREQ_HZ` with `eps_provisional: true`. Memoize on rounded-inputs tuple.
- **Provisional domain** until `global_derive`: depth = 1.2 × Σ thickness_mid (min 0.5 m); width = max(1.0, target x_max + r_max + 0.2).
- **Manifest readers** gated by flags (stale files never shown): `sampled_layers.json` (re-read every time — target_placement rewrites it), `derived_layers.json` (join ε by sample_id + layer index), `global_derive.json` (grid). Cap samples at `SAMPLE_CAP` with `total`/`included`/`truncated` metadata.

### `model_update` payload

```jsonc
{
  "type": "model_update",
  "scene": {
    "phase": "collect", // collect|sampled|derived|grid|placed|emitted (highest flag)
    "stage": "layers",
    "project": "soil_sample",
    "domain": {
      "width_m": 1.0,
      "depth_m": 1.15,
      "dx_m": null,
      "provisional": true,
    },
    "acquisition": {
      "frequency_ghz": 0.7,
      "waveform": "ricker",
      "antenna_kind": "hertzian_dipole",
      "txrx_sep_m": 0.12,
      "time_window_ns": null,
    },
    "ranges": {
      // present once layers section complete
      "layers": [
        {
          "name": "top_sandy_loam",
          "thickness_mid_m": 0.4,
          "thickness_min_m": 0.3,
          "thickness_max_m": 0.5,
          "sand_pct_mid": 35.0,
          "clay_pct_mid": 10.0,
          "silt_pct_mid": 55.0,
          "theta_v_mid": 0.13,
          "eps_mid": 8.8,
          "eps_dry": 5.9,
          "eps_wet": 11.7,
          "eps_freq_hz": 9.0e8,
          "eps_provisional": true,
        },
      ],
      "target": {
        "material": "pec",
        "x_mid_m": 0.5,
        "depth_mid_m": 0.3,
        "radius_mid_m": 0.05,
        "x_min_m": 0.3,
        "x_max_m": 0.7,
        "depth_min_m": 0.2,
        "depth_max_m": 0.4,
        "radius_min_m": 0.03,
        "radius_max_m": 0.07,
      }, // or null
    },
    "samples": {
      // null until flags.sampled
      "total": 500,
      "included": 200,
      "truncated": true,
      "items": [
        {
          "sample_id": 1,
          "layers": [
            {
              "name": "...",
              "thickness_m": 0.218,
              "sand_pct": 45.1,
              "clay_pct": 6.4,
              "theta_v_mid": 0.13,
              "eps_mid": 8.8,
              "eps_dry": 5.93,
              "eps_wet": 11.68,
            },
          ],
          "target": {
            "x_m": 0.5,
            "depth_m": 0.2,
            "radius_m": 0.05,
            "material": "pec",
          },
        },
      ],
    },
    "grid": {
      // null until flags.grid
      "domain_x_m": 1.335,
      "domain_y_m": 1.691,
      "depth_z_m": 1.08,
      "ground_y_m": 1.153,
      "dx_m": 0.00367,
      "source_height_m": 0.445,
      "tx_x_m": 0.607,
      "rx_x_m": 0.727,
      "time_window_ns": 43.82,
      "f_peak_hz": 7.0e8,
    },
  },
}
```

Material → color/pattern mapping stays **frontend-side** (`materialKeyForLayer` in data.jsx) — the MATERIALS presentation catalog lives there; the backend stays physics-only.

## Step 2 — Emission hooks in `backend/api.py`

- `ChatSession` gains `viz_flags` (5 booleans, all False) and `last_scene: Optional[dict]`.
- New `async def _send_model_update(ws, chat, *, stage=None)`: builds scene via `asyncio.to_thread(build_scene, ...)` (resolve `output_dir` from `chat.state["dataset_config"]` with the existing `_resolve_dataset_path`, ~api.py:475); wrap in try/except so viz can never break the chat loop; skip send when `scene is None or scene == chat.last_scene` (dedup).
- **Hook 1** — `_handle_agent_result` (~api.py:266-273), after the AI-text relay loop: emit. Fires after every agent turn, so mid-stage/partial saves and cross-edits render immediately.
- **Hook 2** — `_run_deterministic` (~api.py:446-465), after `chat.state.update(updates)`: map node → flags, then emit:
  - `layer_sampling_node` → `sampled=True`, reset `derived/grid/placed/emitted=False` (covers staleness re-sampling + global-remediation resample automatically)
  - `peplinski_derive_node` → `derived=True`; `global_derive_node` → `grid=True`; `target_placement_node` → `placed=True`; `dataset_generation_node` → `emitted=True`
  - Validation nodes change nothing → dedup drops them.
- **Hook 3** — reconnect (websocket_endpoint `phase=="agent"`/busy branches ~lines 161-170): if `chat.last_scene`, re-send it so a page reload repopulates the canvas.

## Step 3 — Frontend `data.jsx`

1. **Blank start**: rename the current `makeInitialModel` body to `makeUtilityModel()`; new `makeInitialModel()` returns `{ project: "untitled_dataset", domain: {width:1.2, depth:1.0, dx:0.002}, acquisition: {antenna:"custom", frequency:1.0, waveform:"ricker", timeWindow:12, traceStep:0.01, txrxSep:0.05, surveyMode:"B-scan"}, layers: [], targets: [] }`. Repoint `scenarioUtility().patch` (~658) and app.jsx `onLoadPreset("utility")` (~138) to `makeUtilityModel()`; export both on `window`.
2. **`materialKeyForLayer({sandPct, clayPct, thetaV, epsilon})`**: deterministic classifier — dominant texture picks sand/clay/silt/topsoil family; θv ≥ ~0.20 picks the wet variant; fallback = nearest MATERIALS epsilon.
3. **`sceneToModel(scene, vizTab, sampleIdx)`**:
   - `vizTab === "overview"`: layers from `scene.ranges.layers` → `{id:"ly_"+i, thickness: thickness_mid_m, thicknessMin/Max, epsilon: eps_mid, sigma: null, material: materialKeyForLayer(...)}`; target from `scene.ranges.target` midpoints (`type:"metalpipe"` for pec, `diameter: 2*radius_mid_m`).
   - `vizTab === "sample"`: layers/target from `scene.samples.items[clamp(sampleIdx)]`, concrete thicknesses, no thicknessMin/Max.
   - Both: domain from `scene.grid` when present (`width: domain_x_m, depth: depth_z_m, dx: dx_m`; `acquisition.timeWindow = time_window_ns`), else `scene.domain`; acquisition frequency/waveform/txrxSep from `scene.acquisition` with fallbacks. Stable ids so selection/reconciliation survive updates.

## Step 4 — Frontend `chatbot.jsx`

- Destructure new `onModelUpdate` prop; keep it in a ref (`const cbRef = React.useRef(onModelUpdate); cbRef.current = onModelUpdate;`) to dodge the mount-time `ws.onmessage` stale closure.
- In `handleServerEvent`, **before** the `setSending(false)/setTyping(false)` lines (like `pipeline_busy`):
  ```js
  if (msg.type === "model_update") {
    cbRef.current?.(msg.scene);
    return;
  }
  ```
  (must not clear the typing indicator — updates arrive mid-turn).

## Step 5 — Frontend `app.jsx`: scene state, canvas tabs, sample dropdown

- New state: `scene` (raw payload), `vizTab` (`"overview" | "sample"`, default overview), `sampleIdx` (default 0).
- `onModelUpdate = useCallback((s) => setScene(s), [])`, passed to `ChatPane`.
- Effect on `[scene, vizTab, sampleIdx]`: clamp `sampleIdx` to items length; `setModel(sceneToModel(scene, vizTab, idx))`. If `vizTab === "sample"` and samples become empty (re-sample reset), fall back to overview.
- **Canvas tab switch** inside `.stage-canvas` (near the HUD, ~app.jsx:289): two small tabs `Overview | Samples`.
  - **Samples tab disabled** (grayed, non-clickable, tooltip "run sampling first") until `scene?.samples?.items?.length > 0`.
  - When Samples tab active: a compact **dropdown** listing `sample {sample_id}` for each item (plus `· {included}/{total}` when truncated), driving `sampleIdx`. No auto-switch when samples arrive — user switches manually.
- Manual inspector/tree edits get overwritten by the next `model_update` — accepted, out of scope.

## Step 6 — Frontend `viz.jsx`: uncertainty bands + empty-model polish

- In `SubsurfaceView`, alongside the cumulative-depth `rects` loop (~174-181), accumulate `accMin += l.thicknessMin ?? l.thickness` / `accMax += ...Max`; for each layer with a spread push a band `{from: min(accMin, dom.depth), to: min(accMax, dom.depth)}` (cumulative uncertainty is correct by construction). Render after layer rects: semi-transparent accent rect + dashed min/max hairlines per boundary. Skip entirely when no layer carries `thicknessMin` (sample view → clean boundaries).
- Layer label (~391): `εr {l.epsilon ?? "—"}` and only append `σ` when non-null.
- Empty model is already safe (half-space rect covers the domain; legend guarded; `layersDepth([])=0`). Polish: when `layers.length === 0`, caption the half-space "no layers yet — describe the subsurface in chat"; optional empty-state row under Soil Layers in ModelTree (panels.jsx ~505).

## Step 7 — Tests: `backend/tests/test_viz_projection.py` (new, key-free)

Follow `test_single_agent_store.py` conventions (sys.path bootstrap incl. `gprMax/`):

- Store → scene: layers-only store → midpoints + min/max carried, `eps_dry ≤ eps_mid ≤ eps_wet`, `eps_provisional=True` at `PREVIEW_FREQ_HZ`; + waveform → `eps_freq_hz == peak_frequency(...)`, provisional off; target ranges → midpoint cylinder; empty store → `None`.
- Manifests → scene: write fixture `sampled_layers.json` / `derived_layers.json` / `global_derive.json` into `tmp_path` (shapes copied from `dataset/*.json`); flags `{sampled, derived, grid}` → samples joined with real per-sample ε, grid populated, `phase == "grid"`.
- Cap: `SAMPLE_CAP + 5` samples → `included == SAMPLE_CAP`, `truncated`, correct `total`.
- Flag gating: `sampled=True, grid=False` → `grid is None` even with a stale `global_derive.json` on disk.

## Verification (manual, end-to-end)

1. `docker-compose up -d`; `uvicorn backend.api:app --reload --port 8000`; open `frontend/html-design.html`. Canvas blank (air + half-space, empty tree/legend); Samples tab disabled.
2. Complete `dataset_config` → project name updates. Complete `layers` → midpoint layers + uncertainty bands appear; legend/tree populate.
3. Complete `target_ranges` → midpoint cylinder; `layer_sampling` runs → Samples tab enables; switch to it, dropdown flips realizations (concrete thicknesses, no bands); Overview tab still shows ranges+bands.
4. Waveform/antenna stages → frequency label + txrxSep update live.
5. After `global_derive` → domain snaps to real grid, dx + time window set. After placement/emission → placed targets per sample; dropped samples vanish from the dropdown (ids non-contiguous but shown).
6. Trigger a validation failure (e.g. 100 MHz waveform) → remediation edit re-renders; a `RESAMPLE_SECTIONS` edit resets samples (Samples tab disables until re-sampled).
7. Reload page mid-session → `last_scene` re-sent, canvas repopulates.
8. Regressions: presets menu still loads demo scenes (`makeUtilityModel`); inspector/radargram untouched; `pytest backend/tests/ -v` green.

## Files

| File                                   | Change                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| `backend/viz_projection.py`            | **new** — store/manifests → scene projection, preview Peplinski ε                    |
| `backend/api.py`                       | `viz_flags`/`last_scene` on ChatSession; `_send_model_update`; 3 hooks               |
| `frontend/app/data.jsx`                | blank `makeInitialModel` + `makeUtilityModel`; `sceneToModel`; `materialKeyForLayer` |
| `frontend/app/chatbot.jsx`             | `onModelUpdate` prop via ref; `model_update` case                                    |
| `frontend/app/app.jsx`                 | scene/vizTab/sampleIdx state; canvas tabs + sample dropdown; wire ChatPane           |
| `frontend/app/viz.jsx`                 | uncertainty bands; ε-only labels; empty-state caption                                |
| `backend/tests/test_viz_projection.py` | **new** — key-free projection tests                                                  |
