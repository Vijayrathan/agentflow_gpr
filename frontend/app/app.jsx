/* ============================================================
   NL2Sim — application root
   ============================================================ */
const { useState, useEffect, useRef, useCallback } = React;

function ViewToggle({ on, icon, label, onClick }) {
  return (
    <button
      className={"tbtn" + (on ? " on" : "")}
      title={label}
      onClick={onClick}
    >
      <Icon name={icon} className="ic" />
    </button>
  );
}

function App() {
  const [model, setModel] = useState(makeInitialModel);
  const modelRef = useRef(model);
  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  // ── Identity + per-user chat list (multi-chat, DB-backed) ──────────────
  // user id: plain string identity (no auth), remembered locally; the chat
  // LIST always comes from the backend so the same id on another browser
  // recovers everything.
  const [userId, setUserId] = useState(() =>
    localStorage.getItem("nl2sim_user_id"),
  );
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(() =>
    localStorage.getItem("nl2sim_current_chat_id"),
  );

  const fetchChats = useCallback(async (uid) => {
    const u = uid || localStorage.getItem("nl2sim_user_id");
    if (!u) return [];
    try {
      const res = await fetch(
        `${window.getApiHttpBase()}/users/${encodeURIComponent(u)}/chats`,
      );
      if (!res.ok) return [];
      const body = await res.json();
      const list = body.chats || [];
      setChats(list);
      return list;
    } catch (e) {
      return [];
    }
  }, []);

  const [selected, setSelected] = useState(null);
  const [activeModel, setActiveModel] = useState("gprmax");

  const [chatOpen, setChatOpen] = useState(true);
  const [railCollapsed, setRailCollapsed] = useState(false);

  const [dockCollapsed, setDockCollapsed] = useState(false);

  const [view, setView] = useState({
    ruler: true,
    targets: true,
    antenna: true,
  });
  const [zoom, setZoom] = useState(1);

  const [solving, setSolving] = useState(false);
  const [progress, setProgress] = useState(0);
  const [scanFrac, setScanFrac] = useState(0.12);
  // live gprMax run counters ({done, total} while a batch is running)
  const [sim, setSim] = useState(null);
  // pending forward-model reuse recommendation (the /simulate response was
  // "reuse_recommended"): drives the compact Reuse / Simulate-anyway bar
  const [reuseRec, setReuseRec] = useState(null);

  // live scene streamed by the backend agent (model_update events);
  // vizTab: "overview" = range midpoints + uncertainty bands,
  //         "sample"   = one concrete realization (after sampling ran)
  const [scene, setScene] = useState(null);
  const [vizTab, setVizTab] = useState("overview");
  const [sampleIdx, setSampleIdx] = useState(0);
  const onModelUpdate = useCallback((s) => setScene(s), []);
  const sampleItems = scene?.samples?.items || [];

  // Left-rail tab ("model" tree vs generated "dataset" files) and the
  // currently opened .in file (shown in place of the canvas).
  const [railTab, setRailTab] = useState("model");
  const [datasetFiles, setDatasetFiles] = useState([]);
  const [datasetView, setDatasetView] = useState(null); // {filename, content, loading, error}
  // .in filenames whose forward-model .out exists (enables "View outcome");
  // filled live from simulation_progress events + reconciled from the backend
  const [outFiles, setOutFiles] = useState(() => new Set());
  const [outcomeView, setOutcomeView] = useState(null); // {filename, data, loading, error}

  // switching back to the Model tab returns to the canvas — the file/outcome
  // overlays would otherwise keep covering it until closed via their x
  const changeRailTab = useCallback((tab) => {
    setRailTab(tab);
    if (tab === "model") {
      setDatasetView(null);
      setOutcomeView(null);
    }
  }, []);

  const refreshOutputs = useCallback(async () => {
    try {
      const base = window.getApiHttpBase();
      const sid = window.getSessionId();
      if (!sid) return;
      const res = await fetch(`${base}/datasets/${sid}/outputs`);
      // a chat switch mid-flight: never write another chat's outputs here
      if (sid !== window.getSessionId()) return;
      if (!res.ok) return;
      const body = await res.json();
      if (sid !== window.getSessionId()) return;
      setOutFiles(new Set(body.files || []));
    } catch (e) {
      /* no outputs yet / backend unreachable — keep whatever we have */
    }
  }, []);

  const onDatasetReady = useCallback(
    (result) => {
      const files = (result && result.files) || [];
      setDatasetFiles(files);
      if (files.length) setRailTab("dataset");
      // a re-emitted dataset invalidates old outcomes in the UI
      setOutFiles(new Set());
      setOutcomeView(null);
      if (files.length) refreshOutputs();
      // the chat's title may have changed (basename now known)
      fetchChats();
    },
    [refreshOutputs, fetchChats],
  );

  // Upload a zip of gprMax .in files as this session's dataset. The backend
  // syntax-checks every deck (gprMax's own command rules) and only stores the
  // valid ones; per-file rejections come back in the response and are also
  // reported in the chat (dataset_ready over the WS re-populates the tab too).
  const uploadDatasetZip = useCallback(
    async (file) => {
      if (!file) return;
      const esc = (s) =>
        String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
      toast(`Uploading <b>${esc(file.name)}</b> — validating…`, "info");
      try {
        const base = window.getApiHttpBase();
        const sid = window.getSessionId();
        if (!sid) throw new Error("no chat selected");
        const res = await fetch(
          `${base}/datasets/${sid}/upload?filename=${encodeURIComponent(file.name)}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/zip" },
            body: file,
          },
        );
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(
            typeof body.detail === "string" ? body.detail : "HTTP " + res.status,
          );
        }
        // switched chats while uploading — the result belongs to the old one
        if (sid !== window.getSessionId()) return;
        const rejected = body.rejected || [];
        toast(
          rejected.length
            ? `Imported <b>${body.n_written}</b> file(s) · ${rejected.length} failed the syntax check — see chat`
            : `Imported <b>${body.n_written}</b> gprMax input file(s)`,
          rejected.length ? "info" : "ok",
        );
        // The WS dataset_ready event does this too — calling directly keeps
        // the tab correct even if the socket is reconnecting.
        onDatasetReady(body);
      } catch (e) {
        toast("Upload failed — " + esc(e.message || e), "info");
      }
    },
    [toast, onDatasetReady],
  );

  const openDatasetFile = useCallback(async (filename) => {
    // the outcome overlay sits on top — drop it so the newly opened
    // input deck is actually visible
    setOutcomeView(null);
    setDatasetView({ filename, content: "", loading: true });
    try {
      const base = window.getApiHttpBase();
      const sid = window.getSessionId();
      const res = await fetch(
        `${base}/datasets/${sid}/files/${encodeURIComponent(filename)}`,
      );
      if (sid !== window.getSessionId()) return; // switched chats mid-fetch
      if (!res.ok) throw new Error("HTTP " + res.status);
      const text = await res.text();
      if (sid !== window.getSessionId()) return;
      setDatasetView({ filename, content: text, loading: false });
    } catch (e) {
      setDatasetView({
        filename,
        content: "Could not load file — " + e.message,
        loading: false,
        error: true,
      });
    }
  }, []);

  // fetch the A-scan payload for one emitted file and show it on the canvas
  const openDatasetOutcome = useCallback(async (filename) => {
    setOutcomeView({ filename, data: null, loading: true });
    try {
      const base = window.getApiHttpBase();
      const sid = window.getSessionId();
      const res = await fetch(
        `${base}/datasets/${sid}/outputs/${encodeURIComponent(filename)}`,
      );
      if (sid !== window.getSessionId()) return; // switched chats mid-fetch
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      if (sid !== window.getSessionId()) return;
      setOutcomeView({ filename, data, loading: false });
    } catch (e) {
      setOutcomeView({
        filename,
        data: null,
        loading: false,
        error: "Could not load output — " + e.message,
      });
    }
  }, []);

  useEffect(() => {
    if (!scene) return;
    const n = scene.samples?.items?.length || 0;
    let tab = vizTab;
    if (tab === "sample" && n === 0) {
      // re-sampling invalidated the realizations — fall back to ranges
      tab = "overview";
      setVizTab("overview");
    }
    const idx = clamp(sampleIdx, 0, Math.max(0, n - 1));
    if (idx !== sampleIdx) setSampleIdx(idx);
    setModel(sceneToModel(scene, tab, idx));
  }, [scene, vizTab, sampleIdx]);

  const [modal, setModal] = useState(null);
  const [toasts, setToasts] = useState([]);

  const toast = useCallback((msg, kind = "info") => {
    const id = uid("t");
    setToasts((t) => [...t, { id, msg, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 3200);
  }, []);

  // ── Chat switching ──────────────────────────────────────────────────────
  // One reset for everything server-derived; the selected chat's
  // session_restore + model_update then repaint the pane, tab and canvas.
  const resetForChatSwitch = useCallback(() => {
    setScene(null);
    setModel(makeInitialModel());
    setDatasetFiles([]);
    setDatasetView(null);
    setOutcomeView(null);
    setOutFiles(new Set());
    setSim(null);
    setSolving(false);
    setReuseRec(null);
    setProgress(0);
    setSelected(null);
    setRailTab("model");
    setVizTab("overview");
    setSampleIdx(0);
  }, []);

  const selectChat = useCallback(
    (id) => {
      if (!id || id === localStorage.getItem("nl2sim_current_chat_id")) {
        setCurrentChatId(id || null);
        return;
      }
      localStorage.setItem("nl2sim_current_chat_id", id);
      resetForChatSwitch();
      setCurrentChatId(id);
    },
    [resetForChatSwitch],
  );

  const newChat = useCallback(async () => {
    if (!userId) return;
    try {
      const res = await fetch(
        `${window.getApiHttpBase()}/users/${encodeURIComponent(userId)}/chats`,
        { method: "POST" },
      );
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || "HTTP " + res.status);
      await fetchChats(userId);
      selectChat(body.session_id);
    } catch (e) {
      toast("Could not create a new chat — " + (e.message || e), "info");
    }
  }, [userId, fetchChats, selectChat, toast]);

  const switchUser = useCallback(() => {
    localStorage.removeItem("nl2sim_user_id");
    localStorage.removeItem("nl2sim_current_chat_id");
    resetForChatSwitch();
    setChats([]);
    setCurrentChatId(null);
    setUserId(null);
  }, [resetForChatSwitch]);

  const onUserSubmit = useCallback((uid_) => {
    localStorage.setItem("nl2sim_user_id", uid_);
    // Never carry another user's chat pointer across the gate.
    localStorage.removeItem("nl2sim_current_chat_id");
    setCurrentChatId(null);
    setUserId(uid_);
  }, []);

  // Bootstrap per user: load the chat list, then keep the stored pointer if
  // it belongs to this user, else most-recent chat, else mint the first one.
  useEffect(() => {
    if (!userId) return undefined;
    let cancelled = false;
    (async () => {
      const list = await fetchChats(userId);
      if (cancelled) return;
      const stored = localStorage.getItem("nl2sim_current_chat_id");
      if (stored && list.some((c) => c.id === stored)) {
        setCurrentChatId(stored);
        return;
      }
      if (list.length) {
        selectChat(list[0].id);
        return;
      }
      try {
        const res = await fetch(
          `${window.getApiHttpBase()}/users/${encodeURIComponent(userId)}/chats`,
          { method: "POST" },
        );
        const body = await res.json().catch(() => ({}));
        if (!cancelled && res.ok && body.session_id) {
          await fetchChats(userId);
          selectChat(body.session_id);
        }
      } catch (e) {
        /* backend unreachable — the gate stays open, retry by re-submitting */
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [userId]);

  // selecting an item opens inspector
  const onSelect = useCallback((sel) => {
    setSelected(sel);
    if (sel) {
      setDockCollapsed(false);
    }
  }, []);

  // Run the real gprMax forward model on the emitted dataset. The POST only
  // kicks the batch off; per-file progress and the final summary arrive as
  // simulation_* events on the chat WebSocket (onSimulationEvent below).
  const runForward = useCallback(async (force) => {
    // `force === true` only when passed explicitly ("Simulate anyway") —
    // the toolbar onClick hands us a click event, which must not count.
    const forced = force === true;
    if (!datasetFiles.length) {
      toast(
        "Generate the dataset first — the forward model runs the emitted gprMax files",
        "info",
      );
      return;
    }
    const solver = ML_MODELS.find((m) => m.id === activeModel);
    if (!solver || !solver.available) {
      toast(
        (solver ? solver.label : "Selected model") +
          " is not available yet — switch to <b>gprMax</b> to run",
        "info",
      );
      return;
    }
    setReuseRec(null);
    setSolving(true);
    setProgress(0);
    setSim({ done: 0, total: datasetFiles.length });
    try {
      const base = window.getApiHttpBase();
      const sid = window.getSessionId();
      if (!sid) throw new Error("no chat selected");
      const res = await fetch(
        `${base}/datasets/${sid}/simulate${forced ? "?force=true" : ""}`,
        { method: "POST" },
      );
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || "HTTP " + res.status);
      if (body.status === "reuse_recommended") {
        // A ≥threshold-similar simulated dataset exists — the run was NOT
        // started. Offer Reuse / Simulate-anyway instead of a progress bar.
        setSolving(false);
        setSim(null);
        setReuseRec(body.recommendation || {});
      }
    } catch (e) {
      setSolving(false);
      setSim(null);
      toast("Could not start the forward model — " + e.message, "info");
    }
  }, [datasetFiles.length, activeModel, toast]);

  // Execute the pending reuse recommendation: the backend copies the source
  // session's samples + signals into this session (no gprMax run). The
  // resulting dataset_ready / simulation_complete WS events refresh the tab
  // and outcome buttons through their existing handlers.
  const adoptDataset = useCallback(async () => {
    if (!reuseRec || !reuseRec.source_session_id) return;
    setSolving(true);
    setSim(null);
    try {
      const base = window.getApiHttpBase();
      const sid = window.getSessionId();
      if (!sid) throw new Error("no chat selected");
      const res = await fetch(`${base}/datasets/${sid}/adopt`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_session_id: reuseRec.source_session_id }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || "HTTP " + res.status);
      setReuseRec(null);
      toast(
        `Reused <b>${body.num_generated}</b> simulated sample(s) — no run needed`,
        "ok",
      );
    } catch (e) {
      toast("Could not reuse the dataset — " + e.message, "info");
    } finally {
      setSolving(false);
    }
  }, [reuseRec, toast]);

  // simulation_* events from the backend (routed through the chat WS)
  const onSimulationEvent = useCallback(
    (msg) => {
      if (msg.type === "simulation_progress") {
        const total = msg.total || 1;
        // `completed` is a running count of finished files — the only
        // monotonic source once several models solve in parallel and finish
        // out of order (msg.index is the file's position in the batch).
        const done =
          typeof msg.completed === "number"
            ? msg.completed
            : msg.event === "done"
              ? msg.index
              : msg.index - 1;
        const p = clamp(done / total, 0, 1);
        setSolving(true);
        setSim({ done, total });
        setProgress(p);
        setScanFrac(0.06 + p * 0.88);
        if (msg.event === "done" && msg.status === "ok" && msg.filename) {
          // this file's .out just landed — enable its "View outcome" now
          setOutFiles((s) => {
            const next = new Set(s);
            next.add(msg.filename);
            return next;
          });
        }
        return;
      }
      if (msg.type === "simulation_complete") {
        const r = msg.result || {};
        setSolving(false);
        setSim(null);
        const ran = (r.succeeded || 0) + (r.skipped || 0);
        if (ran > 0) {
          setProgress(1);
          setScanFrac(0.94);
        }
        toast(
          r.failed
            ? `Forward model finished · <b>${r.succeeded}/${r.total}</b> ok, ${r.failed} failed`
            : `Forward model complete · <b>${ran}</b> simulation(s)`,
          r.failed ? "info" : "ok",
        );
        refreshOutputs(); // authoritative reconciliation of "View outcome" state
        return;
      }
      if (msg.type === "session_restore") {
        // page refresh: re-hydrate the run indicator + pending recommendation
        if (msg.simulating) {
          setSolving(true);
          setSim(null);
        }
        setReuseRec(msg.reuse || null);
      }
    },
    [toast, refreshOutputs],
  );

  // overview-tab caveats: assumptions that hold until derived truth exists.
  // Collapsed to a chip by default so the panel never sits over the plot
  // (the SVG fills the full canvas width when the dock is collapsed).
  const caveats =
    vizTab === "overview" && scene ? overviewCaveats(scene) : [];
  const [caveatsOpen, setCaveatsOpen] = useState(false);

  // legend data
  const usedMats = [
    ...new Set(
      model.layers.filter((l) => l.visible !== false).map((l) => l.material),
    ),
  ];
  const usedTargets = [
    ...new Set(
      model.targets.filter((t) => t.visible !== false).map((t) => t.type),
    ),
  ];

  const Z = (v) => Math.round(v * 100);

  return (
    <div className="app">
      <MenuBar
        model={model}
        setModel={setModel}
        activeModel={activeModel}
        setActiveModel={setActiveModel}
        datasetCount={datasetFiles.length}
        chatOpen={chatOpen}
        setChatOpen={setChatOpen}
        toast={toast}
        openModal={setModal}
        onUploadZip={uploadDatasetZip}
      />

      <div className="body">
        <div className="viz">
          {/* viz toolbar */}
          <div className="viztoolbar">
            <button
              className="tbtn"
              title={railCollapsed ? "Show model tree" : "Hide model tree"}
              onClick={() => setRailCollapsed((c) => !c)}
            >
              <Icon name="panel" className="ic" />
            </button>
            <div className="crumb">
              <b>{model.project}</b>
              <span className="sl">/</span>Subsurface model
              <span className="sl">·</span>cross-section
            </div>
            <div className="spacer"></div>

            <div className="tgroup">
              <ViewToggle
                on={view.ruler}
                icon="ruler"
                label="Rulers"
                onClick={() => setView((v) => ({ ...v, ruler: !v.ruler }))}
              />
              <ViewToggle
                on={view.targets}
                icon="target"
                label="Targets"
                onClick={() => setView((v) => ({ ...v, targets: !v.targets }))}
              />
              <ViewToggle
                on={view.antenna}
                icon="radar"
                label="Antenna &amp; scan path"
                onClick={() => setView((v) => ({ ...v, antenna: !v.antenna }))}
              />
            </div>
            <div className="tgroup">
              <button
                className="tbtn"
                title="Zoom out"
                onClick={() => setZoom((z) => clamp(z / 1.2, 0.6, 3))}
              >
                <Icon name="zoomout" className="ic" />
              </button>
              <span className="zoomread">{Z(zoom)}%</span>
              <button
                className="tbtn"
                title="Zoom in"
                onClick={() => setZoom((z) => clamp(z * 1.2, 0.6, 3))}
              >
                <Icon name="zoomin" className="ic" />
              </button>
              <button className="tbtn" title="Fit" onClick={() => setZoom(1)}>
                <Icon name="fit" className="ic" />
              </button>
            </div>

            <button
              className="runbtn"
              onClick={runForward}
              disabled={solving}
              title={
                datasetFiles.length
                  ? "Run gprMax on the generated input files"
                  : "Available once the dataset has been generated"
              }
            >
              <Icon name="play" className="ic" />
              {solving
                ? sim && sim.total
                  ? `Running ${sim.done}/${sim.total}…`
                  : "Running…"
                : "Run forward model"}
            </button>
          </div>

          {/* viz body */}
          <div className="vizbody">
            <ModelTree
              model={model}
              setModel={setModel}
              selected={selected}
              onSelect={onSelect}
              collapsed={railCollapsed}
              toast={toast}
              railTab={railTab}
              setRailTab={changeRailTab}
              datasetFiles={datasetFiles}
              onOpenFile={openDatasetFile}
              activeFile={datasetView?.filename}
              outFiles={outFiles}
              onOpenOutcome={openDatasetOutcome}
            />

            <div className="stage">
              <div className="stage-canvas">
                {solving && (
                  <div className="solving-bar">
                    <i style={{ width: progress * 100 + "%" }}></i>
                  </div>
                )}

                {/* forward-model reuse offer — compact overlay (never a
                    permanent panel; the SVG fills the full canvas width) */}
                {reuseRec && !solving && (
                  <div className="reuse-bar">
                    <div className="rb-head">
                      <Icon name="info" size={12} />
                      <b>
                        {reuseRec.similarity_pct}% similar dataset already
                        simulated
                      </b>
                      <button
                        className="rb-x"
                        onClick={() => setReuseRec(null)}
                        title="Dismiss"
                      >
                        <Icon name="x" size={11} />
                      </button>
                    </div>
                    <div className="rb-meta">
                      {reuseRec.num_samples} sample(s)
                      {reuseRec.requested_samples != null &&
                      Number(reuseRec.requested_samples) !==
                        Number(reuseRec.num_samples)
                        ? ` — you requested ${reuseRec.requested_samples}`
                        : ""}
                      {reuseRec.simulated_at
                        ? ` · simulated ${String(reuseRec.simulated_at).slice(0, 10)}`
                        : ""}
                      {reuseRec.source_user_id
                        ? ` · by ${reuseRec.source_user_id}`
                        : ""}
                    </div>
                    {(reuseRec.params_diff || []).length > 0 && (
                      <details className="rb-diffs">
                        <summary>closest differences</summary>
                        {(reuseRec.params_diff || [])
                          .slice(0, 5)
                          .map((d, i) => (
                            <div className="rb-diff" key={i}>
                              <code>{d.param}</code>{" "}
                              {Math.round((d.sim || 0) * 100)}% match
                            </div>
                          ))}
                      </details>
                    )}
                    <div className="rb-actions">
                      <button className="rb-btn primary" onClick={adoptDataset}>
                        Reuse results
                      </button>
                      <button
                        className="rb-btn"
                        onClick={() => runForward(true)}
                      >
                        Simulate anyway
                      </button>
                    </div>
                  </div>
                )}
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    transform: `scale(${zoom})`,
                    transformOrigin: "center center",
                    transition: "transform .15s",
                  }}
                >
                  {scene?.dimensionality === "3D" ? (
                    <OrthogonalSlices scene={scene} sampleIndex={vizTab === "samples" ? sampleIdx : 0} />
                  ) : <SubsurfaceView
                    model={model}
                    selected={selected}
                    onSelect={onSelect}
                    view={view}
                    scanFrac={scanFrac}
                    solving={solving}
                  />}
                </div>

                {/* generated .in file viewer — covers the canvas when a
                    dataset sample is opened from the left rail */}
                {datasetView && (
                  <DatasetFileView
                    view={datasetView}
                    onClose={() => setDatasetView(null)}
                    hasOutcome={outFiles.has(datasetView.filename)}
                    onOpenOutcome={openDatasetOutcome}
                  />
                )}

                {/* forward-model A-scan viewer — layers over the file view */}
                {outcomeView && (
                  <DatasetOutcomeView
                    view={outcomeView}
                    onClose={() => setOutcomeView(null)}
                  />
                )}

                {/* view tabs: ranges overview vs one sampled realization */}
                <div className="viewtabs">
                  <button
                    className={"vt" + (vizTab === "overview" ? " on" : "")}
                    onClick={() => setVizTab("overview")}
                  >
                    Overview
                  </button>
                  <button
                    className={"vt" + (vizTab === "sample" ? " on" : "")}
                    disabled={sampleItems.length === 0}
                    title={
                      sampleItems.length === 0
                        ? "Available once layer sampling has run"
                        : "Inspect one sampled realization"
                    }
                    onClick={() => setVizTab("sample")}
                  >
                    Samples
                  </button>
                  {vizTab === "sample" && sampleItems.length > 0 && (
                    <select
                      className="vsel"
                      value={sampleIdx}
                      onChange={(e) => setSampleIdx(Number(e.target.value))}
                    >
                      {sampleItems.map((it, i) => (
                        <option key={it.sample_id} value={i}>
                          sample {it.sample_id}
                        </option>
                      ))}
                    </select>
                  )}
                  {vizTab === "sample" && scene?.samples?.truncated && (
                    <span
                      title="Realizations shown / total drawn"
                      style={{ color: "var(--muted)", padding: "0 6px" }}
                    >
                      {scene.samples.included}/{scene.samples.total}
                    </span>
                  )}
                </div>

                {/* overview disclaimers — mental model, not ground truth */}
                {caveats.length > 0 && !caveatsOpen && (
                  <button
                    className="caveats-chip"
                    onClick={() => setCaveatsOpen(true)}
                    title="This view is a mental model — click to see its assumptions"
                  >
                    <Icon name="info" size={12} />
                    assumptions ({caveats.length})
                  </button>
                )}
                {caveats.length > 0 && caveatsOpen && (
                  <div className="caveats">
                    <button
                      className="ch"
                      onClick={() => setCaveatsOpen(false)}
                      title="Collapse"
                    >
                      <Icon name="info" size={12} />
                      Overview is a mental model
                      <Icon
                        name="x"
                        size={11}
                        style={{ marginLeft: "auto", flex: "none" }}
                      />
                    </button>
                    {caveats.map((c, i) => (
                      <div className="crow" key={i}>
                        {c}
                      </div>
                    ))}
                  </div>
                )}

                {/* HUD */}
                <div className="hud">
                  <div className="pill">
                    <span className="k">domain</span>
                    <b>
                      {fmt(model.domain.width)}×{fmt(model.domain.depth)} m
                    </b>
                  </div>
                  <div className="pill">
                    <span className="k">modelled</span>
                    <b>{fmt(layersDepth(model))} m</b>
                    <span className="k">
                      {model.layers.length}L · {model.targets.length}T
                    </span>
                  </div>
                  <div className="pill">
                    <span className="k">f</span>
                    <b>{model.acquisition.frequency} GHz</b>
                    <span className="k">
                      {
                        ANTENNAS.find(
                          (a) => a.id === model.acquisition.antenna,
                        )?.label.split(" ")[0]
                      }
                    </span>
                  </div>
                </div>

                {/* legend */}
                {(usedMats.length > 0 || usedTargets.length > 0) && (
                  <div className="legend">
                    <div className="lh">Materials</div>
                    {usedMats.map((k) => (
                      <div className="lrow" key={k}>
                        <span
                          className="ls"
                          style={{ background: MATERIALS[k].color }}
                        ></span>
                        <span className="lname">
                          {MATERIALS[k].label.split(" / ")[0]}
                        </span>
                        <span className="leps mono">
                          εr {MATERIALS[k].epsilon}
                        </span>
                      </div>
                    ))}
                    {usedTargets.length > 0 && (
                      <div className="lh" style={{ marginTop: 8 }}>
                        Targets
                      </div>
                    )}
                    {usedTargets.map((k) => (
                      <div className="lrow" key={k}>
                        <span
                          className="ls"
                          style={{
                            background:
                              TARGET_TYPES[k].kind === "pec"
                                ? "#8a9099"
                                : TARGET_TYPES[k].color,
                            borderRadius: "50%",
                          }}
                        ></span>
                        <span className="lname">{TARGET_TYPES[k].label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <Dock
                model={model}
                selected={selected}
                collapsed={dockCollapsed}
                setCollapsed={setDockCollapsed}
                height="clamp(170px, 30vh, 300px)"
              />
            </div>
          </div>
        </div>

        {/* chat */}
        <ChatPane
          model={model}
          modelRef={modelRef}
          setModel={setModel}
          onModelUpdate={onModelUpdate}
          onDatasetReady={onDatasetReady}
          onSimulationEvent={onSimulationEvent}
          onRun={runForward}
          activeModel={activeModel}
          collapsed={!chatOpen}
          setCollapsed={(v) =>
            setChatOpen(typeof v === "function" ? !v(!chatOpen) : !v)
          }
          toast={toast}
          sessionId={currentChatId}
          userId={userId}
          chats={chats}
          onSelectChat={selectChat}
          onNewChat={newChat}
          onSwitchUser={switchUser}
        />
      </div>

      {/* reopen chat tab when collapsed */}
      {!chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          title="Open assistant"
          style={{
            position: "fixed",
            right: 18,
            bottom: 18,
            zIndex: 50,
            height: 44,
            padding: "0 16px",
            borderRadius: 24,
            border: "1px solid var(--accent-2)",
            background: "var(--accent)",
            color: "#fff",
            display: "flex",
            alignItems: "center",
            gap: 9,
            fontWeight: 600,
            fontSize: 13,
            boxShadow: "var(--shadow-lg)",
          }}
        >
          <Icon name="sparkles" size={16} />
          Ask the assistant
        </button>
      )}

      {modal && (
        <ExportModal
          kind={modal}
          model={model}
          onClose={() => setModal(null)}
          toast={toast}
        />
      )}
      <Toasts items={toasts} />
      {!userId && <UserGate onSubmit={onUserSubmit} />}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
