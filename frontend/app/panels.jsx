/* ============================================================
   NL2Sim — chrome: menu bar, model tree, dock/inspector, modals
   ============================================================ */

/* ---------- model mutation helpers ---------- */
const updLayer = (setModel, id, patch) =>
  setModel((m) => ({
    ...m,
    layers: m.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)),
  }));
const updTarget = (setModel, id, patch) =>
  setModel((m) => ({
    ...m,
    targets: m.targets.map((t) => (t.id === id ? { ...t, ...patch } : t)),
  }));

/* ---------- gprMax input deck generator ---------- */
function generateGprMax(model) {
  const a = model.acquisition,
    d = model.domain;
  const L = [];
  L.push(`#title: NL2Sim — ${model.project}`);
  L.push(`#domain: ${fmt(d.width, 3)} ${fmt(d.depth, 3)} ${fmt(d.dx, 4)}`);
  L.push(`#dx_dy_dz: ${fmt(d.dx, 4)} ${fmt(d.dx, 4)} ${fmt(d.dx, 4)}`);
  L.push(`#time_window: ${a.timeWindow}e-9`);
  L.push("");
  L.push(`#waveform: ${a.waveform} 1 ${a.frequency}e9 src_wave`);
  L.push(`#hertzian_dipole: z 0.040 ${fmt(d.depth - 0.01, 3)} 0 src_wave`);
  L.push(`#rx: ${fmt(0.04 + a.txrxSep, 3)} ${fmt(d.depth - 0.01, 3)} 0`);
  L.push(`#src_steps: ${fmt(a.traceStep, 3)} 0 0`);
  L.push(`#rx_steps: ${fmt(a.traceStep, 3)} 0 0`);
  L.push("");
  let acc = 0;
  model.layers.forEach((l, i) => {
    const m = MATERIALS[l.material];
    L.push(`#material: ${l.epsilon} ${l.sigma} 1 0 ${l.material}`);
    L.push(
      `#box: 0 ${fmt(d.depth - (acc + l.thickness), 3)} 0 ${fmt(d.width, 3)} ${fmt(d.depth - acc, 3)} ${fmt(d.dx, 4)} ${l.material}`,
    );
    acc += l.thickness;
  });
  L.push("");
  model.targets.forEach((t) => {
    const tt = TARGET_TYPES[t.type];
    const yc = fmt(d.depth - t.depth, 3);
    if (tt.kind === "pec") {
      L.push(
        `#cylinder: ${fmt(t.x, 3)} ${yc} 0 ${fmt(t.x, 3)} ${yc} ${fmt(d.dx, 4)} ${fmt(t.diameter / 2, 3)} pec`,
      );
    } else {
      L.push(`#material: ${tt.epsilon} ${tt.sigma} 1 0 ${t.type}`);
      L.push(
        `#cylinder: ${fmt(t.x, 3)} ${yc} 0 ${fmt(t.x, 3)} ${yc} ${fmt(d.dx, 4)} ${fmt(t.diameter / 2, 3)} ${t.type}`,
      );
    }
  });
  L.push("");
  L.push(
    `#geometry_view: 0 0 0 ${fmt(d.width, 3)} ${fmt(d.depth, 3)} ${fmt(d.dx, 4)} ${fmt(d.dx, 4)} ${fmt(d.dx, 4)} ${fmt(d.dx, 4)} ${model.project} n`,
  );
  return L.join("\n");
}

/* ---------- toasts ---------- */
function Toasts({ items }) {
  return (
    <div className="toasts">
      {items.map((t) => (
        <div key={t.id} className={"toast " + (t.kind || "info")}>
          <Icon name={t.kind === "ok" ? "check" : "info"} size={15} />
          <span dangerouslySetInnerHTML={{ __html: t.msg }} />
        </div>
      ))}
    </div>
  );
}

/* ============================================================
   MENU BAR
   ============================================================ */
function MenuBar(props) {
  const {
    model,
    setModel,
    activeModel,
    setActiveModel,
    datasetCount,
    chatOpen,
    setChatOpen,
    toast,
    openModal,
  } = props;
  const [open, setOpen] = React.useState(null);
  // Hidden file input backing "Upload → From file…" (zip of gprMax .in decks).
  const zipInputRef = React.useRef(null);

  // Download all generated gprMax .in files as a zip from the backend.
  const downloadInputDeck = () => {
    if (!datasetCount) {
      toast("Generate a dataset first to download the input deck", "info");
      return;
    }
    const base = window.getApiHttpBase ? window.getApiHttpBase() : "";
    const sid = window.getSessionId ? window.getSessionId() : "";
    if (!sid) {
      toast("Select a chat first", "info");
      return;
    }
    const a = document.createElement("a");
    a.href = `${base}/datasets/${sid}/download`;
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast("Downloading <b>gprMax input deck</b> · .zip", "ok");
  };
  const close = () => setOpen(null);
  React.useEffect(() => {
    const h = () => close();
    if (open) {
      window.addEventListener("click", h);
      return () => window.removeEventListener("click", h);
    }
  }, [open]);
  const stop = (e) => e.stopPropagation();
  const am = ML_MODELS.find((m) => m.id === activeModel) || ML_MODELS[0];

  return (
    <header className="appbar" onClick={close}>
      <div className="brand">
        <svg className="mark" viewBox="0 0 24 24" fill="none">
          <rect x="1" y="1" width="22" height="22" rx="6" fill="#2f6bd4" />
          <path
            d="M5 15 Q9 9 12 15 T19 15"
            stroke="#fff"
            strokeWidth="1.7"
            fill="none"
            strokeLinecap="round"
          />
          <circle cx="12" cy="7.5" r="1.6" fill="#fff" />
          <path
            d="M12 9 Q12 11 12 12.5"
            stroke="#bcd4f6"
            strokeWidth="1.3"
            strokeLinecap="round"
          />
        </svg>
        <div>
          <div className="name">
            NL2<b>Sim</b>
          </div>
          <div className="tag">GPR Scenario Studio</div>
        </div>
      </div>

      <div className="menu" onClick={stop}>
        {/* Upload — zip of gprMax .in files, straight to the file picker */}
        <div style={{ position: "relative" }}>
          <button
            className="mbtn"
            title="Import a .zip of gprMax .in files"
            onClick={() => {
              close();
              if (zipInputRef.current) zipInputRef.current.click();
            }}
          >
            <Icon name="upload" className="ic" />
            Upload
          </button>
          <input
            ref={zipInputRef}
            type="file"
            accept=".zip,application/zip"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files && e.target.files[0];
              e.target.value = ""; // allow re-selecting the same zip
              if (f && props.onUploadZip) props.onUploadZip(f);
            }}
          />
        </div>

        {/* Download */}
        <div style={{ position: "relative" }}>
          <button
            className="mbtn"
            onClick={() => setOpen(open === "dn" ? null : "dn")}
          >
            <Icon name="download" className="ic" />
            Download
            <Icon name="caret" className="caret" />
          </button>
          {open === "dn" && (
            <div className="pop" onClick={stop}>
              <div className="head">Export</div>
              <button
                className="item"
                onClick={() => {
                  close();
                  downloadInputDeck();
                }}
              >
                <Icon name="cpu" className="ic" />
                <div className="col">
                  <span>gprMax input deck</span>
                  <span className="sub">
                    {datasetCount
                      ? datasetCount + " generated .in files"
                      : "generate a dataset first"}
                  </span>
                </div>
                <span className="meta">.zip</span>
              </button>
              <button className="item" disabled title="Coming soon">
                <Icon name="table" className="ic" />
                <div className="col">
                  <span>Dataset labels</span>
                  <span className="sub">not yet available</span>
                </div>
                <span className="meta">.csv</span>
              </button>
            </div>
          )}
        </div>

        <div className="mdiv"></div>

        {/* ML model selection */}
        <div style={{ position: "relative" }}>
          <button
            className={"mbtn" + (open === "ml" ? " on" : "")}
            onClick={() => setOpen(open === "ml" ? null : "ml")}
          >
            <Icon name="cpu" className="ic" />
            {am.label}
            <Icon name="caret" className="caret" />
          </button>
          {open === "ml" && (
            <div className="pop" style={{ minWidth: 288 }} onClick={stop}>
              <div className="head">Forward model · solver</div>
              {ML_MODELS.map((m) => (
                <button
                  key={m.id}
                  className={"item" + (m.id === activeModel ? " sel" : "")}
                  disabled={!m.available}
                  title={m.available ? undefined : "Not available"}
                  onClick={() => {
                    setActiveModel(m.id);
                    close();
                    toast("Forward model → <b>" + m.label + "</b>", "info");
                  }}
                >
                  <Icon
                    name={m.id === activeModel ? "check" : "cpu"}
                    className={"ic " + (m.id === activeModel ? "check" : "")}
                  />
                  <div className="col">
                    <span>{m.label}</span>
                    <span className="sub">
                      {m.arch} · {m.desc}
                    </span>
                  </div>
                  <span className="meta">
                    {m.available ? "" : "Not available"}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* AI assistant */}
        <button
          className={"mbtn" + (chatOpen ? " on" : "")}
          onClick={() => setChatOpen((o) => !o)}
        >
          <Icon name="message" className="ic" />
          AI Assistant
        </button>
      </div>

      <div className="spacer"></div>
    </header>
  );
}

/* ============================================================
   MODEL TREE (left rail)
   ============================================================ */
function ModelTree({
  model,
  setModel,
  selected,
  onSelect,
  collapsed,
  toast,
  railTab,
  setRailTab,
  datasetFiles,
  onOpenFile,
  activeFile,
  outFiles,
  onOpenOutcome,
}) {
  const [openL, setOpenL] = React.useState(true);
  const [openT, setOpenT] = React.useState(true);
  const addLayer = () => {
    const id = uid("ly");
    setModel((m) => ({
      ...m,
      layers: [
        ...m.layers,
        {
          id,
          name: "New Layer",
          material: "drysand",
          thickness: 0.2,
          epsilon: 4,
          sigma: 0.001,
          visible: true,
        },
      ],
    }));
    onSelect({ type: "layer", id });
    toast("Added layer", "ok");
  };
  const addTarget = () => {
    const id = uid("tg");
    setModel((m) => ({
      ...m,
      targets: [
        ...m.targets,
        {
          id,
          name: "New Target",
          type: "metalpipe",
          x: round(m.domain.width / 2, 2),
          depth: 0.3,
          diameter: 0.08,
          visible: true,
        },
      ],
    }));
    onSelect({ type: "target", id });
    toast("Added target", "ok");
  };

  const files = datasetFiles || [];

  return (
    <aside className={"rail" + (collapsed ? " collapsed" : "")}>
      <div className="rail-tabs">
        <button
          className={"rail-tab" + (railTab === "model" ? " on" : "")}
          onClick={() => setRailTab("model")}
        >
          <Icon name="layers" size={13} />
          Model
        </button>
        <button
          className={"rail-tab" + (railTab === "dataset" ? " on" : "")}
          onClick={() => setRailTab("dataset")}
        >
          <Icon name="database" size={13} />
          Dataset
          {files.length > 0 && <span className="rail-tab-count">{files.length}</span>}
        </button>
      </div>

      {railTab === "dataset" ? (
        <div className="rail-scroll">
          <div className="tree-sec">
            <div className="tree-sec-head" style={{ cursor: "default" }}>
              <Icon name="cpu" size={13} style={{ opacity: 0.7 }} />
              Samples
              <span className="count mono">{files.length}</span>
            </div>
            {files.length === 0 && (
              <div
                style={{
                  padding: "8px 10px",
                  fontSize: 11.5,
                  color: "var(--muted)",
                }}
              >
                No dataset yet — ask the assistant to generate one.
              </div>
            )}
            {files.map((f) => (
              <div
                key={f.filename}
                className={"tnode" + (activeFile === f.filename ? " sel" : "")}
                onClick={() => onOpenFile(f.filename)}
                title={f.filename}
              >
                <Icon name="cpu" size={13} style={{ opacity: 0.6 }} />
                <span className="tn-name">{f.filename}</span>
                {outFiles && outFiles.has(f.filename) && (
                  <button
                    className="tn-out"
                    title="View simulation outcome (A-scan)"
                    onClick={(e) => {
                      e.stopPropagation();
                      onOpenOutcome && onOpenOutcome(f.filename);
                    }}
                  >
                    <Icon name="wave" size={12} />
                  </button>
                )}
                <span className="tn-meta mono">#{f.sample_id}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
      <div className="rail-scroll">
        {/* domain summary */}
        <div className="tree-sec">
          <div className="tree-sec-head" style={{ cursor: "default" }}>
            <Icon name="fit" className="chev" style={{ transform: "none" }} />
            Domain
          </div>
          <div
            style={{
              padding: "2px 8px 4px",
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
            }}
          >
            <span className="tn-meta mono">
              {fmt(model.domain.width)}×{fmt(model.domain.depth)} m
            </span>
            <span className="tn-meta mono">
              Δ{(model.domain.dx * 1000).toFixed(1)}mm
            </span>
          </div>
        </div>

        {/* layers */}
        <div className="tree-sec">
          <div
            className={"tree-sec-head" + (openL ? "" : " closed")}
            onClick={() => setOpenL((o) => !o)}
          >
            <Icon name="caret" className="chev" />
            <Icon name="layers" size={13} style={{ opacity: 0.7 }} />
            Soil Layers
            <button
              className="add-mini"
              onClick={(e) => {
                e.stopPropagation();
                addLayer();
              }}
            >
              <Icon name="plus" size={13} />
            </button>
            <span className="count mono">{model.layers.length}</span>
          </div>
          {openL &&
            model.layers.map((l, i) => {
              const mat = MATERIALS[l.material] || { color: "#ccc" };
              return (
                <div
                  key={l.id}
                  className={
                    "tnode" + (selected && selected.id === l.id ? " sel" : "")
                  }
                  onClick={() => onSelect({ type: "layer", id: l.id })}
                >
                  <span className="sw" style={{ background: mat.color }}></span>
                  <span className="tn-name">{l.name}</span>
                  <span className="tn-meta">{fmt(l.thickness)}m</span>
                  <button
                    className={"vis" + (l.visible === false ? " off" : "")}
                    onClick={(e) => {
                      e.stopPropagation();
                      updLayer(setModel, l.id, {
                        visible: l.visible === false,
                      });
                    }}
                  >
                    <Icon
                      name={l.visible === false ? "eyeoff" : "eye"}
                      size={13}
                    />
                  </button>
                </div>
              );
            })}
        </div>

        {/* targets */}
        <div className="tree-sec">
          <div
            className={"tree-sec-head" + (openT ? "" : " closed")}
            onClick={() => setOpenT((o) => !o)}
          >
            <Icon name="caret" className="chev" />
            <Icon name="target" size={13} style={{ opacity: 0.7 }} />
            Buried Targets
            <button
              className="add-mini"
              onClick={(e) => {
                e.stopPropagation();
                addTarget();
              }}
            >
              <Icon name="plus" size={13} />
            </button>
            <span className="count mono">{model.targets.length}</span>
          </div>
          {openT &&
            model.targets.map((t) => {
              const tt = TARGET_TYPES[t.type] || { color: "#ccc" };
              return (
                <div
                  key={t.id}
                  className={
                    "tnode" + (selected && selected.id === t.id ? " sel" : "")
                  }
                  onClick={() => onSelect({ type: "target", id: t.id })}
                >
                  <span
                    className="sw"
                    style={{
                      background: tt.kind === "pec" ? "#8a9099" : tt.color,
                      borderRadius: "50%",
                    }}
                  ></span>
                  <span className="tn-name">{t.name}</span>
                  <span className="tn-meta">
                    {(t.kind === "box" ? "box · " : "") + fmt(t.depth) + "m"}
                  </span>
                  <button
                    className={"vis" + (t.visible === false ? " off" : "")}
                    onClick={(e) => {
                      e.stopPropagation();
                      updTarget(setModel, t.id, {
                        visible: t.visible === false,
                      });
                    }}
                  >
                    <Icon
                      name={t.visible === false ? "eyeoff" : "eye"}
                      size={13}
                    />
                  </button>
                </div>
              );
            })}
          {openT && model.targets.length === 0 && (
            <div
              style={{
                padding: "6px 10px",
                fontSize: 11.5,
                color: "var(--muted)",
              }}
            >
              No targets — ask the assistant to bury one.
            </div>
          )}
        </div>
      </div>
      )}
    </aside>
  );
}

/* ============================================================
   DOCK (inspector)
   ============================================================ */
function ReadField({ label, value, unit }) {
  const displayValue = value ?? "not available";
  return (
    <div className="fld">
      <label>{label}</label>
      <div className="ctl read-ctl">
        <div className="readout">{displayValue}</div>
        {unit && displayValue !== "not available" && (
          <span className="unit">{unit}</span>
        )}
      </div>
    </div>
  );
}

function formatReadNumber(value, digits = 2) {
  return value == null || Number.isNaN(Number(value))
    ? "not available"
    : fmt(Number(value), digits);
}

function Dock({
  model,
  selected,
  collapsed,
  setCollapsed,
  height,
}) {
  const layer =
    selected && selected.type === "layer"
      ? model.layers.find((l) => l.id === selected.id)
      : null;
  const target =
    selected && selected.type === "target"
      ? model.targets.find((t) => t.id === selected.id)
      : null;

  return (
    <div className="dock" style={{ height: collapsed ? 38 : height }}>
      <div className="dock-tabs">
        <button
          className="dtab on"
          onClick={() => {
            setCollapsed(false);
          }}
        >
          <Icon name="info" className="ic" />
          Inspector
        </button>
        <button
          className="dock-collapse"
          onClick={() => setCollapsed((c) => !c)}
          title={collapsed ? "Expand" : "Collapse"}
        >
          <Icon
            name="caret"
            size={14}
            style={{ transform: collapsed ? "rotate(180deg)" : "none" }}
          />
        </button>
      </div>
      {!collapsed && (
        <div className="dock-body">
          <Inspector layer={layer} target={target} />
        </div>
      )}
    </div>
  );
}

function Inspector({ layer, target }) {
  if (layer) {
    const mat = MATERIALS[layer.material] || {
      color: "#ccc",
      label: layer.material,
    };
    const velocity =
      layer.epsilon == null || Number(layer.epsilon) <= 0
        ? null
        : 0.3 / Math.sqrt(Number(layer.epsilon));
    return (
      <div className="insp">
        <div className="insp-title">
          <span className="sw" style={{ background: mat.color }}></span>
          <h4>{layer.name}</h4>
          <span className="badge">soil layer</span>
        </div>
        <ReadField label="Name" value={layer.name} />
        <ReadField label="Material" value={mat.label || layer.material} />
        <ReadField
          label="Thickness"
          value={formatReadNumber(layer.thickness, 3)}
          unit="m"
        />
        {(layer.thicknessMin != null || layer.thicknessMax != null) && (
          <ReadField
            label="Thickness range"
            value={`${formatReadNumber(layer.thicknessMin, 3)} - ${formatReadNumber(layer.thicknessMax, 3)}`}
            unit="m"
          />
        )}
        <ReadField
          label="Permittivity εr"
          value={formatReadNumber(layer.epsilon, 1)}
        />
        <ReadField
          label="Conductivity σ"
          value={formatReadNumber(layer.sigma, 4)}
          unit="S/m"
        />
        <ReadField
          label="Wave velocity"
          value={formatReadNumber(velocity, 3)}
          unit="m/ns"
        />
      </div>
    );
  }
  if (target) {
    const tt = TARGET_TYPES[target.type] || {
      color: "#ccc",
      label: target.type,
      kind: target.material === "pec" ? "pec" : "diel",
      epsilon: target.material === "pec" ? 0 : null,
    };
    const isBox = target.kind === "box" || target.shape === "rect";
    return (
      <div className="insp">
        <div className="insp-title">
          <span
            className="sw"
            style={{
              background: tt.kind === "pec" ? "#8a9099" : tt.color,
              borderRadius: "50%",
            }}
          ></span>
          <h4>{target.name}</h4>
          <span className="badge">
            {tt.kind === "pec" ? "conductor" : "dielectric"}
          </span>
        </div>
        <ReadField label="Name" value={target.name} />
        <ReadField label="Object type" value={tt.label || target.type} />
        <ReadField label="Geometry" value={isBox ? "box" : "cylinder"} />
        <ReadField
          label="Offset x"
          value={formatReadNumber(target.x, 3)}
          unit="m"
        />
        <ReadField
          label="Depth"
          value={formatReadNumber(target.depth, 3)}
          unit="m"
        />
        {isBox ? (
          <React.Fragment>
            <ReadField
              label="Width"
              value={formatReadNumber(target.width, 3)}
              unit="m"
            />
            <ReadField
              label="Height"
              value={formatReadNumber(target.height, 3)}
              unit="m"
            />
          </React.Fragment>
        ) : (
          <ReadField
            label="Diameter"
            value={formatReadNumber(target.diameter, 3)}
            unit="m"
          />
        )}
        <ReadField
          label="EM property"
          value={
            tt.kind === "pec"
              ? "perfect conductor"
              : "εr " + formatReadNumber(tt.epsilon, 1)
          }
        />
      </div>
    );
  }
  return (
    <div className="insp-empty">
      <Icon name="info" className="ic" />
      <div>Select a layer or target to view its properties</div>
    </div>
  );
}

/* ---------- export modal ---------- */
function ExportModal({ kind, model, onClose, toast }) {
  const isGpr = kind === "gprmax";
  const code = isGpr
    ? generateGprMax(model)
    : JSON.stringify(
        model,
        (k, v) => (typeof v === "number" ? round(v, 4) : v),
        2,
      );
  const fname = isGpr ? model.project + ".in" : model.project + ".json";
  const html = code
    .replace(/(#[a-z_]+:)/g, '<span class="k">$1</span>')
    .replace(/(\/\/.*$)/gm, '<span class="c">$1</span>');
  return (
    <div className="scrim" onClick={onClose}>
      <div
        className="modal"
        style={{ width: 600 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mh">
          <Icon
            name={isGpr ? "cpu" : "database"}
            size={18}
            style={{ color: "var(--accent)" }}
          />
          <h3>{isGpr ? "gprMax input deck" : "Scenario definition"}</h3>
          <span
            className="badge mono"
            style={{
              fontFamily: "var(--mono)",
              fontSize: 11,
              color: "var(--ink-3)",
            }}
          >
            {fname}
          </span>
          <button className="x" onClick={onClose}>
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="mb">
          <pre className="codeblk" dangerouslySetInnerHTML={{ __html: html }} />
        </div>
        <div className="mf">
          <button className="btn" onClick={onClose}>
            Close
          </button>
          <button
            className="btn primary"
            onClick={() => {
              toast("Downloaded <b>" + fname + "</b>", "ok");
              onClose();
            }}
          >
            <Icon
              name="download"
              size={14}
              style={{ marginRight: 6, verticalAlign: "-2px" }}
            />
            Download {fname}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------- generated .in file viewer (covers the canvas) ---------- */
function DatasetFileView({ view, onClose, hasOutcome, onOpenOutcome }) {
  const html = String(view.content || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/(#[a-z_]+:)/g, '<span class="k">$1</span>')
    .replace(/(--.*$)/gm, '<span class="c">$1</span>');
  return (
    <div className="fileview">
      <div className="fileview-head">
        <Icon name="cpu" size={15} style={{ color: "var(--accent)" }} />
        <span className="fileview-name mono">{view.filename}</span>
        <span className="badge mono">.in</span>
        {hasOutcome && (
          <button
            className="hbtn outcome-btn"
            title="View the simulated receiver waveform for this file"
            onClick={() => onOpenOutcome && onOpenOutcome(view.filename)}
          >
            <Icon name="wave" size={13} />
            View outcome
          </button>
        )}
        <div className="spacer" style={{ flex: 1 }} />
        <button className="hbtn" title="Close" onClick={onClose}>
          <Icon name="x" size={15} />
        </button>
      </div>
      <div className="fileview-body">
        {view.loading ? (
          <div className="fileview-empty">Loading {view.filename}…</div>
        ) : (
          <pre className="codeblk" dangerouslySetInnerHTML={{ __html: html }} />
        )}
      </div>
    </div>
  );
}

/* ---------- forward-model outcome viewer (A-scan, covers the canvas) ---- */
function fmtAmp(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000 || a < 0.01) return v.toExponential(1);
  return String(round(v, a >= 10 ? 1 : 3));
}

function AscanPlot({ samples, dt, component }) {
  const W = 920;
  const H = 430;
  const padL = 70;
  const padR = 16;
  const padT = 16;
  const padB = 42;
  const n = samples.length;
  // stride so the polyline stays light even for long time windows
  const stride = Math.max(1, Math.ceil(n / 1600));
  const pts = [];
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i += stride) {
    const v = samples[i];
    if (v < min) min = v;
    if (v > max) max = v;
    pts.push([i, v]);
  }
  if (!isFinite(min)) {
    min = -1;
    max = 1;
  }
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const head = (max - min) * 0.06;
  min -= head;
  max += head;
  const tTotalNs = (n - 1) * dt * 1e9;
  const x = (i) => padL + (i / Math.max(1, n - 1)) * (W - padL - padR);
  const y = (v) => padT + ((max - v) / (max - min)) * (H - padT - padB);
  const poly = pts
    .map(([i, v]) => x(i).toFixed(1) + "," + y(v).toFixed(1))
    .join(" ");
  const unit = component && component[0] === "H" ? "A/m" : "V/m";
  const xTicks = [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => ({
    px: padL + f * (W - padL - padR),
    label: round(f * tTotalNs, 2),
  }));
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const v = min + f * (max - min);
    return { py: y(v), label: fmtAmp(v) };
  });
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ width: "100%", height: "100%", display: "block" }}
    >
      {yTicks.map((t, i) => (
        <g key={"y" + i}>
          <line
            x1={padL}
            x2={W - padR}
            y1={t.py}
            y2={t.py}
            stroke="var(--line)"
            strokeWidth="1"
            opacity="0.5"
          />
          <text
            x={padL - 8}
            y={t.py + 3.5}
            textAnchor="end"
            fontSize="11"
            fill="var(--muted)"
            className="mono"
          >
            {t.label}
          </text>
        </g>
      ))}
      {xTicks.map((t, i) => (
        <g key={"x" + i}>
          <line
            x1={t.px}
            x2={t.px}
            y1={padT}
            y2={H - padB}
            stroke="var(--line)"
            strokeWidth="1"
            opacity="0.35"
          />
          <text
            x={t.px}
            y={H - padB + 16}
            textAnchor="middle"
            fontSize="11"
            fill="var(--muted)"
            className="mono"
          >
            {t.label}
          </text>
        </g>
      ))}
      {min < 0 && max > 0 && (
        <line
          x1={padL}
          x2={W - padR}
          y1={y(0)}
          y2={y(0)}
          stroke="var(--muted)"
          strokeWidth="1"
          strokeDasharray="4 3"
          opacity="0.6"
        />
      )}
      <text
        x={(padL + W - padR) / 2}
        y={H - 8}
        textAnchor="middle"
        fontSize="11.5"
        fill="var(--muted)"
      >
        time (ns)
      </text>
      <text
        x={14}
        y={(padT + H - padB) / 2}
        textAnchor="middle"
        fontSize="11.5"
        fill="var(--muted)"
        transform={`rotate(-90 14 ${(padT + H - padB) / 2})`}
      >
        {component} ({unit})
      </text>
      <polyline
        points={poly}
        fill="none"
        stroke="var(--accent)"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function DatasetOutcomeView({ view, onClose }) {
  const comps = view.data ? Object.keys(view.data.components || {}) : [];
  const [comp, setComp] = React.useState(null);
  // fall back gracefully when the chosen component isn't in this file
  const active =
    comp && comps.includes(comp)
      ? comp
      : comps.includes("Ez")
        ? "Ez"
        : comps[0];
  const outName = view.data?.filename || view.filename;
  return (
    <div className="fileview">
      <div className="fileview-head">
        <Icon name="wave" size={15} style={{ color: "var(--accent)" }} />
        <span className="fileview-name mono">{outName}</span>
        <span className="badge mono">.out</span>
        {view.data && <span className="badge">{view.data.receiver || "rx1"} · {view.data.qualification?.status || "legacy unverified"}</span>}
        {view.data?.geometry_url && <a className="hbtn" href={`${window.getApiHttpBase()}${view.data.geometry_url}`} download title="Download the native voxel material map (HDF5)">Native geometry</a>}
        {comps.length > 1 && (
          <select
            className="vsel"
            value={active}
            onChange={(e) => setComp(e.target.value)}
            title="Receiver field component"
          >
            {comps.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}
        {view.data && (
          <span className="tn-meta mono" style={{ marginLeft: 8 }}>
            {view.data.iterations} it · Δt {round(view.data.dt * 1e12, 3)} ps ·{" "}
            {round((view.data.iterations - 1) * view.data.dt * 1e9, 2)} ns
          </span>
        )}
        <div className="spacer" style={{ flex: 1 }} />
        <button className="hbtn" title="Close" onClick={onClose}>
          <Icon name="x" size={15} />
        </button>
      </div>
      <div className="fileview-body">
        {view.loading ? (
          <div className="fileview-empty">Loading {view.filename} output…</div>
        ) : view.error ? (
          <div className="fileview-empty">{view.error}</div>
        ) : !active ? (
          <div className="fileview-empty">
            No receiver components found in {outName}.
          </div>
        ) : (
          <div className="ascan-wrap">
            <AscanPlot
              samples={view.data.components[active]}
              dt={view.data.dt}
              component={active}
            />
          </div>
        )}
      </div>
    </div>
  );
}

Object.assign(window, {
  MenuBar,
  ModelTree,
  Dock,
  Inspector,
  Toasts,
  ExportModal,
  DatasetFileView,
  DatasetOutcomeView,
  generateGprMax,
  updLayer,
  updTarget,
});
