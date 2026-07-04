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
const updAcq = (setModel, patch) =>
  setModel((m) => ({ ...m, acquisition: { ...m.acquisition, ...patch } }));
const updDomain = (setModel, patch) =>
  setModel((m) => ({ ...m, domain: { ...m.domain, ...patch } }));

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
    dataset,
    datasetCount,
    chatOpen,
    setChatOpen,
    toast,
    openModal,
    onManual,
  } = props;
  const [open, setOpen] = React.useState(null);

  // Download all generated gprMax .in files as a zip from the backend.
  const downloadInputDeck = () => {
    if (!datasetCount) {
      toast("Generate a dataset first to download the input deck", "info");
      return;
    }
    const base = window.getApiHttpBase ? window.getApiHttpBase() : "";
    const sid = window.getSessionId ? window.getSessionId() : "";
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
        {/* Upload */}
        <div style={{ position: "relative" }}>
          <button
            className="mbtn"
            onClick={() => setOpen(open === "up" ? null : "up")}
          >
            <Icon name="upload" className="ic" />
            Upload
            <Icon name="caret" className="caret" />
          </button>
          {open === "up" && (
            <div className="pop" onClick={stop}>
              <div className="head">Import scenario</div>
              <button
                className="item"
                onClick={() => {
                  close();
                  props.onLoadPreset("utility");
                  toast("Loaded <b>utility_survey_01</b>", "ok");
                }}
              >
                <Icon name="layers" className="ic" />
                <div className="col">
                  <span>Buried utility survey</span>
                  <span className="sub">3 layers · 2 targets</span>
                </div>
              </button>
              <button
                className="item"
                onClick={() => {
                  close();
                  props.onLoadPreset("rebar");
                  toast("Loaded <b>slab_inspection_01</b>", "ok");
                }}
              >
                <Icon name="grid" className="ic" />
                <div className="col">
                  <span>Concrete slab + rebar</span>
                  <span className="sub">1.5 GHz inspection</span>
                </div>
              </button>
              <button
                className="item"
                onClick={() => {
                  close();
                  props.onLoadPreset("mine");
                  toast("Loaded <b>demining_01</b>", "ok");
                }}
              >
                <Icon name="target" className="ic" />
                <div className="col">
                  <span>Landmine detection</span>
                  <span className="sub">900 MHz · sandy soil</span>
                </div>
              </button>
              <div className="sep"></div>
              <button
                className="item"
                onClick={() => {
                  close();
                  toast(
                    "Drop a <b>.json</b> or gprMax <b>.in</b> file to import",
                    "info",
                  );
                }}
              >
                <Icon name="upload" className="ic" />
                <div className="col">
                  <span>From file…</span>
                  <span className="sub">.json · .in</span>
                </div>
              </button>
            </div>
          )}
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
              <div className="head">Target model · dataset feeds</div>
              {ML_MODELS.map((m) => (
                <button
                  key={m.id}
                  className={"item" + (m.id === activeModel ? " sel" : "")}
                  onClick={() => {
                    setActiveModel(m.id);
                    close();
                    toast("Target model → <b>" + m.label + "</b>", "info");
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
                  <span className="meta">{m.samples}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="mdiv"></div>

        {/* Manual input */}
        <button
          className="mbtn"
          onClick={() => {
            onManual();
            toast(
              "Manual input — edit any parameter in the <b>Parameters</b> panel",
              "info",
            );
          }}
        >
          <Icon name="edit" className="ic" />
          Manual Input
        </button>

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

      <div className="statuschip" title="Solver engine">
        <span className="dot"></span>gprMax&nbsp;<b>v3.1</b>&nbsp;ready
      </div>
      <div className="statuschip">
        <Icon name="database" size={13} style={{ opacity: 0.6 }} />
        Dataset&nbsp;<b>{dataset.toLocaleString()}</b>
      </div>
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
                  <span className="tn-meta">{fmt(t.depth)}m</span>
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
   DOCK (inspector / parameters / radargram)
   ============================================================ */
function NumField({ label, value, unit, step, onChange, disabled }) {
  return (
    <div className="fld">
      <label>{label}</label>
      <div className="ctl">
        <input
          className="inp"
          type="number"
          step={step || 0.01}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        {unit && <span className="unit">{unit}</span>}
      </div>
    </div>
  );
}

function Dock({
  model,
  setModel,
  selected,
  onSelect,
  tab,
  setTab,
  solved,
  progress,
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
          className={"dtab" + (tab === "inspector" ? " on" : "")}
          onClick={() => {
            setTab("inspector");
            setCollapsed(false);
          }}
        >
          <Icon name="edit" className="ic" />
          Inspector
        </button>
        <button
          className={"dtab" + (tab === "acq" ? " on" : "")}
          onClick={() => {
            setTab("acq");
            setCollapsed(false);
          }}
        >
          <Icon name="settings" className="ic" />
          Parameters
        </button>
        <button
          className={"dtab" + (tab === "radar" ? " on" : "")}
          onClick={() => {
            setTab("radar");
            setCollapsed(false);
          }}
        >
          <Icon name="radar" className="ic" />
          Radargram
          {solved && (
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "var(--green)",
                marginLeft: 2,
              }}
            ></span>
          )}
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
          {tab === "inspector" && (
            <Inspector
              layer={layer}
              target={target}
              model={model}
              setModel={setModel}
              onSelect={onSelect}
            />
          )}
          {tab === "acq" && <Acquisition model={model} setModel={setModel} />}
          {tab === "radar" && (
            <Radargram model={model} solved={solved} progress={progress} />
          )}
        </div>
      )}
    </div>
  );
}

function Inspector({ layer, target, model, setModel, onSelect }) {
  if (layer) {
    const mat = MATERIALS[layer.material];
    return (
      <div className="insp">
        <div className="insp-title">
          <span className="sw" style={{ background: mat.color }}></span>
          <h4>{layer.name}</h4>
          <span className="badge">soil layer</span>
          <button
            className="del"
            onClick={() => {
              setModel((m) => ({
                ...m,
                layers: m.layers.filter((l) => l.id !== layer.id),
              }));
              onSelect(null);
            }}
          >
            <Icon name="trash" size={13} />
            Delete
          </button>
        </div>
        <div className="fld">
          <label>Name</label>
          <input
            className="inp"
            style={{ fontFamily: "var(--sans)" }}
            value={layer.name}
            onChange={(e) =>
              updLayer(setModel, layer.id, { name: e.target.value })
            }
          />
        </div>
        <div className="fld">
          <label>Material</label>
          <select
            className="sel-ctl"
            value={layer.material}
            onChange={(e) => {
              const k = e.target.value;
              const mm = MATERIALS[k];
              updLayer(setModel, layer.id, {
                material: k,
                epsilon: mm.epsilon,
                sigma: mm.sigma,
                name: mm.label.split(" / ")[0],
              });
            }}
          >
            {MAT_KEYS.map((k) => (
              <option key={k} value={k}>
                {MATERIALS[k].label}
              </option>
            ))}
          </select>
        </div>
        <NumField
          label="Thickness"
          value={layer.thickness}
          unit="m"
          step={0.01}
          onChange={(v) =>
            updLayer(setModel, layer.id, {
              thickness: clamp(v || 0.01, 0.01, 5),
            })
          }
        />
        <NumField
          label="Permittivity εr"
          value={layer.epsilon}
          step={0.5}
          onChange={(v) =>
            updLayer(setModel, layer.id, { epsilon: clamp(v || 1, 1, 90) })
          }
        />
        <NumField
          label="Conductivity σ"
          value={layer.sigma}
          unit="S/m"
          step={0.001}
          onChange={(v) =>
            updLayer(setModel, layer.id, { sigma: clamp(v || 0, 0, 10) })
          }
        />
        <div className="fld">
          <label>Wave velocity</label>
          <div className="ctl">
            <input
              className="inp"
              disabled
              value={fmt(0.3 / Math.sqrt(layer.epsilon), 3)}
            />
            <span className="unit">m/ns</span>
          </div>
        </div>
      </div>
    );
  }
  if (target) {
    const tt = TARGET_TYPES[target.type];
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
          <button
            className="del"
            onClick={() => {
              setModel((m) => ({
                ...m,
                targets: m.targets.filter((t) => t.id !== target.id),
              }));
              onSelect(null);
            }}
          >
            <Icon name="trash" size={13} />
            Delete
          </button>
        </div>
        <div className="fld">
          <label>Name</label>
          <input
            className="inp"
            style={{ fontFamily: "var(--sans)" }}
            value={target.name}
            onChange={(e) =>
              updTarget(setModel, target.id, { name: e.target.value })
            }
          />
        </div>
        <div className="fld">
          <label>Object type</label>
          <select
            className="sel-ctl"
            value={target.type}
            onChange={(e) => {
              const k = e.target.value;
              updTarget(setModel, target.id, {
                type: k,
                name: TARGET_TYPES[k].label,
              });
            }}
          >
            {TARGET_KEYS.map((k) => (
              <option key={k} value={k}>
                {TARGET_TYPES[k].label}
              </option>
            ))}
          </select>
        </div>
        <NumField
          label="Offset x"
          value={target.x}
          unit="m"
          step={0.01}
          onChange={(v) =>
            updTarget(setModel, target.id, {
              x: clamp(v || 0, 0, model.domain.width),
            })
          }
        />
        <NumField
          label="Depth"
          value={target.depth}
          unit="m"
          step={0.01}
          onChange={(v) =>
            updTarget(setModel, target.id, {
              depth: clamp(v || 0.02, 0.02, model.domain.depth),
            })
          }
        />
        <NumField
          label="Diameter"
          value={target.diameter}
          unit="m"
          step={0.005}
          onChange={(v) =>
            updTarget(setModel, target.id, {
              diameter: clamp(v || 0.01, 0.005, 1),
            })
          }
        />
        <div className="fld">
          <label>EM property</label>
          <div className="ctl">
            <input
              className="inp"
              disabled
              value={
                tt.kind === "pec" ? "perfect conductor" : "εr " + tt.epsilon
              }
            />
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="insp-empty">
      <Icon name="edit" className="ic" />
      <div>Select a layer or target to edit its parameters</div>
      <div style={{ fontSize: 11.5, color: "var(--muted)" }}>
        or open <b style={{ color: "var(--ink-3)" }}>Parameters</b> for
        acquisition settings
      </div>
    </div>
  );
}

function Acquisition({ model, setModel }) {
  const a = model.acquisition,
    d = model.domain;
  return (
    <div className="insp">
      <div className="insp-title">
        <Icon name="settings" size={16} style={{ color: "var(--accent)" }} />
        <h4>Acquisition &amp; Domain</h4>
        <span className="badge mono">manual input</span>
      </div>
      <div className="fld">
        <label>GPR system</label>
        <select
          className="sel-ctl"
          value={a.antenna}
          onChange={(e) => {
            const id = e.target.value;
            const an = ANTENNAS.find((x) => x.id === id);
            updAcq(setModel, {
              antenna: id,
              frequency: an.freq,
              txrxSep: an.sep,
            });
          }}
        >
          {ANTENNAS.map((an) => (
            <option key={an.id} value={an.id}>
              {an.label}
            </option>
          ))}
        </select>
      </div>
      <NumField
        label="Centre frequency"
        value={a.frequency}
        unit="GHz"
        step={0.05}
        onChange={(v) =>
          updAcq(setModel, { frequency: clamp(v || 0.1, 0.05, 5) })
        }
      />
      <div className="fld">
        <label>Source waveform</label>
        <select
          className="sel-ctl"
          value={a.waveform}
          onChange={(e) => updAcq(setModel, { waveform: e.target.value })}
        >
          {WAVEFORMS.map((w) => (
            <option key={w} value={w}>
              {w}
            </option>
          ))}
        </select>
      </div>
      <div className="fld">
        <label>Survey mode</label>
        <div className="seg">
          {["A-scan", "B-scan"].map((s) => (
            <button
              key={s}
              className={a.surveyMode === s ? "on" : ""}
              onClick={() => updAcq(setModel, { surveyMode: s })}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <NumField
        label="Time window"
        value={a.timeWindow}
        unit="ns"
        step={0.5}
        onChange={(v) =>
          updAcq(setModel, { timeWindow: clamp(v || 1, 1, 200) })
        }
      />
      <NumField
        label="Trace step"
        value={a.traceStep}
        unit="m"
        step={0.005}
        onChange={(v) =>
          updAcq(setModel, { traceStep: clamp(v || 0.005, 0.002, 0.2) })
        }
      />
      <NumField
        label="Tx–Rx separation"
        value={a.txrxSep}
        unit="m"
        step={0.01}
        onChange={(v) => updAcq(setModel, { txrxSep: clamp(v || 0, 0, 0.5) })}
      />
      <div className="insp-title" style={{ marginTop: 6 }}>
        <Icon name="fit" size={15} style={{ color: "var(--ink-3)" }} />
        <h4 style={{ fontSize: 12.5 }}>Domain</h4>
      </div>
      <NumField
        label="Width"
        value={d.width}
        unit="m"
        step={0.05}
        onChange={(v) =>
          updDomain(setModel, { width: clamp(v || 0.2, 0.2, 10) })
        }
      />
      <NumField
        label="Depth"
        value={d.depth}
        unit="m"
        step={0.05}
        onChange={(v) =>
          updDomain(setModel, { depth: clamp(v || 0.2, 0.2, 10) })
        }
      />
      <div className="fld">
        <label>Spatial step Δx</label>
        <div className="ctl">
          <input
            className="inp"
            type="number"
            step={0.0005}
            value={d.dx}
            onChange={(e) =>
              updDomain(setModel, {
                dx: clamp(parseFloat(e.target.value) || 0.002, 0.0005, 0.02),
              })
            }
          />
          <span className="unit">m</span>
        </div>
      </div>
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
function DatasetFileView({ view, onClose }) {
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

Object.assign(window, {
  MenuBar,
  ModelTree,
  Dock,
  Inspector,
  Acquisition,
  Toasts,
  ExportModal,
  DatasetFileView,
  generateGprMax,
  updLayer,
  updTarget,
  updAcq,
  updDomain,
});
