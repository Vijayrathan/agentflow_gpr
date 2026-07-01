/* ============================================================
   NL2Sim — AI assistant pane
   ============================================================ */

function mdToHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br/>");
}

function nextChips(kind, model) {
  const hasTargets = model.targets.length > 0;
  if (kind === "welcome")
    return [
      { t: "Buried utility survey", ic: "layers" },
      { t: "Landmine detection", ic: "target" },
      { t: "Rebar in concrete", ic: "grid" },
      { t: "Soil moisture study", ic: "wave" },
    ];
  if (kind === "run")
    return [
      { t: "Generate 500 variations", ic: "database" },
      { t: "Why is there a hyperbola?", ic: "info" },
      { t: "Make the soil wet", ic: "wave" },
    ];
  if (kind === "dataset")
    return [
      { t: "Set 900 MHz", ic: "bolt" },
      { t: "Add a metal pipe at 0.3 m", ic: "target" },
      { t: "Run forward model", ic: "play" },
    ];
  // default contextual
  const c = [];
  if (!hasTargets) c.push({ t: "Bury a PVC pipe at 0.4 m", ic: "target" });
  else c.push({ t: "Add a wet clay layer 0.4 m thick", ic: "layers" });
  c.push({ t: "Run forward model", ic: "play" });
  c.push({ t: "Generate 500 variations", ic: "database" });
  c.push({ t: "Explain the permittivity", ic: "info" });
  return c;
}

function ChatPane({
  model,
  modelRef,
  setModel,
  onRun,
  dataset,
  addDataset,
  activeModel,
  collapsed,
  setCollapsed,
  toast,
}) {
  const [messages, setMessages] = React.useState([
    {
      id: uid("m"),
      role: "bot",
      html: mdToHtml(
        "Hi — I'm the **NL2Sim** assistant. Describe a ground-penetrating-radar scene in plain language and I'll build the subsurface model, wire up the gprMax parameters, and turn it into labelled training data.\n\nWhat would you like to simulate?",
      ),
    },
  ]);
  const [chips, setChips] = React.useState(nextChips("welcome", model));
  const [typing, setTyping] = React.useState(false);
  const [draft, setDraft] = React.useState("");
  const scrollRef = React.useRef(null);
  const taRef = React.useRef(null);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, typing, chips]);

  function pushBot(html, card) {
    setMessages((m) => [...m, { id: uid("m"), role: "bot", html, card }]);
  }

  function handle(text) {
    const clean = text.trim();
    if (!clean) return;
    setMessages((m) => [
      ...m,
      { id: uid("m"), role: "user", html: mdToHtml(clean) },
    ]);
    setChips([]);
    setDraft("");
    setTyping(true);
    const cur = modelRef.current;
    const res = parseCommand(clean, cur);
    const delay = 620 + Math.random() * 420;
    setTimeout(() => {
      setTyping(false);
      if (res.patch) setModel(res.patch);
      if (res.run) onRun();
      if (res.datasetAdd) addDataset(res.datasetAdd);
      let html,
        card = res.actions;
      if (res.answer) {
        html = mdToHtml(localAnswer(clean, cur));
      } else {
        html = mdToHtml(res.reply);
      }
      pushBot(html, card);
      setTimeout(
        () =>
          setChips(
            nextChips(
              res.kind === "scenario" ? "default" : res.kind || "default",
              modelRef.current,
            ),
          ),
        60,
      );
    }, delay);
  }

  return (
    <section className={"chat" + (collapsed ? " collapsed" : "")}>
      <div className="chat-head">
        <div className="chat-av">
          <Icon name="sparkles" size={17} style={{ color: "#fff" }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="ct">Simulation Assistant</div>
          <div className="cs">
            <span className="dot"></span>building{" "}
            <b style={{ color: "var(--ink-2)", fontWeight: 600 }}>
              &nbsp;{ML_MODELS.find((m) => m.id === activeModel)?.label}
            </b>
            &nbsp;data
          </div>
        </div>
        <button
          className="hbtn"
          title="Collapse"
          onClick={() => setCollapsed(true)}
        >
          <Icon name="chev" size={15} />
        </button>
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.map((m) => (
          <div key={m.id} className={"msg " + m.role}>
            <div className="av">
              {m.role === "bot" ? (
                <Icon name="sparkles" size={13} style={{ color: "#fff" }} />
              ) : (
                <span
                  className="mono"
                  style={{ fontSize: 10, fontWeight: 600 }}
                >
                  YOU
                </span>
              )}
            </div>
            <div className="bubble">
              <div dangerouslySetInnerHTML={{ __html: m.html }} />
              {m.card && m.card.length > 0 && (
                <div className="act-card">
                  <div className="ah">
                    <Icon name="check" size={12} />
                    Applied to model
                  </div>
                  {m.card.map((a, i) => (
                    <div className="ar" key={i}>
                      {a.sw && (
                        <span
                          className="sw"
                          style={{ background: a.sw }}
                        ></span>
                      )}
                      <span>{a.label}</span>
                      <span className="v">{a.v}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {typing && (
          <div className="msg bot">
            <div className="av">
              <Icon name="sparkles" size={13} style={{ color: "#fff" }} />
            </div>
            <div className="bubble">
              <div className="typing">
                <i></i>
                <i></i>
                <i></i>
              </div>
            </div>
          </div>
        )}
        {chips.length > 0 && !typing && (
          <div className="chips">
            {chips.map((c, i) => (
              <button key={i} className="chip" onClick={() => handle(c.t)}>
                <Icon name={c.ic} className="ic" />
                {c.t}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="chat-input">
        <div className="chat-inwrap">
          <textarea
            ref={taRef}
            rows={1}
            placeholder="Describe a scene, ask a question, or tweak a parameter…"
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height =
                Math.min(120, e.target.scrollHeight) + "px";
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handle(draft);
                if (taRef.current) taRef.current.style.height = "auto";
              }
            }}
          />
          <button
            className="send"
            disabled={!draft.trim()}
            onClick={() => {
              handle(draft);
              if (taRef.current) taRef.current.style.height = "auto";
            }}
          >
            <Icon name="send" size={15} />
          </button>
        </div>
        <div className="chat-hint">
          natural language → gprMax scenario · press ↵ to send
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { ChatPane, mdToHtml, nextChips });
