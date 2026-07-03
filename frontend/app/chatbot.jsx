/* ============================================================
   NL2Sim — backend-connected AI assistant pane
   ============================================================ */

function mdToHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br/>");
}

function nextChips(kind) {
  if (kind === "choice")
    return [
      { t: "dataset_config", ic: "grid" },
      { t: "layers", ic: "layers" },
      { t: "waveform", ic: "wave" },
      { t: "antenna", ic: "radar" },
      { t: "advanced_params", ic: "target" },
    ];
  if (kind === "starter")
    return [
      { t: "Use defaults", ic: "check" },
      { t: "Skip target", ic: "target" },
      { t: "Continue", ic: "play" },
    ];
  return [];
}

function getSessionId() {
  const key = "nl2sim_chat_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id =
      window.crypto && crypto.randomUUID
        ? crypto.randomUUID()
        : "session-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    localStorage.setItem(key, id);
  }
  return id;
}

function getWsUrl(sessionId) {
  if (window.NL2SIM_WS_URL) return window.NL2SIM_WS_URL + "/" + sessionId;
  const isFile = window.location.protocol === "file:";
  const isStaticDevServer =
    window.location.hostname === "127.0.0.1" &&
    ["5173", "8001", "8080"].includes(window.location.port);
  const host =
    isFile || isStaticDevServer ? "127.0.0.1:8000" : window.location.host;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${host}/ws/${sessionId}`;
}

function ChatPane({
  activeModel,
  collapsed,
  setCollapsed,
  toast,
  onModelUpdate,
}) {
  const [messages, setMessages] = React.useState([
    {
      id: uid("m"),
      role: "bot",
      html: mdToHtml("Connecting to the **NL2Sim** LangGraph pipeline..."),
    },
  ]);
  const [chips, setChips] = React.useState([]);
  const [typing, setTyping] = React.useState(false);
  const [draft, setDraft] = React.useState("");
  const [status, setStatus] = React.useState("connecting");
  const [busy, setBusy] = React.useState(false);
  const [sending, setSending] = React.useState(false);
  const scrollRef = React.useRef(null);
  const taRef = React.useRef(null);
  const wsRef = React.useRef(null);
  const paneRef = React.useRef(null);

  // user-resizable pane width; null = CSS default (clamp on .chat).
  // CSS min/max-width still bound the inline value, so it stays responsive
  // when the window shrinks.
  const [chatW, setChatW] = React.useState(() => {
    const v = parseInt(localStorage.getItem("nl2sim_chat_w") || "", 10);
    return Number.isFinite(v) && v > 0 ? v : null;
  });
  const [resizing, setResizing] = React.useState(false);

  React.useEffect(() => {
    if (chatW != null) localStorage.setItem("nl2sim_chat_w", String(chatW));
    else localStorage.removeItem("nl2sim_chat_w");
  }, [chatW]);

  function startResize(e) {
    e.preventDefault();
    const startX = e.clientX;
    const startW = paneRef.current ? paneRef.current.offsetWidth : 392;
    setResizing(true);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const onMove = (ev) => {
      const max = Math.round(window.innerWidth * 0.72);
      setChatW(clamp(startW + (startX - ev.clientX), 280, max));
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      setResizing(false);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
  // ws.onmessage is bound once at mount and would capture the first render's
  // handleServerEvent; route the callback through a ref so it stays current.
  const onModelUpdateRef = React.useRef(onModelUpdate);
  onModelUpdateRef.current = onModelUpdate;

  React.useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, typing, chips]);

  React.useEffect(() => {
    const sessionId = getSessionId();
    const ws = new WebSocket(getWsUrl(sessionId));
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      setChips(nextChips("starter"));
    };
    ws.onclose = () => {
      setStatus("disconnected");
      setTyping(false);
      setBusy(false);
      setSending(false);
      pushBot(mdToHtml("Connection closed. Restart the API server and reload the page."));
    };
    ws.onerror = () => {
      setStatus("error");
      setTyping(false);
      setSending(false);
    };
    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      handleServerEvent(msg);
    };

    return () => {
      ws.close();
    };
  }, []);

  function pushBot(html, card) {
    setMessages((m) => [...m, { id: uid("m"), role: "bot", html, card }]);
  }

  function pushStatus(text) {
    setMessages((m) => [
      ...m,
      { id: uid("m"), role: "bot", html: mdToHtml(text), status: true },
    ]);
  }

  function handleServerEvent(msg) {
    if (msg.type === "pipeline_busy") {
      setBusy(Boolean(msg.busy));
      return;
    }
    if (msg.type === "model_update") {
      // Mid-turn canvas update — must not clear the typing indicator.
      if (onModelUpdateRef.current) onModelUpdateRef.current(msg.scene);
      return;
    }

    setSending(false);
    setTyping(false);

    if (msg.type === "agent_message") {
      pushBot(mdToHtml(msg.content));
      setChips([]);
      return;
    }
    if (msg.type === "stage_change") {
      pushStatus("**Stage:** " + (msg.stage_name || "Pipeline step"));
      setChips([]);
      return;
    }
    if (msg.type === "progress") {
      pushStatus(msg.content || "Step complete.");
      return;
    }
    if (msg.type === "validation_failed") {
      const errors = (msg.errors || []).map((e) => "- " + e).join("\n");
      pushBot(mdToHtml("**Validation failed:** " + (msg.stage_name || "") + (errors ? "\n" + errors : "")));
      return;
    }
    if (msg.type === "choice_required") {
      pushBot(mdToHtml(msg.content));
      setChips(
        (msg.choices || []).map((c) => ({
          t: c,
          ic: c === "layers" ? "layers" : c === "waveform" ? "wave" : "grid",
        })),
      );
      return;
    }
    if (msg.type === "dataset_ready") {
      pushBot(mdToHtml(msg.content || "Dataset created."));
      setChips([]);
      if (toast) toast("Dataset created and stored", "ok");
      return;
    }
    if (msg.type === "error") {
      pushBot(mdToHtml("**Backend error:** " + (msg.message || "Unknown error")));
      return;
    }
  }

  function handle(text) {
    const clean = text.trim();
    if (!clean) return;

    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      pushBot(mdToHtml("The backend WebSocket is not connected."));
      return;
    }

    setMessages((m) => [
      ...m,
      { id: uid("m"), role: "user", html: mdToHtml(clean) },
    ]);
    setChips([]);
    setDraft("");
    setTyping(true);
    setSending(true);
    ws.send(JSON.stringify({ type: "user_message", content: clean }));
  }

  const connected = status === "connected";
  const inputDisabled = !connected || busy || sending;

  // The textarea is disabled while the pipeline works, which drops focus.
  // Re-focus whenever it becomes editable again so the user can keep typing.
  React.useEffect(() => {
    if (!inputDisabled && taRef.current) taRef.current.focus();
  }, [inputDisabled]);
  const statusText =
    status === "connected"
      ? busy
        ? "running pipeline step"
        : "connected"
      : status;

  return (
    <section
      ref={paneRef}
      className={
        "chat" + (collapsed ? " collapsed" : "") + (resizing ? " resizing" : "")
      }
      style={!collapsed && chatW != null ? { width: chatW } : undefined}
    >
      <div
        className={"chat-resizer" + (resizing ? " active" : "")}
        onPointerDown={startResize}
        onDoubleClick={() => setChatW(null)}
        title="Drag to resize · double-click to reset"
      />
      <div className="chat-head">
        <div className="chat-av">
          <Icon name="sparkles" size={17} style={{ color: "#fff" }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="ct">Simulation Assistant</div>
          <div className="cs">
            <span className="dot"></span>
            {statusText}{" "}
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
        {chips.length > 0 && !typing && !busy && (
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
            placeholder={
              connected
                ? "Reply to the LangGraph agent..."
                : "Waiting for backend connection..."
            }
            value={draft}
            disabled={inputDisabled}
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
            disabled={inputDisabled || !draft.trim()}
            onClick={() => {
              handle(draft);
              if (taRef.current) taRef.current.style.height = "auto";
            }}
          >
            <Icon name="send" size={15} />
          </button>
        </div>
        <div className="chat-hint">
          LangGraph agent pipeline · press ↵ to send
        </div>
      </div>
    </section>
  );
}

Object.assign(window, { ChatPane, mdToHtml, nextChips });
