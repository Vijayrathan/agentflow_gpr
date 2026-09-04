/* ============================================================
   NL2Sim — user identity gate + per-user chat list strip
   Identity is a plain user id (no auth): remembered in
   localStorage, and the SAME id typed on any browser recovers
   every chat/dataset from the DB-backed list.
   ============================================================ */

function UserGate({ onSubmit }) {
  const [value, setValue] = React.useState("");
  const [err, setErr] = React.useState("");

  const submit = () => {
    const v = value.trim();
    if (v.length < 1 || v.length > 64 || !/[A-Za-z0-9]/.test(v)) {
      setErr("1–64 characters with at least one letter or digit.");
      return;
    }
    onSubmit(v);
  };

  return (
    <div className="user-gate">
      <div className="user-gate-card">
        <div className="ugt">Who is simulating?</div>
        <div className="ugs">
          Enter a user id — your chats and datasets are saved under it.
          Type the same id on any machine to pick them back up.
        </div>
        <input
          autoFocus
          value={value}
          placeholder="e.g. vijay"
          maxLength={64}
          onChange={(e) => {
            setValue(e.target.value);
            setErr("");
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") submit();
          }}
        />
        {err && <div className="uge">{err}</div>}
        <button className="ugbtn" onClick={submit}>
          Continue
        </button>
      </div>
    </div>
  );
}

function relTime(iso) {
  if (!iso) return "";
  const dt = Date.now() - new Date(iso).getTime();
  const m = Math.floor(dt / 60000);
  if (m < 1) return "just now";
  if (m < 60) return m + "m ago";
  const h = Math.floor(m / 60);
  if (h < 24) return h + "h ago";
  return Math.floor(h / 24) + "d ago";
}

function ChatStrip({ chats, currentChatId, userId, onSelect, onNew, onSwitchUser }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    if (!open) return undefined;
    const onDoc = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDoc);
    return () => document.removeEventListener("pointerdown", onDoc);
  }, [open]);

  const list = chats || [];
  const current = list.find((c) => c.id === currentChatId);

  return (
    <div className="chat-strip" ref={ref}>
      <button
        className={"cs-picker" + (open ? " open" : "")}
        onClick={() => setOpen((o) => !o)}
        title="Your chats"
      >
        <span className="cs-name">{(current && current.title) || "New chat"}</span>
        <Icon name="chev" size={12} className="cs-chev" />
      </button>
      <button className="cs-new" onClick={onNew} title="Start a new simulation chat">
        + New
      </button>
      <button
        className="cs-user"
        onClick={onSwitchUser}
        title={"Signed in as " + userId + " — click to switch user"}
      >
        {userId}
      </button>
      {open && (
        <div className="cs-pop">
          {list.map((c) => (
            <button
              key={c.id}
              className={"cs-item" + (c.id === currentChatId ? " active" : "")}
              onClick={() => {
                setOpen(false);
                if (c.id !== currentChatId) onSelect(c.id);
              }}
            >
              <span className="cs-item-title">{c.title || "New chat"}</span>
              <span className="cs-item-meta">
                {c.complete ? "✓ dataset · " : ""}
                {relTime(c.updated_at)}
              </span>
            </button>
          ))}
          {!list.length && <div className="cs-empty">No chats yet</div>}
        </div>
      )}
    </div>
  );
}

Object.assign(window, { UserGate, ChatStrip });
