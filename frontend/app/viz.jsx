/* ============================================================
   NL2Sim — visualization: subsurface cross-section + radargram
   ============================================================ */

/* color utils */
function hexToRgb(h) {
  h = h.replace("#", "");
  if (h.length === 3)
    h = h
      .split("")
      .map((c) => c + c)
      .join("");
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function shade(hex, amt) {
  const [r, g, b] = hexToRgb(hex);
  const f = (v) => Math.max(0, Math.min(255, Math.round(v + amt)));
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}
function niceStep(extent, target = 7) {
  const raw = extent / target || 0.1;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const c = raw / p;
  let s = c < 1.5 ? 1 : c < 3.5 ? 2 : c < 7.5 ? 5 : 10;
  return s * p;
}

/* ---------- pattern defs (dark overlays, color-independent) ---------- */
function PatternDefs() {
  const stroke = "rgba(40,32,22,0.16)";
  return (
    <defs>
      <pattern id="p-dots" width="11" height="11" patternUnits="userSpaceOnUse">
        <circle cx="3" cy="3" r="1.1" fill={stroke} />
        <circle cx="8" cy="8" r="1.1" fill={stroke} />
      </pattern>
      <pattern id="p-soil" width="16" height="16" patternUnits="userSpaceOnUse">
        <circle cx="4" cy="6" r="0.9" fill={stroke} />
        <circle cx="11" cy="3" r="0.7" fill={stroke} />
        <circle cx="9" cy="12" r="0.8" fill={stroke} />
        <circle cx="14" cy="10" r="0.6" fill={stroke} />
      </pattern>
      <pattern
        id="p-hatch"
        width="9"
        height="9"
        patternUnits="userSpaceOnUse"
        patternTransform="rotate(45)"
      >
        <line x1="0" y1="0" x2="0" y2="9" stroke={stroke} strokeWidth="1.1" />
      </pattern>
      <pattern id="p-grid" width="12" height="12" patternUnits="userSpaceOnUse">
        <path
          d="M12 0H0V12"
          fill="none"
          stroke="rgba(40,40,46,0.13)"
          strokeWidth="1"
        />
      </pattern>
      <pattern
        id="p-gravel"
        width="20"
        height="18"
        patternUnits="userSpaceOnUse"
      >
        <circle
          cx="4"
          cy="5"
          r="2.4"
          fill="none"
          stroke={stroke}
          strokeWidth="1"
        />
        <circle
          cx="13"
          cy="11"
          r="3"
          fill="none"
          stroke={stroke}
          strokeWidth="1"
        />
        <circle
          cx="17"
          cy="3"
          r="1.7"
          fill="none"
          stroke={stroke}
          strokeWidth="1"
        />
        <circle
          cx="8"
          cy="14"
          r="1.6"
          fill="none"
          stroke={stroke}
          strokeWidth="1"
        />
      </pattern>
      <pattern id="p-rock" width="18" height="18" patternUnits="userSpaceOnUse">
        <path
          d="M2 4l4 2M9 2l3 4M13 9l4 1M3 12l5 2M11 13l4 3"
          stroke={stroke}
          strokeWidth="1.1"
          fill="none"
        />
      </pattern>
      <pattern id="p-wave" width="22" height="10" patternUnits="userSpaceOnUse">
        <path
          d="M0 5 Q5.5 0 11 5 T22 5"
          fill="none"
          stroke="rgba(40,60,80,0.18)"
          strokeWidth="1"
        />
      </pattern>
      <radialGradient id="g-metal" cx="38%" cy="32%" r="75%">
        <stop offset="0%" stopColor="#c5cbd2" />
        <stop offset="55%" stopColor="#8a9099" />
        <stop offset="100%" stopColor="#5b626d" />
      </radialGradient>
      <radialGradient id="g-air" cx="50%" cy="0%" r="90%">
        <stop offset="0%" stopColor="#eef4fb" />
        <stop offset="100%" stopColor="#f7f8fa" />
      </radialGradient>
      <linearGradient id="g-anten" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stopColor="#3d4350" />
        <stop offset="100%" stopColor="#252a34" />
      </linearGradient>
    </defs>
  );
}

const PATTERN_OF = {
  dots: "p-dots",
  soil: "p-soil",
  hatch: "p-hatch",
  grid: "p-grid",
  gravel: "p-gravel",
  rock: "p-rock",
  wave: "p-wave",
  solid: null,
};

/* ============================================================
   SubsurfaceView
   ============================================================ */
function SubsurfaceView({
  model,
  selected,
  onSelect,
  view,
  scanFrac,
  solving,
}) {
  const VW = 1000,
    VH = 660;
  const mL = 58,
    mR = 176,
    mT = 78,
    mB = 30;
  const dom = model.domain;
  const maxW = VW - mL - mR,
    maxH = VH - mT - mB;
  const scale = Math.min(maxW / dom.width, maxH / dom.depth);
  const plotW = dom.width * scale,
    plotH = dom.depth * scale;
  const oX = mL + (maxW - plotW) / 2,
    oY = mT;
  const mx = (m) => oX + m * scale,
    my = (d) => oY + d * scale;

  const visLayers = model.layers.filter((l) => l.visible !== false);
  // cumulative layer rects
  let acc = 0;
  const rects = [];
  for (const l of model.layers) {
    const top = acc;
    acc += l.thickness;
    rects.push({ l, top, bot: acc });
  }
  const filled = acc;

  // thickness-uncertainty bands (ranges overview): boundary k's spread is the
  // cumulative min vs max of all layers above it, so uncertainty compounds
  // downward by construction. Absent thicknessMin/Max (sample view) => none.
  let accMin = 0,
    accMax = 0;
  const bands = [];
  for (const l of model.layers) {
    accMin += l.thicknessMin ?? l.thickness;
    accMax += l.thicknessMax ?? l.thickness;
    if (l.thicknessMin != null && accMax - accMin > 1e-6)
      bands.push({
        from: Math.min(accMin, dom.depth),
        to: Math.min(accMax, dom.depth),
      });
  }

  const stepX = niceStep(dom.width),
    stepY = niceStep(dom.depth);
  const xticks = [];
  for (let v = 0; v <= dom.width + 1e-9; v += stepX) xticks.push(round(v, 3));
  const yticks = [];
  for (let v = 0; v <= dom.depth + 1e-9; v += stepY) yticks.push(round(v, 3));

  const antX = oX + (scanFrac ?? 0.12) * plotW;

  return (
    <svg viewBox={`0 0 ${VW} ${VH}`} preserveAspectRatio="xMidYMid meet">
      <PatternDefs />
      {/* air */}
      <rect
        x={oX}
        y={mT - 40}
        width={plotW}
        height={40}
        fill="url(#g-air)"
        stroke="var(--line-2)"
        strokeWidth="1"
      />
      <text
        x={oX + 8}
        y={mT - 26}
        fontFamily="var(--mono)"
        fontSize="10"
        fill="var(--ink-3)"
      >
        AIR · εr 1.0
      </text>

      {/* surface ruler ticks */}
      {view.ruler &&
        xticks.map((v, i) => (
          <g key={i}>
            <line
              x1={mx(v)}
              y1={oY - 4}
              x2={mx(v)}
              y2={oY}
              stroke="var(--ink-3)"
              strokeWidth="1"
            />
            <text
              x={mx(v)}
              y={mT - 46}
              textAnchor="middle"
              fontFamily="var(--mono)"
              fontSize="9.5"
              fill="var(--ink-3)"
            >
              {fmt(v, 2)}
            </text>
          </g>
        ))}
      <text
        x={oX + plotW / 2}
        y={20}
        textAnchor="middle"
        fontFamily="var(--mono)"
        fontSize="9.5"
        fill="var(--muted)"
        letterSpacing="1"
      >
        SURFACE OFFSET (m) →
      </text>

      {/* depth ruler */}
      {view.ruler &&
        yticks.map((v, i) => (
          <g key={i}>
            <line
              x1={oX - 4}
              y1={my(v)}
              x2={oX}
              y2={my(v)}
              stroke="var(--ink-3)"
              strokeWidth="1"
            />
            <text
              x={oX - 9}
              y={my(v) + 3}
              textAnchor="end"
              fontFamily="var(--mono)"
              fontSize="9.5"
              fill="var(--ink-3)"
            >
              {fmt(v, 2)}
            </text>
          </g>
        ))}
      <text
        x={16}
        y={oY + plotH / 2}
        fontFamily="var(--mono)"
        fontSize="9.5"
        fill="var(--muted)"
        letterSpacing="1"
        transform={`rotate(-90 16 ${oY + plotH / 2})`}
        textAnchor="middle"
      >
        ↓ DEPTH (m)
      </text>

      {/* layers */}
      {rects.map(({ l, top, bot }) => {
        const mat = MATERIALS[l.material] || {
          color: "#cccccc",
          pattern: "solid",
        };
        const y = my(top),
          h = (bot - top) * scale;
        const isSel = selected && selected.id === l.id;
        const hidden = l.visible === false;
        if (hidden) return null;
        const pat = PATTERN_OF[mat.pattern];
        return (
          <g
            key={l.id}
            onClick={() => onSelect({ type: "layer", id: l.id })}
            style={{ cursor: "pointer" }}
          >
            <rect x={oX} y={y} width={plotW} height={h} fill={mat.color} />
            {pat && (
              <rect
                x={oX}
                y={y}
                width={plotW}
                height={h}
                fill={`url(#${pat})`}
              />
            )}
            <line
              x1={oX}
              y1={y}
              x2={oX + plotW}
              y2={y}
              stroke={shade(mat.color, -55)}
              strokeWidth="1"
              opacity="0.7"
            />
            {isSel && (
              <rect
                x={oX + 1}
                y={y + 1}
                width={plotW - 2}
                height={h - 2}
                fill="var(--accent)"
                opacity="0.10"
              />
            )}
            {isSel && (
              <rect
                x={oX}
                y={y}
                width={plotW}
                height={h}
                fill="none"
                stroke="var(--accent)"
                strokeWidth="2"
              />
            )}
            {/* right-edge label */}
            <g transform={`translate(${oX + plotW + 10}, ${y + h / 2})`}>
              <line
                x1={-10}
                y1={0}
                x2={-2}
                y2={0}
                stroke="var(--line-3)"
                strokeWidth="1"
              />
              <rect
                x={0}
                y={-15}
                width={158}
                height={30}
                rx={6}
                fill="var(--panel)"
                stroke={isSel ? "var(--accent-line)" : "var(--line-2)"}
                strokeWidth="1"
              />
              <rect
                x={8}
                y={-8}
                width={14}
                height={16}
                rx={3}
                fill={mat.color}
                stroke="rgba(0,0,0,.15)"
              />
              <text
                x={29}
                y={-1}
                fontSize="11"
                fontWeight="600"
                fill="var(--ink)"
              >
                {l.name}
              </text>
              <text
                x={29}
                y={10}
                fontFamily="var(--mono)"
                fontSize="9"
                fill="var(--ink-3)"
              >
                εr {l.epsilon ?? "—"}
                {l.sigma != null ? ` · σ ${l.sigma}` : ""}
              </text>
            </g>
          </g>
        );
      })}

      {/* half-space below modelled layers */}
      {filled < dom.depth - 1e-6 && (
        <g>
          <rect
            x={oX}
            y={my(filled)}
            width={plotW}
            height={(dom.depth - filled) * scale}
            fill="#e7e3da"
          />
          <rect
            x={oX}
            y={my(filled)}
            width={plotW}
            height={(dom.depth - filled) * scale}
            fill="url(#p-hatch)"
            opacity="0.5"
          />
          <text
            x={oX + plotW / 2}
            y={my(filled) + 18}
            textAnchor="middle"
            fontFamily="var(--mono)"
            fontSize="10"
            fill="var(--ink-3)"
          >
            {model.layers.length === 0
              ? "no layers yet — describe the subsurface in chat"
              : "half-space (background medium)"}
          </text>
        </g>
      )}

      {/* thickness-uncertainty bands around layer boundaries */}
      {bands.map((b, i) => (
        <g key={"band" + i} pointerEvents="none">
          <rect
            x={oX}
            y={my(b.from)}
            width={plotW}
            height={(b.to - b.from) * scale}
            fill="var(--accent)"
            opacity="0.10"
          />
          <line
            x1={oX}
            y1={my(b.from)}
            x2={oX + plotW}
            y2={my(b.from)}
            stroke="var(--accent)"
            strokeWidth="1"
            strokeDasharray="4 4"
            opacity="0.55"
          />
          <line
            x1={oX}
            y1={my(b.to)}
            x2={oX + plotW}
            y2={my(b.to)}
            stroke="var(--accent)"
            strokeWidth="1"
            strokeDasharray="4 4"
            opacity="0.55"
          />
        </g>
      ))}

      {/* plot frame */}
      <rect
        x={oX}
        y={oY}
        width={plotW}
        height={plotH}
        fill="none"
        stroke="var(--line-3)"
        strokeWidth="1.25"
      />

      {/* targets */}
      {view.targets &&
        model.targets
          .filter((t) => t.visible !== false)
          .map((t) => (
            <TargetGlyph
              key={t.id}
              t={t}
              mx={mx}
              my={my}
              scale={scale}
              selected={selected && selected.id === t.id}
              onSelect={() => onSelect({ type: "target", id: t.id })}
            />
          ))}

      {/* GPR antenna + scan path */}
      {view.antenna && (
        <g>
          <line
            x1={oX}
            y1={oY - 2}
            x2={oX + plotW}
            y2={oY - 2}
            stroke="var(--accent)"
            strokeWidth="1.25"
            strokeDasharray="2 4"
            opacity="0.6"
          />
          <polygon
            points={`${oX + plotW},${oY - 2} ${oX + plotW - 7},${oY - 5.5} ${oX + plotW - 7},${oY + 1.5}`}
            fill="var(--accent)"
            opacity="0.7"
          />
          {/* wavefronts */}
          {solving &&
            [16, 30, 44].map((r, i) => (
              <path
                key={i}
                d={`M ${antX - r} ${oY} A ${r} ${r} 0 0 0 ${antX + r} ${oY}`}
                fill="none"
                stroke="var(--accent)"
                strokeWidth="1.5"
                opacity={0.5 - i * 0.13}
              />
            ))}
          <g transform={`translate(${antX},${oY})`}>
            <line
              x1="0"
              y1="0"
              x2="0"
              y2="8"
              stroke="#252a34"
              strokeWidth="2"
            />
            <rect
              x="-34"
              y="-26"
              width="68"
              height="22"
              rx="5"
              fill="url(#g-anten)"
            />
            <rect
              x="-30"
              y="-22"
              width="60"
              height="6"
              rx="2"
              fill="#5a6270"
              opacity="0.5"
            />
            <text
              x="0"
              y="-10"
              textAnchor="middle"
              fontFamily="var(--mono)"
              fontSize="9"
              fontWeight="600"
              fill="#dfe6ef"
            >
              {model.acquisition.frequency}GHz
            </text>
            <circle cx="-18" cy="2" r="2" fill="#7fb0f5" />
            <circle cx="18" cy="2" r="2" fill="#5fd6a0" />
          </g>
        </g>
      )}
    </svg>
  );
}

/* ---------- single target glyph ---------- */
function TargetGlyph({ t, mx, my, scale, selected, onSelect }) {
  const tt = TARGET_TYPES[t.type] || TARGET_TYPES.boulder;
  const cx = mx(t.x),
    cy = my(t.depth);
  const r = Math.max(4, (t.diameter / 2) * scale);
  const label = `${t.name} · ${fmt(t.depth)} m`;
  let glyph;
  if (t.shape === "rect") {
    // box target: rectangle with the same metal treatment as PEC circles
    const w = Math.max(6, (t.width ?? t.diameter ?? 0.1) * scale);
    const h = Math.max(6, (t.height ?? t.diameter ?? 0.1) * scale);
    glyph = (
      <g>
        <rect
          x={cx - w / 2}
          y={cy - h / 2}
          width={w}
          height={h}
          rx={2}
          fill={tt.kind === "pec" ? "url(#g-metal)" : tt.color}
          stroke="#3c4350"
          strokeWidth="1.25"
        />
        {tt.kind === "pec" && (
          <rect
            x={cx - w * 0.35}
            y={cy - h * 0.35}
            width={w * 0.3}
            height={h * 0.2}
            rx={2}
            fill="rgba(255,255,255,.4)"
          />
        )}
      </g>
    );
  } else if (tt.shape === "rebar") {
    const n = 5,
      span = Math.max(r * 4, 60);
    glyph = (
      <g>
        {Array.from({ length: n }).map((_, i) => {
          const gx = cx - span / 2 + (span / (n - 1)) * i;
          return (
            <circle
              key={i}
              cx={gx}
              cy={cy}
              r={Math.max(3.5, r)}
              fill="url(#g-metal)"
              stroke="#3c4350"
              strokeWidth="1"
            />
          );
        })}
      </g>
    );
  } else if (tt.shape === "circle") {
    glyph = (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill={tt.kind === "pec" ? "url(#g-metal)" : tt.color}
          stroke={shade(tt.color, -50)}
          strokeWidth="1.25"
        />
        {tt.kind === "pec" && (
          <circle
            cx={cx - r * 0.3}
            cy={cy - r * 0.3}
            r={r * 0.28}
            fill="rgba(255,255,255,.5)"
          />
        )}
      </g>
    );
  } else if (tt.shape === "ring") {
    glyph = (
      <g>
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill={tt.color}
          stroke={shade(tt.color, -45)}
          strokeWidth="1.5"
        />
        <circle
          cx={cx}
          cy={cy}
          r={Math.max(2, r * 0.55)}
          fill="#f4f6f9"
          stroke={shade(tt.color, -30)}
          strokeWidth="1"
        />
      </g>
    );
  } else if (tt.shape === "dash") {
    glyph = (
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill="rgba(255,255,255,.7)"
        stroke="var(--accent)"
        strokeWidth="1.5"
        strokeDasharray="3 3"
      />
    );
  } else {
    // blob
    glyph = (
      <ellipse
        cx={cx}
        cy={cy}
        rx={r * 1.15}
        ry={r * 0.9}
        fill={tt.color}
        stroke={shade(tt.color, -45)}
        strokeWidth="1.25"
      />
    );
  }
  return (
    <g onClick={onSelect} style={{ cursor: "pointer" }}>
      {selected && (
        <circle
          cx={cx}
          cy={cy}
          r={r + 7}
          fill="none"
          stroke="var(--amber)"
          strokeWidth="1.75"
          strokeDasharray="3 3"
        />
      )}
      {glyph}
      {/* leader tag */}
      <g transform={`translate(${cx + r + 9}, ${cy})`}>
        <rect
          x={0}
          y={-11}
          width={Math.max(96, label.length * 5.6)}
          height={22}
          rx={5}
          fill={selected ? "var(--amber-soft)" : "rgba(255,255,255,.94)"}
          stroke={selected ? "var(--amber-line)" : "var(--line-2)"}
          strokeWidth="1"
        />
        <circle
          cx={11}
          cy={0}
          r={4}
          fill={tt.kind === "pec" ? "#8a9099" : tt.color}
          stroke="rgba(0,0,0,.2)"
        />
        <text
          x={20}
          y={3.5}
          fontFamily="var(--mono)"
          fontSize="9.5"
          fill="var(--ink)"
        >
          {label}
        </text>
      </g>
    </g>
  );
}

/* ============================================================
   Radargram (synthetic B-scan, canvas)
   ============================================================ */
function ricker(tau, f) {
  const a = Math.PI * f * tau;
  const a2 = a * a;
  return (1 - 2 * a2) * Math.exp(-a2);
}

function Radargram({ model, solved, progress }) {
  const ref = React.useRef(null);
  const W = 560,
    H = 300,
    padL = 42,
    padT = 16,
    padB = 26,
    padR = 12;
  React.useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.style.width = W + "px";
    cv.style.height = H + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);
    const pw = W - padL - padR,
      ph = H - padT - padB;

    const c = 0.3; // m/ns
    const dom = model.domain;
    // build interface list (cumulative depth, time, refl coeff)
    let acc = 0,
      accT = 0;
    const interfaces = [];
    let prevEps = 1;
    for (const l of model.layers) {
      const v = c / Math.sqrt(l.epsilon);
      const R =
        (Math.sqrt(prevEps) - Math.sqrt(l.epsilon)) /
        (Math.sqrt(prevEps) + Math.sqrt(l.epsilon));
      interfaces.push({ t: accT, R });
      accT += (2 * l.thickness) / v;
      acc += l.thickness;
      prevEps = l.epsilon;
    }
    // velocity & time helper to a depth
    const timeToDepth = (z) => {
      let a = 0,
        t = 0;
      for (const l of model.layers) {
        const v = c / Math.sqrt(l.epsilon);
        const seg = Math.min(l.thickness, Math.max(0, z - a));
        t += (2 * seg) / v;
        a += l.thickness;
        if (a >= z) break;
      }
      if (z > a) {
        t += (2 * (z - a)) / (c / Math.sqrt(prevEps));
      }
      return t;
    };
    const avgEpsAbove = (z) => {
      let a = 0,
        s = 0;
      for (const l of model.layers) {
        const seg = Math.min(l.thickness, Math.max(0, z - a));
        s += seg * Math.sqrt(l.epsilon);
        a += l.thickness;
        if (a >= z) break;
      }
      return Math.pow(s / Math.max(z, 0.01), 2) || 9;
    };

    const targets = model.targets
      .filter((t) => t.visible !== false)
      .map((t) => {
        const eps = avgEpsAbove(t.depth);
        const v = c / Math.sqrt(eps);
        const td = timeToDepth(t.depth);
        const tt = TARGET_TYPES[t.type];
        const amp = tt && tt.kind === "pec" ? 1.0 : 0.6;
        return { x0: t.x, td, v, amp };
      });
    const Tmax = Math.max(2, timeToDepth(dom.depth) * 1.08);
    const f = model.acquisition.frequency; // GHz -> ns wavelet
    const colMax = Math.floor(pw * (solved ? 1 : progress || 0));

    const img = ctx.createImageData(pw, ph);
    for (let px = 0; px < pw; px++) {
      const xpos = (px / pw) * dom.width;
      for (let py = 0; py < ph; py++) {
        const t = (py / ph) * Tmax;
        let amp = 0;
        for (const itf of interfaces) amp += itf.R * 1.4 * ricker(t - itf.t, f);
        for (const tg of targets) {
          const th = Math.sqrt(
            tg.td * tg.td + Math.pow((2 * (xpos - tg.x0)) / tg.v, 2),
          );
          amp += tg.amp * ricker(t - th, f);
        }
        // depth gain + faint clutter
        amp *= 1 + t * 0.18;
        amp += Math.sin(px * 12.9 + py * 7.1) * 0.5 * 0.05;
        let v = Math.max(-1, Math.min(1, amp * 1.3));
        const g = (v + 1) / 2;
        const idx = (py * pw + px) * 4;
        const lit = px <= colMax || (colMax <= 0 && false);
        const base = lit ? g : 0.07;
        img.data[idx] = 255 * base * 0.86;
        img.data[idx + 1] = 255 * base * 0.95;
        img.data[idx + 2] = 255 * base * 1.0;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, padL, padT);
    // scanning edge
    if (!solved && progress > 0 && progress < 1) {
      const ex = padL + colMax;
      ctx.fillStyle = "rgba(127,176,245,0.9)";
      ctx.fillRect(ex, padT, 1.5, ph);
    }
    // axes
    ctx.fillStyle = "#8a93a3";
    ctx.font = "9px 'IBM Plex Mono', monospace";
    ctx.textBaseline = "middle";
    const stepT = niceStep(Tmax, 5);
    for (let tv = 0; tv <= Tmax; tv += stepT) {
      const y = padT + (tv / Tmax) * ph;
      ctx.textAlign = "right";
      ctx.fillText(tv.toFixed(0), padL - 6, y);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
    }
    ctx.save();
    ctx.translate(11, padT + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("two-way time (ns)", 0, 0);
    ctx.restore();
    const stepX = niceStep(dom.width, 6);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let xv = 0; xv <= dom.width + 1e-9; xv += stepX) {
      const x = padL + (xv / dom.width) * pw;
      ctx.fillText(xv.toFixed(1), x, H - padB + 6);
    }
  }, [model, solved, progress]);

  if (!solved && !(progress > 0)) {
    return (
      <div className="radar-empty">
        <Icon name="radar" size={26} />
        <div>
          No synthetic B-scan yet.
          <br />
          Press <b style={{ color: "var(--accent-2)" }}>Run forward model</b> to
          generate the radargram preview.
        </div>
      </div>
    );
  }
  return (
    <div className="radar-wrap">
      <div className="radar-canvas-box">
        <canvas ref={ref} />
      </div>
      <div className="radar-side">
        <div>
          <div className="rail-head" style={{ padding: 0, marginBottom: 8 }}>
            <span className="t">Acquisition</span>
          </div>
          <div className="radar-stat">
            <span className="k">Centre freq</span>
            <span className="v">{model.acquisition.frequency} GHz</span>
          </div>
          <div className="radar-stat">
            <span className="k">Waveform</span>
            <span className="v">{model.acquisition.waveform}</span>
          </div>
          <div className="radar-stat">
            <span className="k">Time window</span>
            <span className="v">{model.acquisition.timeWindow} ns</span>
          </div>
          <div className="radar-stat">
            <span className="k">Trace step</span>
            <span className="v">
              {(model.acquisition.traceStep * 100).toFixed(1)} cm
            </span>
          </div>
          <div className="radar-stat">
            <span className="k">Traces</span>
            <span className="v">
              {Math.round(model.domain.width / model.acquisition.traceStep)}
            </span>
          </div>
        </div>
        <div className="radar-note">
          {solved
            ? "Synthetic B-scan rendered from a 1-D convolutional approximation. Hyperbolae mark each buried target; flat bands are layer reflections. For the full FDTD result, export and run in gprMax."
            : "Sweeping the antenna across the scan line…"}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { SubsurfaceView, Radargram, shade, niceStep });
