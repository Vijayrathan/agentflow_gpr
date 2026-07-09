/* ============================================================
   NL2Sim — visualization: subsurface cross-section
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

Object.assign(window, { SubsurfaceView, shade, niceStep });
