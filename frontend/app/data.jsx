/* ============================================================
   NL2Sim — data layer: catalogs, initial model, icons, NLU
   ============================================================ */

/* ---------- tiny helpers ---------- */
const uid = (p = "id") => p + "_" + Math.random().toString(36).slice(2, 8);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round = (v, d = 2) => {
  const f = Math.pow(10, d);
  return Math.round(v * f) / f;
};
const fmt = (v, d = 2) => Number(v).toFixed(d);

/* ---------- feather-style icon set ---------- */
const ICONS = {
  download: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
  sparkles:
    "M12 3l1.6 4.6L18 9l-4.4 1.4L12 15l-1.6-4.6L6 9l4.4-1.4L12 3 M19 14l.6 1.8 1.8.6-1.8.6-.6 1.8-.6-1.8-1.8-.6 1.8-.6z",
  cpu: "M9 3v3 M15 3v3 M9 18v3 M15 18v3 M3 9h3 M3 15h3 M18 9h3 M18 15h3 M6 6h12v12H6z M9 9h6v6H9z",
  edit: "M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7 M18.5 2.5a2.12 2.12 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z",
  caret: "M6 9l6 6 6-6",
  chev: "M9 6l6 6-6 6",
  check: "M20 6L9 17l-5-5",
  x: "M18 6L6 18 M6 6l12 12",
  layers: "M12 2L2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5",
  target:
    "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z M12 11a1 1 0 1 0 0 2 1 1 0 0 0 0-2z",
  radar: "M12 12l7-4 M12 22a10 10 0 1 1 0-20 M12 12l4 8 M3 12h3 M12 3v3",
  zoomin:
    "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z M20 20l-3.5-3.5 M11 8v6 M8 11h6",
  zoomout: "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14z M20 20l-3.5-3.5 M8 11h6",
  fit: "M4 8V5a1 1 0 0 1 1-1h3 M16 4h3a1 1 0 0 1 1 1v3 M20 16v3a1 1 0 0 1-1 1h-3 M8 20H5a1 1 0 0 1-1-1v-3",
  grid: "M3 9h18 M3 15h18 M9 3v18 M15 3v18",
  ruler: "M3 8h18v8H3z M7 8v3 M11 8v4 M15 8v3 M19 8v4",
  eye: "M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z",
  eyeoff:
    "M17.9 18A10.9 10.9 0 0 1 12 19c-7 0-11-7-11-7a18 18 0 0 1 5.1-5.9 M9.9 4.2A11 11 0 0 1 12 4c7 0 11 7 11 7a18 18 0 0 1-2.2 3.2 M1 1l22 22 M9.5 9.5a3 3 0 0 0 4 4",
  play: "M5 3l14 9-14 9V3z",
  plus: "M12 5v14 M5 12h14",
  trash:
    "M3 6h18 M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2 M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6",
  panel: "M3 4h18v16H3z M15 4v16",
  send: "M22 2L11 13 M22 2l-7 20-4-9-9-4 20-7z",
  refresh:
    "M23 4v6h-6 M1 20v-6h6 M3.5 9a9 9 0 0 1 14.9-3.4L23 10 M1 14l4.6 4.4A9 9 0 0 0 20.5 15",
  table: "M3 3h18v18H3z M3 9h18 M3 15h18 M9 3v18",
  settings:
    "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 7 19.4a1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.6 1.6 0 0 0-1.1-2.7H1a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 2.6 7a1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3H7a1.6 1.6 0 0 0 1-1.5V1a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1 1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8V7a1.6 1.6 0 0 0 1.5 1H23a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z",
  beaker:
    "M9 2h6 M10 2v6l-5 9a2 2 0 0 0 1.8 3h10.4A2 2 0 0 0 19 17l-5-9V2 M7 14h10",
  pipe: "M4 8a4 4 0 0 1 8 0v8a4 4 0 0 0 8 0",
  database:
    "M12 2c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3z M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5 M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
  info: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z M12 16v-4 M12 8h.01",
  wave: "M2 12c2 0 2-4 4-4s2 8 4 8 2-8 4-8 2 4 4 4",
  chevdown: "M6 9l6 6 6-6",
  message:
    "M21 11.5a8.4 8.4 0 0 1-9 8.4 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.2A8.4 8.4 0 0 1 12 3a8.4 8.4 0 0 1 9 8.5z",
  bolt: "M13 2L3 14h8l-1 8 10-12h-8l1-8z",
};

function Icon({ name, size = 16, className = "", style }) {
  return (
    <svg
      className={"feather " + className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      style={style}
    >
      {(ICONS[name] || "").split(" M").map((seg, i) => (
        <path key={i} d={i === 0 ? seg : "M" + seg} />
      ))}
    </svg>
  );
}

/* ---------- materials catalog (typical GPR dielectric props) ----------
   epsilon = relative permittivity, sigma = conductivity (S/m)        */
const MATERIALS = {
  topsoil: {
    label: "Topsoil / Loam",
    epsilon: 9,
    sigma: 0.012,
    color: "#b39c78",
    pattern: "soil",
  },
  drysand: {
    label: "Dry Sand",
    epsilon: 4,
    sigma: 0.001,
    color: "#dcc99c",
    pattern: "dots",
  },
  wetsand: {
    label: "Saturated Sand",
    epsilon: 25,
    sigma: 0.03,
    color: "#c3b184",
    pattern: "dots",
  },
  silt: {
    label: "Silt",
    epsilon: 14,
    sigma: 0.02,
    color: "#cabfa3",
    pattern: "soil",
  },
  dryclay: {
    label: "Dry Clay",
    epsilon: 6,
    sigma: 0.05,
    color: "#c0987f",
    pattern: "hatch",
  },
  wetclay: {
    label: "Wet Clay",
    epsilon: 28,
    sigma: 0.3,
    color: "#a8806b",
    pattern: "hatch",
  },
  gravel: {
    label: "Gravel",
    epsilon: 5,
    sigma: 0.005,
    color: "#aca799",
    pattern: "gravel",
  },
  limestone: {
    label: "Limestone",
    epsilon: 7,
    sigma: 0.003,
    color: "#a6abb0",
    pattern: "rock",
  },
  bedrock: {
    label: "Bedrock / Granite",
    epsilon: 6,
    sigma: 0.001,
    color: "#969ba2",
    pattern: "rock",
  },
  concrete: {
    label: "Concrete",
    epsilon: 6,
    sigma: 0.01,
    color: "#cdd1d5",
    pattern: "grid",
  },
  asphalt: {
    label: "Asphalt",
    epsilon: 5,
    sigma: 0.001,
    color: "#9a9ba0",
    pattern: "solid",
  },
  freshwater: {
    label: "Fresh Water",
    epsilon: 81,
    sigma: 0.005,
    color: "#9fb6c4",
    pattern: "wave",
  },
};
const MAT_KEYS = Object.keys(MATERIALS);

/* ---------- target (buried object) types ---------- */
const TARGET_TYPES = {
  metalpipe: {
    label: "Metal Pipe",
    epsilon: 0,
    sigma: 1e7,
    color: "#7e858f",
    kind: "pec",
    shape: "circle",
  },
  pvcpipe: {
    label: "PVC Pipe",
    epsilon: 3,
    sigma: 0,
    color: "#d8d2c6",
    kind: "diel",
    shape: "ring",
  },
  claypipe: {
    label: "Clay Pipe",
    epsilon: 5,
    sigma: 0.01,
    color: "#bf9a82",
    kind: "diel",
    shape: "ring",
  },
  cable: {
    label: "Buried Cable",
    epsilon: 0,
    sigma: 1e7,
    color: "#6c7785",
    kind: "pec",
    shape: "circle",
  },
  rebar: {
    label: "Rebar Mesh",
    epsilon: 0,
    sigma: 1e7,
    color: "#6c7785",
    kind: "pec",
    shape: "rebar",
  },
  void: {
    label: "Air Void",
    epsilon: 1,
    sigma: 0,
    color: "#f4f6f9",
    kind: "diel",
    shape: "dash",
  },
  boulder: {
    label: "Boulder",
    epsilon: 6,
    sigma: 0.001,
    color: "#9aa0a6",
    kind: "diel",
    shape: "blob",
  },
  landmine: {
    label: "Landmine",
    epsilon: 3,
    sigma: 0.01,
    color: "#b7794a",
    kind: "diel",
    shape: "circle",
  },
};
const TARGET_KEYS = Object.keys(TARGET_TYPES);

/* ---------- antennas / GPR systems ---------- */
const ANTENNAS = [
  { id: "gssi1500", label: "GSSI 1.5 GHz", freq: 1.5, sep: 0.04 },
  { id: "gssi900", label: "GSSI 900 MHz", freq: 0.9, sep: 0.06 },
  { id: "malå450", label: "MALÅ 450 MHz", freq: 0.45, sep: 0.12 },
  { id: "malå250", label: "MALÅ 250 MHz", freq: 0.25, sep: 0.18 },
  { id: "custom", label: "Custom Hertzian", freq: 1.2, sep: 0.05 },
];

/* ---------- downstream ML models the dataset feeds ---------- */
const ML_MODELS = [
  {
    id: "permnet",
    label: "Permittivity Estimator",
    arch: "CNN · regression",
    desc: "Predicts layer εr from B-scan",
    samples: "32k",
  },
  {
    id: "clf",
    label: "Target Classifier",
    arch: "ResNet-18",
    desc: "Pipe / rebar / void / clutter",
    samples: "58k",
  },
  {
    id: "unet",
    label: "Clutter Removal",
    arch: "U-Net",
    desc: "Suppresses surface clutter",
    samples: "41k",
  },
  {
    id: "depthreg",
    label: "Depth Regressor",
    arch: "XGBoost",
    desc: "Estimates target burial depth",
    samples: "27k",
  },
  {
    id: "seg",
    label: "Subsurface Segmenter",
    arch: "SegFormer",
    desc: "Layer boundary segmentation",
    samples: "19k",
  },
];

/* ---------- waveforms ---------- */
const WAVEFORMS = ["ricker", "gaussian", "gaussiandot", "sine"];

/* ============================================================
   initial model — blank canvas; the agent conversation builds the
   scene. The old demo scenario lives on as the "utility" preset.
   coordinates in metres; depth grows downward from surface (0)
   ============================================================ */
function makeInitialModel() {
  return {
    project: "untitled_dataset",
    domain: { width: 1.2, depth: 1.0, dx: 0.002 },
    acquisition: {
      antenna: "custom",
      frequency: 1.0,
      waveform: "ricker",
      timeWindow: 12,
      traceStep: 0.01,
      txrxSep: 0.05,
      surveyMode: "B-scan",
    },
    layers: [],
    targets: [],
  };
}

function makeUtilityModel() {
  return {
    project: "utility_survey_01",
    domain: { width: 1.4, depth: 0.9, dx: 0.002 },
    acquisition: {
      antenna: "gssi1500",
      frequency: 1.5,
      waveform: "ricker",
      timeWindow: 12,
      traceStep: 0.01,
      txrxSep: 0.04,
      surveyMode: "B-scan",
    },
    layers: [
      {
        id: uid("ly"),
        name: "Topsoil",
        material: "topsoil",
        thickness: 0.18,
        epsilon: 9,
        sigma: 0.012,
        visible: true,
      },
      {
        id: uid("ly"),
        name: "Dry Sand",
        material: "drysand",
        thickness: 0.3,
        epsilon: 4,
        sigma: 0.001,
        visible: true,
      },
      {
        id: uid("ly"),
        name: "Wet Clay",
        material: "wetclay",
        thickness: 0.42,
        epsilon: 28,
        sigma: 0.3,
        visible: true,
      },
    ],
    targets: [
      {
        id: uid("tg"),
        name: "PVC Water Main",
        type: "pvcpipe",
        x: 0.45,
        depth: 0.4,
        diameter: 0.11,
        visible: true,
      },
      {
        id: uid("tg"),
        name: "Power Conduit",
        type: "metalpipe",
        x: 0.95,
        depth: 0.28,
        diameter: 0.08,
        visible: true,
      },
    ],
  };
}

/* total modelled depth from stacked layers */
const layersDepth = (m) => m.layers.reduce((s, l) => s + l.thickness, 0);

/* ============================================================
   NATURAL-LANGUAGE COMMAND PARSER
   returns { reply, patch?(model)=>model, actions:[{sw,label,v}], run?, kind }
   Deterministic intent matching so the prototype is reliable offline.
   ============================================================ */
function matchMaterial(text) {
  const t = text.toLowerCase();
  for (const k of MAT_KEYS) {
    const lbl = MATERIALS[k].label.toLowerCase();
    if (
      t.includes(k) ||
      lbl
        .split(" / ")[0]
        .split(" ")
        .every((w) => t.includes(w))
    )
      return k;
  }
  if (/\bsand\b/.test(t)) return /wet|satur/.test(t) ? "wetsand" : "drysand";
  if (/\bclay\b/.test(t))
    return /wet|moist|satur/.test(t) ? "wetclay" : "dryclay";
  if (/\bsoil|loam|dirt|top\b/.test(t)) return "topsoil";
  if (/\bgravel\b/.test(t)) return "gravel";
  if (/\bsilt\b/.test(t)) return "silt";
  if (/\brock|bed ?rock|granite\b/.test(t)) return "bedrock";
  if (/\blimestone\b/.test(t)) return "limestone";
  if (/\bconcrete\b/.test(t)) return "concrete";
  if (/\basphalt|pavement\b/.test(t)) return "asphalt";
  if (/\bwater\b/.test(t)) return "freshwater";
  return null;
}
function matchTarget(text) {
  const t = text.toLowerCase();
  if (/metal|steel|copper|conduit|power/.test(t)) return "metalpipe";
  if (/pvc|plastic|water main|water pipe/.test(t)) return "pvcpipe";
  if (/clay pipe|sewer|terracotta/.test(t)) return "claypipe";
  if (/rebar|mesh|reinforc/.test(t)) return "rebar";
  if (/cable|wire|fiber|fibre/.test(t)) return "cable";
  if (/void|cavity|air pocket|sinkhole/.test(t)) return "void";
  if (/boulder|rock|stone/.test(t)) return "boulder";
  if (/mine|landmine|uxo|ordnance/.test(t)) return "landmine";
  if (/pipe/.test(t)) return "pvcpipe";
  return null;
}
function num(re, text, dflt) {
  const m = text.match(re);
  return m ? parseFloat(m[1]) : dflt;
}

function parseCommand(text, model) {
  const t = text.toLowerCase().trim();

  /* run / simulate */
  if (
    /^(run|simulate|solve|generate.*scan|preview|forward)\b/.test(t) ||
    /\brun (the )?(sim|simulation|forward|solver)\b/.test(t)
  ) {
    return {
      kind: "run",
      run: true,
      reply:
        "Launching the gprMax forward model across the scan line. I'll preview the synthetic B-scan below — this configuration becomes a labelled sample in the dataset.",
    };
  }

  /* generate dataset variations */
  let mGen = t.match(
    /(?:generate|create|add|make)\s+(\d+)?\s*(?:variation|sample|scenario|realiz)/,
  );
  if (mGen) {
    const n = mGen[1] ? parseInt(mGen[1]) : 200;
    return {
      kind: "dataset",
      datasetAdd: n,
      reply: `On it — sweeping permittivity, depth and geometry within plausible ranges to synthesise **${n.toLocaleString()} labelled variations** of this scenario. Each is tagged with ground-truth layer and target parameters for training.`,
    };
  }

  /* add a layer */
  if (/\badd|insert|append|put\b/.test(t) && /\blayer\b/.test(t)) {
    const mat = matchMaterial(t) || "drysand";
    const mm = MATERIALS[mat];
    const thk = num(
      /([\d.]+)\s*(?:m|metre|meter|cm)/,
      t,
      /cm/.test(t) ? 30 : 0.3,
    );
    const thickness = /cm/.test(t) ? thk / 100 : thk;
    return {
      kind: "layer",
      reply: `Added a **${mm.label}** layer (${fmt(thickness)} m). Typical dielectric: εr ≈ ${mm.epsilon}, σ ≈ ${mm.sigma} S/m — I've filled those from the reference catalogue, edit anytime.`,
      actions: [
        { sw: mm.color, label: mm.label, v: fmt(thickness) + " m" },
        { label: "Permittivity εr", v: String(mm.epsilon) },
        { label: "Conductivity σ", v: mm.sigma + " S/m" },
      ],
      patch: (m) => ({
        ...m,
        layers: [
          ...m.layers,
          {
            id: uid("ly"),
            name: mm.label,
            material: mat,
            thickness,
            epsilon: mm.epsilon,
            sigma: mm.sigma,
            visible: true,
          },
        ],
      }),
    };
  }

  /* add a target */
  const tk = matchTarget(t);
  if (tk && /\badd|insert|bury|put|place|there'?s|include\b/.test(t)) {
    const tt = TARGET_TYPES[tk];
    const depth = (() => {
      const d = num(
        /([\d.]+)\s*(?:m|metre|meter|cm)\s*(?:deep|depth|down|below)?/,
        t,
        /cm/.test(t) ? 35 : 0.35,
      );
      return /cm/.test(t) ? d / 100 : d;
    })();
    const diameter = (() => {
      const d = num(
        /(?:diam|width|wide|across|size)\D{0,8}([\d.]+)\s*(?:m|cm|mm)?/,
        t,
        null,
      );
      if (d == null) return 0.1;
      return /mm/.test(t)
        ? d / 1000
        : /cm/.test(t)
          ? d / 100
          : d > 1
            ? d / 100
            : d;
    })();
    const x = num(
      /(?:at\s*x\s*=?|position|x\s*=)\s*([\d.]+)/,
      t,
      round(0.2 + Math.random() * (model.domain.width - 0.4)),
    );
    return {
      kind: "target",
      reply: `Buried a **${tt.label}** at ${fmt(depth)} m depth${diameter ? `, ⌀ ${Math.round(diameter * 100)} cm` : ""}. ${tt.kind === "pec" ? "Modelled as a perfect electrical conductor — expect a bright hyperbolic reflection." : `Dielectric contrast: εr ≈ ${tt.epsilon}.`}`,
      actions: [
        { sw: tt.color, label: tt.label, v: fmt(depth) + " m deep" },
        { label: "Diameter", v: Math.round(diameter * 100) + " cm" },
      ],
      patch: (m) => ({
        ...m,
        targets: [
          ...m.targets,
          {
            id: uid("tg"),
            name: tt.label,
            type: tk,
            x: clamp(x, 0.05, m.domain.width - 0.05),
            depth,
            diameter,
            visible: true,
          },
        ],
      }),
    };
  }

  /* set frequency / antenna */
  let mFreq = t.match(/([\d.]+)\s*(ghz|mhz)/);
  if (mFreq && /freq|antenn|ghz|mhz/.test(t)) {
    let f = parseFloat(mFreq[1]);
    if (mFreq[2] === "mhz") f = f / 1000;
    return {
      kind: "acq",
      reply: `Set centre frequency to **${f >= 1 ? f + " GHz" : f * 1000 + " MHz"}**. ${f >= 1 ? "Higher frequency → finer resolution but shallower penetration." : "Lower frequency → deeper penetration, coarser resolution."}`,
      actions: [
        {
          label: "Centre frequency",
          v: f >= 1 ? f + " GHz" : f * 1000 + " MHz",
        },
      ],
      patch: (m) => ({ ...m, acquisition: { ...m.acquisition, frequency: f } }),
    };
  }

  /* moisture: make wet / dry */
  if (
    /(make|set).*(wet|moist|satur|dry)/.test(t) ||
    /(wet|dry|moist|satur)\w*\s+(soil|ground|conditions)/.test(t)
  ) {
    const wet = /wet|moist|satur/.test(t);
    return {
      kind: "moisture",
      reply: wet
        ? "Raised the water table — bumping permittivity and conductivity on the soil layers to saturated values. This strengthens reflections and increases attenuation."
        : "Set dry conditions — lowered permittivity and conductivity across soil layers for a low-loss subsurface.",
      patch: (m) => ({
        ...m,
        layers: m.layers.map((l) => {
          const isSoil = [
            "topsoil",
            "drysand",
            "wetsand",
            "silt",
            "dryclay",
            "wetclay",
            "gravel",
          ].includes(l.material);
          if (!isSoil) return l;
          return wet
            ? {
                ...l,
                epsilon: round(Math.min(30, l.epsilon * 2.4), 0),
                sigma: round(Math.min(0.5, l.sigma * 6), 3),
              }
            : {
                ...l,
                epsilon: round(Math.max(3, l.epsilon * 0.5), 0),
                sigma: round(Math.max(0.001, l.sigma * 0.25), 3),
              };
        }),
      }),
    };
  }

  /* remove last */
  if (/\b(remove|delete|drop|undo)\b/.test(t)) {
    if (/target|pipe|object|rebar|void/.test(t))
      return {
        kind: "rm",
        reply: "Removed the most recently added target.",
        patch: (m) => ({ ...m, targets: m.targets.slice(0, -1) }),
      };
    if (/layer/.test(t))
      return {
        kind: "rm",
        reply: "Removed the bottom layer.",
        patch: (m) => ({ ...m, layers: m.layers.slice(0, -1) }),
      };
  }

  /* clear / reset */
  if (/^(clear|reset|new|start over|blank)\b/.test(t)) {
    return {
      kind: "reset",
      reply:
        "Cleared the scene to bare ground. Describe the subsurface you want and I'll build it.",
      patch: (m) => ({
        ...m,
        layers: [
          {
            id: uid("ly"),
            name: "Subgrade",
            material: "topsoil",
            thickness: m.domain.depth,
            epsilon: 9,
            sigma: 0.012,
            visible: true,
          },
        ],
        targets: [],
      }),
    };
  }

  /* scenario presets */
  if (/utility|pipe.*survey|locate.*pipe/.test(t)) return scenarioUtility();
  if (/landmine|demining|uxo|ordnance/.test(t)) return scenarioMine();
  if (/rebar|concrete.*inspect|bridge|slab/.test(t)) return scenarioRebar();
  if (/moisture|agric|root|farm/.test(t)) return scenarioMoisture();

  /* questions about the sim — informational */
  if (
    /\?$/.test(t) ||
    /^(what|why|how|explain|which|tell me|can you|is the|are the)\b/.test(t)
  ) {
    return { kind: "answer", answer: true, raw: text };
  }

  return { kind: "fallback", answer: true, raw: text };
}

/* ---------- scenario presets ---------- */
function scenarioUtility() {
  return {
    kind: "scenario",
    reply:
      "Built a **buried-utility survey**: topsoil over dry sand over wet clay, with a PVC water main and a metal power conduit at different depths. Frequency set to 1.5 GHz for shallow utilities. Hit **Run** to preview the radargram.",
    actions: [
      { sw: MATERIALS.topsoil.color, label: "3 soil layers", v: "0.90 m" },
      { sw: TARGET_TYPES.pvcpipe.color, label: "PVC water main", v: "0.40 m" },
      { sw: TARGET_TYPES.metalpipe.color, label: "Power conduit", v: "0.28 m" },
    ],
    patch: () => makeUtilityModel(),
  };
}
function scenarioMine() {
  return {
    kind: "scenario",
    reply:
      "Configured a **landmine detection** scene: dry sandy soil with a shallow plastic-cased landmine and a clutter rock. Switched to 900 MHz for depth/clutter balance.",
    actions: [
      { sw: TARGET_TYPES.landmine.color, label: "Landmine", v: "0.10 m" },
      { sw: TARGET_TYPES.boulder.color, label: "Clutter rock", v: "0.18 m" },
    ],
    patch: (m) => ({
      ...m,
      project: "demining_01",
      acquisition: { ...m.acquisition, antenna: "gssi900", frequency: 0.9 },
      domain: { width: 1.0, depth: 0.6, dx: 0.002 },
      layers: [
        {
          id: uid("ly"),
          name: "Dry Sand",
          material: "drysand",
          thickness: 0.6,
          epsilon: 4,
          sigma: 0.001,
          visible: true,
        },
      ],
      targets: [
        {
          id: uid("tg"),
          name: "Landmine",
          type: "landmine",
          x: 0.4,
          depth: 0.1,
          diameter: 0.12,
          visible: true,
        },
        {
          id: uid("tg"),
          name: "Clutter Rock",
          type: "boulder",
          x: 0.72,
          depth: 0.18,
          diameter: 0.07,
          visible: true,
        },
      ],
    }),
  };
}
function scenarioRebar() {
  return {
    kind: "scenario",
    reply:
      "Set up a **concrete inspection** scene: a 0.30 m concrete slab over subgrade, with a rebar mesh at 0.06 m cover. Bumped to 1.5 GHz for resolution.",
    actions: [
      { sw: MATERIALS.concrete.color, label: "Concrete slab", v: "0.30 m" },
      { sw: TARGET_TYPES.rebar.color, label: "Rebar mesh", v: "0.06 m cover" },
    ],
    patch: (m) => ({
      ...m,
      project: "slab_inspection_01",
      acquisition: { ...m.acquisition, antenna: "gssi1500", frequency: 1.5 },
      domain: { width: 1.2, depth: 0.55, dx: 0.0015 },
      layers: [
        {
          id: uid("ly"),
          name: "Concrete",
          material: "concrete",
          thickness: 0.3,
          epsilon: 6,
          sigma: 0.01,
          visible: true,
        },
        {
          id: uid("ly"),
          name: "Subgrade",
          material: "gravel",
          thickness: 0.25,
          epsilon: 5,
          sigma: 0.005,
          visible: true,
        },
      ],
      targets: [
        {
          id: uid("tg"),
          name: "Rebar mesh",
          type: "rebar",
          x: 0.6,
          depth: 0.06,
          diameter: 0.016,
          visible: true,
        },
      ],
    }),
  };
}
function scenarioMoisture() {
  return {
    kind: "scenario",
    reply:
      "Built a **soil moisture** scene: a moisture gradient from dry topsoil into saturated clay, no hard targets. Great for permittivity-regression training data.",
    actions: [
      { sw: MATERIALS.topsoil.color, label: "Dry topsoil", v: "εr 9" },
      { sw: MATERIALS.wetclay.color, label: "Saturated clay", v: "εr 28" },
    ],
    patch: (m) => ({
      ...m,
      project: "moisture_profile_01",
      domain: { width: 1.4, depth: 1.0, dx: 0.003 },
      layers: [
        {
          id: uid("ly"),
          name: "Dry Topsoil",
          material: "topsoil",
          thickness: 0.25,
          epsilon: 9,
          sigma: 0.012,
          visible: true,
        },
        {
          id: uid("ly"),
          name: "Moist Silt",
          material: "silt",
          thickness: 0.35,
          epsilon: 14,
          sigma: 0.02,
          visible: true,
        },
        {
          id: uid("ly"),
          name: "Saturated Clay",
          material: "wetclay",
          thickness: 0.4,
          epsilon: 28,
          sigma: 0.3,
          visible: true,
        },
      ],
      targets: [],
    }),
  };
}

/* informational answers (no live LLM dependency, deterministic) */
function localAnswer(text, model) {
  const t = text.toLowerCase();
  if (/permittiv|dielectric|epsilon|εr/.test(t))
    return (
      "Relative permittivity (εr) governs how fast EM waves travel through a medium — velocity ≈ c/√εr. In this scene it ranges from " +
      Math.min(...model.layers.map((l) => l.epsilon)) +
      " to " +
      Math.max(...model.layers.map((l) => l.epsilon)) +
      ". Higher εr (wet soils) slows the wave and produces stronger reflections at interfaces."
    );
  if (/conductiv|sigma|σ|atten/.test(t))
    return (
      "Conductivity (σ, S/m) drives signal attenuation. The lossy layer here is the wet clay at σ = " +
      (model.layers.find((l) => l.sigma >= 0.1)?.sigma ?? "—") +
      " S/m, which will rapidly damp energy and limit penetration below it."
    );
  if (/depth|penetrat|how deep/.test(t))
    return (
      "Penetration depth depends on frequency and soil loss. At " +
      model.acquisition.frequency +
      " GHz through these layers, expect usable returns to roughly " +
      fmt(layersDepth(model)) +
      " m before the wet clay attenuates the signal. Drop to 450 MHz if you need to see deeper."
    );
  if (/frequenc|antenna|resolution/.test(t))
    return (
      "You're at " +
      model.acquisition.frequency +
      " GHz (" +
      (ANTENNAS.find((a) => a.id === model.acquisition.antenna)?.label ||
        "custom") +
      "). Rule of thumb: vertical resolution ≈ a quarter wavelength. Higher frequency sharpens targets but won't see as deep."
    );
  if (/hyperbola|why.*curve|reflection/.test(t))
    return "Each buried point target produces a diffraction hyperbola in the B-scan: directly above the object the travel-time is shortest, and it increases as the antenna moves away. The hyperbola's curvature encodes the wave velocity — and therefore the soil permittivity.";
  if (/dataset|training|ml|label/.test(t))
    return 'Every scenario you configure here is a fully-labelled sample: the soil/target parameters are the ground truth, and a Run produces the synthetic B-scan input. Use **"generate 500 variations"** to sweep the parameters and mass-produce training data.';
  if (/gprmax|solver/.test(t))
    return "gprMax is the open-source FDTD solver doing the forward electromagnetic modelling. NL2Sim translates your description into a gprMax input deck (#material, #cylinder, #box, #waveform…), runs it, and tags the output for the dataset. Use **Download** to export the .in file.";
  return 'I can build and tweak GPR scenarios, explain the physics, or mass-produce labelled training data. Try: "add a wet clay layer 0.4 m thick", "bury a metal pipe at 0.3 m", "set 900 MHz", or "generate 500 variations".';
}

/* ============================================================
   backend scene -> viz model mapping (model_update WebSocket event)
   ============================================================ */

/* Pick a MATERIALS key (color/pattern only — physics stays backend-side)
   from a layer's texture + moisture, falling back to nearest catalog εr. */
function materialKeyForLayer({ sandPct, clayPct, thetaV, epsilon }) {
  const wet = thetaV != null && thetaV >= 0.2;
  if (sandPct != null && clayPct != null) {
    const siltPct = 100 - sandPct - clayPct;
    if (clayPct >= 35 || (clayPct >= sandPct && clayPct >= siltPct))
      return wet ? "wetclay" : "dryclay";
    if (sandPct >= 50 || (sandPct >= clayPct && sandPct >= siltPct))
      return wet ? "wetsand" : "drysand";
    if (siltPct >= 50) return "silt";
    return "topsoil";
  }
  if (epsilon != null) {
    let best = "topsoil";
    let bestD = Infinity;
    for (const k of MAT_KEYS) {
      const d = Math.abs(MATERIALS[k].epsilon - epsilon);
      if (d < bestD) {
        bestD = d;
        best = k;
      }
    }
    return best;
  }
  return "topsoil";
}

/* One scene target -> viz model target. `x` is ABSOLUTE canvas metres — the
   caller resolves the backend's center-relative offset against the current
   domain width (x = width/2 + x_offset), mirroring the backend's own
   resolution against the derived grid. Boxes render as rects (shape). */
function sceneTargetToModel(id, kind, name, material, x, depth, radius, width, height) {
  const isBox = kind === "box";
  return {
    id,
    name: name || "Target",
    type: material === "pec" ? "metalpipe" : "pvcpipe",
    kind: kind || "cylinder",
    shape: isBox ? "rect" : "circle",
    x: x ?? 0,
    depth: depth ?? 0,
    diameter: isBox ? (width ?? 0.1) : 2 * (radius ?? 0.05),
    width: isBox ? (width ?? 0.1) : undefined,
    height: isBox ? (height ?? 0.1) : undefined,
    visible: true,
  };
}

/* Map the backend `scene` payload to the viz model shape.
   vizTab: "overview" (range midpoints + thickness uncertainty bands)
         | "sample"   (one concrete realization from scene.samples)   */
function sceneToModel(scene, vizTab, sampleIdx) {
  const base = makeInitialModel();
  if (!scene) return base;

  const grid = scene.grid;
  const domain = grid
    ? {
        width: round(grid.domain_x_m, 3),
        depth: round(grid.depth_z_m, 3),
        dx: grid.dx_m,
      }
    : {
        width: round(scene.domain?.width_m ?? base.domain.width, 3),
        depth: round(scene.domain?.depth_m ?? base.domain.depth, 3),
        dx: scene.domain?.dx_m ?? base.domain.dx,
      };

  const acq = scene.acquisition || {};
  const acquisition = {
    ...base.acquisition,
    frequency: acq.frequency_ghz ?? base.acquisition.frequency,
    waveform: acq.waveform ?? base.acquisition.waveform,
    txrxSep: acq.txrx_sep_m ?? base.acquisition.txrxSep,
    timeWindow: acq.time_window_ns ?? base.acquisition.timeWindow,
  };

  let layers = [];
  let targets = [];
  const items = scene.samples?.items || [];

  if (vizTab === "sample" && items.length > 0) {
    const item = items[clamp(sampleIdx || 0, 0, items.length - 1)];
    layers = item.layers.map((l, i) => ({
      id: "ly_" + i,
      name: l.name || "Layer " + (i + 1),
      material: materialKeyForLayer({
        sandPct: l.sand_pct,
        clayPct: l.clay_pct,
        thetaV: l.theta_v_mid,
        epsilon: l.eps_mid,
      }),
      thickness: round(l.thickness_m, 3),
      epsilon: l.eps_mid != null ? round(l.eps_mid, 1) : null,
      sigma: null,
      visible: true,
    }));
    targets = (item.targets || []).map((t, i) =>
      sceneTargetToModel(
        "tg_" + i,
        t.kind,
        t.name,
        t.material,
        domain.width / 2 + (t.x_offset_m ?? 0),
        t.depth_m,
        t.radius_m,
        t.width_m,
        t.height_m,
      ),
    );
  } else if (scene.ranges) {
    layers = (scene.ranges.layers || []).map((l, i) => ({
      id: "ly_" + i,
      name: l.name || "Layer " + (i + 1),
      material: materialKeyForLayer({
        sandPct: l.sand_pct_mid,
        clayPct: l.clay_pct_mid,
        thetaV: l.theta_v_mid,
        epsilon: l.eps_mid,
      }),
      thickness: round(l.thickness_mid_m, 3),
      thicknessMin: l.thickness_min_m,
      thicknessMax: l.thickness_max_m,
      epsilon: l.eps_mid != null ? round(l.eps_mid, 1) : null,
      sigma: null,
      visible: true,
    }));
    targets = (scene.ranges.targets || []).map((t, i) =>
      sceneTargetToModel(
        "tg_" + i,
        t.kind,
        t.name,
        t.material,
        domain.width / 2 + (t.x_offset_mid_m ?? 0),
        t.depth_mid_m,
        t.radius_mid_m,
        t.width_mid_m,
        t.height_mid_m,
      ),
    );
  }

  return {
    project: scene.project || base.project,
    domain,
    acquisition,
    layers,
    targets,
  };
}

/* Caveats shown beside the canvas on the Overview tab: the overview is a
   mental model built from range midpoints and placeholders. Each caveat
   lists an assumption that holds ONLY until the pipeline derives the real
   value — items drop out as the truth becomes available. */
function overviewCaveats(scene) {
  if (!scene) return [];
  const out = [];
  const layers = scene.ranges?.layers || [];
  const rangeTargets = scene.ranges?.targets || [];
  const nSamples = scene.samples?.items?.length || 0;
  const epsProvisional = layers.some((l) => l.eps_provisional);

  if (scene.domain?.provisional)
    out.push(
      "Domain size is a placeholder (~1.2× the layer stack) — the real grid is derived later from the wavelength budget.",
    );
  if (layers.length > 0) {
    out.push(
      "Layer thicknesses are range midpoints; the shaded bands show the min–max spread each sample is drawn from.",
    );
    out.push(
      epsProvisional
        ? "εr is a preview at midpoint composition, evaluated at a placeholder 0.9 GHz until the waveform frequency is set."
        : "εr is a preview at midpoint composition — each sample gets its own derived εr.",
    );
    out.push(
      "σ is not previewed at all — gprMax derives conductivity at model-build time.",
    );
    out.push(
      "Layer colors are a display classification from texture, not a physical material assignment.",
    );
  }
  if (rangeTargets.length > 0) {
    out.push(
      "Objects are drawn at the midpoints of their ranges; actual positions and sizes are drawn per sample.",
    );
    if (rangeTargets.some((t) => t.static))
      out.push(
        "Fixed objects (min = max ranges) appear identically in every sample.",
      );
  }
  if (!scene.acquisition?.frequency_ghz)
    out.push(
      "Frequency and antenna values are defaults until the waveform and antenna stages complete.",
    );
  if (nSamples > 0)
    out.push(
      `${scene.samples.total} concrete realization(s) exist — the Samples tab is the ground truth.`,
    );
  return out;
}

Object.assign(window, {
  uid,
  clamp,
  round,
  fmt,
  ICONS,
  Icon,
  MATERIALS,
  MAT_KEYS,
  TARGET_TYPES,
  TARGET_KEYS,
  ANTENNAS,
  ML_MODELS,
  WAVEFORMS,
  makeInitialModel,
  makeUtilityModel,
  layersDepth,
  parseCommand,
  localAnswer,
  materialKeyForLayer,
  sceneToModel,
  overviewCaveats,
});
