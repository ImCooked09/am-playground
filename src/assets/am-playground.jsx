import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";

/* ==================================================================
   How AM radio works

   s(t) = [1 + m·cos(2π·fm·t)] · cos(2π·fc·t)

   Eleven steps. Each declares target parameters and moving between
   steps glides toward them, so the animation carries the explanation
   rather than decorating it. Figures are annotated on the canvas with
   leader lines instead of sitting under a legend, which is how a good
   textbook plate works and how this reads as a figure rather than a
   chart in a box.
   ================================================================== */

const FS = 2048;
const N = 8192;
const BIN = FS / N;
const F_VIEW = 110;
const PAD = { l: 46, r: 18, t: 30, b: 26 };
const TWEEN_MS = 1500;

/* ---------------- DSP ---------------- */

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const vr = re[b] * cr - im[b] * ci;
        const vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr; im[a] += vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildSignals({ fc, fm, m, snr }, seed, needSpectrum) {
  const msg = new Float32Array(N);
  const car = new Float32Array(N);
  const env = new Float32Array(N);
  const envNeg = new Float32Array(N);
  const sig = new Float32Array(N);
  let power = 0;

  for (let n = 0; n < N; n++) {
    const t = n / FS;
    const mv = Math.cos(2 * Math.PI * fm * t);
    const cv = Math.cos(2 * Math.PI * fc * t);
    const ev = 1 + m * mv;
    const s = ev * cv;
    msg[n] = mv; car[n] = cv; env[n] = ev; envNeg[n] = -ev; sig[n] = s;
    power += s * s;
  }
  power /= N;

  const sigma = Math.sqrt(power / Math.pow(10, snr / 10));
  if (sigma > 1e-9) {
    const rnd = mulberry32(seed);
    for (let n = 0; n < N; n += 2) {
      const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
      const r = Math.sqrt(-2 * Math.log(u1));
      sig[n] += sigma * r * Math.cos(2 * Math.PI * u2);
      if (n + 1 < N) sig[n + 1] += sigma * r * Math.sin(2 * Math.PI * u2);
    }
  }

  /* The FFT is the costly part, so skip it while the spectrum is
     hidden. During a glide this runs on every frame. */
  let spec = null;
  if (needSpectrum) {
    const re = new Float64Array(N), im = new Float64Array(N);
    for (let n = 0; n < N; n++) re[n] = sig[n];
    fft(re, im);
    const bins = Math.ceil(F_VIEW / BIN) + 1;
    spec = new Float32Array(bins);
    for (let k = 0; k < bins; k++) spec[k] = (2 * Math.hypot(re[k], im[k])) / N;
  }

  const fCut = Math.sqrt(Math.max(fm, 0.5) * fc);
  const a = 1 - Math.exp((-2 * Math.PI * fCut) / FS);
  const rec = new Float32Array(N);
  let y = Math.abs(sig[0]), mean = 0;
  for (let n = 0; n < N; n++) {
    y += a * (Math.abs(sig[n]) - y);
    rec[n] = y; mean += y;
  }
  mean /= N;
  const scale = Math.PI / (2 * Math.max(m, 0.05));
  for (let n = 0; n < N; n++) rec[n] = (rec[n] - mean) * scale;

  return { msg, car, env, envNeg, sig, rec, spec };
}

/* ---------------- the lesson ---------------- */

const LESSON = [
  {
    id: "message", title: "The message",
    body: "This slow wave is what you want to send. A voice, some music, data. On its own it cannot travel: radiating a 3 kHz signal directly would need an antenna about 100 km long.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["msg"], mark: [],
  },
  {
    id: "carrier", title: "So borrow a faster wave",
    body: "A carrier oscillates far more quickly, and a fast wave needs only a short antenna. It can travel anywhere. It just carries nothing yet.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["msg", "car"], mark: [],
  },
  {
    id: "ride", title: "Ride one on the other",
    body: "Let the message control the carrier's height. Where the message rises the carrier grows tall, where it dips the carrier shrinks. Its frequency never changes. Only the amplitude does.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["mod"], mark: [],
  },
  {
    id: "envelope", title: "The outline is the message",
    body: "Trace the peaks of the carrier and the message comes straight back. That outline is called the envelope. The whole of AM is this single idea: hide the message in the envelope.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["mod"], mark: ["envelope"],
  },
  {
    id: "index", title: "How deep to go",
    body: "The modulation index m sets the depth. At 0.35 the envelope barely breathes. Watch it open up as m climbs toward 1.",
    p: { fc: 40, fm: 4, m: 0.95, snr: 50 }, show: ["mod"], mark: ["envelope"],
  },
  {
    id: "over", title: "Push it too far",
    body: "Past m = 1 the envelope crosses zero and the carrier flips phase. A receiver cannot tell a flip from a dip, so it folds those troughs back upward. The information in them is gone for good.",
    p: { fc: 40, fm: 4, m: 1.35, snr: 50 }, show: ["mod", "rec"], mark: ["flip", "humps"],
    predict: {
      q: "Before it happens: what will the recovered message do?",
      options: ["Just get louder", "Flatten off at the peaks", "Grow extra humps where the troughs were"],
      answer: 2,
      note: "The detector measures size, not sign, so a negative envelope comes back positive.",
    },
  },
  {
    id: "spectrum", title: "Only three frequencies leave",
    body: "Multiply the two cosines and the product identity splits them into exactly three: the carrier in the middle, plus one sideband either side at fc + fm and fc − fm. Nothing else is transmitted.",
    p: { fc: 40, fm: 4, m: 0.7, snr: 50 }, show: ["mod", "spec"], mark: ["spikes"],
  },
  {
    id: "bandwidth", title: "How much room it needs",
    body: "Raise the message frequency and the sidebands slide outward. The gap between them is the bandwidth, and it always works out to twice the message frequency.",
    p: { fc: 40, fm: 10, m: 0.7, snr: 50 }, show: ["spec"], mark: ["bw"],
    predict: {
      q: "Raise the message frequency. What do the sidebands do?",
      options: ["Move closer together", "Slide further apart", "Stay put and grow taller"],
      answer: 1,
      note: "They sit at fc ± fm, so raising fm pushes them out symmetrically.",
    },
  },
  {
    id: "cost", title: "What the carrier costs you",
    body: "The middle spike is the tallest by far, and it carries no information at all. Only the sidebands do. Even at m = 1, barely a third of your transmitter power does useful work. That is exactly why single sideband exists.",
    p: { fc: 40, fm: 6, m: 1.0, snr: 50 }, show: ["spec"], mark: ["waste"],
  },
  {
    id: "noise", title: "Then the channel gets in the way",
    body: "Real links add noise. Watch where it shows up first: the spectrum floor lifts long before the waveform looks damaged. This is why engineers watch spectra rather than waveforms.",
    p: { fc: 40, fm: 6, m: 0.7, snr: 7 }, show: ["mod", "spec"], mark: ["floor"],
  },
  {
    id: "recover", title: "Getting it back",
    body: "The receiver rectifies the signal, low-passes away the carrier ripple, then drops the DC. If the green curve lands on the blue one, the link worked.",
    p: { fc: 40, fm: 5, m: 0.7, snr: 30 }, show: ["mod", "rec"], mark: ["match"],
  },
];

/* ---------------- ink ---------------- */

const C = {
  field: "#F6F4EF",
  fine: "#E0DCD0",
  major: "#CFC9B8",
  frame: "#B6AF9C",
  tick: "#7C7566",
  label: "#3A4550",
  message: "#1F5F8B",
  carrier: "#24313A",
  envelope: "#B3306B",
  bars: "#8E9AA2",
  spike: "#BE7414",
  recovered: "#1B6E58",
};
const FONT = "'SF Pro Display', -apple-system, Inter, system-ui, sans-serif";

/* ---------------- canvas helpers ---------------- */

function useLoop(drawRef) {
  const ref = useRef(null);
  useEffect(() => {
    let raf = 0;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const cv = ref.current;
      if (!cv || !drawRef.current) return;
      const box = cv.parentElement;
      if (!box) return;
      const w = box.clientWidth, h = box.clientHeight;
      if (w <= 0 || h <= 0) return;
      const dpr = window.devicePixelRatio || 1;
      if (cv.width !== Math.round(w * dpr)) {
        cv.width = Math.round(w * dpr);
        cv.height = Math.round(h * dpr);
        cv.style.width = w + "px";
        cv.style.height = h + "px";
      }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      drawRef.current(ctx, w, h);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [drawRef]);
  return ref;
}

/* Module scope on purpose. Declared inside the component it becomes a
   new type every render, and React would rebuild the canvas sixty
   times a second. */
function Figure({ open, caption, height, cref }) {
  return (
    <figure className={open ? "fig open" : "fig"} aria-hidden={!open}>
      <div className="plate" style={{ height }}><canvas ref={cref} /></div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

const rectOf = (w, h) => ({
  x: PAD.l, y: PAD.t,
  w: Math.max(10, w - PAD.l - PAD.r),
  h: Math.max(10, h - PAD.t - PAD.b),
});

/* Two-level graticule, the way plotting paper actually rules up. */
function plate(ctx, w, h, r) {
  ctx.fillStyle = C.field;
  ctx.fillRect(0, 0, w, h);
  ctx.lineWidth = 1;
  ctx.strokeStyle = C.fine;
  ctx.beginPath();
  for (let i = 1; i < 20; i++) {
    if (i % 5 === 0) continue;
    const x = Math.round(r.x + (i / 20) * r.w) + 0.5;
    ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h);
  }
  for (let i = 1; i < 8; i++) {
    if (i % 2 === 0) continue;
    const y = Math.round(r.y + (i / 8) * r.h) + 0.5;
    ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y);
  }
  ctx.stroke();
  ctx.strokeStyle = C.major;
  ctx.beginPath();
  for (let i = 1; i < 4; i++) {
    const x = Math.round(r.x + (i / 4) * r.w) + 0.5;
    ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h);
  }
  for (let i = 1; i < 4; i++) {
    const y = Math.round(r.y + (i / 4) * r.h) + 0.5;
    ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y);
  }
  ctx.stroke();
  ctx.strokeStyle = C.frame;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h);
}

function yScale(ctx, r, range) {
  ctx.fillStyle = C.tick;
  ctx.font = `12px ${FONT}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [-1, 0, 1].forEach((v) => {
    const y = r.y + r.h / 2 - (v / range) * (r.h / 2 - 4);
    ctx.fillText(String(v), r.x - 8, y);
  });
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function zeroRule(ctx, r) {
  ctx.strokeStyle = C.major;
  ctx.beginPath();
  const y = Math.round(r.y + r.h / 2) + 0.5;
  ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y);
  ctx.stroke();
}

function curve(ctx, r, data, count, off, range, color, lw, dash) {
  ctx.save();
  ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  const mid = r.y + r.h / 2, k = (r.h / 2 - 4) / range;
  for (let i = 0; i < count; i++) {
    const x = r.x + (i / (count - 1)) * r.w;
    const y = mid - data[(off + i) % N] * k;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

/* Leader line from a point on a curve out to a label. */
function callout(ctx, tx, ty, lx, ly, text, color, align) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(tx, ty, 2.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(tx, ty);
  ctx.lineTo(lx, ly);
  ctx.stroke();
  ctx.font = `12.5px ${FONT}`;
  ctx.textAlign = align || "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, lx + (align === "right" ? -6 : 6), ly);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

/* Trace names sit inside the plate, so no legend block is needed. */
function keys(ctx, r, items) {
  ctx.font = `12.5px ${FONT}`;
  ctx.textBaseline = "alphabetic";
  let x = r.x + 2;
  items.forEach(({ label, color }) => {
    ctx.fillStyle = color;
    ctx.fillRect(x, r.y - 13, 13, 2.5);
    ctx.fillText(label, x + 18, r.y - 9);
    x += 18 + ctx.measureText(label).width + 20;
  });
}

/* ---------------- app ---------------- */

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function readUrl() {
  try {
    const q = new URLSearchParams(window.location.search);
    if (q.has("fc") || q.has("m")) {
      return {
        mode: "explore",
        p: {
          fc: clamp(+q.get("fc") || 40, 8, 80),
          fm: clamp(+q.get("fm") || 4, 1, 12),
          m: clamp(+q.get("m") || 0.6, 0, 1.6),
          snr: clamp(+q.get("snr") || 45, 0, 50),
        },
      };
    }
    if (q.has("step")) {
      const i = clamp(parseInt(q.get("step"), 10) - 1, 0, LESSON.length - 1);
      return { mode: "lesson", step: i, p: LESSON[i].p };
    }
  } catch (e) { /* sandboxed frame */ }
  return null;
}

export default function AmLesson() {
  const initial = useMemo(readUrl, []);
  const [mode, setMode] = useState(initial?.mode || "lesson");
  const [step, setStep] = useState(initial?.step ?? 0);
  const [params, setParams] = useState(initial?.p || LESSON[0].p);
  const [answered, setAnswered] = useState({});
  const [picked, setPicked] = useState(null);
  const [running, setRunning] = useState(true);
  const [off, setOff] = useState(0);
  const [periods, setPeriods] = useState(2);
  const [seed] = useState(1234);
  const [copied, setCopied] = useState(false);

  const paramsRef = useRef(params);
  paramsRef.current = params;
  const tween = useRef(null);
  const runRef = useRef(running);
  runRef.current = running;

  const L = LESSON[step];
  const inLesson = mode === "lesson";
  const gated = inLesson && L.predict && !answered[L.id];
  const show = inLesson ? L.show : ["msg", "mod", "spec", "rec"];
  const mark = inLesson ? L.mark : [];
  const needSpec = show.includes("spec");

  const sig = useMemo(() => buildSignals(params, seed, needSpec), [params, seed, needSpec]);
  const count = Math.min(N, Math.max(64, Math.round((periods / params.fm) * FS)));
  const over = params.m > 1;
  const eff = (100 * params.m * params.m) / (2 + params.m * params.m);

  useEffect(() => {
    let raf = 0;
    const loop = (now) => {
      const tw = tween.current;
      if (tw) {
        const t = clamp((now - tw.t0) / tw.ms, 0, 1);
        const k = ease(t);
        const mix = (a, b) => a + (b - a) * k;
        setParams({
          fc: mix(tw.from.fc, tw.to.fc),
          fm: mix(tw.from.fm, tw.to.fm),
          m: mix(tw.from.m, tw.to.m),
          snr: mix(tw.from.snr, tw.to.snr),
        });
        if (t >= 1) tween.current = null;
      }
      if (runRef.current) setOff((o) => (o + 4) % N);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const glideTo = useCallback((to) => {
    const quick = window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (quick) { setParams(to); return; }
    tween.current = { from: { ...paramsRef.current }, to, t0: performance.now(), ms: TWEEN_MS };
  }, []);

  const goStep = useCallback((i) => {
    const n = clamp(i, 0, LESSON.length - 1);
    setStep(n);
    setPicked(null);
    const s = LESSON[n];
    if (!s.predict || answered[s.id]) glideTo(s.p);
  }, [answered, glideTo]);

  useEffect(() => {
    const onKey = (e) => {
      if (!inLesson) return;
      if (e.key === "ArrowRight") goStep(step + 1);
      if (e.key === "ArrowLeft") goStep(step - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inLesson, step, goStep]);

  useEffect(() => {
    try {
      const q = inLesson
        ? `?step=${step + 1}`
        : `?fc=${Math.round(params.fc)}&fm=${Math.round(params.fm)}` +
          `&m=${params.m.toFixed(2)}&snr=${Math.round(params.snr)}`;
      window.history.replaceState(null, "", q);
    } catch (e) { /* sandboxed frame */ }
  }, [inLesson, step, params]);

  const answer = (i) => {
    setPicked(i);
    setAnswered((a) => ({ ...a, [L.id]: true }));
    setTimeout(() => glideTo(L.p), 700);
  };

  const share = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch (e) { setCopied(false); }
  };

  const set = (k) => (v) => {
    tween.current = null;
    setParams((p) => ({ ...p, [k]: v }));
  };

  const view = useRef({});
  view.current = { sig, count, off, params, over, mark };

  /* ---- figures ---- */

  const drawMsg = useRef(null);
  drawMsg.current = (ctx, w, h) => {
    const { sig, count, off } = view.current;
    const r = rectOf(w, h);
    plate(ctx, w, h, r); zeroRule(ctx, r); yScale(ctx, r, 1.05);
    keys(ctx, r, [{ label: "message", color: C.message }]);
    curve(ctx, r, sig.msg, count, off, 1.05, C.message, 2.4);
  };

  const drawCar = useRef(null);
  drawCar.current = (ctx, w, h) => {
    const { sig, count, off } = view.current;
    const r = rectOf(w, h);
    plate(ctx, w, h, r); zeroRule(ctx, r); yScale(ctx, r, 1.05);
    keys(ctx, r, [{ label: "carrier", color: C.carrier }]);
    curve(ctx, r, sig.car, count, off, 1.05, C.carrier, 1.4);
  };

  const drawMod = useRef(null);
  drawMod.current = (ctx, w, h) => {
    const { sig, count, off, params, over, mark } = view.current;
    const r = rectOf(w, h);
    const range = 1 + Math.max(params.m, 0.2) + 0.25;
    plate(ctx, w, h, r); zeroRule(ctx, r); yScale(ctx, r, range);
    keys(ctx, r, [
      { label: "modulated signal", color: C.carrier },
      { label: "envelope", color: C.envelope },
    ]);
    curve(ctx, r, sig.sig, count, off, range, C.carrier, 1.2);
    curve(ctx, r, sig.env, count, off, range, C.envelope, 2, [6, 5]);
    curve(ctx, r, sig.envNeg, count, off, range, C.envelope, 2, [6, 5]);

    const mid = r.y + r.h / 2, k = (r.h / 2 - 4) / range;
    const xAt = (i) => r.x + (i / (count - 1)) * r.w;
    const yAt = (v) => mid - v * k;

    if (mark.includes("envelope")) {
      let bi = 0, bv = -9;
      for (let i = 0; i < count; i++) {
        const v = sig.env[(off + i) % N];
        if (v > bv) { bv = v; bi = i; }
      }
      const tx = xAt(bi), ty = yAt(bv);
      const right = tx > r.x + r.w * 0.6;
      callout(ctx, tx, ty, right ? tx - 60 : tx + 60, Math.max(r.y + 16, ty - 26),
        "the envelope", C.envelope, right ? "right" : "left");
    }

    if (mark.includes("flip")) {
      let bi = -1, bv = 9;
      for (let i = 0; i < count; i++) {
        const v = sig.env[(off + i) % N];
        if (v < bv) { bv = v; bi = i; }
      }
      if (bi >= 0 && bv < 0) {
        const tx = xAt(bi), ty = yAt(bv);
        const right = tx > r.x + r.w * 0.55;
        callout(ctx, tx, ty, right ? tx - 52 : tx + 52, Math.min(r.y + r.h - 16, ty + 24),
          "envelope below zero", C.envelope, right ? "right" : "left");
      }
    }
  };

  const drawSpec = useRef(null);
  drawSpec.current = (ctx, w, h) => {
    const { sig, params, mark } = view.current;
    const r = rectOf(w, h);
    plate(ctx, w, h, r);
    if (!sig.spec) return;
    const { fc, fm } = params;
    const fx = (f) => r.x + (f / F_VIEW) * r.w;
    const toY = (a) => r.y + r.h - Math.min(a / 1.15, 1) * (r.h - 10);

    for (let k = 0; k < sig.spec.length; k++) {
      const f = k * BIN;
      const near = Math.abs(f - fc) < 0.7 ||
                   Math.abs(f - (fc + fm)) < 0.7 ||
                   Math.abs(f - (fc - fm)) < 0.7;
      ctx.strokeStyle = near ? C.spike : C.bars;
      ctx.lineWidth = near ? 3 : 1;
      ctx.beginPath();
      ctx.moveTo(fx(f), r.y + r.h);
      ctx.lineTo(fx(f), toY(sig.spec[k]));
      ctx.stroke();
    }

    const xa = fx(fc - fm), xb = fx(fc + fm), xc = fx(fc);
    const yb = r.y + 22;
    ctx.strokeStyle = C.spike; ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xa, yb + 7); ctx.lineTo(xa, yb);
    ctx.lineTo(xb, yb); ctx.lineTo(xb, yb + 7);
    ctx.stroke();
    ctx.fillStyle = C.spike;
    ctx.font = `12.5px ${FONT}`;
    ctx.textAlign = "center";
    ctx.fillText(`bandwidth ${(2 * fm).toFixed(0)} Hz`, (xa + xb) / 2, yb - 6);
    ctx.textAlign = "left";

    if (mark.includes("spikes")) {
      callout(ctx, xc, toY(1), xc + 46, r.y + r.h * 0.34, "carrier", C.spike);
      callout(ctx, xa, toY(params.m / 2), xa - 40, r.y + r.h * 0.58, "sideband", C.spike, "right");
      callout(ctx, xb, toY(params.m / 2), xb + 34, r.y + r.h * 0.58, "sideband", C.spike);
    }
    if (mark.includes("waste")) {
      callout(ctx, xc, toY(0.85), xc + 46, r.y + r.h * 0.3, "carries nothing", C.spike);
    }
    if (mark.includes("floor")) {
      const k = Math.round(18 / BIN);
      callout(ctx, fx(18), toY(sig.spec[k] || 0), fx(18) + 40, r.y + r.h * 0.55,
        "noise floor", C.label);
    }

    ctx.fillStyle = C.tick;
    ctx.font = `12px ${FONT}`;
    for (let f = 0; f <= F_VIEW; f += 20) ctx.fillText(String(f), fx(f) - 7, r.y + r.h + 18);
    ctx.fillText("Hz", r.x + r.w - 20, r.y + r.h + 18);
  };

  const drawRec = useRef(null);
  drawRec.current = (ctx, w, h) => {
    const { sig, count, off, over, mark } = view.current;
    const r = rectOf(w, h);
    const range = over ? 1.7 : 1.05;
    plate(ctx, w, h, r); zeroRule(ctx, r); yScale(ctx, r, range);
    keys(ctx, r, [
      { label: "sent", color: C.message },
      { label: "recovered", color: C.recovered },
    ]);
    curve(ctx, r, sig.msg, count, off, range, C.message, 1.5, [5, 5]);
    curve(ctx, r, sig.rec, count, off, range, C.recovered, 2.4);

    if (mark.includes("humps") && over) {
      const mid = r.y + r.h / 2, k = (r.h / 2 - 4) / range;
      let bi = 0, bv = -9;
      for (let i = 0; i < count; i++) {
        const idx = (off + i) % N;
        if (sig.msg[idx] < -0.5 && sig.rec[idx] > bv) { bv = sig.rec[idx]; bi = i; }
      }
      const tx = r.x + (bi / (count - 1)) * r.w, ty = mid - bv * k;
      const right = tx > r.x + r.w * 0.55;
      callout(ctx, tx, ty, right ? tx - 56 : tx + 56, Math.max(r.y + 18, ty - 26),
        "not in the original", C.recovered, right ? "right" : "left");
    }
  };

  const refMsg = useLoop(drawMsg);
  const refCar = useLoop(drawCar);
  const refMod = useLoop(drawMod);
  const refSpec = useLoop(drawSpec);
  const refRec = useLoop(drawRec);

  return (
    <div className="page">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .page{
          --sf:'SF Pro Display',-apple-system,BlinkMacSystemFont,'Inter',system-ui,sans-serif;
          --bg:#E7E4DB; --ink:#1B2A33; --mute:#6E7883; --hair:#C8C2B3;
          --accent:#BE7414; --plum:#B3306B;
          background:var(--bg); color:var(--ink); min-height:100%;
          font-family:var(--sf); font-size:16px; line-height:1.6;
          padding:26px 28px 40px; box-sizing:border-box;
          -webkit-font-smoothing:antialiased;
        }
        .page *,.page *::before,.page *::after{box-sizing:border-box;}

        .bar{display:flex;justify-content:space-between;align-items:baseline;gap:20px;flex-wrap:wrap;}
        .bar h1{margin:0;font-size:19px;font-weight:600;letter-spacing:-.022em;}
        .bar .who{margin:0;font-size:13.5px;color:var(--mute);}
        .tools{display:flex;gap:6px;}

        button{font-family:inherit;font-size:13.5px;cursor:pointer;color:var(--ink);
          background:transparent;border:1px solid var(--hair);border-radius:2px;
          padding:6px 12px;transition:background .14s,border-color .14s,color .14s;}
        button:hover:not(:disabled){border-color:var(--ink);}
        button:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
        button[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:var(--bg);}
        button:disabled{opacity:.32;cursor:not-allowed;}

        /* the ruler doubles as progress and as a jump control */
        .ruler{display:flex;gap:0;margin:22px 0 26px;border-top:1px solid var(--hair);
               padding-top:9px;}
        .ruler button{flex:1;border:0;border-radius:0;padding:0;height:22px;background:none;
                      position:relative;}
        .ruler button::after{content:"";position:absolute;left:0;top:0;width:100%;height:2px;
                             background:var(--hair);}
        .ruler button[data-done=true]::after{background:var(--mute);}
        .ruler button[data-on=true]::after{background:var(--accent);height:4px;}
        .ruler button:hover::after{background:var(--ink);}

        .stage{display:grid;grid-template-columns:minmax(300px,370px) minmax(0,1fr);
               gap:46px;align-items:start;}
        @media(max-width:960px){.stage{grid-template-columns:1fr;gap:26px;}}

        .col{position:sticky;top:20px;}
        @media(max-width:960px){.col{position:static;}}
        .count{font-size:13px;color:var(--mute);font-variant-numeric:tabular-nums;
               letter-spacing:.04em;}
        .col h2{margin:4px 0 14px;font-size:36px;line-height:1.08;font-weight:600;
                letter-spacing:-.032em;max-width:15ch;}
        @media(max-width:960px){.col h2{font-size:29px;}}
        .col .body{margin:0;font-size:16.5px;color:#2E3B45;max-width:42ch;}

        .quiz{margin-top:22px;padding-top:18px;border-top:1px solid var(--hair);}
        .quiz .q{margin:0 0 12px;font-size:15px;color:var(--plum);max-width:38ch;}
        .quiz button{display:block;width:100%;text-align:left;margin-bottom:7px;
                     padding:9px 13px;font-size:14.5px;}
        .quiz button[data-r=hit]{background:#DCE9DF;border-color:#3C7A56;color:#1D4632;}
        .quiz button[data-r=miss]{background:#F0DCD8;border-color:#B3564A;color:#6E2A22;}
        .quiz .note{margin:11px 0 0;font-size:14px;color:var(--mute);max-width:40ch;}

        .nav{display:flex;gap:8px;margin-top:24px;}

        .eq{margin-top:26px;font-size:15px;color:var(--mute);font-variant-numeric:tabular-nums;}
        .eq b{color:var(--accent);font-weight:500;}

        .fig{margin:0 0 4px;max-height:0;opacity:0;overflow:hidden;
             transition:max-height .45s ease,opacity .3s ease,margin .45s ease;}
        .fig.open{max-height:400px;opacity:1;margin-bottom:26px;}
        .plate{overflow:hidden;}
        .plate canvas{display:block;}
        figcaption{margin-top:7px;font-size:13px;color:var(--mute);}

        .dials{margin-bottom:26px;}
        .ctl{margin-bottom:16px;}
        .ctl-top{display:flex;justify-content:space-between;align-items:baseline;}
        .ctl-top label{font-size:14.5px;}
        .ctl-top output{font-size:14.5px;color:var(--accent);font-variant-numeric:tabular-nums;}
        input[type=range]{-webkit-appearance:none;appearance:none;width:100%;
          background:transparent;margin:8px 0 0;cursor:grab;}
        input[type=range]:active{cursor:grabbing;}
        input[type=range]::-webkit-slider-runnable-track{height:1px;background:var(--hair);}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;
          width:15px;height:15px;margin-top:-7px;border-radius:50%;
          background:var(--accent);border:2px solid var(--bg);}
        input[type=range]::-moz-range-track{height:1px;background:var(--hair);}
        input[type=range]::-moz-range-thumb{width:13px;height:13px;border-radius:50%;
          background:var(--accent);border:2px solid var(--bg);}
        input[type=range]:focus-visible{outline:2px solid var(--accent);outline-offset:4px;}

        .facts{display:flex;gap:30px;flex-wrap:wrap;padding-top:18px;
               border-top:1px solid var(--hair);}
        .facts div{font-size:13px;color:var(--mute);}
        .facts b{display:block;font-size:26px;color:var(--ink);font-weight:500;
                 font-variant-numeric:tabular-nums;line-height:1.2;letter-spacing:-.02em;}

        .warn{margin:0 0 20px;padding-left:13px;border-left:2px solid var(--plum);
              font-size:14.5px;color:#7A2350;max-width:60ch;}

        @media(prefers-reduced-motion:reduce){.fig{transition:none;}button{transition:none;}}
      `}</style>

      <div className="bar">
        <div>
          <h1>How AM radio works</h1>
          <p className="who">TC-307 Communication Systems</p>
        </div>
        <div className="tools">
          <button aria-pressed={inLesson} onClick={() => { setMode("lesson"); goStep(step); }}>Lesson</button>
          <button aria-pressed={!inLesson} onClick={() => { tween.current = null; setMode("explore"); }}>Explore</button>
          <button aria-pressed={running} onClick={() => setRunning((r) => !r)}>{running ? "Freeze" : "Run"}</button>
          <button onClick={share}>{copied ? "Link copied" : "Copy link"}</button>
        </div>
      </div>

      {inLesson && (
        <div className="ruler">
          {LESSON.map((s, i) => (
            <button key={s.id} title={`${i + 1}. ${s.title}`}
              data-on={i === step} data-done={i < step}
              aria-label={`Step ${i + 1}: ${s.title}`}
              onClick={() => goStep(i)} />
          ))}
        </div>
      )}

      <div className="stage">
        <div className="col">
          {inLesson ? (
            <>
              <div className="count">{String(step + 1).padStart(2, "0")} of {LESSON.length}</div>
              <h2>{L.title}</h2>
              <p className="body">{L.body}</p>

              {L.predict && (
                <div className="quiz">
                  <p className="q">{L.predict.q}</p>
                  {L.predict.options.map((o, i) => (
                    <button key={i} onClick={() => picked == null && answer(i)}
                      data-r={picked == null ? undefined
                        : i === L.predict.answer ? "hit"
                        : i === picked ? "miss" : undefined}>
                      {o}
                    </button>
                  ))}
                  {picked != null && <p className="note">{L.predict.note}</p>}
                </div>
              )}

              <div className="nav">
                <button onClick={() => goStep(step - 1)} disabled={step === 0}>Back</button>
                <button onClick={() => goStep(step + 1)}
                  disabled={step === LESSON.length - 1 || gated}>
                  {gated ? "Pick one first" : "Next"}
                </button>
              </div>

              <p className="eq">
                s(t) = [1 + <b>{params.m.toFixed(2)}</b>·cos(2π·<b>{params.fm.toFixed(1)}</b>·t)]
                {" "}· cos(2π·<b>{params.fc.toFixed(0)}</b>·t)
              </p>
            </>
          ) : (
            <>
              <div className="count">Explore</div>
              <h2>Turn the dials</h2>
              <div className="dials" style={{ marginTop: 18 }}>
                {[
                  ["Carrier frequency", "fc", 8, 80, 1, `${params.fc.toFixed(0)} Hz`],
                  ["Message frequency", "fm", 1, 12, 1, `${params.fm.toFixed(0)} Hz`],
                  ["Modulation index", "m", 0, 1.6, 0.01, params.m.toFixed(2)],
                  ["Signal to noise", "snr", 0, 50, 1, `${params.snr.toFixed(0)} dB`],
                ].map(([label, key, lo, hi, st, out]) => (
                  <div className="ctl" key={key}>
                    <div className="ctl-top">
                      <label htmlFor={key}>{label}</label><output>{out}</output>
                    </div>
                    <input id={key} type="range" min={lo} max={hi} step={st}
                      value={params[key]} onChange={(e) => set(key)(+e.target.value)} />
                  </div>
                ))}
                <div className="ctl">
                  <div className="ctl-top">
                    <label htmlFor="pz">Message periods shown</label><output>{periods}</output>
                  </div>
                  <input id="pz" type="range" min="1" max="6" step="1" value={periods}
                    onChange={(e) => setPeriods(+e.target.value)} />
                </div>
              </div>
              <div className="facts">
                <div><b>{eff.toFixed(0)}%</b>power carrying the message</div>
                <div><b>{(2 * params.fm).toFixed(0)} Hz</b>bandwidth</div>
                <div><b>{(params.m / 2).toFixed(2)}</b>sideband height</div>
              </div>
            </>
          )}
        </div>

        <div>
          {over && (
            <p className="warn">
              Overmodulated at m = {params.m.toFixed(2)}. The envelope crosses zero and the
              detector folds those troughs back up. That information cannot be recovered.
            </p>
          )}
          <Figure open={show.includes("msg")} height={148} cref={refMsg}
            caption="The signal you want to send, one clean tone." />
          <Figure open={show.includes("car")} height={148} cref={refCar}
            caption="The carrier on its own. Fast, steady, empty." />
          <Figure open={show.includes("mod")} height={248} cref={refMod}
            caption="Carrier and message combined. The dashed outline is the envelope." />
          <Figure open={show.includes("spec")} height={216} cref={refSpec}
            caption="What actually leaves the antenna, measured by FFT." />
          <Figure open={show.includes("rec")} height={172} cref={refRec}
            caption="Rectify, low pass, drop the DC. Overlaid on the original." />
        </div>
      </div>
    </div>
  );
}
