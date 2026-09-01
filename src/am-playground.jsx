import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";

/* ==================================================================
   How AM Radio Works — an eleven-step walkthrough

   s(t) = [1 + m·cos(2π·fm·t)] · cos(2π·fc·t)

   Two modes. Lesson walks through the idea one animated step at a
   time; Explore hands over every control. Each lesson step declares
   target parameters, and moving between steps tweens toward them, so
   the animation IS the explanation rather than decoration on top.
   ================================================================== */

const FS = 2048;
const N = 8192;                  // 4 s buffer -> FFT bin = 0.25 Hz
const BIN = FS / N;
const F_VIEW = 110;
const PAD = { l: 42, r: 14, t: 14, b: 24 };
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

  /* The FFT is the expensive part, so skip it whenever the spectrum
     panel is hidden. During a tween this runs every frame. */
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
    body: "This slow wave is what you want to send — a voice, some music, data. On its own it cannot travel. Radiating a 3 kHz signal directly would need an antenna around 100 km long.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["msg"],
  },
  {
    id: "carrier", title: "So borrow a faster wave",
    body: "A carrier oscillates far more quickly, and a fast wave needs only a short antenna. It can travel anywhere. It just carries nothing yet.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["msg", "car"],
  },
  {
    id: "ride", title: "Ride one on the other",
    body: "Let the message control the carrier's height. Where the message rises the carrier grows tall; where it dips the carrier shrinks. Its frequency never changes — only the amplitude.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["mod"],
  },
  {
    id: "envelope", title: "The outline is the message",
    body: "Trace the peaks of the carrier and the message comes straight back. That outline is called the envelope, and the whole of AM is this single idea: hide the message in the envelope.",
    p: { fc: 40, fm: 4, m: 0.35, snr: 50 }, show: ["mod"],
  },
  {
    id: "index", title: "How deep to go",
    body: "The modulation index m sets the depth. At 0.35 the envelope barely breathes. Watch it open up as m climbs toward 1.",
    p: { fc: 40, fm: 4, m: 0.95, snr: 50 }, show: ["mod"],
  },
  {
    id: "over", title: "Push it too far",
    body: "Past m = 1 the envelope crosses zero and the carrier flips phase. A receiver cannot tell a flip from a dip, so it folds those troughs back upward. The information in them is gone for good.",
    p: { fc: 40, fm: 4, m: 1.35, snr: 50 }, show: ["mod", "rec"],
    predict: {
      q: "Before it happens — what do you think the recovered message will do?",
      options: ["Just get louder", "Flatten off at the peaks", "Grow extra humps where the troughs were"],
      answer: 2,
      note: "The detector can only measure size, not sign, so a negative envelope comes back positive.",
    },
  },
  {
    id: "spectrum", title: "Only three frequencies leave",
    body: "Multiply the two cosines and the product identity splits them into exactly three: the carrier in the middle, and one sideband either side at fc + fm and fc − fm. Nothing else is transmitted.",
    p: { fc: 40, fm: 4, m: 0.7, snr: 50 }, show: ["mod", "spec"],
  },
  {
    id: "bandwidth", title: "How much room it needs",
    body: "Raise the message frequency and the two sidebands slide outward. The gap between them is the bandwidth, and it always works out to twice the message frequency.",
    p: { fc: 40, fm: 10, m: 0.7, snr: 50 }, show: ["spec"],
    predict: {
      q: "Raise the message frequency. What do the sidebands do?",
      options: ["Move closer together", "Slide further apart", "Stay put and grow taller"],
      answer: 1,
      note: "They sit at fc ± fm, so raising fm pushes them out symmetrically.",
    },
  },
  {
    id: "cost", title: "What the carrier costs you",
    body: "The middle spike is the tallest by far, and it carries no information at all. Only the sidebands do. Even at m = 1, barely a third of your transmitter power does useful work — which is exactly why single sideband exists.",
    p: { fc: 40, fm: 6, m: 1.0, snr: 50 }, show: ["spec"],
  },
  {
    id: "noise", title: "Then the channel gets in the way",
    body: "Real links add noise. Watch where it shows up first: the spectrum floor lifts long before the waveform looks damaged. This is why engineers watch spectra rather than waveforms.",
    p: { fc: 40, fm: 6, m: 0.7, snr: 7 }, show: ["mod", "spec"],
  },
  {
    id: "recover", title: "Getting it back",
    body: "The receiver rectifies the signal, low-passes away the carrier ripple, and drops the DC. If the green curve lands on the blue one, the link worked.",
    p: { fc: 40, fm: 5, m: 0.7, snr: 30 }, show: ["mod", "rec"],
  },
];

/* ---------------- palette ---------------- */

const C = {
  paper: "#f8f6f0", grid: "#e3dfd2", rule: "#b9b19c", tick: "#8c8471",
  message: "#2f6491", carrier: "#243740", envelope: "#a52a6d",
  bars: "#6d8fa3", spike: "#b57a10", recovered: "#1d6f5b",
};
const FONT = "'SF Pro Display', -apple-system, Inter, system-ui, sans-serif";

/* ---------------- canvas ---------------- */

/* One rAF loop per canvas, reading the latest draw function from a ref.
   Nothing re-subscribes when parameters change, which matters when
   they change sixty times a second during a tween. */
function useLoop(drawRef) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const box = cv.parentElement;
    let raf = 0;
    const frame = () => {
      const w = box.clientWidth, h = box.clientHeight;
      if (w > 0 && h > 0 && drawRef.current) {
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
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [drawRef]);
  return ref;
}

const rectOf = (w, h) => ({
  x: PAD.l, y: PAD.t,
  w: Math.max(10, w - PAD.l - PAD.r),
  h: Math.max(10, h - PAD.t - PAD.b),
});

function field(ctx, w, h, r) {
  ctx.fillStyle = C.paper;
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < 10; i++) {
    const x = Math.round(r.x + (i / 10) * r.w) + 0.5;
    ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h);
  }
  for (let i = 1; i < 4; i++) {
    const y = Math.round(r.y + (i / 4) * r.h) + 0.5;
    ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y);
  }
  ctx.stroke();
  ctx.strokeStyle = C.rule;
  ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w, r.h);
}

function yTicks(ctx, r, range) {
  ctx.fillStyle = C.tick;
  ctx.font = `12px ${FONT}`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  [-1, 0, 1].forEach((v) => {
    const y = r.y + r.h / 2 - (v / range) * (r.h / 2 - 4);
    ctx.fillText(String(v), r.x - 7, y);
  });
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function midline(ctx, r) {
  ctx.strokeStyle = C.rule;
  ctx.setLineDash([2, 3]);
  ctx.beginPath();
  const y = Math.round(r.y + r.h / 2) + 0.5;
  ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function trace(ctx, r, data, count, off, range, color, lw, dash) {
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
  } catch (e) { /* sandboxed frame, ignore */ }
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
  const [seed, setSeed] = useState(1234);
  const [copied, setCopied] = useState(false);

  const paramsRef = useRef(params);
  paramsRef.current = params;
  const tween = useRef(null);
  const runRef = useRef(running);
  runRef.current = running;

  const lessonStep = LESSON[step];
  const inLesson = mode === "lesson";
  const gated = inLesson && lessonStep.predict && !answered[lessonStep.id];
  const show = inLesson
    ? lessonStep.show
    : ["msg", "mod", "spec", "rec"];
  const needSpec = show.includes("spec");

  const sig = useMemo(
    () => buildSignals(params, seed, needSpec),
    [params, seed, needSpec]
  );

  const count = Math.min(N, Math.max(64, Math.round((periods / params.fm) * FS)));
  const over = params.m > 1;
  const eff = (100 * params.m * params.m) / (2 + params.m * params.m);

  /* one master loop: parameter tween plus the scrolling trace */
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
    tween.current = { from: { ...paramsRef.current }, to, t0: performance.now(), ms: TWEEN_MS };
  }, []);

  const goStep = useCallback((i) => {
    const n = clamp(i, 0, LESSON.length - 1);
    setStep(n);
    setPicked(null);
    const s = LESSON[n];
    if (!s.predict || answered[s.id]) glideTo(s.p);
  }, [answered, glideTo]);

  /* keyboard paging */
  useEffect(() => {
    const onKey = (e) => {
      if (!inLesson) return;
      if (e.key === "ArrowRight") goStep(step + 1);
      if (e.key === "ArrowLeft") goStep(step - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inLesson, step, goStep]);

  /* keep the address bar in sync so any view can be linked */
  useEffect(() => {
    try {
      const q = inLesson
        ? `?step=${step + 1}`
        : `?fc=${Math.round(params.fc)}&fm=${Math.round(params.fm)}` +
          `&m=${params.m.toFixed(2)}&snr=${Math.round(params.snr)}`;
      window.history.replaceState(null, "", q);
    } catch (e) { /* sandboxed frame, ignore */ }
  }, [inLesson, step, params]);

  const answer = (i) => {
    setPicked(i);
    setAnswered((a) => ({ ...a, [lessonStep.id]: true }));
    setTimeout(() => glideTo(lessonStep.p), 700);
  };

  const share = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch (e) {
      setCopied(false);
    }
  };

  const set = (k) => (v) => {
    tween.current = null;
    setParams((p) => ({ ...p, [k]: v }));
  };

  /* ---- draw functions, kept in refs ---- */

  const view = useRef({});
  view.current = { sig, count, off, params, over, periods };

  const mkDraw = (fn) => {
    const r = useRef(fn);
    r.current = fn;
    return r;
  };

  const drawMsg = mkDraw((ctx, w, h) => {
    const { sig, count, off } = view.current;
    const r = rectOf(w, h);
    field(ctx, w, h, r); midline(ctx, r); yTicks(ctx, r, 1.05);
    trace(ctx, r, sig.msg, count, off, 1.05, C.message, 2.4);
  });

  const drawCar = mkDraw((ctx, w, h) => {
    const { sig, count, off } = view.current;
    const r = rectOf(w, h);
    field(ctx, w, h, r); midline(ctx, r); yTicks(ctx, r, 1.05);
    trace(ctx, r, sig.car, count, off, 1.05, C.carrier, 1.4);
  });

  const drawMod = mkDraw((ctx, w, h) => {
    const { sig, count, off, params, over } = view.current;
    const r = rectOf(w, h);
    const range = 1 + Math.max(params.m, 0.2) + 0.2;
    field(ctx, w, h, r); midline(ctx, r); yTicks(ctx, r, range);
    if (over) {
      ctx.save();
      ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
      const k = (r.h / 2 - 4) / range, mid = r.y + r.h / 2;
      ctx.fillStyle = "rgba(165,42,109,0.09)";
      ctx.fillRect(r.x, mid - k, r.w, 2 * k);
      ctx.restore();
    }
    trace(ctx, r, sig.sig, count, off, range, C.carrier, 1.2);
    trace(ctx, r, sig.env, count, off, range, C.envelope, 2, [6, 5]);
    trace(ctx, r, sig.envNeg, count, off, range, C.envelope, 2, [6, 5]);
  });

  const drawSpec = mkDraw((ctx, w, h) => {
    const { sig, params } = view.current;
    const r = rectOf(w, h);
    field(ctx, w, h, r);
    if (!sig.spec) return;
    const { fc, fm } = params;
    const fx = (f) => r.x + (f / F_VIEW) * r.w;
    const toY = (a) => r.y + r.h - Math.min(a / 1.15, 1) * (r.h - 8);

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

    const xa = fx(fc - fm), xb = fx(fc + fm), yb = r.y + 20;
    ctx.strokeStyle = C.spike; ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(xa, yb + 6); ctx.lineTo(xa, yb);
    ctx.lineTo(xb, yb); ctx.lineTo(xb, yb + 6);
    ctx.stroke();
    ctx.fillStyle = C.spike;
    ctx.font = `12px ${FONT}`;
    ctx.textAlign = "center";
    ctx.fillText(`bandwidth ${(2 * fm).toFixed(0)} Hz`, (xa + xb) / 2, yb - 5);
    ctx.textAlign = "left";

    ctx.fillStyle = C.tick;
    for (let f = 0; f <= F_VIEW; f += 20) ctx.fillText(String(f), fx(f) - 7, r.y + r.h + 17);
    ctx.fillText("Hz", r.x + r.w - 18, r.y + r.h + 17);
  });

  const drawRec = mkDraw((ctx, w, h) => {
    const { sig, count, off, over } = view.current;
    const r = rectOf(w, h);
    const range = over ? 1.7 : 1.05;
    field(ctx, w, h, r); midline(ctx, r); yTicks(ctx, r, range);
    trace(ctx, r, sig.msg, count, off, range, C.message, 1.5, [5, 5]);
    trace(ctx, r, sig.rec, count, off, range, C.recovered, 2.4);
  });

  const refMsg = useLoop(drawMsg);
  const refCar = useLoop(drawCar);
  const refMod = useLoop(drawMod);
  const refSpec = useLoop(drawSpec);
  const refRec = useLoop(drawRec);

  const Plot = ({ id, title, legend, height, cref }) => (
    <div className={show.includes(id) ? "wrap open" : "wrap"} aria-hidden={!show.includes(id)}>
      <section className="panel">
        <header className="p-head">
          <h2>{title}</h2>
          <ul className="legend">
            {legend.map((l) => (
              <li key={l.label}>
                <span className="sw" style={l.dash
                  ? { borderTop: `2px dashed ${l.color}` }
                  : { background: l.color, height: 3 }} />
                {l.label}
              </li>
            ))}
          </ul>
        </header>
        <div className="plot" style={{ height }}><canvas ref={cref} /></div>
      </section>
    </div>
  );

  return (
    <div className="app">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        .app{
          --sf:'SF Pro Display',-apple-system,BlinkMacSystemFont,'Inter',system-ui,sans-serif;
          --deep:#10272c;--deep-2:#17343a;--deep-3:#1e4048;--line:#2c525b;
          --fg:#e8efee;--fg-dim:#93aeb2;--brass:#dda63f;--rose:#d97a63;--go:#6fbfa0;
          background:var(--deep);color:var(--fg);min-height:100%;font-family:var(--sf);
          font-size:15px;line-height:1.55;padding:24px 22px 32px;box-sizing:border-box;
        }
        .app *,.app *::before,.app *::after{box-sizing:border-box;}

        .top{display:flex;justify-content:space-between;align-items:center;gap:20px;
             flex-wrap:wrap;padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:22px;}
        .top h1{margin:0;font-size:17px;font-weight:600;letter-spacing:-.015em;}
        .top .sub{margin:1px 0 0;font-size:13px;color:var(--fg-dim);}
        .tools{display:flex;gap:7px;align-items:center;}

        button{font-family:inherit;font-size:13px;cursor:pointer;color:var(--fg);
          background:transparent;border:1px solid var(--line);border-radius:3px;padding:7px 13px;
          transition:border-color .15s,color .15s,background .15s;}
        button:hover:not(:disabled){border-color:var(--brass);color:var(--brass);}
        button:focus-visible{outline:2px solid var(--brass);outline-offset:2px;}
        button[aria-pressed=true]{background:var(--brass);border-color:var(--brass);color:#14262b;}
        button:disabled{opacity:.35;cursor:not-allowed;}

        .stage{display:grid;grid-template-columns:340px minmax(0,1fr);gap:26px;align-items:start;}
        @media(max-width:940px){.stage{grid-template-columns:1fr;}}

        .rail{position:sticky;top:16px;}
        @media(max-width:940px){.rail{position:static;}}
        .steps{list-style:none;margin:0 0 20px;padding:0;}
        .steps li{display:flex;gap:11px;align-items:baseline;padding:5px 0;cursor:pointer;
                  color:var(--fg-dim);font-size:13.5px;border:0;background:none;width:100%;
                  text-align:left;font-family:inherit;}
        .steps li .num{font-variant-numeric:tabular-nums;font-size:11.5px;width:16px;flex:none;
                       color:var(--line);}
        .steps li:hover{color:var(--fg);}
        .steps li[data-on=true]{color:var(--brass);}
        .steps li[data-on=true] .num{color:var(--brass);}
        .steps li[data-done=true]{color:var(--fg);}

        .card{background:var(--deep-3);border:1px solid var(--line);border-radius:4px;padding:20px;}
        .card h2{margin:0 0 9px;font-size:22px;font-weight:600;letter-spacing:-.02em;line-height:1.2;}
        .card p{margin:0;font-size:14.5px;color:#cfdedd;max-width:44ch;}
        .nav{display:flex;gap:8px;margin-top:18px;}
        .nav button{flex:1;}

        .quiz{margin-top:16px;padding-top:16px;border-top:1px solid var(--line);}
        .quiz p{font-size:14px;color:var(--brass);margin:0 0 11px;}
        .quiz button{display:block;width:100%;text-align:left;margin-bottom:7px;font-size:13.5px;}
        .quiz button[data-r=hit]{background:rgba(111,191,160,.16);border-color:var(--go);color:#cdefe1;}
        .quiz button[data-r=miss]{background:rgba(217,122,99,.14);border-color:var(--rose);color:#f6ded6;}
        .quiz .note{margin:10px 0 0;font-size:13px;color:var(--fg-dim);}

        .eq{font-size:19px;margin-top:20px;color:var(--fg-dim);font-variant-numeric:tabular-nums;}
        .eq .v{color:var(--brass);}

        .wrap{max-height:0;opacity:0;overflow:hidden;transform:translateY(-6px);
              transition:max-height .45s ease,opacity .35s ease,transform .35s ease,margin .45s ease;
              margin-bottom:0;}
        .wrap.open{max-height:420px;opacity:1;transform:none;margin-bottom:14px;}

        .panel{background:var(--deep-2);border:1px solid var(--line);border-radius:4px;padding:14px;}
        .p-head{display:flex;justify-content:space-between;align-items:baseline;gap:14px;
                flex-wrap:wrap;margin-bottom:10px;}
        .p-head h2{margin:0;font-size:14px;font-weight:600;}
        .legend{display:flex;gap:14px;list-style:none;margin:0;padding:0;}
        .legend li{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--fg-dim);}
        .sw{width:15px;display:inline-block;border-radius:2px;}
        .plot{border-radius:3px;overflow:hidden;}
        .plot canvas{display:block;}

        .dials{background:var(--deep-2);border:1px solid var(--line);border-radius:4px;
               padding:17px;margin-bottom:20px;}
        .ctl{margin-bottom:15px;}.ctl:last-child{margin-bottom:0;}
        .ctl-top{display:flex;justify-content:space-between;align-items:baseline;}
        .ctl-top label{font-size:13.5px;}
        .ctl-top output{font-size:13.5px;color:var(--brass);font-variant-numeric:tabular-nums;}
        input[type=range]{-webkit-appearance:none;appearance:none;width:100%;background:transparent;
          margin:9px 0 0;cursor:grab;}
        input[type=range]:active{cursor:grabbing;}
        input[type=range]::-webkit-slider-runnable-track{height:2px;background:var(--line);border-radius:2px;}
        input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:16px;height:16px;
          margin-top:-7px;border-radius:50%;background:var(--brass);border:2px solid var(--deep-2);}
        input[type=range]::-moz-range-track{height:2px;background:var(--line);border-radius:2px;}
        input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;
          background:var(--brass);border:2px solid var(--deep-2);}
        input[type=range]:focus-visible{outline:2px solid var(--brass);outline-offset:4px;}

        .facts{display:flex;gap:28px;flex-wrap:wrap;margin-top:6px;}
        .facts div{font-size:12.5px;color:var(--fg-dim);}
        .facts b{display:block;font-size:23px;color:var(--fg);font-weight:500;
                 font-variant-numeric:tabular-nums;line-height:1.25;}

        .flag{margin:0 0 14px;padding:11px 14px;border-radius:3px;font-size:13.5px;
              background:rgba(217,122,99,.14);border-left:3px solid var(--rose);color:#f7e2da;}

        @media(prefers-reduced-motion:reduce){.wrap{transition:none;}button{transition:none;}}
      `}</style>

      <div className="top">
        <div>
          <h1>How AM radio works</h1>
          <p className="sub">TC-307 Communication Systems</p>
        </div>
        <div className="tools">
          <button aria-pressed={inLesson} onClick={() => { setMode("lesson"); goStep(step); }}>Lesson</button>
          <button aria-pressed={!inLesson} onClick={() => { tween.current = null; setMode("explore"); }}>Explore</button>
          <button aria-pressed={running} onClick={() => setRunning((r) => !r)}>{running ? "Freeze" : "Run"}</button>
          <button onClick={share}>{copied ? "Link copied" : "Copy link"}</button>
        </div>
      </div>

      <div className="stage">
        <div className="rail">
          {inLesson ? (
            <>
              <ul className="steps">
                {LESSON.map((s, i) => (
                  <li key={s.id} role="button" tabIndex={0}
                      data-on={i === step} data-done={i < step}
                      onClick={() => goStep(i)}
                      onKeyDown={(e) => e.key === "Enter" && goStep(i)}>
                    <span className="num">{i + 1}</span>{s.title}
                  </li>
                ))}
              </ul>

              <div className="card">
                <h2>{lessonStep.title}</h2>
                <p>{lessonStep.body}</p>

                {lessonStep.predict && (
                  <div className="quiz">
                    <p>{lessonStep.predict.q}</p>
                    {lessonStep.predict.options.map((o, i) => (
                      <button key={i} onClick={() => picked == null && answer(i)}
                        data-r={picked == null ? undefined
                          : i === lessonStep.predict.answer ? "hit"
                          : i === picked ? "miss" : undefined}>
                        {o}
                      </button>
                    ))}
                    {picked != null && <p className="note">{lessonStep.predict.note}</p>}
                  </div>
                )}

                <div className="nav">
                  <button onClick={() => goStep(step - 1)} disabled={step === 0}>Back</button>
                  <button onClick={() => goStep(step + 1)} disabled={step === LESSON.length - 1 || gated}>
                    {gated ? "Pick one first" : "Next"}
                  </button>
                </div>
              </div>

              <div className="eq">
                s(t) = [1 + <span className="v">{params.m.toFixed(2)}</span>·cos(2π·
                <span className="v">{params.fm.toFixed(1)}</span>·t)] · cos(2π·
                <span className="v">{params.fc.toFixed(0)}</span>·t)
              </div>
            </>
          ) : (
            <>
              <div className="dials">
                {[
                  ["Carrier frequency", "fc", 8, 80, 1, `${params.fc.toFixed(0)} Hz`],
                  ["Message frequency", "fm", 1, 12, 1, `${params.fm.toFixed(0)} Hz`],
                  ["Modulation index", "m", 0, 1.6, 0.01, params.m.toFixed(2)],
                  ["Signal-to-noise", "snr", 0, 50, 1, `${params.snr.toFixed(0)} dB`],
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
                <div><b>{eff.toFixed(1)}%</b>power carrying the message</div>
                <div><b>{(2 * params.fm).toFixed(0)} Hz</b>bandwidth</div>
                <div><b>{(params.m / 2).toFixed(2)}</b>sideband height</div>
              </div>
            </>
          )}
        </div>

        <div>
          {over && (
            <p className="flag">
              Overmodulated at m = {params.m.toFixed(2)}. The envelope crosses zero and the
              detector folds those troughs back up. That information cannot be recovered.
            </p>
          )}
          <Plot id="msg" title="Message" height={150} cref={refMsg}
            legend={[{ label: "m(t)", color: C.message }]} />
          <Plot id="car" title="Carrier" height={150} cref={refCar}
            legend={[{ label: "cos(2π·fc·t)", color: C.carrier }]} />
          <Plot id="mod" title="Modulated signal" height={252} cref={refMod}
            legend={[{ label: "s(t)", color: C.carrier }, { label: "envelope", color: C.envelope, dash: true }]} />
          <Plot id="spec" title="Spectrum" height={210} cref={refSpec}
            legend={[{ label: "carrier and sidebands", color: C.spike }, { label: "noise floor", color: C.bars }]} />
          <Plot id="rec" title="Recovered message" height={172} cref={refRec}
            legend={[{ label: "sent", color: C.message, dash: true }, { label: "detected", color: C.recovered }]} />
        </div>
      </div>
    </div>
  );
}
