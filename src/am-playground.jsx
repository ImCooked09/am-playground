import React, { useState, useMemo, useRef, useEffect, useCallback } from "react";

/* ==================================================================
   AM Modulation Playground — TC-307 Communication Systems

   s(t) = [1 + m·cos(2π·fm·t)] · cos(2π·fc·t)

   Chain: build -> add noise -> FFT -> rectify + low-pass (detector)
   ================================================================== */

const FS = 2048;                 // sample rate, Hz
const N = 8192;                  // 4.0 s buffer -> FFT bin = 0.25 Hz
const BIN = FS / N;
const F_VIEW = 110;              // spectrum axis limit, Hz
const PAD = { l: 40, r: 12, t: 12, b: 22 };

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

function buildSignals(fc, fm, m, snrDb, seed) {
  const msg = new Float32Array(N);
  const env = new Float32Array(N);
  const envNeg = new Float32Array(N);
  const sig = new Float32Array(N);
  let power = 0;

  for (let n = 0; n < N; n++) {
    const t = n / FS;
    const mv = Math.cos(2 * Math.PI * fm * t);
    const ev = 1 + m * mv;
    const s = ev * Math.cos(2 * Math.PI * fc * t);
    msg[n] = mv; env[n] = ev; envNeg[n] = -ev; sig[n] = s;
    power += s * s;
  }
  power /= N;

  const sigma = Math.sqrt(power / Math.pow(10, snrDb / 10));
  if (sigma > 1e-9) {
    const rnd = mulberry32(seed);
    for (let n = 0; n < N; n += 2) {
      const u1 = Math.max(rnd(), 1e-12), u2 = rnd();
      const r = Math.sqrt(-2 * Math.log(u1));
      sig[n] += sigma * r * Math.cos(2 * Math.PI * u2);
      if (n + 1 < N) sig[n + 1] += sigma * r * Math.sin(2 * Math.PI * u2);
    }
  }

  const re = new Float64Array(N), im = new Float64Array(N);
  for (let n = 0; n < N; n++) re[n] = sig[n];
  fft(re, im);
  const bins = Math.ceil(F_VIEW / BIN) + 1;
  const spec = new Float32Array(bins);
  for (let k = 0; k < bins; k++) spec[k] = (2 * Math.hypot(re[k], im[k])) / N;

  // Envelope detector: rectify, one-pole low-pass, strip DC.
  // Cutoff at sqrt(fm*fc) always satisfies 1/fc << RC << 1/fm.
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

  return { msg, env, envNeg, sig, rec, spec };
}

/* ---------------- palette ---------------- */

const C = {
  paper: "#f8f6f0",
  grid: "#e3dfd2",
  rule: "#b9b19c",
  tick: "#8c8471",
  message: "#2f6491",
  carrier: "#243740",
  envelope: "#a52a6d",
  bars: "#6d8fa3",
  spike: "#b57a10",
  recovered: "#1d6f5b",
  hair: "#3f5259",
};

/* ---------------- canvas ---------------- */

function usePlot(draw, deps) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const box = cv.parentElement;
    let raf = 0;
    const render = () => {
      const w = box.clientWidth, h = box.clientHeight;
      if (!w || !h) return;
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
      draw(ctx, w, h);
    };
    render();
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(render);
    });
    ro.observe(box);
    return () => { ro.disconnect(); cancelAnimationFrame(raf); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return ref;
}

const rect = (w, h) => ({
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

function yTicks(ctx, r, range, labels) {
  ctx.fillStyle = C.tick;
  ctx.font = "11px 'SF Pro Display', -apple-system, Inter, system-ui, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.strokeStyle = C.rule;
  labels.forEach((v) => {
    const y = r.y + r.h / 2 - (v / range) * (r.h / 2 - 4);
    ctx.fillText(String(v), r.x - 7, y);
    ctx.beginPath();
    ctx.moveTo(r.x - 3, Math.round(y) + 0.5);
    ctx.lineTo(r.x, Math.round(y) + 0.5);
    ctx.stroke();
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
  ctx.beginPath();
  ctx.rect(r.x, r.y, r.w, r.h);
  ctx.clip();
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

function crosshair(ctx, r, hx, lines) {
  if (hx == null) return;
  const x = Math.round(hx) + 0.5;
  ctx.save();
  ctx.strokeStyle = C.hair;
  ctx.globalAlpha = 0.45;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h);
  ctx.stroke();
  ctx.restore();

  ctx.font = "11px 'SF Pro Display', -apple-system, Inter, system-ui, sans-serif";
  ctx.textBaseline = "top";
  const wBox = Math.max(...lines.map((t) => ctx.measureText(t).width)) + 14;
  const hBox = lines.length * 14 + 9;
  const bx = Math.min(x + 9, r.x + r.w - wBox - 3);
  const by = r.y + 5;
  ctx.fillStyle = "rgba(30,48,56,0.93)";
  ctx.fillRect(bx, by, wBox, hBox);
  ctx.fillStyle = "#f2efe6";
  lines.forEach((t, i) => ctx.fillText(t, bx + 7, by + 5 + i * 14));
  ctx.textBaseline = "alphabetic";
}

/* ---------------- panel shell ---------------- */

function Panel({ title, sub, legend, hint, height, canvasRef, onPointerMove, onPointerLeave, onPointerDown, grab, feature }) {
  return (
    <section className={feature ? "panel panel-feature" : "panel"}>
      <header className="p-head">
        <div>
          <h2>{title}</h2>
          {sub && <p className="p-sub">{sub}</p>}
        </div>
        <div className="p-meta">
          {hint && <span className="hint">{hint}</span>}
          {legend && (
            <ul className="legend">
              {legend.map((l) => (
                <li key={l.label}>
                  <span
                    className="sw"
                    style={
                      l.dash
                        ? { borderTop: `2px dashed ${l.color}` }
                        : { background: l.color, height: 3 }
                    }
                  />
                  {l.label}
                </li>
              ))}
            </ul>
          )}
        </div>
      </header>
      <div className="plot" style={{ height, cursor: grab || "crosshair" }}>
        <canvas
          ref={canvasRef}
          onPointerMove={onPointerMove}
          onPointerLeave={onPointerLeave}
          onPointerDown={onPointerDown}
        />
      </div>
    </section>
  );
}

/* ---------------- app ---------------- */

export default function AmPlayground() {
  const [fc, setFc] = useState(40);
  const [fm, setFm] = useState(4);
  const [m, setM] = useState(0.6);
  const [snr, setSnr] = useState(45);
  const [periods, setPeriods] = useState(2);
  const [logScale, setLogScale] = useState(false);
  const [seed, setSeed] = useState(1234);
  const [playing, setPlaying] = useState(false);
  const [off, setOff] = useState(0);
  const [hover, setHover] = useState({ id: null, x: 0 });

  const sig = useMemo(() => buildSignals(fc, fm, m, snr, seed), [fc, fm, m, snr, seed]);

  const count = Math.min(N, Math.max(64, Math.round((periods / fm) * FS)));
  const span = count / FS;
  const cycles = Math.round((periods / fm) * fc);
  const over = m > 1;
  const eff = (100 * m * m) / (2 + m * m);

  /* the buffer holds a whole number of cycles, so wrapping is seamless */
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const step = () => {
      setOff((o) => (o + 5) % N);
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const hoverAt = (id) => (e) => {
    const b = e.currentTarget.getBoundingClientRect();
    setHover({ id, x: e.clientX - b.left });
  };
  const clearHover = () => setHover({ id: null, x: 0 });

  const sampleAt = (id, w) => {
    if (hover.id !== id) return null;
    const r = rect(w, 0);
    const f = (hover.x - r.x) / r.w;
    if (f < 0 || f > 1) return null;
    return Math.round(f * (count - 1));
  };

  /* drag the envelope to change modulation index */
  const dragIndex = (e) => {
    const startY = e.clientY, startM = m;
    const move = (ev) => {
      const next = startM - (ev.clientY - startY) * 0.0065;
      setM(Math.min(1.6, Math.max(0, Math.round(next * 100) / 100)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  /* drag a spike in the spectrum to retune */
  const dragSpike = (e) => {
    const b = e.currentTarget.getBoundingClientRect();
    const r = rect(b.width, b.height);
    const toF = (clientX) => ((clientX - b.left - r.x) / r.w) * F_VIEW;
    const f0 = toF(e.clientX);
    const targets = [{ k: "fc", f: fc }, { k: "fm", f: fc + fm }, { k: "fm", f: fc - fm }];
    let best = null, bd = 7;
    targets.forEach((t) => {
      const d = Math.abs(t.f - f0);
      if (d < bd) { bd = d; best = t; }
    });
    if (!best) return;
    const move = (ev) => {
      const f = toF(ev.clientX);
      if (best.k === "fc") setFc(Math.min(80, Math.max(8, Math.round(f))));
      else setFm(Math.min(12, Math.max(1, Math.round(Math.abs(f - fc)))));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  /* ---- plots ---- */

  const modRef = usePlot((ctx, w, h) => {
    const r = rect(w, h);
    const range = 1 + Math.max(m, 0.2) + 0.2;
    field(ctx, w, h, r);
    midline(ctx, r);
    yTicks(ctx, r, range, [-1, 0, 1]);

    if (over) {
      ctx.save();
      ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
      const k = (r.h / 2 - 4) / range, mid = r.y + r.h / 2;
      ctx.fillStyle = "rgba(165,42,109,0.08)";
      ctx.fillRect(r.x, mid - k, r.w, 2 * k);
      ctx.restore();
    }

    trace(ctx, r, sig.sig, count, off, range, C.carrier, 1.1);
    trace(ctx, r, sig.env, count, off, range, C.envelope, 1.7, [5, 4]);
    trace(ctx, r, sig.envNeg, count, off, range, C.envelope, 1.7, [5, 4]);

    ctx.fillStyle = C.tick;
    ctx.font = "11px 'SF Pro Display', -apple-system, Inter, system-ui, sans-serif";
    ctx.fillText(`${(span * 1000).toFixed(0)} ms shown`, r.x + 3, r.y + r.h + 15);

    const i = sampleAt("mod", w);
    if (i != null) {
      crosshair(ctx, r, r.x + (i / (count - 1)) * r.w, [
        `t = ${((i / FS) * 1000).toFixed(1)} ms`,
        `s(t) = ${sig.sig[(off + i) % N].toFixed(3)}`,
        `envelope = ${sig.env[(off + i) % N].toFixed(3)}`,
      ]);
    }
  }, [sig, count, m, off, hover, over, span]);

  const specRef = usePlot((ctx, w, h) => {
    const r = rect(w, h);
    field(ctx, w, h, r);
    const floorDb = -70;
    const toY = (a) => {
      if (!logScale) return r.y + r.h - Math.min(a / 1.15, 1) * (r.h - 6);
      const db = 20 * Math.log10(Math.max(a, 1e-9));
      const f = Math.min(Math.max((db - floorDb) / (6 - floorDb), 0), 1);
      return r.y + r.h - f * (r.h - 6);
    };
    const fx = (f) => r.x + (f / F_VIEW) * r.w;

    for (let k = 0; k < sig.spec.length; k++) {
      const f = k * BIN;
      const near =
        Math.abs(f - fc) < 0.6 ||
        Math.abs(f - (fc + fm)) < 0.6 ||
        Math.abs(f - (fc - fm)) < 0.6;
      ctx.strokeStyle = near ? C.spike : C.bars;
      ctx.lineWidth = near ? 2.4 : 1;
      ctx.beginPath();
      ctx.moveTo(fx(f), r.y + r.h);
      ctx.lineTo(fx(f), toY(sig.spec[k]));
      ctx.stroke();
    }

    // bandwidth bracket across the two sidebands
    const xa = fx(fc - fm), xb = fx(fc + fm), yb = r.y + 18;
    ctx.strokeStyle = C.spike;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(xa, yb + 5); ctx.lineTo(xa, yb);
    ctx.lineTo(xb, yb); ctx.lineTo(xb, yb + 5);
    ctx.stroke();
    ctx.fillStyle = C.spike;
    ctx.font = "11px 'SF Pro Display', -apple-system, Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(`bandwidth ${2 * fm} Hz`, (xa + xb) / 2, yb - 4);
    ctx.textAlign = "left";

    ctx.strokeStyle = C.rule;
    ctx.fillStyle = C.tick;
    for (let f = 0; f <= F_VIEW; f += 20) {
      ctx.beginPath();
      ctx.moveTo(Math.round(fx(f)) + 0.5, r.y + r.h);
      ctx.lineTo(Math.round(fx(f)) + 0.5, r.y + r.h + 3);
      ctx.stroke();
      ctx.fillText(String(f), fx(f) - 6, r.y + r.h + 15);
    }
    ctx.fillText("Hz", r.x + r.w - 16, r.y + r.h + 15);

    if (hover.id === "spec") {
      const f = ((hover.x - r.x) / r.w) * F_VIEW;
      if (f >= 0 && f <= F_VIEW) {
        const amp = sig.spec[Math.round(f / BIN)] || 0;
        crosshair(ctx, r, hover.x, [
          `f = ${f.toFixed(1)} Hz`,
          logScale
            ? `${(20 * Math.log10(Math.max(amp, 1e-9))).toFixed(1)} dB`
            : `amplitude = ${amp.toFixed(3)}`,
        ]);
      }
    }
  }, [sig, logScale, fc, fm, hover]);

  const msgRef = usePlot((ctx, w, h) => {
    const r = rect(w, h);
    field(ctx, w, h, r);
    midline(ctx, r);
    yTicks(ctx, r, 1.05, [-1, 0, 1]);
    trace(ctx, r, sig.msg, count, off, 1.05, C.message, 2);
    const i = sampleAt("msg", w);
    if (i != null) {
      crosshair(ctx, r, r.x + (i / (count - 1)) * r.w, [
        `t = ${((i / FS) * 1000).toFixed(1)} ms`,
        `m(t) = ${sig.msg[(off + i) % N].toFixed(3)}`,
      ]);
    }
  }, [sig, count, off, hover]);

  const recRef = usePlot((ctx, w, h) => {
    const r = rect(w, h);
    const range = over ? 1.7 : 1.05;
    field(ctx, w, h, r);
    midline(ctx, r);
    yTicks(ctx, r, range, [-1, 0, 1]);
    trace(ctx, r, sig.msg, count, off, range, C.message, 1.3, [4, 4]);
    trace(ctx, r, sig.rec, count, off, range, C.recovered, 2);
    const i = sampleAt("rec", w);
    if (i != null) {
      const a = sig.msg[(off + i) % N], b = sig.rec[(off + i) % N];
      crosshair(ctx, r, r.x + (i / (count - 1)) * r.w, [
        `sent = ${a.toFixed(3)}`,
        `received = ${b.toFixed(3)}`,
        `error = ${(b - a).toFixed(3)}`,
      ]);
    }
  }, [sig, count, off, hover, over]);

  const preset = useCallback((a, b, c, d) => {
    setFc(a); setFm(b); setM(c); setSnr(d);
  }, []);

  return (
    <div className="bench">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

        .bench {
          --sf: "SF Pro Display", -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
          --deep: #10272c;
          --deep-2: #17343a;
          --deep-3: #1e4048;
          --line: #2c525b;
          --fg: #e4eceb;
          --fg-dim: #93aeb2;
          --brass: #dda63f;
          --rose: #d97a63;
          background: var(--deep);
          color: var(--fg);
          font-family: var(--sf);
          font-size: 14px; line-height: 1.55;
          padding: 26px 24px 30px;
          min-height: 100%;
          box-sizing: border-box;
        }
        .bench *, .bench *::before, .bench *::after { box-sizing: border-box; }

        .mast { display: flex; justify-content: space-between; align-items: flex-end;
                gap: 26px; flex-wrap: wrap; padding-bottom: 18px;
                border-bottom: 1px solid var(--line); margin-bottom: 20px; }
        .mast h1 { margin: 0; font-size: 16px; font-weight: 600; letter-spacing: -.01em; }
        .mast .course { margin: 2px 0 16px; font-size: 13px; color: var(--fg-dim); }
        .eq { font-family: var(--sf);
              font-size: clamp(20px, 3.6vw, 31px); font-weight: 400; }
        .eq i { font-style: italic; }
        .eq .v { color: var(--brass); font-style: normal; font-variant-numeric: tabular-nums; }
        .eq .o { color: var(--fg-dim); }
        .stat { text-align: right; }
        .stat .n { font-family: var(--sf); font-size: 36px;
                   line-height: 1; font-variant-numeric: tabular-nums; }
        .stat .l { font-size: 12px; color: var(--fg-dim); margin-top: 5px;
                   max-width: 200px; margin-left: auto; line-height: 1.4; }

        .flag { margin: 0 0 20px; padding: 11px 14px; border-radius: 2px;
                background: rgba(217,122,99,.13); border-left: 3px solid var(--rose);
                font-size: 13px; color: #f7e2da; max-width: 66ch; }

        .grid { display: grid; grid-template-columns: 264px minmax(0,1fr); gap: 20px; align-items: start; }
        .stack { display: grid; gap: 16px; }
        .pair { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 16px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } .pair { grid-template-columns: 1fr; } }

        .rail { background: var(--deep-2); border: 1px solid var(--line);
                border-radius: 3px; padding: 17px; position: sticky; top: 14px; }
        @media (max-width: 900px) { .rail { position: static; } }
        .rail h3 { margin: 0 0 12px; font-size: 12px; font-weight: 500; color: var(--fg-dim); }
        .ctl { margin-bottom: 15px; }
        .ctl:last-child { margin-bottom: 0; }
        .ctl-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
        .ctl-top label { font-size: 13px; }
        .ctl-top output { font-size: 13px; color: var(--brass); font-variant-numeric: tabular-nums; }
        .ctl small { display: block; font-size: 11.5px; color: var(--fg-dim); margin-top: 4px; line-height: 1.4; }

        input[type=range] { -webkit-appearance: none; appearance: none;
          width: 100%; background: transparent; margin: 8px 0 0; cursor: grab; }
        input[type=range]:active { cursor: grabbing; }
        input[type=range]::-webkit-slider-runnable-track { height: 2px; background: var(--line); border-radius: 2px; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none;
          width: 15px; height: 15px; margin-top: -6.5px; border-radius: 50%;
          background: var(--brass); border: 2px solid var(--deep-2); }
        input[type=range]::-moz-range-track { height: 2px; background: var(--line); border-radius: 2px; }
        input[type=range]::-moz-range-thumb { width: 13px; height: 13px; border-radius: 50%;
          background: var(--brass); border: 2px solid var(--deep-2); }
        input[type=range]:focus-visible { outline: 2px solid var(--brass); outline-offset: 4px; border-radius: 2px; }

        .sep { height: 1px; background: var(--line); margin: 17px -17px; }
        .btns { display: flex; flex-wrap: wrap; gap: 6px; }
        button { font-family: inherit; font-size: 12px; cursor: pointer; color: var(--fg);
                 background: transparent; border: 1px solid var(--line);
                 border-radius: 2px; padding: 6px 10px;
                 transition: border-color .12s ease, color .12s ease; }
        button:hover { border-color: var(--brass); color: var(--brass); }
        button:focus-visible { outline: 2px solid var(--brass); outline-offset: 2px; }
        button[aria-pressed=true] { background: var(--brass); border-color: var(--brass); color: #14262b; }
        .play { width: 100%; margin-top: 8px; }

        .rd { list-style: none; margin: 0; padding: 0; }
        .rd li { display: flex; justify-content: space-between; gap: 12px; font-size: 12.5px;
                 padding: 6px 0; border-bottom: 1px solid rgba(44,82,91,.55); }
        .rd li:last-child { border-bottom: 0; padding-bottom: 0; }
        .rd span { color: var(--fg-dim); }
        .rd b { font-weight: 500; font-variant-numeric: tabular-nums; }

        .panel { background: var(--deep-2); border: 1px solid var(--line);
                 border-radius: 3px; padding: 14px; }
        .panel-feature { background: var(--deep-3); }
        .p-head { display: flex; justify-content: space-between; align-items: flex-start;
                  gap: 14px; flex-wrap: wrap; margin-bottom: 11px; }
        .p-head h2 { margin: 0; font-size: 14px; font-weight: 600; }
        .p-sub { margin: 3px 0 0; font-size: 12px; color: var(--fg-dim); max-width: 54ch; }
        .p-meta { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
        .hint { font-size: 11.5px; color: var(--brass); padding: 2px 8px;
                border: 1px dashed rgba(221,166,63,.45); border-radius: 2px; }
        .legend { display: flex; gap: 13px; list-style: none; margin: 0; padding: 0; }
        .legend li { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--fg-dim); }
        .sw { width: 14px; display: inline-block; border-radius: 2px; }
        .plot { border-radius: 2px; overflow: hidden; touch-action: none; }
        .plot canvas { display: block; }

        .foot { margin-top: 22px; padding-top: 14px; border-top: 1px solid var(--line);
                font-size: 12px; color: var(--fg-dim); max-width: 78ch; }

        @media (prefers-reduced-motion: reduce) { button { transition: none; } }
      `}</style>

      <div className="mast">
        <div>
          <h1>AM modulation playground</h1>
          <p className="course">TC-307 Communication Systems</p>
          <div className="eq">
            <i>s</i>(<i>t</i>) <span className="o">=</span> [1 <span className="o">+</span>{" "}
            <span className="v">{m.toFixed(2)}</span> cos(2π<span className="v">{fm}</span>
            <i>t</i>)] <span className="o">·</span> cos(2π<span className="v">{fc}</span><i>t</i>)
          </div>
        </div>
        <div className="stat">
          <div className="n" style={{ color: over ? "#d97a63" : "#dda63f" }}>{eff.toFixed(1)}%</div>
          <div className="l">of transmitted power actually carries the message</div>
        </div>
      </div>

      {over && (
        <p className="flag">
          Overmodulated at m = {m.toFixed(2)}. The envelope crosses zero, the carrier flips phase,
          and the detector folds those troughs back up as extra humps. That information is gone —
          no receiver can recover it.
        </p>
      )}

      <div className="grid">
        <div className="rail">
          <h3>Transmitter</h3>

          <div className="ctl">
            <div className="ctl-top"><label htmlFor="fc">Carrier frequency</label><output>{fc} Hz</output></div>
            <input id="fc" type="range" min="8" max="80" step="1" value={fc} onChange={(e) => setFc(+e.target.value)} />
          </div>
          <div className="ctl">
            <div className="ctl-top"><label htmlFor="fm">Message frequency</label><output>{fm} Hz</output></div>
            <input id="fm" type="range" min="1" max="12" step="1" value={fm} onChange={(e) => setFm(+e.target.value)} />
            <small>Sidebands slide apart as this rises.</small>
          </div>
          <div className="ctl">
            <div className="ctl-top"><label htmlFor="mi">Modulation index</label><output>{m.toFixed(2)}</output></div>
            <input id="mi" type="range" min="0" max="1.6" step="0.01" value={m} onChange={(e) => setM(+e.target.value)} />
            <small>Past 1.00 the wave breaks.</small>
          </div>

          <div className="sep" />
          <h3>Channel</h3>
          <div className="ctl">
            <div className="ctl-top"><label htmlFor="snr">Signal-to-noise</label><output>{snr} dB</output></div>
            <input id="snr" type="range" min="0" max="50" step="1" value={snr} onChange={(e) => setSnr(+e.target.value)} />
            <small>Noise lifts the spectrum floor before it ruins the wave.</small>
          </div>
          <div className="btns">
            <button onClick={() => setSeed(Math.floor(Math.random() * 1e6))}>Reroll noise</button>
          </div>

          <div className="sep" />
          <h3>View</h3>
          <div className="ctl">
            <div className="ctl-top"><label htmlFor="pz">Message periods</label><output>{periods}</output></div>
            <input id="pz" type="range" min="1" max="6" step="1" value={periods} onChange={(e) => setPeriods(+e.target.value)} />
            <small>{cycles} carrier cycles on screen.</small>
          </div>
          <div className="btns">
            <button aria-pressed={!logScale} onClick={() => setLogScale(false)}>Linear</button>
            <button aria-pressed={logScale} onClick={() => setLogScale(true)}>Decibels</button>
          </div>
          <button className="play" aria-pressed={playing} onClick={() => setPlaying((p) => !p)}>
            {playing ? "Freeze the trace" : "Run the signal"}
          </button>

          <div className="sep" />
          <h3>Starting points</h3>
          <div className="btns">
            <button onClick={() => preset(40, 4, 0.6, 45)}>Textbook</button>
            <button onClick={() => preset(40, 4, 1.3, 45)}>Overmodulate</button>
            <button onClick={() => preset(40, 12, 0.6, 45)}>Widen band</button>
            <button onClick={() => preset(40, 4, 0.6, 6)}>Heavy noise</button>
          </div>

          <div className="sep" />
          <ul className="rd">
            <li><span>Lower sideband</span><b>{fc - fm} Hz</b></li>
            <li><span>Carrier</span><b>{fc} Hz</b></li>
            <li><span>Upper sideband</span><b>{fc + fm} Hz</b></li>
            <li><span>Bandwidth, 2·fm</span><b>{2 * fm} Hz</b></li>
            <li><span>Sideband height, m/2</span><b>{(m / 2).toFixed(3)}</b></li>
            <li><span>Detector cutoff</span><b>{Math.sqrt(fm * fc).toFixed(1)} Hz</b></li>
          </ul>
        </div>

        <div className="stack">
          <Panel
            feature
            title="Modulated signal"
            sub="The carrier squeezed inside the dashed envelope. The envelope is the message."
            hint="drag up or down"
            grab="ns-resize"
            height={272}
            legend={[{ label: "s(t)", color: C.carrier }, { label: "envelope", color: C.envelope, dash: true }]}
            canvasRef={modRef}
            onPointerDown={dragIndex}
            onPointerMove={hoverAt("mod")}
            onPointerLeave={clearHover}
          />

          <Panel
            title="Spectrum"
            sub="Three cosines and nothing else — the carrier, plus one sideband either side at fc ± fm."
            hint="drag a gold spike"
            grab="ew-resize"
            height={212}
            legend={[{ label: "carrier and sidebands", color: C.spike }, { label: "noise floor", color: C.bars }]}
            canvasRef={specRef}
            onPointerDown={dragSpike}
            onPointerMove={hoverAt("spec")}
            onPointerLeave={clearHover}
          />

          <div className="pair">
            <Panel
              title="Message sent"
              sub="What you are trying to transmit."
              height={158}
              legend={[{ label: "m(t)", color: C.message }]}
              canvasRef={msgRef}
              onPointerMove={hoverAt("msg")}
              onPointerLeave={clearHover}
            />
            <Panel
              title="Message recovered"
              sub="Rectify, low-pass, drop the DC."
              height={158}
              legend={[{ label: "sent", color: C.message, dash: true }, { label: "detected", color: C.recovered }]}
              canvasRef={recRef}
              onPointerMove={hoverAt("rec")}
              onPointerLeave={clearHover}
            />
          </div>
        </div>
      </div>

      <p className="foot">
        Sampled at {FS} Hz over {(N / FS).toFixed(0)} s, so the FFT resolves {BIN} Hz per bin and every
        spike lands dead centre in a bin — no leakage, no window function needed. The detector cutoff
        sits at √(fm·fc), which keeps it between the two frequencies the way the RC rule asks.
      </p>
    </div>
  );
}
