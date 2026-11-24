/* Caller page: one call over LiveKit WebRTC, a voice ring driven by the real audio, live captions,
   booking progress, and what happened at the end. */
const LK = window.LivekitClient;
const $ = (id) => document.getElementById(id);
const dial = $("dial");
const main = $("main");
let room = null;
let startedAt = 0, timerId = 0, callId = null;

const STATUS = {
  idle: "Tap to talk to our appointment assistant.",
  connecting: "Connecting… allow the microphone when your browser asks.",
  live: "Connected. The assistant will greet you.",
  listening: "Listening. Go ahead and speak.",
  thinking: "One moment…",
  speaking: "The assistant is speaking. You can interrupt.",
};

function setState(s) {
  dial.dataset.state = s;
  $("statusText").textContent = STATUS[s] || STATUS.live;
  $("callLabel").textContent = s === "idle" ? "Start call" : s === "connecting" ? "Connecting" : "End call";
}

const langOf = (t) => (/[ऀ-ॿ]/.test(t) ? "hi" : "en");

function caption(role, text) {
  const el = role === "agent" ? $("agentLine") : $("youLine");
  el.textContent = text;
  el.lang = langOf(text);
  el.classList.remove("fresh"); void el.offsetWidth; el.classList.add("fresh");
  if (role === "agent") $("youLine").textContent = "";
  const li = document.createElement("li");
  li.className = role;
  li.lang = langOf(text);
  li.innerHTML = `<b>${role === "agent" ? "Assistant" : "You"}</b>`;
  li.append(document.createTextNode(text));
  $("history").append(li);
  $("historyBox").hidden = false;
}

/* ---------------------------------------------------------------- booking progress */
const ORDER = ["doctor", "time", "details", "done"];
const fmtTime = (hhmm) => { const [h, m] = hhmm.split(":").map(Number); return `${h % 12 || 12}:${String(m).padStart(2, "0")} ${h < 12 ? "am" : "pm"}`; };
const fmtDate = (iso) => new Date(iso + "T00:00:00").toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long" });

function progress(ev) {
  const step = ["booked", "rescheduled", "cancelled"].includes(ev.step) ? "done" : ev.step;
  const at = ORDER.indexOf(step);
  if (at < 0) return;
  $("progress").hidden = false;
  const cur = Math.max(at, Number($("progress").dataset.at ?? -1));
  $("progress").dataset.at = cur;
  document.querySelectorAll(".steps li").forEach((li, i) => {
    li.classList.toggle("done", i < cur || (i === cur && step === "done"));
    li.classList.toggle("now", i === cur && step !== "done");
  });
  const parts = [];
  if (ev.doctor) parts.push(ev.doctor);
  if (ev.date) parts.push(fmtDate(ev.date));
  if (ev.time && /^\d{1,2}:\d{2}$/.test(ev.time)) parts.push(fmtTime(ev.time));
  if (parts.length) $("picked").textContent = parts.join(", ");
}

function resetProgress() {
  $("progress").hidden = true;
  delete $("progress").dataset.at;
  $("picked").textContent = "";
  document.querySelectorAll(".steps li").forEach((li) => li.classList.remove("done", "now"));
}

function showOutcome(r) {
  const box = $("outcome");
  const detail = (r.summary || "").replace(/^Caller\s+/, "").replace(/^./, (c) => c.toUpperCase())
    .replace(/\b(\d{1,2}):(\d{2})\b/g, (_, h, m) => `${(+h % 12) || 12}:${m} ${+h < 12 ? "am" : "pm"}`);
  let cls = "", title = "", body = detail;
  switch (r.outcome) {
    case "booked": title = "Your appointment is booked"; body = `${detail} Please arrive 15 minutes early.`; break;
    case "rescheduled": title = "Your appointment has been moved"; break;
    case "cancelled": title = "Your appointment is cancelled"; break;
    case "transferred":
      if (r.transfer_reason === "emergency") {
        cls = "urgent"; title = "Call 108 or come to Cedarbrook Emergency now";
        body = "Our staff have been alerted to your call.";
      } else {
        cls = "handoff"; title = "Passed to our front desk";
        body = "A member of staff will follow up with you.";
      }
      break;
    default: title = "Call ended"; body = "Nothing was booked or changed. Call again whenever you're ready.";
  }
  box.className = "outcome " + cls;
  box.innerHTML = "";
  const h = document.createElement("h2"); h.textContent = title;
  const p = document.createElement("p"); p.textContent = body;
  box.append(h, p);
  box.hidden = false;
}

/* ---------------------------------------------------------------- voice ring (canvas) */
const ring = (() => {
  const cv = $("voice"), ctx = cv.getContext("2d");
  const BARS = 72;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  let actx = null, agentAn = null, micAn = null, buf = null;
  const level = new Float32Array(BARS); // smoothed bar lengths 0..1
  let thinkAngle = 0, mix = 0; // mix: 0 = caller colour, 1 = agent colour
  let raf = 0, last = performance.now();

  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  let colours = {};
  const readColours = () => { colours = { teal: css("--teal"), mari: css("--marigold"), rule: css("--rule"), soft: css("--ink-soft") }; };
  readColours();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readColours);

  function size() {
    const r = cv.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = Math.round(r.width * dpr); cv.height = Math.round(r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  new ResizeObserver(size).observe(cv);

  function ensureCtx() {
    if (!actx) actx = new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === "suspended") actx.resume();
    return actx;
  }
  function analyser(track) {
    const a = ensureCtx();
    const src = a.createMediaStreamSource(new MediaStream([track]));
    const an = a.createAnalyser(); an.fftSize = 256; an.smoothingTimeConstant = 0.72;
    src.connect(an); // not connected to output: the <audio> element plays the sound
    buf = buf || new Uint8Array(an.frequencyBinCount);
    return an;
  }

  // energy per bar from the speech band, mirrored so the ring is symmetric
  function bands(an) {
    an.getByteFrequencyData(buf);
    const out = new Float32Array(BARS), half = BARS / 2, lo = 2, hi = 56;
    for (let i = 0; i < half; i++) {
      const b = Math.floor(lo + (hi - lo) * (i / half));
      const v = (buf[b] + buf[b + 1]) / 510;
      out[i] = out[BARS - 1 - i] = Math.pow(v, 1.4);
    }
    return out;
  }

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    const st = dial.dataset.state;
    const W = cv.clientWidth, H = cv.clientHeight, cx = W / 2, cy = H / 2;
    const inner = W * 0.27, span = W * 0.2;
    ctx.clearRect(0, 0, W, H);

    let target = null, toward = mix;
    if (st === "speaking" && agentAn) { target = bands(agentAn); toward = 1; }
    else if ((st === "listening" || st === "live") && micAn) { target = bands(micAn); toward = 0; }
    mix += (toward - mix) * Math.min(1, dt * 6);

    // idle / waiting: a slow two-beat pulse, like a heartbeat on a monitor
    const t = now / 1000, beatT = t % 1.6;
    const beat = Math.exp(-Math.pow((beatT - 0.15) / 0.07, 2)) + 0.6 * Math.exp(-Math.pow((beatT - 0.42) / 0.08, 2));

    for (let i = 0; i < BARS; i++) {
      let goal;
      if (target) goal = Math.max(0.04, target[i]);
      else if (st === "thinking") goal = 0.06;
      else if (st === "connecting") goal = 0.05 + 0.05 * Math.sin(t * 6 + i * 0.5);
      else goal = 0.05 + 0.16 * beat * (0.6 + 0.4 * Math.sin(i * 0.9));
      const k = goal > level[i] ? 18 : 6; // fast attack, slow release
      level[i] += (goal - level[i]) * Math.min(1, dt * k);
    }

    const active = st !== "idle" && st !== "connecting";
    const col = st === "thinking" || st === "idle" ? colours.teal
      : active ? (mix > 0.5 ? colours.teal : colours.mari) : colours.soft;
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(2.5, W / 110);

    if (st === "thinking") thinkAngle += dt * 3.2;
    for (let i = 0; i < BARS; i++) {
      const a = (i / BARS) * Math.PI * 2 - Math.PI / 2;
      let len = level[i] * span;
      let alpha = active ? 0.35 + 0.65 * Math.min(1, level[i] * 2.2) : 0.55;
      if (st === "thinking") {
        const d = Math.abs(((a - thinkAngle) % (Math.PI * 2) + Math.PI * 3) % (Math.PI * 2) - Math.PI);
        const glow = Math.max(0, 1 - d / 1.1);
        len += glow * span * 0.35; alpha = 0.25 + 0.75 * glow;
      }
      const r0 = inner + 4, r1 = inner + 4 + Math.max(3, len);
      ctx.strokeStyle = col; ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(frame);
  }

  function run() { cancelAnimationFrame(raf); if (reduce.matches) { drawStatic(); return; } last = performance.now(); raf = requestAnimationFrame(frame); }
  function drawStatic() {
    const W = cv.clientWidth, H = cv.clientHeight, cx = W / 2, cy = H / 2, inner = W * 0.27;
    ctx.clearRect(0, 0, W, H); ctx.strokeStyle = colours.teal; ctx.globalAlpha = 0.5; ctx.lineWidth = 3; ctx.lineCap = "round";
    for (let i = 0; i < BARS; i++) { const a = (i / BARS) * Math.PI * 2; ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * (inner + 4), cy + Math.sin(a) * (inner + 4)); ctx.lineTo(cx + Math.cos(a) * (inner + 12), cy + Math.sin(a) * (inner + 12)); ctx.stroke(); }
    ctx.globalAlpha = 1;
  }
  reduce.addEventListener("change", run);
  requestAnimationFrame(() => { size(); run(); });

  return {
    unlock: ensureCtx,
    agent(track) { try { agentAn = analyser(track); } catch { agentAn = null; } },
    mic(track) { try { micAn = analyser(track); } catch { micAn = null; } },
    clear() { agentAn = null; micAn = null; },
  };
})();

/* ---------------------------------------------------------------- call */
function onEvent(ev) {
  if (ev.type === "transcript") caption(ev.role, ev.text);
  else if (ev.type === "agent_state" && room) setState(ev.state === "initializing" || ev.state === "idle" ? "live" : ev.state);
  else if (ev.type === "progress") progress(ev);
  else if (ev.type === "call_ended") showOutcome(ev.record || {});
}

function tick() {
  const s = Math.floor((Date.now() - startedAt) / 1000);
  $("timer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

async function start() {
  ring.unlock(); // the audio context must start inside the click
  main.dataset.phase = "call";
  callId = null;
  $("historyBox").open = false;
  setState("connecting");
  $("outcome").hidden = true;
  $("agentLine").textContent = ""; $("youLine").textContent = "";
  $("history").innerHTML = ""; $("historyBox").hidden = true;
  resetProgress();
  try {
    const res = await fetch("/api/voice/token", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (!res.ok) throw new Error(res.status === 503 ? "The phone line is not set up yet." : "The line is busy. Please try again.");
    const { url, token, call_id } = await res.json();
    callId = call_id;
    room = new LK.Room({ adaptiveStream: true, dynacast: true });
    room.on(LK.RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== "audio") return;
      const el = track.attach(); el.dataset.agent = "1"; document.body.append(el);
      ring.agent(track.mediaStreamTrack);
    });
    room.on(LK.RoomEvent.DataReceived, (payload, _p, _k, topic) => {
      if (topic !== "cedarbrook.events") return;
      try { onEvent(JSON.parse(new TextDecoder().decode(payload))); } catch {}
    });
    room.on(LK.RoomEvent.Disconnected, cleanup);
    room.on(LK.RoomEvent.ParticipantDisconnected, (p) => { if (p.isAgent) end(); });
    await room.connect(url, token);
    await room.localParticipant.setMicrophoneEnabled(true, { echoCancellation: true, noiseSuppression: true, autoGainControl: true });
    const mic = room.localParticipant.getTrackPublication(LK.Track.Source.Microphone);
    if (mic && mic.track) ring.mic(mic.track.mediaStreamTrack);
    setState("live");
    startedAt = Date.now(); tick(); timerId = setInterval(tick, 1000);
  } catch (e) {
    if (room) { try { await room.disconnect(); } catch {} }
    room = null;
    main.dataset.phase = "idle";
    setState("idle");
    const micDenied = /permission|notallowed/i.test(String(e && (e.name + e.message)));
    $("statusText").textContent = micDenied
      ? "Microphone access is blocked. Allow it in your browser's address bar, then start the call again."
      : e.message || "Could not connect. Check your internet and try again.";
  }
}

async function end() { if (room) await room.disconnect(); cleanup(); }

function cleanup() {
  if (main.dataset.phase !== "call") return; // runs once per call (disconnect fires it too)
  document.querySelectorAll("audio[data-agent]").forEach((a) => a.remove());
  ring.clear();
  clearInterval(timerId); timerId = 0; $("timer").textContent = "";
  room = null;
  main.dataset.phase = "ended";
  setState("idle");
  $("statusText").textContent = "Call ended. Tap to call again.";
  startedAt = 0;
  // the live view is over: clear it, keep the full conversation in "Conversation so far"
  $("agentLine").textContent = ""; $("youLine").textContent = "";
  $("progress").hidden = true;
  if ($("outcome").hidden) fetchOutcome(callId);
}

// When the caller hangs up first, the agent can't send its call record over the room any more:
// ask the API for it (the record is written a moment after the call closes).
async function fetchOutcome(id) {
  for (let i = 0; i < 6 && id; i++) {
    await new Promise((r) => setTimeout(r, 800));
    if (!$("outcome").hidden || main.dataset.phase !== "ended") return;
    try {
      const res = await fetch(`/api/calls/${encodeURIComponent(id)}`);
      if (res.ok) { const c = await res.json(); if (c.record && c.record.outcome) { showOutcome(c.record); return; } }
    } catch {}
  }
  if ($("outcome").hidden && main.dataset.phase === "ended") showOutcome({});
}

$("callBtn").addEventListener("click", () => (dial.dataset.state === "idle" ? start() : dial.dataset.state !== "connecting" && end()));
