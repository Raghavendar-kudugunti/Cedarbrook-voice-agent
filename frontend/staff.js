/* Staff console: who is waiting for a person, how fast the agent replies, and today's calls. */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sec = (ms) => (ms == null ? "–" : ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms");
const TARGET = 1200, SCALE = 2500;

const REASON = {
  emergency: "Emergency",
  caller_request: "Asked for a person",
  complaint: "Complaint or billing",
  agent_stuck: "Assistant couldn't continue",
};
const OUTCOME = {
  booked: "Booked", rescheduled: "Moved", cancelled: "Cancelled", transferred: "Transferred",
  no_action: "No change", abandoned: "Hung up", live: "On call",
};
const LANG = { en: "English", hi: "Hindi", mr: "Marathi" };

function waited(ts) {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min`;
}

let lastTransfers = [];
function renderQueue(list) {
  lastTransfers = list;
  const sorted = [...list].sort((a, b) => (b.reason === "emergency") - (a.reason === "emergency") || a.created_at - b.created_at);
  $("qcount").textContent = list.length ? `${list.length} waiting` : "";
  $("transfers").innerHTML = sorted.length ? sorted.map((t) => `
    <li class="${t.reason === "emergency" ? "emergency" : ""}">
      <div class="who">
        <div class="reason">${esc(REASON[t.reason] || t.reason)}</div>
        <div class="meta">Call ${esc(t.call_id)}${t.note ? ", " + esc(t.note) : ""}</div>
      </div>
      <div class="side">
        <span class="wait" data-since="${t.created_at}">waiting ${waited(t.created_at)}</span>
        <button class="btn" data-handle="${esc(t.id)}">Mark handled</button>
      </div>
    </li>`).join("")
    : `<li class="empty">No one is waiting for a person.</li>`;
}

function renderLatency(L) {
  const t = (L.recent || []).find((x) => x.total_ms != null);
  const stack = $("stack");
  stack.querySelectorAll(".seg").forEach((n) => n.remove());
  const scale = Math.max(SCALE, t ? t.total_ms * 1.05 : 0);
  stack.querySelector(".target").style.left = `${(TARGET / scale) * 100}%`;
  if (t) {
    const known = (t.eou_ms || 0) + (t.llm_ttft_ms || 0) + (t.tts_ttfb_ms || 0);
    const parts = [["eou", t.eou_ms], ["llm", t.llm_ttft_ms], ["tts", t.tts_ttfb_ms], ["wait", Math.max(0, t.total_ms - known)]];
    for (const [k, v] of parts) {
      if (!v) continue;
      const seg = document.createElement("span");
      seg.className = "seg " + k;
      seg.style.width = `${(v / scale) * 100}%`;
      stack.insertBefore(seg, stack.querySelector(".target"));
    }
  }
  const key = (cls) => `<span class="key ${cls}"></span>`;
  const waitMs = t ? Math.max(0, t.total_ms - ((t.eou_ms || 0) + (t.llm_ttft_ms || 0) + (t.tts_ttfb_ms || 0))) : 0;
  $("stages").innerHTML = t ? `
    <div><dt>${key("eou")}Hearing you finish</dt><dd>${sec(t.eou_ms)}</dd></div>
    <div><dt>${key("llm")}Thinking</dt><dd>${sec(t.llm_ttft_ms)}</dd></div>
    <div><dt>${key("tts")}Voice starts</dt><dd>${sec(t.tts_ttfb_ms)}</dd></div>
    <div><dt>${waitMs > 300 ? key("wait") + "Waiting, retries" : "Total"}</dt><dd class="${t.total_ms > TARGET ? "over" : ""}">${waitMs > 300 ? sec(waitMs) : sec(t.total_ms)}</dd></div>
    ${waitMs > 300 ? `<div class="total-line"><dt>Total</dt><dd class="over">${sec(t.total_ms)}</dd></div>` : ""}`
    : `<div><dt>No caller turns yet today</dt><dd>–</dd></div>`;
  $("todayLat").innerHTML = `
    <div><dt>Typical today (p50)</dt><dd class="${L.total.p50 > TARGET ? "over" : ""}">${sec(L.total.p50)}</dd></div>
    <div><dt>Slowest 5% (p95)</dt><dd class="${L.total.p95 > 2000 ? "over" : ""}">${sec(L.total.p95)}</dd></div>
    <div><dt>Stops when interrupted</dt><dd class="${L.barge_in.p50 > 300 ? "over" : ""}">${sec(L.barge_in.p50)}</dd></div>
    <div><dt>Turns measured</dt><dd>${L.turns}</dd></div>`;
}

const inr = (v) => "₹" + (Math.round(v * 100) / 100).toFixed(2);

function renderCalls(calls, counts) {
  const n = (k) => counts[k] || 0;
  $("tally").innerHTML = calls.length
    ? `<b>${calls.length}</b> calls today: <b>${n("booked")}</b> booked, <b>${n("rescheduled")}</b> moved, <b>${n("cancelled")}</b> cancelled, <b>${n("transferred")}</b> transferred, <b>${n("no_action") + n("abandoned")}</b> with no change${n("live") ? `, <b>${n("live")}</b> on a call now` : ""}.`
    : "No calls yet today.";
  const costs = calls.map((c) => (c.record || {}).cost_inr).filter((v) => typeof v === "number");
  const total = costs.reduce((a, b) => a + b, 0);
  const booked = n("booked") + n("rescheduled");
  $("costs").innerHTML = costs.length
    ? `<div><dt>Spent today</dt><dd>${inr(total)}</dd></div>
       <div><dt>Average per call</dt><dd>${inr(total / costs.length)}</dd></div>
       <div><dt>Per booking</dt><dd>${booked ? inr(total / booked) : "–"}</dd></div>
       <div><dt>Most expensive call</dt><dd>${inr(Math.max(...costs))}</dd></div>`
    : "";
  $("calls").innerHTML = calls.length ? calls.map((c) => {
    const r = c.record || {};
    const o = r.outcome || "live";
    const langs = (r.language || []).map((l) => LANG[l] || l).join(", ");
    const time = new Date(c.started_at * 1000).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
    const what = o === "live" ? "In progress" : (r.summary || "").replace(/^Caller\s+/, "").replace(/^./, (x) => x.toUpperCase());
    return `<tr tabindex="0" data-call="${esc(c.call_id)}"><td class="when">${time}</td>
      <td><span class="out ${o}">${OUTCOME[o] || o}</span></td><td>${esc(what)}</td><td>${esc(langs)}</td>
      <td class="num">${sec(r.latency_ms && r.latency_ms.p50)}</td>
      <td class="num">${typeof r.cost_inr === "number" ? inr(r.cost_inr) : "–"}</td></tr>`;
  }).join("") : `<tr><td colspan="6" class="empty">Calls appear here as they come in.</td></tr>`;
}

const last = { q: "", l: "", c: "" };
async function refresh() {
  let d;
  try { d = await (await fetch("/api/dashboard")).json(); } catch { $("health").textContent = "Can't reach the server"; $("health").className = "health bad"; return; }
  // only touch the DOM when something changed, so focus and hover survive the 2.5 s refresh
  const sig = (x) => JSON.stringify(x);
  if (sig(d.transfers_waiting) !== last.q) { last.q = sig(d.transfers_waiting); renderQueue(d.transfers_waiting); }
  if (sig(d.latency) !== last.l) { last.l = sig(d.latency); renderLatency(d.latency); }
  if (sig([d.calls, d.outcome_counts]) !== last.c) { last.c = sig([d.calls, d.outcome_counts]); renderCalls(d.calls, d.outcome_counts || {}); }
}

async function health() {
  try {
    const h = await (await fetch("/api/health")).json();
    $("today").textContent = new Date(h.today_ist + "T00:00:00").toLocaleDateString("en-IN", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
    $("health").textContent = h.voice_ready ? `Voice line on, ${h.llm_model.replace("openai/", "")}` : `Voice line off: missing ${h.missing.join(", ")}`;
    $("health").className = "health" + (h.voice_ready ? "" : " bad");
  } catch {}
}

async function tests() {
  try {
    const t = await (await fetch("/api/test-results")).json();
    if (!t.available) return;
    const rows = t.results.map((r) => {
      const [cls, word] = r.blocked ? ["blocked", "Blocked (rate limit)"] : r.passed ? ["pass", "Pass"] : ["fail", "Fail"];
      return `<tr><td>${esc(r.id)}</td><td>${esc(r.title)}</td><td class="${cls}">${word}</td></tr>`;
    }).join("");
    $("tests").innerHTML = `<p class="sub">Last run ${esc(t.ran_at)} on ${esc(t.model)}. Public scripts passed: <b>${t.public_passed} of ${t.public_total}</b>.</p>
      <div class="table-wrap"><table><tbody>${rows}</tbody></table></div>`;
  } catch {}
}

async function openCall(id) {
  const d = await (await fetch(`/api/calls/${encodeURIComponent(id)}`)).json();
  const r = d.record || {};
  $("dh").textContent = `Call ${id}`;
  const facts = d.record ? `
    <dl class="facts">
      <dt>Outcome</dt><dd>${esc(OUTCOME[r.outcome] || r.outcome)}${r.transfer_reason ? ` (${esc(REASON[r.transfer_reason] || r.transfer_reason)})` : ""}</dd>
      <dt>Summary</dt><dd>${esc(r.summary)}</dd>
      ${r.appointment_id ? `<dt>Appointment</dt><dd>${esc(r.appointment_id)}</dd>` : ""}
      <dt>Read back before change</dt><dd>${r.confirmed_before_commit == null ? "No change made" : r.confirmed_before_commit ? "Yes" : "No"}</dd>
      <dt>Caller turns</dt><dd>${r.turns}, interrupted ${r.barge_ins} times</dd>
      <dt>Reply time</dt><dd>typical ${sec(r.latency_ms.p50)}, slowest ${sec(r.latency_ms.p95)}</dd>
      <dt>Length</dt><dd>${Math.round(r.duration_s)} s, ended by ${esc((r.end_reason || "").replace("_", " "))}</dd>
      <dt>Estimated cost</dt><dd>₹${r.cost_inr}</dd>
    </dl>` : `<p class="sub">This call is still in progress.</p>`;
  const convo = d.transcript.map((t) => `<li class="${t.role}"><b>${t.role === "agent" ? "Assistant" : "Caller"}</b>${esc(t.text)}</li>`).join("");
  const tools = d.tools.map((t) => `<li>${esc(t.name)} ${t.ok ? "" : "(refused)"} in ${Math.round(t.duration_ms)} ms</li>`).join("");
  $("drawerBody").innerHTML = `${facts}
    <h3>Conversation</h3><p class="sub">Phone numbers show only the last four digits.</p><ol class="convo">${convo || "<li>No speech recorded.</li>"}</ol>
    <h3>Scheduling actions</h3><ul class="tools">${tools || "<li>None</li>"}</ul>
    <details class="raw"><summary>Call record (JSON)</summary><pre>${esc(JSON.stringify(d.record, null, 2))}</pre></details>
    <details class="raw"><summary>Turn timings</summary><pre>${esc(d.turns.map((t) => JSON.stringify(t)).join("\n"))}</pre></details>`;
  $("drawer").showModal();
}

document.addEventListener("click", async (e) => {
  const h = e.target.closest("[data-handle]");
  if (h) { h.disabled = true; await fetch(`/api/transfers/${h.dataset.handle}/handled`, { method: "POST" }); refresh(); return; }
  const row = e.target.closest("tr[data-call]");
  if (row) openCall(row.dataset.call);
});
document.addEventListener("keydown", (e) => {
  const row = e.target.closest && e.target.closest("tr[data-call]");
  if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openCall(row.dataset.call); }
});
setInterval(() => document.querySelectorAll(".wait[data-since]").forEach((w) => { w.textContent = "waiting " + waited(+w.dataset.since); }), 1000);

health(); refresh(); tests();
setInterval(refresh, 2500);
