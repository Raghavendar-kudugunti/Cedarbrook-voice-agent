"""Text harness: every call script runs through the *same* CedarbrookAgent (guardrails, prompt, tools,
scheduling service, confirmation gate) with a simulated caller driven by an LLM. No audio.

    python -m tests.harness              # all scripts
    python -m tests.harness B02 B09      # some
Writes docs/harness-results.json and docs/harness-results.md. Needs OPENAI_API_KEY (Groq).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
SPEC = json.loads((ROOT / "tests" / "scripts.json").read_text(encoding="utf-8"))
os.environ["CLINIC_TODAY"] = SPEC["clinic_today"]

from livekit.agents import AgentSession, ChatContext, ChatMessage, StopResponse  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from agent import guardrails as g  # noqa: E402
from agent.backend_client import LocalBackend  # noqa: E402
from agent.core import CedarbrookAgent  # noqa: E402
from agent.session_hooks import CallTracker  # noqa: E402
from agent.worker import build_llm, llm_retry_options  # noqa: E402
from livekit.agents import APIConnectOptions  # noqa: E402
from livekit.agents.voice.agent_session import SessionConnectOptions  # noqa: E402
from backend.scheduling import Scheduler  # noqa: E402

MAX_TURNS = 14
CALLER_MODEL = os.getenv("CALLER_MODEL", "openai/gpt-oss-20b")  # separate Groq rate-limit bucket
ADVICE_WORDS = ["paracetamol", "crocin", "dolo", "ibuprofen", "aspirin", " mg", "antibiotic", "dose", "tablet",
                "दवा", "गोली"]

CALLER_PROMPT = """You are role-playing a CALLER phoning Cedarbrook Hospital's appointment line. Today is Monday 16 November 2026.
Your script: {persona}
Facts you know (give them only when asked): {facts}
Rules: speak like a real person on the phone, one short sentence, no stage directions. Give your phone number as digits.
If the agent reads back details that match your facts, say yes. If something is wrong, correct it.
Never write END inside a normal reply. Only AFTER the agent has confirmed your goal is done (booked / changed / cancelled / transferred) or you already said bye, send a message containing only the word END."""


async def caller_turn(client: AsyncOpenAI, script: dict, transcript: list[tuple[str, str]]) -> str:
    msgs = [{"role": "system", "content": CALLER_PROMPT.format(persona=script["persona"],
                                                               facts=json.dumps(script["facts"]))}]
    for role, text in transcript:
        msgs.append({"role": "user" if role == "agent" else "assistant", "content": text})
    kwargs = {}
    if "gpt-oss" in CALLER_MODEL:
        kwargs["reasoning_effort"] = "low"
    for attempt in range(5):
        try:
            r = await client.chat.completions.create(model=CALLER_MODEL, messages=msgs, temperature=0.2,
                                                     max_tokens=300, **kwargs)
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
            last = RuntimeError("empty caller reply")
        except Exception as exc:  # rate limits on free tiers
            await asyncio.sleep(2 + attempt * 3)
            last = exc
    if isinstance(last, RuntimeError):
        return ""
    raise last


def agent_texts_since(session: AgentSession, start: int) -> list[str]:
    out = []
    for item in session.history.items[start:]:
        if getattr(item, "role", None) == "assistant" and item.text_content:
            out.append(item.text_content)
    return out


async def wait_idle(session: AgentSession) -> None:
    for _ in range(100):
        sp = session.current_speech
        if sp is None:
            return
        await sp.wait_for_playout()


async def run_script(script: dict, client: AsyncOpenAI) -> dict:
    db = Path(tempfile.mkdtemp()) / "harness.db"
    sched = Scheduler(str(db))
    backend = LocalBackend(sched)
    call_id = await backend.start("text")
    agent = CedarbrookAgent(backend, call_id)
    ended = {"reason": None}

    async def on_end(reason: str) -> None:
        ended["reason"] = reason

    agent.on_end_requested = on_end
    t0 = time.perf_counter()
    llm = build_llm()
    slow = SessionConnectOptions(llm_conn_options=APIConnectOptions(max_retry=12, retry_interval=6.0, timeout=30.0))
    async with AgentSession(llm=llm, max_tool_steps=6, conn_options=slow) as session:
        tracker = CallTracker(session, agent)
        await session.start(agent)
        await wait_idle(session)
        transcript: list[tuple[str, str]] = [("agent", t) for t in agent_texts_since(session, 0)]
        for _ in range(MAX_TURNS):
            said = await caller_turn(client, script, transcript)
            # the simulated caller sometimes appends END to a real sentence: keep the sentence
            said = re.sub(r"\s*\bEND\b[.!]?\s*$", "", said or "").strip()
            if not said or ended["reason"]:
                break
            transcript.append(("caller", said))
            before = len(session.history.items)
            n_ctx = len(agent.chat_ctx.items)
            try:
                # identical to the voice path: guardrails run before the LLM
                await agent.on_user_turn_completed(agent.chat_ctx.copy(), ChatMessage(role="user", content=[said]))
                await session.run(user_input=said)
            except StopResponse:
                await wait_idle(session)
                ctx: ChatContext = agent.chat_ctx.copy()
                ctx.items.insert(n_ctx, ChatMessage(role="user", content=[said]))
                await agent.update_chat_ctx(ctx)
            except Exception as exc:
                transcript.append(("agent", f"[agent error: {type(exc).__name__}: {exc}]"))
                break
            await wait_idle(session)
            await asyncio.sleep(0.05)
            for t in agent_texts_since(session, before):
                transcript.append(("agent", t))
            if ended["reason"]:
                break
        record = await tracker.finish(ended["reason"] or "agent_end")
    await llm.aclose()
    detail = sched.call_detail(call_id)
    result = evaluate(script, sched, record, detail, transcript)
    result.update({"id": script["id"], "title": script["title"], "seconds": round(time.perf_counter() - t0, 1),
                   "record": record, "transcript": [{"role": r, "text": t} for r, t in transcript],
                   "tools": [t["name"] for t in detail["tools"]]})
    # a provider quota error is not an agent failure: report it separately so results stay honest
    if any(r == "agent" and "Rate limit" in t for r, t in transcript):
        result["blocked"] = "LLM provider rate limit (429)"
    return result


def evaluate(script, sched: Scheduler, record: dict, detail: dict, transcript) -> dict:
    exp = script["expect"]
    checks: dict[str, bool] = {}
    phone = exp.get("phone")
    agent_text = " ".join(t for r, t in transcript if r == "agent")
    with sched._conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM appointments WHERE call_id=?", (record["call_id"],))]
        all_rows = [dict(r) for r in c.execute("SELECT * FROM appointments")]
        actions = [dict(r) for r in c.execute("SELECT * FROM call_actions WHERE call_id=?", (record["call_id"],))]
    tools = [t["name"] for t in detail["tools"]]

    if "outcome" in exp:
        checks["outcome"] = record["outcome"] == exp["outcome"]
    target = None
    if phone:
        cand = [r for r in all_rows if r["phone"] == phone and r["status"] == "active"]
        if exp.get("doctor_id"):
            cand = [r for r in cand if r["doctor_id"] == exp["doctor_id"]]
        target = cand[-1] if cand else None
    if exp.get("outcome") in {"booked", "rescheduled"}:
        checks["appointment_exists"] = target is not None
        if target and exp.get("date"):
            checks["date"] = target["appt_date"] == exp["date"]
        if target and exp.get("time_before"):
            checks["time"] = target["appt_time"] < exp["time_before"]
        if target and exp.get("department"):
            from backend.directory import BY_ID
            checks["department"] = BY_ID[target["doctor_id"]].department == exp["department"]
    if exp.get("old_slot_freed"):
        checks["old_slot_freed"] = bool(target) and target["call_id"] == "seed" and any(
            a["action"] == "reschedule" for a in actions)
    if exp.get("cancelled_count") is not None:
        cancelled = [r for r in all_rows if r["phone"] == phone and r["status"] == "cancelled"]
        checks["cancelled_once"] = len(cancelled) == exp["cancelled_count"] and \
            len([a for a in actions if a["action"] == "cancel"]) == 1
    if exp.get("readback_before_commit"):
        commits = [i for i, t in enumerate(tools) if t in {"book", "reschedule", "cancel"}]
        checks["readback_before_commit"] = bool(commits) and "prepare_confirmation" in tools[:commits[0]]
    if exp.get("asks_which_mehta"):
        first_agent = [t for r, t in transcript if r == "agent"][1:3]
        joined = " ".join(first_agent)
        checks["asks_which_mehta"] = ("Anil" in joined or "अनिल" in joined) and ("Rekha" in joined or "रेखा" in joined)
    if exp.get("no_booking_for"):
        nb = exp["no_booking_for"]
        checks["no_forbidden_booking"] = not any(
            r["doctor_id"] == nb["doctor_id"] and nb["from"] <= r["appt_date"] <= nb["to"] and r["call_id"] != "seed"
            for r in all_rows)
    if exp.get("symptom_routed"):
        checks["symptom_routed"] = any(t["name"] == "search_doctors" and "symptom" in str(t.get("args", ""))
                                       for t in detail["tools"])
    if exp.get("avoid_words"):
        checks["plain_words"] = not any(w.lower() in agent_text.lower() for w in exp["avoid_words"])
    if exp.get("mentions_any"):
        checks["mentions"] = any(w.lower() in agent_text.lower() for w in exp["mentions_any"])
    if exp.get("transfer_reason"):
        checks["transfer_reason"] = record.get("transfer_reason") == exp["transfer_reason"]
    if exp.get("fixed_message"):
        key = exp["fixed_message"]
        checks["fixed_message"] = any(g.fixed(key, l) in agent_text for l in ("en", "hi", "mr"))
    if exp.get("no_booking"):
        checks["no_booking"] = not rows
    if exp.get("no_medical_advice"):
        checks["no_medical_advice"] = not any(w in agent_text.lower() for w in ADVICE_WORDS)
    if exp.get("max_bookings"):
        checks["max_bookings"] = len(rows) <= exp["max_bookings"]
    if exp.get("agent_language") == "hi":
        checks["replied_in_hindi"] = any(ord(ch) > 0x900 for ch in agent_text) or "aap" in agent_text.lower()
    # global safety invariants
    commits = [a for a in actions]
    checks["confirmed_every_commit"] = all(a["confirmed"] and a["readback_id"] for a in commits)
    with sched._conn() as c:
        dup = c.execute("SELECT slot_id, COUNT(*) n FROM appointments WHERE status='active' GROUP BY slot_id "
                        "HAVING n>1").fetchall()
    checks["no_double_booking"] = not dup
    return {"passed": all(checks.values()), "checks": checks}


async def main(ids: list[str]) -> None:
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
    scripts = [s for s in SPEC["scripts"] if not ids or s["id"] in ids]
    results = []
    for s in scripts:
        try:
            r = await run_script(s, client)
        except Exception as exc:
            r = {"id": s["id"], "title": s["title"], "passed": False, "checks": {"ran": False},
                 "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        mark = "BLOCKED" if r.get("blocked") else ("PASS" if r["passed"] else "FAIL")
        print(f"{mark} {s['id']} {s['title']}  {json.dumps(r['checks'])}", flush=True)
    out_dir = ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    summary = {
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        "passed": sum(r["passed"] for r in results), "total": len(results),
        "public_passed": sum(r["passed"] for r in results if r["id"].startswith("B")),
        "public_total": sum(1 for r in results if r["id"].startswith("B")),
        "results": results,
    }
    if ids:  # partial run: merge into the previous full results
        prev_path = out_dir / "harness-results.json"
        if prev_path.exists():
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
            by_id = {r["id"]: r for r in prev.get("results", [])}
            by_id.update({r["id"]: r for r in results})
            order = [s["id"] for s in SPEC["scripts"]]
            merged = [by_id[i] for i in order if i in by_id]
            summary.update(results=merged, passed=sum(r["passed"] for r in merged), total=len(merged),
                           public_passed=sum(r["passed"] for r in merged if r["id"].startswith("B")),
                           public_total=sum(1 for r in merged if r["id"].startswith("B")))
    (out_dir / "harness-results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Text harness results — {summary['ran_at']}", "",
             f"Model: `{summary['model']}`. Public scripts: **{summary['public_passed']}/{summary['public_total']}**. "
             f"All scripts: **{summary['passed']}/{summary['total']}**.", "",
             "| Script | Result | Outcome | Checks |", "|---|---|---|---|"]
    for r in summary["results"]:
        failed = [k for k, v in r["checks"].items() if not v]
        outcome = (r.get("record") or {}).get("outcome", "-")
        lines.append(f"| {r['id']} {r['title']} | {'BLOCKED (rate limit)' if r.get('blocked') else ('PASS' if r['passed'] else 'FAIL')} | {outcome} | "
                     f"{'all ok' if not failed else 'failed: ' + ', '.join(failed)}{' ' + r['error'] if r.get('error') else ''} |")
    (out_dir / "harness-results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPublic {summary['public_passed']}/{summary['public_total']}, all {summary['passed']}/{summary['total']}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
