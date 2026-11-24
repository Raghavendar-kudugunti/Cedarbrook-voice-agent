"""FastAPI: scheduling tools for the agent, LiveKit tokens for the browser, staff dashboard."""
from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:  # local runs read .env; Docker passes env_file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from . import directory as d
from . import symptoms, telemetry
from .scheduling import Scheduler, ToolError

app = FastAPI(title="Cedarbrook Hospital appointments", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8000").split(","),
                   allow_methods=["*"], allow_headers=["*"])

_sched: Scheduler | None = None


def sched() -> Scheduler:
    global _sched
    if _sched is None:
        _sched = Scheduler()
    return _sched


def internal(authorization: str = Header(default="")) -> None:
    """Agent-only endpoints (tool calls, metrics) need the shared AGENT_INTERNAL_TOKEN."""
    token = os.getenv("AGENT_INTERNAL_TOKEN", "")
    if not token or not secrets.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(401, "agent token required")


@app.exception_handler(ToolError)
async def tool_error(_req, exc: ToolError):
    return JSONResponse(exc.as_dict(), status_code=409)


# ---------------- public ----------------
@app.get("/api/health")
def health():
    google = os.getenv("LLM_PROVIDER", "groq").strip().lower() == "google"
    voice_keys = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY",
                  "GOOGLE_API_KEY" if google else "OPENAI_API_KEY"]
    missing = [k for k in voice_keys if not os.getenv(k)]
    model = os.getenv("LLM_MODEL") or ("gemini-3.8-flash" if google else "openai/gpt-oss-120b")
    return {"ok": True, "today_ist": d.today_ist().isoformat(), "voice_ready": not missing, "missing": missing,
            "langfuse": telemetry.enabled(), "llm_provider": "google" if google else "groq", "llm_model": model}


@app.get("/api/doctors")
def doctors():
    return [doc.public() for doc in d.DOCTORS]


class TokenReq(BaseModel):
    language: str = "en"


@app.post("/api/voice/token")
async def voice_token(req: TokenReq):
    url, key, secret = os.getenv("LIVEKIT_URL"), os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        raise HTTPException(503, "LiveKit is not configured (LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET).")
    from livekit import api as lk

    call_id = sched().start_call("voice")
    room = f"cedarbrook-{call_id}"
    identity = f"caller-{call_id}"
    token = (lk.AccessToken(key, secret).with_identity(identity).with_name("Caller")
             .with_grants(lk.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True,
                                         can_publish_data=True))
             .with_room_config(lk.RoomConfiguration(agents=[
                 lk.RoomAgentDispatch(agent_name=os.getenv("AGENT_NAME", "cedarbrook-voice"),
                                      metadata=json.dumps({"call_id": call_id, "language": req.language}))]))
             .to_jwt())
    return {"call_id": call_id, "room": room, "url": url, "token": token}


@app.get("/api/dashboard")
def dashboard():
    return sched().dashboard()


@app.get("/api/calls/{call_id}")
def call_detail(call_id: str):
    return sched().call_detail(call_id)


@app.get("/api/appointments")
def appointments(active_only: bool = True):
    return sched().appointments(active_only)


@app.post("/api/transfers/{transfer_id}/handled")
def transfer_handled(transfer_id: str):
    sched().resolve_transfer(transfer_id)
    return {"ok": True}


@app.get("/api/test-results")
def test_results():
    p = Path(os.getenv("TEST_RESULTS_PATH", "docs/harness-results.json"))
    if not p.exists():
        return {"available": False}
    return {"available": True, **json.loads(p.read_text(encoding="utf-8"))}


# ---------------- agent-only (tool layer) ----------------
class StartCall(BaseModel):
    call_id: str | None = None
    transport: str = "voice"


class ToolCall(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)


class Utterance(BaseModel):
    role: str
    text: str
    language: str | None = None


class TurnMetrics(BaseModel):
    data: dict


class Usage(BaseModel):
    usage: dict = Field(default_factory=dict)
    barge_in: bool = False


class EndCall(BaseModel):
    end_reason: str = "caller_hangup"
    trace: dict | None = None


@app.post("/api/internal/calls", dependencies=[Depends(internal)])
def start_call(req: StartCall):
    return {"call_id": sched().start_call(req.transport, req.call_id)}


@app.post("/api/internal/calls/{call_id}/tool", dependencies=[Depends(internal)])
def run_tool(call_id: str, req: ToolCall):
    return dispatch(sched(), call_id, req.name, req.args)


@app.post("/api/internal/calls/{call_id}/utterance", dependencies=[Depends(internal)])
def utterance(call_id: str, req: Utterance):
    s = sched()
    masked = s.add_transcript(call_id, req.role, req.text)
    verdict = None
    if req.role == "caller":
        verdict = s.record_caller_reply(call_id, req.text)
        if req.language:
            s.add_language(call_id, req.language)
    return {"masked": masked, "confirmation": verdict}


@app.post("/api/internal/calls/{call_id}/turn", dependencies=[Depends(internal)])
def turn(call_id: str, req: TurnMetrics):
    sched().add_turn(call_id, req.data)
    return {"ok": True}


@app.post("/api/internal/calls/{call_id}/usage", dependencies=[Depends(internal)])
def usage(call_id: str, req: Usage):
    s = sched()
    if req.usage:
        s.add_usage(call_id, req.usage)
    if req.barge_in:
        s.add_barge_in(call_id)
    return {"ok": True}


@app.post("/api/internal/calls/{call_id}/end", dependencies=[Depends(internal)])
def end_call(call_id: str, req: EndCall):
    s = sched()
    record = s.end_call(call_id, req.end_reason)
    trace_id = telemetry.export_call(s.call_detail(call_id), record, req.trace or {})
    return {"record": record, "trace_id": trace_id}


def search_doctors_tool(s: Scheduler, args: dict) -> dict:
    """search_doctors, plus symptom routing: the caller's own words ("ghutnon mein dard") pick the
    department in code, with a plain name to say back. Routing only, never a diagnosis."""
    dept, name, lang, symptom = (args.get(k) for k in ("department", "name", "language", "symptom"))
    if not symptom or dept or name:
        return {"ok": True, "doctors": s.search_doctors(dept, name, lang)}
    t = symptoms.triage(symptom)
    res: dict = {"ok": True}
    if t["maybe_child"]:
        res["ask_age_first"] = ("The patient may be a child. Ask their age first: under 16 -> Paediatrics "
                                "(children's doctor); 16 or older -> the department below.")
    if t["not_available"] and not t["departments"]:
        res.update(doctors=[], not_available=t["not_available"],
                   guide=("This OPD line has no doctor for that. Say so simply and offer the front desk "
                          "(transfer_to_human reason caller_request) or a general physician."))
        return res
    depts = t["departments"] or ["General Medicine"]
    if t["maybe_child"]:
        depts = list(dict.fromkeys(depts + ["Paediatrics"]))
    res["departments"] = [{"department": dep, "say_as": symptoms.PLAIN[dep]} for dep in depts]
    res["doctors"] = [doc for dep in depts for doc in s.search_doctors(dep, None, lang)]
    if not t["departments"]:
        res["guide"] = ("No body part or problem recognised. Ask in simple words where it hurts or what the "
                        "problem is; if still unclear, the general physician is the right first doctor.")
    elif len(t["departments"]) > 1:
        res["guide"] = "More than one problem: ask which one they want to see a doctor for first."
    else:
        res["guide"] = "Tell the caller in plain words which doctor (say_as), then offer a time."
    return res


def dispatch(s: Scheduler, call_id: str, name: str, args: dict) -> dict:
    """Single entry point for every agent tool, so HTTP and in-process harness share behaviour."""
    t0 = time.perf_counter()
    try:
        if name == "search_doctors":
            res = search_doctors_tool(s, args)
        elif name == "get_slots":
            res = s.get_slots(call_id, args["doctor_id"], args["date_from"], args.get("date_to"),
                              args.get("part_of_day"))
        elif name == "hold_slot":
            res = s.hold_slot(call_id, args["slot_id"])
        elif name == "find_appointments":
            res = {"ok": True, "appointments": s.find_appointments(args["patient_phone"])}
        elif name == "prepare_confirmation":
            res = s.prepare_confirmation(call_id, args["action"], hold_id=args.get("hold_id"),
                                         appointment_id=args.get("appointment_id"),
                                         patient_name=args.get("patient_name"),
                                         patient_phone=args.get("patient_phone"),
                                         patient_age=args.get("patient_age"))
        elif name == "book":
            res = s.book(call_id, args["hold_id"], args.get("patient_name"), args.get("patient_phone"))
        elif name == "reschedule":
            res = s.reschedule(call_id, args["appointment_id"], args["new_hold_id"])
        elif name == "cancel":
            res = s.cancel(call_id, args["appointment_id"])
        elif name == "transfer_to_human":
            res = s.transfer_to_human(call_id, args.get("reason", "agent_stuck"), args.get("note"))
        else:
            raise ToolError("unknown_tool", f"Unknown tool {name}")
        ok = bool(res.get("ok", True))
    except ToolError as e:
        res, ok = e.as_dict(), False
    except (KeyError, ValueError) as e:
        res, ok = {"ok": False, "error": "bad_arguments", "message": f"Bad or missing argument: {e}"}, False
    s.log_tool(call_id, name, args, ok, (time.perf_counter() - t0) * 1000, res)
    return res


# ---------------- frontend ----------------
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/staff")
    def staff():
        return FileResponse(FRONTEND / "staff.html")
