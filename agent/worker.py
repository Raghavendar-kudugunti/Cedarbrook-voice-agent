"""LiveKit voice worker: browser WebRTC audio -> Deepgram Nova-3 (multi) + Silero VAD + multilingual turn
detector -> Groq LLM with tools -> Cartesia Sonic-3, streamed sentence by sentence.

Run:  python -m agent.worker dev      (or `start` in production, `download-files` once at build time)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import AgentServer, AgentSession, APIConnectOptions, JobContext, JobProcess, cli, llm  # noqa: E402
from livekit.agents.voice.agent_session import SessionConnectOptions  # noqa: E402
from livekit.plugins import cartesia, deepgram, openai, silero  # noqa: E402
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # noqa: E402

from . import guardrails as g  # noqa: E402
from .backend_client import HttpBackend  # noqa: E402
from .core import TTS_LANG, CedarbrookAgent  # noqa: E402
from .session_hooks import CallTracker  # noqa: E402

log = logging.getLogger("cedarbrook.worker")
SILENCE_S = float(os.getenv("SILENCE_SECONDS", "6"))
MAX_CALL_S = float(os.getenv("MAX_CALL_SECONDS", "300"))


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()  # groq (any OpenAI-compatible API) or google


def default_model() -> str:
    return os.getenv("LLM_MODEL") or ("gemini-3.8-flash" if LLM_PROVIDER == "google" else "openai/gpt-oss-120b")


def build_llm(model: str | None = None) -> llm.LLM:
    model = model or default_model()
    if LLM_PROVIDER == "google":
        # Google Gemini through its own plugin (native tool calling); key from Google AI Studio
        from livekit.plugins import google
        return google.LLM(model=model, api_key=os.environ["GOOGLE_API_KEY"], temperature=0.3)
    kwargs = dict(model=model, base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1"),
                  api_key=os.environ["OPENAI_API_KEY"], temperature=0.3, parallel_tool_calls=False,
                  _strict_tool_schema=False)
    if "gpt-oss" in model:
        kwargs["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT", "low")
    return openai.LLM(**kwargs)


def build_voice_llm():
    """One model by default. Groq's free tier is per-minute (8k tokens/min) and a 429 says 'retry in
    1-3 s', so waiting is right. A FallbackAdapter gave up after 0.2 s, and its background recovery probes
    spent the same token budget: that is what sent calls to the front desk mid-booking. Set
    LLM_FALLBACK_MODELS to opt back into fallbacks (e.g. on a paid tier)."""
    names = [default_model()] + [
        m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    models = [build_llm(m) for m in dict.fromkeys(names)]
    if len(models) == 1:
        return models[0]
    return llm.FallbackAdapter(models, attempt_timeout=float(os.getenv("LLM_ATTEMPT_TIMEOUT_S", "6")),
                               max_retry_per_llm=1, retry_interval=2.0)


def llm_retry_options() -> SessionConnectOptions:
    # 429 retries: 0.1 s, then every LLM_RETRY_S, up to LLM_MAX_RETRY times (about 10 s in all by default)
    return SessionConnectOptions(llm_conn_options=APIConnectOptions(
        max_retry=int(os.getenv("LLM_MAX_RETRY", "5")), retry_interval=float(os.getenv("LLM_RETRY_S", "2.5")),
        timeout=15.0))


def build_session(vad, language: str) -> AgentSession:
    return AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi", smart_format=True, punctuate=True, interim_results=True),
        llm=build_voice_llm(),
        tts=cartesia.TTS(model=os.getenv("CARTESIA_MODEL", "sonic-3"), voice=os.environ["CARTESIA_VOICE_ID"],
                         language=TTS_LANG.get(language, "en")),
        vad=vad,
        turn_handling={
            "turn_detection": MultilingualModel(),
            # start thinking on a likely end of turn instead of a long fixed pause
            "endpointing": {"mode": "fixed", "min_delay": float(os.getenv("MIN_ENDPOINT_S", "0.3")),
                            "max_delay": float(os.getenv("MAX_ENDPOINT_S", "2.0"))},
            # barge-in: stop agent audio quickly on caller speech
            "interruption": {"enabled": True, "mode": "vad", "min_duration": float(os.getenv("MIN_INTERRUPT_S", "0.2")),
                             "resume_false_interruption": True, "false_interruption_timeout": 1.5},
            # preemptive generation sends an extra LLM request per turn that is often thrown away; on an
            # 8k tokens/min tier that alone trips the limit. Turn on with PREEMPTIVE=1 on a paid tier.
            "preemptive_generation": {"enabled": os.getenv("PREEMPTIVE", "0") == "1"},
        },
        max_tool_steps=6,
        user_away_timeout=SILENCE_S,
        conn_options=llm_retry_options(),
    )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.25, activation_threshold=0.5)


server = AgentServer()
server.setup_fnc = prewarm


@server.rtc_session(agent_name=os.getenv("AGENT_NAME", "cedarbrook-voice"))
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    call_id = meta.get("call_id")
    if not call_id:
        raise ValueError("call_id missing from dispatch metadata (start calls through /api/voice/token)")
    language = meta.get("language", "en")
    backend = HttpBackend(call_id)
    await backend.start("voice")

    session = build_session(ctx.proc.userdata["vad"], language)
    agent = CedarbrookAgent(backend, call_id, language)

    async def publish(payload: dict) -> None:
        try:
            await ctx.room.local_participant.publish_data(json.dumps(payload, ensure_ascii=False).encode(),
                                                          reliable=True, topic="cedarbrook.events")
        except Exception:
            pass

    tracker = CallTracker(session, agent, publish)
    silence = {"prompted": False, "task": None}

    async def finish(reason: str) -> None:
        if tracker.ended:
            return
        t = silence["task"]
        if t and t is not asyncio.current_task() and not t.done():
            t.cancel()
        await tracker.finish(reason)
        session.shutdown()
        await asyncio.sleep(0.5)
        ctx.shutdown(reason=reason)

    agent.on_end_requested = finish

    async def progress(step: str, info: dict) -> None:
        await publish({"type": "progress", "step": step, **info})

    agent.on_progress = progress

    async def silence_watch() -> None:
        handle = session.say(g.fixed("still_there", agent.language))
        await handle.wait_for_playout()
        await asyncio.sleep(SILENCE_S)
        if session.user_state != "speaking" and not tracker.ended:
            bye = session.say(g.fixed("goodbye_silence", agent.language), allow_interruptions=False)
            await bye.wait_for_playout()
            await finish("silence_timeout")

    @session.on("user_state_changed")
    def _user_state(ev) -> None:
        if ev.new_state == "speaking":
            silence["prompted"] = False
            if silence["task"] and not silence["task"].done():
                silence["task"].cancel()
        elif ev.new_state == "away" and not tracker.ended and not agent.end_requested:
            if session.agent_state in ("speaking", "thinking"):
                return
            if not silence["prompted"]:
                silence["prompted"] = True
                silence["task"] = tracker.launch(silence_watch())

    async def max_duration() -> None:
        await asyncio.sleep(MAX_CALL_S)
        if not tracker.ended and not agent.end_requested:
            session.say(g.fixed("max_duration", agent.language))

    @session.on("error")
    def _error(ev) -> None:
        # only a final failure (all retries spent) counts; a recoverable 429 is still being retried
        if getattr(ev.error, "recoverable", False):
            return
        if isinstance(ev.source, llm.LLM) or type(ev.error).__name__ == "LLMError":
            log.warning("LLM unavailable for this turn (%s)", type(getattr(ev.error, "error", ev.error)).__name__)
            tracker.launch(agent.llm_unavailable())

    @session.on("metrics_collected")
    def _llm_ok(ev) -> None:
        # an LLM request completed, so the provider is answering again
        if type(getattr(ev, "metrics", None)).__name__ == "LLMMetrics":
            agent.llm_failed_turns = 0

    @session.on("close")
    def _closed(_ev) -> None:
        async def _end() -> None:
            await tracker.finish(agent.end_requested or "caller_hangup")
            ctx.shutdown(reason="session closed")
        asyncio.create_task(_end())

    async def on_shutdown(reason: str = "") -> None:
        await tracker.finish("caller_hangup")

    ctx.add_shutdown_callback(on_shutdown)
    await session.start(room=ctx.room, agent=agent)
    tracker.launch(max_duration())


if __name__ == "__main__":
    cli.run_app(server)
