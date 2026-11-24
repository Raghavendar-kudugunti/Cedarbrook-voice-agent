"""Session event wiring shared by the voice worker and the text harness:
masked transcript, per-turn latency breakdown, barge-ins, usage, and the call-ending sequence."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from livekit.agents import AgentSession
from livekit.agents.metrics.usage import LLMModelUsage, STTModelUsage, TTSModelUsage

from backend.scheduling import mask_text

from .core import CedarbrookAgent

log = logging.getLogger("cedarbrook.hooks")
Publish = Callable[[dict], Awaitable[None]]


async def _noop(_: dict) -> None:
    return None


def _ms(v) -> float | None:
    return round(v * 1000, 1) if isinstance(v, (int, float)) and v >= 0 else None


class CallTracker:
    def __init__(self, session: AgentSession, agent: CedarbrookAgent, publish: Publish = _noop):
        self.session = session
        self.agent = agent
        self.backend = agent.backend
        self.publish = publish
        self.tasks: set[asyncio.Task] = set()
        self.user_metrics: dict | None = None
        self.user_text = ""
        self.barge_started: float | None = None
        self.turns: list[dict] = []
        self.ended = False
        self.end_record: dict | None = None
        self._finish_task: asyncio.Future | None = None
        session.on("conversation_item_added", self._on_item)
        session.on("user_state_changed", self._on_user_state)
        session.on("agent_state_changed", self._on_agent_state)

    def launch(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self._done)
        return t

    def _done(self, t: asyncio.Task) -> None:
        self.tasks.discard(t)
        if not t.cancelled() and t.exception():
            log.warning("background task failed: %r", t.exception())

    # ---------------- transcript + latency ----------------
    def _on_item(self, ev) -> None:
        item = ev.item
        role = getattr(item, "role", None)
        text = getattr(item, "text_content", None) or ""
        if role == "user":
            self.user_metrics = dict(getattr(item, "metrics", {}) or {})
            self.user_metrics.setdefault("stopped_speaking_at", time.time())
            self.user_text = mask_text(text)
            self.launch(self.publish({"type": "transcript", "role": "caller", "text": self.user_text,
                                      "language": self.agent.language}))
        elif role == "assistant" and text:
            masked = mask_text(text)
            self.launch(self.publish({"type": "transcript", "role": "agent", "text": masked,
                                      "interrupted": bool(getattr(item, "interrupted", False))}))
            self.launch(self.backend.utterance("agent", text))
            m = dict(getattr(item, "metrics", {}) or {})
            if self.user_metrics is not None:
                self._record_turn(self.user_metrics, m, masked, bool(getattr(item, "interrupted", False)))
                self.user_metrics = None

    def _record_turn(self, um: dict, am: dict, agent_text: str, interrupted: bool) -> None:
        stopped = um.get("stopped_speaking_at")
        total = am.get("e2e_latency")
        if total is None and stopped and am.get("started_speaking_at"):
            total = am["started_speaking_at"] - stopped
        data = {
            "started_at": stopped,
            "stt_ms": _ms(um.get("transcription_delay")),
            "eou_ms": _ms(um.get("end_of_turn_delay")),
            "llm_ttft_ms": _ms(am.get("llm_node_ttft")),
            "tts_ttfb_ms": _ms(am.get("tts_node_ttfb")),
            "total_ms": _ms(total),
            "tools": list(self.agent.turn_tools),
            "caller_text": self.user_text,
            "agent_text": agent_text,
            "interrupted": interrupted,
        }
        self.turns.append(data)
        self.launch(self.backend.turn(data))
        self.launch(self.publish({"type": "latency", **{k: data[k] for k in
                                                        ("stt_ms", "eou_ms", "llm_ttft_ms", "tts_ttfb_ms", "total_ms")},
                                  "tools": [t["name"] for t in data["tools"]]}))

    # ---------------- barge-in ----------------
    def _on_user_state(self, ev) -> None:
        if ev.new_state == "speaking" and self.session.agent_state == "speaking":
            self.barge_started = time.perf_counter()

    def _on_agent_state(self, ev) -> None:
        self.launch(self.publish({"type": "agent_state", "state": ev.new_state}))
        if ev.old_state == "speaking" and self.barge_started is not None:
            ms = (time.perf_counter() - self.barge_started) * 1000
            self.barge_started = None
            if ms < 5000:
                self.launch(self.backend.usage(barge_in=True))
                self.launch(self.backend.turn({"barge_in_ms": round(ms, 1), "started_at": time.time()}))
                self.launch(self.publish({"type": "barge_in", "ms": round(ms, 1)}))

    # ---------------- end ----------------
    def usage(self) -> dict:
        out = {"stt_audio_seconds": 0.0, "tts_characters": 0, "llm_input_tokens": 0, "llm_output_tokens": 0}
        try:
            for u in self.session.usage.model_usage:
                if isinstance(u, LLMModelUsage):
                    out["llm_input_tokens"] += u.input_tokens
                    out["llm_output_tokens"] += u.output_tokens
                elif isinstance(u, STTModelUsage):
                    out["stt_audio_seconds"] += u.audio_duration
                elif isinstance(u, TTSModelUsage):
                    out["tts_characters"] += u.characters_count
        except Exception as exc:
            log.warning("usage unavailable: %s", type(exc).__name__)
        return out

    async def finish(self, reason: str) -> dict | None:
        """Idempotent; every caller (close event, shutdown callback, hang-up) awaits the same shielded task,
        so a cancelled caller can never leave a call without its record."""
        if self._finish_task is None:
            self.ended = True
            self._finish_task = asyncio.ensure_future(self._finish(reason))
        try:
            return await asyncio.shield(self._finish_task)
        except asyncio.CancelledError:
            return self.end_record

    async def _finish(self, reason: str) -> dict | None:
        current = asyncio.current_task()
        pending = [t for t in self.tasks if t is not current]  # finish() may run inside a tracked task
        if pending:
            await asyncio.wait(pending, timeout=5)
        try:
            await self.backend.usage(self.usage())
            res = await self.backend.end(reason, {"llm_model": ""})
            self.end_record = res.get("record")
            await self.publish({"type": "call_ended", "record": self.end_record, "trace_id": res.get("trace_id")})
        except Exception as exc:
            log.warning("could not store call record: %s", type(exc).__name__)
        return self.end_record
