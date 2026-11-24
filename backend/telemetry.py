"""One Langfuse trace per call, sent over OTLP at call end.

Shape:  call  ->  turn-N  ->  stt | llm | tool.<name> | tts
Only masked data is exported: transcripts are already masked by the scheduler (last 4 phone digits,
caller first name only), tool arguments are masked in the tool log. No audio is ever stored.
No-op unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
"""
from __future__ import annotations

import base64
import json
import logging
import os

log = logging.getLogger("cedarbrook.telemetry")
_provider = None


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _tracer():
    global _provider
    if _provider is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        base = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
        auth = base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
        exporter = OTLPSpanExporter(endpoint=f"{base}/api/public/otel/v1/traces",
                                    headers={"Authorization": f"Basic {auth}"})
        _provider = TracerProvider(resource=Resource.create({"service.name": "cedarbrook-voice-agent"}))
        _provider.add_span_processor(BatchSpanProcessor(exporter))
    return _provider.get_tracer("cedarbrook")


def _ns(seconds: float) -> int:
    return int(seconds * 1e9)


def export_call(detail: dict, record: dict, extra: dict) -> str | None:
    if not enabled():
        return None
    try:
        from opentelemetry import trace

        tracer = _tracer()
        turns = detail.get("turns") or []
        transcript = detail.get("transcript") or []
        t_start = transcript[0]["ts"] if transcript else (turns[0].get("started_at") if turns else None)
        import time as _t
        t_start = t_start or _t.time() - (record.get("duration_s") or 1)
        t_end = t_start + float(record.get("duration_s") or 1)

        root = tracer.start_span("call", start_time=_ns(t_start), attributes={
            "langfuse.trace.name": f"call {record['call_id']}",
            "langfuse.session.id": record["call_id"],
            "langfuse.trace.tags": json.dumps(["cedarbrook", record["outcome"]]),
            "langfuse.trace.input": json.dumps([t for t in transcript if t["role"] == "caller"][:1], ensure_ascii=False),
            "langfuse.trace.output": json.dumps(record, ensure_ascii=False),
            "langfuse.trace.metadata.outcome": record["outcome"],
            "langfuse.trace.metadata.cost_inr": str(record.get("cost_inr")),
            "langfuse.trace.metadata.model": extra.get("llm_model", os.getenv("LLM_MODEL", "")),
        })
        ctx = trace.set_span_in_context(root)
        for i, t in enumerate(turns):
            ts = float(t.get("started_at") or t_start)
            total = (t.get("total_ms") or 0) / 1000
            turn = tracer.start_span(f"turn-{i + 1}", context=ctx, start_time=_ns(ts), attributes={
                "langfuse.observation.input": t.get("caller_text", ""),
                "langfuse.observation.output": t.get("agent_text", ""),
                "langfuse.observation.metadata.total_ms": str(t.get("total_ms")),
                "langfuse.observation.metadata.interrupted": str(t.get("interrupted", False)),
            })
            tctx = trace.set_span_in_context(turn)
            cursor = ts - (t.get("stt_ms") or 0) / 1000  # transcription finished shortly after speech end
            stt = tracer.start_span("stt", context=tctx, start_time=_ns(cursor), attributes={
                "langfuse.observation.metadata.stt_ms": str(t.get("stt_ms")),
                "langfuse.observation.metadata.eou_ms": str(t.get("eou_ms")),
                "langfuse.observation.metadata.audio_seconds": str(t.get("audio_s")),
                "langfuse.observation.metadata.provider": "deepgram nova-3 multi",
            })
            stt.end(end_time=_ns(ts + (t.get("eou_ms") or 0) / 1000))
            llm_start = ts + (t.get("eou_ms") or 0) / 1000
            llm = tracer.start_span("llm", context=tctx, start_time=_ns(llm_start), attributes={
                "langfuse.observation.type": "generation",
                "gen_ai.request.model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
                "gen_ai.usage.input_tokens": int(t.get("llm_input_tokens") or 0),
                "gen_ai.usage.output_tokens": int(t.get("llm_output_tokens") or 0),
                "langfuse.observation.metadata.ttft_ms": str(t.get("llm_ttft_ms")),
            })
            tool_cursor = llm_start + (t.get("llm_ttft_ms") or 0) / 1000
            for tool in t.get("tools") or []:
                sp = tracer.start_span(f"tool.{tool['name']}", context=tctx,
                                       start_time=_ns(tool.get("ts") or tool_cursor), attributes={
                    "langfuse.observation.input": json.dumps(tool.get("args", {}), ensure_ascii=False),
                    "langfuse.observation.metadata.ok": str(tool.get("ok")),
                })
                sp.end(end_time=_ns((tool.get("ts") or tool_cursor) + (tool.get("ms") or 0) / 1000))
                tool_cursor = (tool.get("ts") or tool_cursor) + (tool.get("ms") or 0) / 1000
            llm.end(end_time=_ns(llm_start + (t.get("llm_ms") or t.get("llm_ttft_ms") or 0) / 1000))
            tts_start = ts + max(0.0, total - (t.get("tts_ttfb_ms") or 0) / 1000)
            tts = tracer.start_span("tts", context=tctx, start_time=_ns(tts_start), attributes={
                "langfuse.observation.metadata.ttfb_ms": str(t.get("tts_ttfb_ms")),
                "langfuse.observation.metadata.characters": str(t.get("tts_chars")),
                "langfuse.observation.metadata.provider": "cartesia sonic",
            })
            tts.end(end_time=_ns(ts + total))
            turn.end(end_time=_ns(ts + max(total, 0.001)))
        root.end(end_time=_ns(t_end))
        _provider.force_flush(5000)
        return format(root.get_span_context().trace_id, "032x")
    except Exception as exc:  # telemetry must never break a call
        log.warning("Langfuse export failed: %s", type(exc).__name__)
        return None
