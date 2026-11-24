from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from backend import telemetry


def test_trace_shape_and_masking(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(telemetry, "_provider", provider)
    monkeypatch.setattr(telemetry, "_tracer", lambda: provider.get_tracer("t"))
    detail = {"transcript": [{"ts": 1000.0, "role": "caller", "text": "my number is ******3210"}],
              "turns": [{"started_at": 1001.0, "stt_ms": 120, "eou_ms": 300, "llm_ttft_ms": 350, "tts_ttfb_ms": 180,
                         "total_ms": 900, "tools": [{"name": "get_slots", "args": {"patient_phone": "******3210"},
                                                     "ok": True, "ms": 40, "ts": 1001.4}]}]}
    record = {"call_id": "c_x", "outcome": "booked", "duration_s": 30, "cost_inr": 4.2}
    assert telemetry.export_call(detail, record, {})
    names = [s.name for s in exporter.get_finished_spans()]
    assert {"call", "turn-1", "stt", "llm", "tool.get_slots", "tts"} <= set(names)
    blob = str([dict(s.attributes) for s in exporter.get_finished_spans()])
    assert "9876543210" not in blob
