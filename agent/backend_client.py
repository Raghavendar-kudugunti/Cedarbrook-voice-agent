"""How the agent talks to the scheduling service.

HttpBackend  - the LiveKit worker (separate container) calls the FastAPI service.
LocalBackend - the text harness runs the same agent in-process against a scratch database.
Both go through backend.api.dispatch, so tool behaviour is identical.
"""
from __future__ import annotations

import asyncio
import os
from typing import Protocol

import httpx


class Backend(Protocol):
    async def start(self, transport: str) -> str: ...
    async def tool(self, name: str, args: dict) -> dict: ...
    async def utterance(self, role: str, text: str, language: str | None = None) -> dict: ...
    async def turn(self, data: dict) -> None: ...
    async def usage(self, usage: dict | None = None, barge_in: bool = False) -> None: ...
    async def end(self, reason: str, trace: dict | None = None) -> dict: ...


class HttpBackend:
    def __init__(self, call_id: str, base_url: str | None = None, token: str | None = None):
        self.call_id = call_id
        self.client = httpx.AsyncClient(
            base_url=base_url or os.getenv("API_URL", "http://localhost:8000"), timeout=10,
            headers={"Authorization": "Bearer " + (token or os.getenv("AGENT_INTERNAL_TOKEN", ""))})

    async def _post(self, path: str, body: dict) -> dict:
        r = await self.client.post(f"/api/internal/calls/{self.call_id}{path}", json=body)
        if r.status_code == 401:
            raise RuntimeError("AGENT_INTERNAL_TOKEN mismatch between worker and API")
        return r.json()

    async def start(self, transport: str = "voice") -> str:
        r = await self.client.post("/api/internal/calls", json={"call_id": self.call_id, "transport": transport})
        r.raise_for_status()
        return self.call_id

    async def tool(self, name: str, args: dict) -> dict:
        return await self._post("/tool", {"name": name, "args": args})

    async def utterance(self, role: str, text: str, language: str | None = None) -> dict:
        return await self._post("/utterance", {"role": role, "text": text, "language": language})

    async def turn(self, data: dict) -> None:
        await self._post("/turn", {"data": data})

    async def usage(self, usage: dict | None = None, barge_in: bool = False) -> None:
        await self._post("/usage", {"usage": usage or {}, "barge_in": barge_in})

    async def end(self, reason: str, trace: dict | None = None) -> dict:
        res = await self._post("/end", {"end_reason": reason, "trace": trace or {}})
        await self.client.aclose()
        return res


class LocalBackend:
    def __init__(self, scheduler, call_id: str | None = None):
        from backend.api import dispatch
        self.s = scheduler
        self._dispatch = dispatch
        self.call_id = call_id or ""

    async def start(self, transport: str = "text") -> str:
        self.call_id = await asyncio.to_thread(self.s.start_call, transport, self.call_id or None)
        return self.call_id

    async def tool(self, name: str, args: dict) -> dict:
        return await asyncio.to_thread(self._dispatch, self.s, self.call_id, name, args)

    async def utterance(self, role: str, text: str, language: str | None = None) -> dict:
        masked = await asyncio.to_thread(self.s.add_transcript, self.call_id, role, text)
        verdict = None
        if role == "caller":
            verdict = await asyncio.to_thread(self.s.record_caller_reply, self.call_id, text)
            if language:
                await asyncio.to_thread(self.s.add_language, self.call_id, language)
        return {"masked": masked, "confirmation": verdict}

    async def turn(self, data: dict) -> None:
        await asyncio.to_thread(self.s.add_turn, self.call_id, data)

    async def usage(self, usage: dict | None = None, barge_in: bool = False) -> None:
        if usage:
            await asyncio.to_thread(self.s.add_usage, self.call_id, usage)
        if barge_in:
            await asyncio.to_thread(self.s.add_barge_in, self.call_id)

    async def end(self, reason: str, trace: dict | None = None) -> dict:
        record = await asyncio.to_thread(self.s.end_call, self.call_id, reason)
        return {"record": record}
