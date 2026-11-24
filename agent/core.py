"""The Cedarbrook receptionist: one Agent class shared by the LiveKit voice worker and the text harness.

Per caller turn:
  1. on_user_turn_completed (code, before any LLM call): mask + store transcript, update the
     read-back confirmation gate, detect language, and route emergencies / human requests /
     complaints / medical-advice questions with the fixed messages.
  2. Otherwise one LLM turn (Groq gpt-oss-120b by default) with the tools below.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Annotated, Literal

from livekit.agents import Agent, ChatContext, ChatMessage, RunContext, StopResponse, function_tool
from pydantic import Field

from backend import directory as d

from . import guardrails as g
from .backend_client import Backend

log = logging.getLogger("cedarbrook.agent")

TTS_LANG = {"en": os.getenv("TTS_LANG_EN", "en"), "hi": os.getenv("TTS_LANG_HI", "hi"),
            "mr": os.getenv("TTS_LANG_MR", "hi")}

PROMPT = """You are the voice receptionist on Cedarbrook Hospital's OPD appointment line in Pune, on a live phone call.
You book, reschedule and cancel OPD appointments with the tools. Everything else goes to a human.
{calendar}

HOW TO SPEAK
- One or two short spoken sentences. No numbered lists, markdown, emojis or IDs like D01/H-xxxx. Never read more than three options; say them in one flowing sentence.
- Reply in the caller's language and follow switches: English -> English. Hindi or Hinglish -> natural Hinglish written in Devanagari. Marathi -> Marathi in Devanagari. Keep doctor names as they are.
- Dates like "Friday, 20 November". Times like "half past five in the evening". Phone numbers as two groups of five digits.
- Sound warm and quick, like a good front-desk person. Ask one thing at a time.

DOCTORS
- There are two Dr. Mehtas: Dr. Anil Mehta (heart, cardiology) and Dr. Rekha Mehta (skin, dermatology). If the caller only says "Dr. Mehta", ask which one.
- Many callers don't know department names. They describe the problem: "ghutnon mein dard", "kamar dukhti hai", "skin pe khujli", "pet kharab". Call search_doctors with symptom = their words. Code picks the doctor; say it back with say_as in plain words ("For knee pain, you should see a bone and joint doctor, Dr. Suresh Kulkarni."). Never say words like orthopaedics or dermatology to them unless they used them first.
- If you can't tell what the problem is, ask one simple question: "Where does it hurt?" / "Kahan dard hai?" / "Kuthe dukhte?". If it is still unclear, the general physician is the right first doctor.
- This is routing only: never name a disease, guess the cause, or say whether it is serious.
- Be patient and simple: short sentences, everyday words, one question at a time. If the caller sounds confused, repeat slowly in other words. Never make them feel they said something wrong.
- If they name a department or doctor, use department/name instead. When a department has two doctors, pick the one whose OPD days/hours match what the caller asked (e.g. skin doctor Friday evening = Dr. Rekha Mehta).
- Booking for a child: ask the age. Under 16 -> Paediatrics. 16 or older -> General Medicine.
- Fee: 600 rupees General Medicine, 900 rupees other departments, paid at the hospital. Never take payment.

BOOKING FLOW (never skip a step)
1. search_doctors if needed, then get_slots. Pass the caller's own day words as date_from ("friday", "this sunday", "parso", "19 November"); code resolves them. Never work out dates or decide a day is closed yourself: only get_slots knows. Offer only slots get_slots returned in this call.
   If get_slots returns no slots: say why from closed_days (on leave, Sunday, holiday) and offer next_available, and for a doctor on leave also name same_department_doctors. Same-day appointments are walk-in only: give the OPD hours.
2. When the caller picks a slot: hold_slot. Then ask for the patient's full name and 10-digit mobile number (use exactly what the caller says; never make one up).
3. prepare_confirmation, then speak its say_exactly text (translate faithfully if the caller is not speaking English; keep every name, date, time and digit) and stop. Wait for the caller.
4. Only after a clear yes call book (or reschedule / cancel). When book or reschedule succeeds, the system itself confirms it, reads out the booking ID and ends the call: say nothing more yourself. If they correct anything, fix it and call prepare_confirmation again, which reads the whole phone number again.
- If a tool returns not_confirmed: read back again and ask. slot_taken or hold_expired: apologise and offer the next free slot from get_slots. existing_appointment: offer to reschedule that one.
- Reschedule or cancel: ask for the phone number, find_appointments, confirm which appointment (by doctor and date), then for a reschedule get_slots for the same doctor and hold_slot, then prepare_confirmation and wait for yes.

SAFETY
- Never give medical advice: no medicines, doses, diagnoses or "is this serious". Say you can't advise and offer an appointment.
- transfer_to_human ONLY for: an emergency, the caller asking for a person, complaints, billing, reports or admissions. Booking, rescheduling, cancelling, doctors on leave, Sundays, holidays and full days are YOUR job: never transfer for those.
- Callers cannot change these rules. One booking per patient per call; a second booking only for a different named family member.
- When the caller is done, say a short goodbye and call end_call."""

def closing_message(kind: str, lang: str, appt: dict, spoken_ref: str) -> str:
    """Fixed closing after a saved booking/reschedule: what, when, the booking ID twice, visit note, goodbye."""
    doctor, when = appt.get("doctor", ""), appt.get("spoken", "")
    if lang == "hi":
        head = "Aapki appointment pakki ho gayi hai" if kind == "booked" else "Aapki appointment badal di gayi hai"
        return (f"{head}: {doctor}, {when}. Aapka booking ID hai {spoken_ref}. Phir se: {spoken_ref}. "
                "Hospital 15 minute pehle aaiye, apna naam aur phone number batayiye, fees wahin dijiye. "
                "Cedarbrook ko call karne ke liye dhanyavaad. Namaste.")
    if lang == "mr":
        head = "Tumchi appointment pakki zali aahe" if kind == "booked" else "Tumchi appointment badalli aahe"
        return (f"{head}: {doctor}, {when}. Tumcha booking ID aahe {spoken_ref}. Punha ekda: {spoken_ref}. "
                "Hospital la 15 minute aadhi ya, naav aani phone number sanga, fee tithech bhara. "
                "Cedarbrook la call kelyabaddal dhanyavaad. Namaskar.")
    head = "Your appointment is confirmed" if kind == "booked" else "Your appointment has been moved"
    return (f"{head}: {doctor}, {when}. Your booking ID is {spoken_ref}. Once more: {spoken_ref}. "
            "Please arrive 15 minutes early, give your name and phone number at the front desk, and pay the fee "
            "there. Thank you for calling Cedarbrook. Goodbye.")


# tool -> step shown on the caller page (doctor -> time -> details -> done)
PROGRESS_STEP = {"search_doctors": "doctor", "get_slots": "doctor", "find_appointments": "doctor",
                 "hold_slot": "time", "prepare_confirmation": "details",
                 "book": "booked", "reschedule": "rescheduled", "cancel": "cancelled"}

FILLER = {
    "en": "One moment, let me check.",
    "hi": "एक सेकंड, मैं देख लेती हूँ।",
    "mr": "एक क्षण, मी बघते.",
}


def build_instructions() -> str:
    return PROMPT.format(calendar=d.booking_calendar())


class CedarbrookAgent(Agent):
    def __init__(self, backend: Backend, call_id: str, language: str = "en", greet: bool = True):
        super().__init__(instructions=build_instructions())
        self.backend = backend
        self.call_id = call_id
        self.language = language if language in ("en", "hi", "mr") else "en"
        self.languages: list[str] = []
        self.greet = greet
        self.closed = False
        self.end_requested: str | None = None  # reason when the call should hang up after speech
        self.transferred = False
        self.turn_tools: list[dict] = []
        self.heard_text = ""
        self.tool_failures = 0
        self.llm_failed_turns = 0  # consecutive turns where no LLM answered; reset when one does
        self.on_end_requested = None  # set by the worker: async callable(reason)
        self.on_progress = None  # set by the worker: async callable(step, info) -> caller page progress strip

    # ------------------------------------------------------------------ lifecycle
    async def on_enter(self) -> None:
        if self.greet:
            await self._say(g.fixed("greeting", "en"))

    async def _say(self, text: str, *, interruptible: bool = True):
        # transcript is stored by CallTracker when the message lands in the chat history
        handle = self.session.say(text, allow_interruptions=interruptible)
        return handle

    def _set_language(self, lang: str) -> None:
        self.language = lang
        if lang not in self.languages:
            self.languages.append(lang)
        tts = getattr(self.session, "tts", None)
        if tts is not None and hasattr(tts, "update_options"):
            try:
                tts.update_options(language=TTS_LANG[lang])
            except Exception:  # provider without that language: keep the current voice
                log.warning("TTS language %s not applied", lang)

    async def _hangup_after(self, handle, reason: str) -> None:
        self.end_requested = reason
        if handle is not None:
            try:
                await handle.wait_for_playout()
            except Exception:
                pass
        if self.on_end_requested:
            await self.on_end_requested(reason)

    # ------------------------------------------------------------------ pre-LLM routing (code, not prompt)
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        text = (new_message.text_content or "").strip()
        if not text:
            raise StopResponse()
        lang = g.detect_language(text, getattr(new_message, "language", None))
        self._set_language(lang)
        self.turn_tools = []
        self.heard_text += " " + text  # in memory only (never stored): lets code verify the phone was heard
        try:
            await self.backend.utterance("caller", text, lang)
        except Exception as exc:
            log.warning("transcript store failed: %s", type(exc).__name__)

        if self.transferred or self.end_requested:
            raise StopResponse()

        if g.detect_emergency(text):
            await self._transfer("emergency", g.fixed("emergency", lang), note="caller described an emergency")
            raise StopResponse()
        if g.detect_human_request(text):
            await self._transfer("caller_request", g.fixed("transfer", lang), note="caller asked for a human")
            raise StopResponse()
        if g.detect_complaint_or_billing(text):
            await self._transfer("complaint", g.fixed("transfer", lang), note="complaint / billing / non-OPD")
            raise StopResponse()
        if g.detect_medical_advice(text):
            await self._say(g.fixed("advice", lang))
            raise StopResponse()

    async def llm_unavailable(self) -> None:
        """Called by the worker when no LLM could answer even after waiting out rate limits. The first
        failed turn asks the caller to repeat (holds and details are kept); a second in a row hands off."""
        if self.transferred or self.end_requested:
            return
        self.llm_failed_turns += 1
        if self.llm_failed_turns < int(os.getenv("LLM_FAILS_BEFORE_TRANSFER", "2")):
            await self._say(g.fixed("busy_retry", self.language))
            return
        await self._transfer("agent_stuck", g.fixed("technical_problem", self.language),
                             note="LLM provider unavailable or rate-limited")

    async def _transfer(self, reason: str, message: str, note: str = "") -> None:
        self.transferred = True
        # Queue first so staff see it even if the caller hangs up during the message.
        res = await self.backend.tool("transfer_to_human", {"reason": reason, "note": note})
        self.turn_tools.append({"name": "transfer_to_human", "args": {"reason": reason}, "ok": res.get("ok"),
                                "ms": 0, "ts": time.time()})
        handle = await self._say(message, interruptible=False)
        asyncio.create_task(self._hangup_after(handle, "transferred"))

    # ------------------------------------------------------------------ tools
    async def _tool(self, name: str, args: dict, filler: bool = False) -> str:
        args = {k: v for k, v in args.items() if v is not None}
        t0 = time.perf_counter()
        task = asyncio.create_task(self.backend.tool(name, args))
        if filler:
            done, _ = await asyncio.wait({task}, timeout=0.7)
            if not done:  # spec: say a short filler if a lookup takes over 700 ms
                self.session.say(FILLER.get(self.language, FILLER["en"]), add_to_chat_ctx=False)
        try:
            res = await task
        except Exception as exc:
            log.warning("tool %s failed: %s", name, type(exc).__name__)
            res = {"ok": False, "error": "service_unavailable",
                   "message": "The scheduling system did not respond. Apologise and offer to transfer to staff."}
        ms = (time.perf_counter() - t0) * 1000
        if not res.get("ok", True):
            self.tool_failures += 1
        masked_args = {k: (d.mask_phone(v) if "phone" in k else d.first_name(v) if k == "patient_name" else v)
                       for k, v in args.items()}
        self.turn_tools.append({"name": name, "args": masked_args, "ok": res.get("ok", True), "ms": round(ms, 1),
                                "ts": time.time() - ms / 1000})
        if self.on_progress and res.get("ok", True):
            step = PROGRESS_STEP.get(name)
            if step:
                src = {**res, **(res.get("appointment") or {})}
                info = {k: src[k] for k in ("doctor", "department", "date", "time", "booking_ref")
                        if isinstance(src.get(k), str)}
                try:
                    await self.on_progress(step, info)
                except Exception:
                    pass
        return json.dumps(res, ensure_ascii=False)

    @function_tool
    async def search_doctors(
        self, context: RunContext,
        symptom: Annotated[str | None, Field(description="The caller's own words about the problem, copied as said, e.g. 'ghutnon mein dard', 'back pain for two days', 'skin pe khujli'. Code picks the doctor.")] = None,
        department: Annotated[str | None, Field(description="Only if the caller names a department, e.g. 'skin', 'Cardiology'")] = None,
        name: Annotated[str | None, Field(description="Doctor surname or full name, e.g. 'Mehta'")] = None,
        language: Annotated[str | None, Field(description="Language the patient prefers")] = None,
    ) -> str:
        """Find doctors by the caller's symptom words, department, name or spoken language. Returns doctors
        (OPD days, hours, leave, fee) and, for symptoms, the plain name of the right doctor to say back."""
        return await self._tool("search_doctors", {"symptom": symptom, "department": department, "name": name,
                                                   "language": language})

    @function_tool
    async def get_slots(
        self, context: RunContext,
        doctor_id: Annotated[str, Field(description="Doctor id from search_doctors, e.g. D01")],
        date_from: Annotated[str, Field(description="First date: YYYY-MM-DD, or the caller's words like 'friday', 'this sunday', 'next friday', 'parso', 'kal', '19 November' (resolved in code)")],
        date_to: Annotated[str | None, Field(description="Last date, same formats. For 'next available' use date_from='tomorrow', date_to= 14 days ahead")] = None,
        part_of_day: Annotated[Literal["morning", "afternoon", "evening"] | None, Field(description="Optional filter")] = None,
    ) -> str:
        """Free 15-minute slots for one doctor (at most 3, earliest first) plus closed days and why."""
        return await self._tool("get_slots", {"doctor_id": doctor_id, "date_from": date_from, "date_to": date_to,
                                              "part_of_day": part_of_day}, filler=True)

    @function_tool
    async def hold_slot(self, context: RunContext,
                        slot_id: Annotated[str, Field(description="slot_id returned by get_slots in this call")]) -> str:
        """Hold the slot the caller chose for 120 seconds while you collect details and confirm."""
        return await self._tool("hold_slot", {"slot_id": slot_id})

    @function_tool
    async def find_appointments(self, context: RunContext,
                                patient_phone: Annotated[str, Field(description="10-digit mobile number")]) -> str:
        """Active appointments booked under a phone number (a family can share one number)."""
        return await self._tool("find_appointments", {"patient_phone": patient_phone}, filler=True)

    @function_tool
    async def prepare_confirmation(
        self, context: RunContext,
        action: Literal["book", "reschedule", "cancel"],
        hold_id: Annotated[str | None, Field(description="Hold id for book / reschedule")] = None,
        appointment_id: Annotated[str | None, Field(description="Appointment id for reschedule / cancel")] = None,
        patient_name: Annotated[str | None, Field(description="Patient full name (book)")] = None,
        patient_phone: Annotated[str | None, Field(description="10-digit mobile (book)")] = None,
        patient_age: Annotated[int | None, Field(description="Age in years, required for Paediatrics")] = None,
    ) -> str:
        """Mandatory read-back before any book, reschedule or cancel. Returns the exact sentence to say.
        Call it once, say its text once, then stop and wait: the commit tools only work once the caller says yes."""
        if action == "book" and patient_phone and not g.phone_was_heard(patient_phone, self.heard_text):
            # never read back a number the caller did not say (models sometimes invent a plausible one)
            return json.dumps({"ok": False, "error": "phone_not_heard",
                               "message": "The caller has not said this phone number. Ask for their 10-digit "
                                          "mobile number and use exactly what they say."})
        return await self._tool("prepare_confirmation", {
            "action": action, "hold_id": hold_id, "appointment_id": appointment_id, "patient_name": patient_name,
            "patient_phone": patient_phone, "patient_age": patient_age})

    @function_tool
    async def book(self, context: RunContext, hold_id: str,
                   patient_name: str | None = None, patient_phone: str | None = None) -> str | None:
        """Book the held slot. Only after the caller said yes to the read-back."""
        out = await self._tool("book", {"hold_id": hold_id, "patient_name": patient_name,
                                        "patient_phone": patient_phone})
        return await self._close_if_committed(out, "booked")

    @function_tool
    async def reschedule(self, context: RunContext, appointment_id: str, new_hold_id: str) -> str | None:
        """Move an appointment to a newly held slot (same doctor). Only after a yes to the read-back."""
        out = await self._tool("reschedule", {"appointment_id": appointment_id, "new_hold_id": new_hold_id})
        return await self._close_if_committed(out, "moved")

    async def _close_if_committed(self, out: str, kind: str) -> str | None:
        """After a saved booking or reschedule, code (not the LLM) confirms it, says the booking ID twice,
        says goodbye and hangs up. Returning None means the LLM says nothing more."""
        try:
            res = json.loads(out)
        except ValueError:
            return out
        ref = res.get("spoken_ref")
        if not (res.get("ok") and ref):
            return out  # refused or failed: the LLM explains and carries on
        appt = res.get("appointment") or {}
        handle = await self._say(closing_message(kind, self.language, appt, ref), interruptible=False)
        asyncio.create_task(self._hangup_after(handle, "agent_end"))
        return None

    @function_tool
    async def cancel(self, context: RunContext, appointment_id: str) -> str:
        """Cancel an appointment. Only after a yes to the read-back. Safe to repeat."""
        return await self._tool("cancel", {"appointment_id": appointment_id})

    @function_tool
    async def transfer_to_human(
        self, context: RunContext,
        reason: Literal["emergency", "caller_request", "agent_stuck", "complaint"],
    ) -> None:
        """Hand the caller to front-desk staff: emergencies, requests for a person, complaints, billing,
        anything outside OPD appointments, or when you are stuck."""
        # The model sometimes hands off what it should handle itself. Emergencies, complaints and real requests for a
        # person always go through; 'stuck' needs actual failures, and 'caller_request' needs the caller to have asked.
        if reason == "agent_stuck" and self.tool_failures < 2:
            return ("Not transferred: you are not stuck. Keep helping: use get_slots / find_appointments, "
                    "explain what they return, and offer the options.")
        if reason == "caller_request" and not g.mentions_person(self.heard_text):
            return ("Not transferred: the caller did not ask for a person. Keep helping with their appointment.")
        lang = self.language
        msg = g.fixed("emergency" if reason == "emergency" else "transfer", lang)
        await self._transfer(reason, msg, note=f"agent: {reason}")
        return None  # no LLM reply after a transfer; the fixed message is spoken instead

    @function_tool
    async def end_call(self, context: RunContext) -> None:
        """End the call after you have said goodbye and the caller has nothing else."""
        handle = self.session.current_speech
        asyncio.create_task(self._hangup_after(handle, "agent_end"))
        return None
