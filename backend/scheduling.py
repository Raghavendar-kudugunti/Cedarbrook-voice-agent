"""Scheduling service: SQLite store, holds, confirmation gate, idempotent writes, call records.

Safety-critical rules are enforced here, not in the prompt:
  * a slot can only be held if get_slots offered it in this call (never invent a slot)
  * the database primary key on slot_locks makes double booking impossible
  * book / reschedule / cancel refuse to run without a confirmed read-back in this call
  * every write has an idempotency key derived from call id + slot/appointment
  * no bookings after an emergency in the same call
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import statistics
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path

from . import directory as d

HOLD_SECONDS = 120
MAX_BOOKINGS_PER_CALL = 3

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS calls(
  id TEXT PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, transport TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'live', end_reason TEXT, emergency INTEGER NOT NULL DEFAULT 0,
  languages TEXT NOT NULL DEFAULT '[]', usage TEXT NOT NULL DEFAULT '{}', barge_ins INTEGER NOT NULL DEFAULT 0,
  record TEXT);
CREATE TABLE IF NOT EXISTS turns(
  call_id TEXT NOT NULL, idx INTEGER NOT NULL, data TEXT NOT NULL, created_at REAL NOT NULL,
  PRIMARY KEY(call_id, idx));
CREATE TABLE IF NOT EXISTS transcript(
  call_id TEXT NOT NULL, ts REAL NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tool_log(
  call_id TEXT NOT NULL, ts REAL NOT NULL, name TEXT NOT NULL, args TEXT NOT NULL, ok INTEGER NOT NULL,
  duration_ms REAL NOT NULL, result TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS offered_slots(call_id TEXT NOT NULL, slot_id TEXT NOT NULL, PRIMARY KEY(call_id, slot_id));
-- One row per occupied slot. PRIMARY KEY = the database-level guarantee against double booking.
CREATE TABLE IF NOT EXISTS slot_locks(
  slot_id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('hold','appointment')),
  ref TEXT NOT NULL, call_id TEXT, expires_at REAL);
CREATE TABLE IF NOT EXISTS holds(
  id TEXT PRIMARY KEY, slot_id TEXT NOT NULL, call_id TEXT NOT NULL, expires_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS appointments(
  id TEXT PRIMARY KEY, slot_id TEXT NOT NULL, doctor_id TEXT NOT NULL, appt_date TEXT NOT NULL, appt_time TEXT NOT NULL,
  patient_name TEXT NOT NULL, patient_key TEXT NOT NULL, phone TEXT NOT NULL, age INTEGER,
  status TEXT NOT NULL, call_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_patient_doctor ON appointments(patient_key, doctor_id) WHERE status='active';
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_slot ON appointments(slot_id) WHERE status='active';
CREATE TABLE IF NOT EXISTS readbacks(
  id TEXT PRIMARY KEY, call_id TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL, text TEXT NOT NULL,
  status TEXT NOT NULL, issued_at REAL NOT NULL, confirmed_at REAL, used_at REAL);
CREATE TABLE IF NOT EXISTS idempotency(key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS call_actions(
  call_id TEXT NOT NULL, ts REAL NOT NULL, action TEXT NOT NULL, appointment_id TEXT NOT NULL, readback_id TEXT,
  confirmed INTEGER NOT NULL, summary TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transfers(
  id TEXT PRIMARY KEY, call_id TEXT NOT NULL, reason TEXT NOT NULL, note TEXT, created_at REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'waiting', handled_at REAL);
"""

TRANSFER_REASONS = {"emergency", "caller_request", "agent_stuck", "complaint"}

YES_WORDS = {
    "yes", "yeah", "yep", "yup", "ya", "yah", "yes please", "correct", "right", "that's right", "thats right",
    "confirm", "confirmed", "sure", "ok", "okay", "go ahead", "please do", "book it", "do it", "perfect", "fine",
    "absolutely", "of course", "exactly", "that's correct", "thats correct", "all correct", "yes correct",
    "haan", "han", "haa", "ha", "haanji", "haan ji", "ji haan", "ji", "ji ha", "hanji", "theek hai", "thik hai",
    "theek", "sahi hai", "sahi", "bilkul", "kar do", "kardo", "karo", "chalega", "ho", "hoy", "barobar",
    "हाँ", "हां", "हा", "जी", "जी हाँ", "हाँ जी", "ठीक है", "सही है", "बिलकुल", "हो", "होय", "बरोबर", "कर दो",
}
NO_WORDS = {
    "no", "nope", "nah", "not", "wrong", "incorrect", "wait", "change", "instead", "cancel that", "don't", "dont",
    "nahi", "nahin", "nai", "na", "mat", "galat", "ruko", "nako", "naahi", "नहीं", "नही", "ना", "गलत", "रुको", "नको",
}


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return " ".join(re.sub(r"[^\w\s'ऀ-ॿ]", " ", text).split())


def classify_confirmation(text: str) -> str:
    """'yes' | 'no' | 'unclear'. Deterministic so the commit gate never depends on the LLM.

    A digit in the reply ('No, 43120') or any negation means the caller is correcting something.
    """
    t = _norm(text)
    if not t:
        return "unclear"
    words = t.split()
    if any(w in NO_WORDS for w in words) or any(p in t for p in ("not right", "not correct", "cancel that")):
        return "no"
    if re.search(r"\d", t):
        return "no"
    if t in YES_WORDS:
        return "yes"
    # 'Yes, that's correct. Please book it. Thank you!' -> starts with a clear yes, no negation, no digits
    for phrase in sorted(YES_WORDS, key=len, reverse=True):
        if (t == phrase or t.startswith(phrase + " ")) and phrase not in {"ok", "okay", "ji", "ha", "na", "fine",
                                                                         "right", "sure", "ho"}:
            return "yes"
    # 'yes yes', 'haan ji bilkul', 'yes please go ahead book it' -> all tokens/phrases are affirmative
    rest = t
    for phrase in sorted(YES_WORDS, key=len, reverse=True):
        rest = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", rest)
    leftover = [w for w in rest.split() if w not in {"please", "that", "is", "it", "the", "one", "thank", "you",
                                                     "thanks", "do", "go", "book", "na", "toh", "to"}]
    if len(leftover) == 0 and len(words) <= 8:
        return "yes"
    return "unclear"


def normalize_phone(phone: str) -> str:
    dev = str(phone).translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    digits = "".join(c for c in dev if c.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    if len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    return digits


def patient_key(name: str, phone: str) -> str:
    return " ".join(name.lower().split()) + "|" + normalize_phone(phone)


def mask_text(text: str) -> str:
    """Keep only the last 4 digits of any phone-like number in free text."""
    def repl(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "******" + digits[-4:] if len(digits) >= 7 else m.group(0)
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    return re.sub(r"(?<!\d)\d(?:[\s·.\-,]*\d){6,}(?!\d)", repl, text)


class ToolError(Exception):
    """Business-rule refusal. `code` is machine readable; message is for the LLM to rephrase."""

    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code = code
        self.message = message
        self.extra = extra

    def as_dict(self) -> dict:
        return {"ok": False, "error": self.code, "message": self.message, **self.extra}


def slot_view(sid: str) -> dict:
    doc_id, day, hhmm = d.parse_slot_id(sid)
    doc = d.BY_ID[doc_id]
    return {
        "slot_id": sid, "doctor_id": doc_id, "doctor": doc.name, "department": doc.department,
        "date": day.isoformat(), "time": hhmm,
        "spoken": f"{d.spoken_date(day)} at {d.spoken_time(hhmm)}",
    }


class Scheduler:
    def __init__(self, path: str | None = None, seed_demo: bool = True):
        self.path = path or os.getenv("DATABASE_PATH", "data/cedarbrook.db")
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
        if seed_demo:
            self.seed_demo_patients()

    # ---------------- infrastructure ----------------
    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        return c

    @contextmanager
    def _conn(self):
        c = self._connect()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def tx(self):
        """BEGIN IMMEDIATE serialises writers across threads and processes."""
        c = self._connect()
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    @staticmethod
    def _expire_holds(c: sqlite3.Connection) -> None:
        now = time.time()
        c.execute("DELETE FROM slot_locks WHERE kind='hold' AND expires_at <= ?", (now,))

    def _live_hold(self, c: sqlite3.Connection, hold_id: str, call_id: str) -> sqlite3.Row:
        """Return this call's hold, renewing it for another 120 s. A hold that lapsed while the caller was still
        giving details is re-taken if the slot is still free; if someone else took it, the caller gets an apology."""
        hold = c.execute("SELECT * FROM holds WHERE id=? AND call_id=?", (hold_id or "", call_id)).fetchone()
        if not hold:
            raise ToolError("hold_expired", "That hold does not exist. Call get_slots and hold_slot again.")
        self._expire_holds(c)
        lock = c.execute("SELECT * FROM slot_locks WHERE slot_id=?", (hold["slot_id"],)).fetchone()
        exp = time.time() + HOLD_SECONDS
        if lock and lock["ref"] == hold_id:
            c.execute("UPDATE slot_locks SET expires_at=? WHERE slot_id=?", (exp, hold["slot_id"]))
        elif lock or d.is_prebooked(hold["slot_id"]):
            raise ToolError("slot_taken", "Sorry, another caller took that slot while we were talking. Apologise, "
                                          "call get_slots and offer the next free slot.")
        else:
            c.execute("INSERT INTO slot_locks VALUES(?,?,?,?,?)", (hold["slot_id"], "hold", hold_id, call_id, exp))
        c.execute("UPDATE holds SET expires_at=? WHERE id=?", (exp, hold_id))
        return hold

    def _call(self, c: sqlite3.Connection, call_id: str) -> sqlite3.Row:
        row = c.execute("SELECT * FROM calls WHERE id=?", (call_id,)).fetchone()
        if not row:
            raise ToolError("unknown_call", "This call is not registered.")
        return row

    def _idem_get(self, c, key: str) -> dict | None:
        row = c.execute("SELECT response FROM idempotency WHERE key=?", (key,)).fetchone()
        return json.loads(row["response"]) if row else None

    def _idem_put(self, c, key: str, response: dict) -> None:
        c.execute("INSERT OR IGNORE INTO idempotency VALUES(?,?,?)", (key, json.dumps(response), time.time()))

    def _next_appt_id(self, c, day: date) -> str:
        prefix = f"AP-{day.strftime('%Y%m%d')}-"
        n = c.execute("SELECT COUNT(*) FROM appointments WHERE id LIKE ?", (prefix + "%",)).fetchone()[0]
        return f"{prefix}{n + 1:04d}"

    # ---------------- calls ----------------
    def start_call(self, transport: str = "voice", call_id: str | None = None) -> str:
        call_id = call_id or "c_" + uuid.uuid4().hex[:6]
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO calls(id, started_at, transport) VALUES(?,?,?)",
                      (call_id, time.time(), transport))
        return call_id

    def call_is_emergency(self, call_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT emergency FROM calls WHERE id=?", (call_id,)).fetchone()
            return bool(row and row["emergency"])

    def add_transcript(self, call_id: str, role: str, text: str) -> str:
        masked = mask_text(text)
        with self.tx() as c:
            c.execute("INSERT INTO transcript VALUES(?,?,?,?)", (call_id, time.time(), role, masked))
        return masked

    def add_language(self, call_id: str, lang: str) -> None:
        if not lang:
            return
        with self.tx() as c:
            row = self._call(c, call_id)
            langs = json.loads(row["languages"])
            if lang not in langs:
                langs.append(lang)
                c.execute("UPDATE calls SET languages=? WHERE id=?", (json.dumps(langs), call_id))

    def add_turn(self, call_id: str, data: dict) -> None:
        with self.tx() as c:
            idx = c.execute("SELECT COUNT(*) FROM turns WHERE call_id=?", (call_id,)).fetchone()[0]
            c.execute("INSERT INTO turns VALUES(?,?,?,?)", (call_id, idx, json.dumps(data), time.time()))

    def add_usage(self, call_id: str, usage: dict) -> None:
        with self.tx() as c:
            row = self._call(c, call_id)
            cur = json.loads(row["usage"])
            for k, v in usage.items():
                if isinstance(v, (int, float)):
                    cur[k] = cur.get(k, 0) + v
            c.execute("UPDATE calls SET usage=? WHERE id=?", (json.dumps(cur), call_id))

    def add_barge_in(self, call_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE calls SET barge_ins = barge_ins + 1 WHERE id=?", (call_id,))

    def log_tool(self, call_id: str, name: str, args: dict, ok: bool, duration_ms: float, result: dict) -> None:
        safe_args = {k: (d.mask_phone(v) if "phone" in k else (d.first_name(v) if k == "patient_name" else v))
                     for k, v in args.items()}
        safe_result = json.loads(mask_text(json.dumps(result, ensure_ascii=False)))
        with self.tx() as c:
            c.execute("INSERT INTO tool_log VALUES(?,?,?,?,?,?,?)",
                      (call_id, time.time(), name, json.dumps(safe_args, ensure_ascii=False), int(ok), duration_ms,
                       json.dumps(safe_result, ensure_ascii=False)[:4000]))

    # ---------------- tools ----------------
    def search_doctors(self, department: str | None = None, name: str | None = None,
                       language: str | None = None) -> list[dict]:
        dept = d.normalize_department(department)
        out = []
        for doc in d.DOCTORS:
            if dept and dept.lower() not in doc.department.lower():
                continue
            if name:
                tokens = [t for t in re.split(r"[\s.]+", name.lower()) if t and t not in {"dr", "doctor", "doc"}]
                if not all(t in doc.name.lower() for t in tokens):
                    continue
            if language and not any(language.lower() in l.lower() for l in doc.languages):
                continue
            out.append(doc.public())
        return out

    def get_slots(self, call_id: str, doctor_id: str, date_from: str, date_to: str | None = None,
                  part_of_day: str | None = None, limit: int = 3) -> dict:
        doc = d.BY_ID.get(doctor_id)
        if not doc:
            raise ToolError("unknown_doctor", f"No doctor with id {doctor_id}. Use search_doctors first.")
        today = d.today_ist()
        start = d.resolve_day(date_from, today)
        if start is None:
            raise ToolError("bad_date", f"Could not understand the date '{date_from}'. Use YYYY-MM-DD or a day name.")
        end = d.resolve_day(date_to, today) if date_to else start
        end = end or start
        if end < start:
            start, end = end, start
        notes: list[dict] = []
        if start <= today:
            notes.append({"date": start.isoformat(), "reason": "same_day_walk_in" if start == today else "past_date",
                          "message": f"Same-day appointments are walk-in only. {doc.name}'s OPD hours are "
                                     f"{doc.start} to {doc.end}." if start == today else "That date has passed."})
            start = today + timedelta(days=1)
            end = max(end, start)
        last = today + timedelta(days=d.BOOKING_WINDOW_DAYS)
        if end > last:
            notes.append({"date": end.isoformat(), "reason": "beyond_14_days",
                          "message": f"Bookings open only up to 14 days ahead (until {d.spoken_date(last)})."})
            end = last
        slots: list[dict] = []
        with self.tx() as c:
            self._call(c, call_id)
            self._expire_holds(c)
            taken = {r[0] for r in c.execute("SELECT slot_id FROM slot_locks")}

            def scan(first: date, last_day: date, record_notes: bool) -> list[dict]:
                found, day = [], first
                while day <= last_day:
                    reason = d.closure_reason(doc, day, today)
                    if reason:
                        if record_notes and (reason != "doctor_not_in_opd_this_day" or start == end):
                            notes.append({"date": day.isoformat(), "reason": reason})
                    else:
                        for hhmm in d.slot_times(doc):
                            h = int(hhmm[:2])
                            if part_of_day == "morning" and h >= 12:
                                continue
                            if part_of_day == "afternoon" and not (12 <= h < 16):
                                continue
                            if part_of_day == "evening" and h < 16:
                                continue
                            sid = d.slot_id(doc.id, day, hhmm)
                            if not (d.is_prebooked(sid) or sid in taken):
                                found.append(slot_view(sid))
                    day += timedelta(days=1)
                return found

            slots = scan(start, end, True)
            shown = slots[: max(1, min(limit, 3))]
            next_available: list[dict] = []
            if not slots and end < last:
                # nothing on the requested day(s): hand the agent the next real options instead of a dead end
                next_available = scan(end + timedelta(days=1), last, False)[:3]
            for s in shown + next_available:
                c.execute("INSERT OR IGNORE INTO offered_slots VALUES(?,?)", (call_id, s["slot_id"]))
        result = {"ok": True, "doctor": doc.name, "department": doc.department, "slots": shown,
                  "more_available": max(0, len(slots) - len(shown)), "closed_days": notes[:5]}
        if not slots:
            result["message"] = "No free slots on the requested day(s). Say why (closed_days), then offer next_available."
            result["next_available"] = next_available
            if any(n["reason"].startswith("doctor_on_leave") for n in notes):
                result["same_department_doctors"] = [
                    {"doctor_id": o.id, "doctor": o.name, "days": [d.DAY_NAMES[x] for x in o.days],
                     "opd_hours": f"{o.start}-{o.end}"}
                    for o in d.DOCTORS if o.department == doc.department and o.id != doc.id]
        return result

    def hold_slot(self, call_id: str, slot_id: str) -> dict:
        with self.tx() as c:
            call = self._call(c, call_id)
            if call["emergency"]:
                raise ToolError("emergency_call", "No booking in an emergency call. Transfer to staff.")
            if not c.execute("SELECT 1 FROM offered_slots WHERE call_id=? AND slot_id=?", (call_id, slot_id)).fetchone():
                raise ToolError("slot_not_offered", "Only slots returned by get_slots in this call can be held. "
                                                     "Call get_slots and offer one of those.")
            self._expire_holds(c)
            mine = c.execute("SELECT id FROM holds WHERE slot_id=? AND call_id=? ORDER BY expires_at DESC",
                             (slot_id, call_id)).fetchone()
            if mine:
                self._live_hold(c, mine["id"], call_id)
                return {"ok": True, "hold_id": mine["id"], "expires_in_s": HOLD_SECONDS, **slot_view(slot_id)}
            if d.is_prebooked(slot_id):
                raise ToolError("slot_taken", "Sorry, that slot was just taken. Offer the next free slot.")
            hold_id = "H-" + uuid.uuid4().hex[:8]
            exp = time.time() + HOLD_SECONDS
            try:
                c.execute("INSERT INTO slot_locks VALUES(?,?,?,?,?)", (slot_id, "hold", hold_id, call_id, exp))
            except sqlite3.IntegrityError:
                raise ToolError("slot_taken", "Sorry, another caller just took that slot. Apologise and offer the "
                                              "next free slot from get_slots.") from None
            c.execute("INSERT INTO holds VALUES(?,?,?,?)", (hold_id, slot_id, call_id, exp))
        return {"ok": True, "hold_id": hold_id, "expires_in_s": HOLD_SECONDS, **slot_view(slot_id)}

    def find_appointments(self, patient_phone: str) -> list[dict]:
        phone = normalize_phone(patient_phone)
        with self._conn() as c:
            rows = c.execute("SELECT * FROM appointments WHERE phone=? AND status='active' ORDER BY appt_date, appt_time",
                             (phone,)).fetchall()
        return [self._appt_view(r) for r in rows]

    @staticmethod
    def _appt_view(r: sqlite3.Row) -> dict:
        doc = d.BY_ID[r["doctor_id"]]
        day = date.fromisoformat(r["appt_date"])
        return {
            "appointment_id": r["id"], "patient_first_name": d.first_name(r["patient_name"]),
            "doctor_id": doc.id, "doctor": doc.name, "department": doc.department,
            "date": r["appt_date"], "time": r["appt_time"],
            "spoken": f"{d.spoken_date(day)} at {d.spoken_time(r['appt_time'])}",
            "status": r["status"], "phone_last4": r["phone"][-4:],
        }

    # ---------- read-back / confirmation gate ----------
    def prepare_confirmation(self, call_id: str, action: str, *, hold_id: str | None = None,
                             appointment_id: str | None = None, patient_name: str | None = None,
                             patient_phone: str | None = None, patient_age: int | None = None) -> dict:
        """Builds the exact read-back sentence and records it. The commit tools refuse to run until the
        caller's *next* utterance is a clear yes (checked in code by record_caller_reply)."""
        if action not in {"book", "reschedule", "cancel"}:
            raise ToolError("bad_action", "action must be book, reschedule or cancel")
        with self.tx() as c:
            call = self._call(c, call_id)
            if call["emergency"]:
                raise ToolError("emergency_call", "No booking in an emergency call.")
            target: dict = {"action": action}
            if action in {"book", "reschedule"}:
                hold = self._live_hold(c, hold_id, call_id)
                slot = slot_view(hold["slot_id"])
                target.update(hold_id=hold["id"], slot_id=hold["slot_id"])
            if action in {"reschedule", "cancel"}:
                appt = c.execute("SELECT * FROM appointments WHERE id=?", (appointment_id or "",)).fetchone()
                if not appt:
                    raise ToolError("unknown_appointment", "Appointment not found. Use find_appointments.")
                if appt["status"] != "active":
                    raise ToolError("already_cancelled", "That appointment is already cancelled; nothing to do.")
                target["appointment_id"] = appt["id"]
                patient_name = appt["patient_name"]
                patient_phone = appt["phone"]
                if action == "reschedule" and appt["doctor_id"] != slot["doctor_id"]:
                    raise ToolError("different_doctor", "A reschedule keeps the same doctor. To see a different "
                                                        "doctor, book a new appointment instead.")
            if action == "book":
                if not patient_name or len(patient_name.strip()) < 2:
                    raise ToolError("need_name", "Ask for the patient's name first.")
                phone = normalize_phone(patient_phone or "")
                if len(phone) != 10 or phone[0] not in "6789":
                    raise ToolError("bad_phone", "Need a valid 10-digit Indian mobile number. Ask the caller to say "
                                                 "it again slowly.")
                doc = d.BY_ID[slot["doctor_id"]]
                if doc.max_age is not None:
                    if patient_age is None:
                        raise ToolError("need_age", "Paediatrics is for patients under 16. Ask the patient's age.")
                    if patient_age > doc.max_age:
                        raise ToolError("too_old_for_paediatrics", "Patients 16 and older see General Medicine "
                                                                   "(Dr. Meenal Joshi mornings, Dr. Arjun Nair "
                                                                   "evenings). Release this and offer that.")
                target.update(patient_name=" ".join(patient_name.split()).title(), patient_phone=phone,
                              patient_age=patient_age)
                patient_phone = phone
            phone_spoken = d.spoken_phone(patient_phone)
            if action == "book":
                doc = d.BY_ID[slot["doctor_id"]]
                text = (f"Let me confirm: an appointment for {target['patient_name']} with {slot['doctor']}, "
                        f"{slot['department']}, on {slot['spoken']}. Phone number {phone_spoken}. "
                        f"The consultation fee is {doc.fee_inr} rupees, paid at the hospital. Shall I book it?")
            elif action == "reschedule":
                appt_v = self._appt_view(appt)
                text = (f"Let me confirm: move {d.first_name(patient_name)}'s appointment with {slot['doctor']} from "
                        f"{appt_v['spoken']} to {slot['spoken']}. Phone number {phone_spoken}. Shall I go ahead?")
            else:
                appt_v = self._appt_view(appt)
                text = (f"Let me confirm: cancel {d.first_name(patient_name)}'s appointment with {appt_v['doctor']} on "
                        f"{appt_v['spoken']}. Phone number {phone_spoken}. Shall I cancel it?")
            # A new read-back supersedes any earlier unconfirmed one.
            c.execute("UPDATE readbacks SET status='superseded' WHERE call_id=? AND status IN ('issued','confirmed')",
                      (call_id,))
            rb_id = "R-" + uuid.uuid4().hex[:8]
            c.execute("INSERT INTO readbacks VALUES(?,?,?,?,?,?,?,?,?)",
                      (rb_id, call_id, action, json.dumps(target), text, "issued", time.time(), None, None))
        return {"ok": True, "readback_id": rb_id, "say_exactly": text,
                "next": "Say the read-back exactly, then wait. Do not call the commit tool until the caller says yes."}

    def pending_readback(self, call_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM readbacks WHERE call_id=? AND status IN ('issued','confirmed') "
                            "ORDER BY issued_at DESC LIMIT 1", (call_id,)).fetchone()
            return dict(row) if row else None

    def record_caller_reply(self, call_id: str, text: str) -> str | None:
        """Called by agent code (not the LLM) with every final caller utterance.
        Confirms the latest issued read-back on a clear yes; rejects it on a no/correction."""
        with self.tx() as c:
            row = c.execute("SELECT * FROM readbacks WHERE call_id=? AND status='issued' ORDER BY issued_at DESC LIMIT 1",
                            (call_id,)).fetchone()
            if not row:
                return None
            verdict = classify_confirmation(text)
            if verdict == "yes":
                c.execute("UPDATE readbacks SET status='confirmed', confirmed_at=? WHERE id=?", (time.time(), row["id"]))
            elif verdict == "no":
                c.execute("UPDATE readbacks SET status='rejected' WHERE id=?", (row["id"],))
            return verdict

    def _take_confirmed(self, c, call_id: str, action: str, match: dict) -> str:
        row = c.execute("SELECT * FROM readbacks WHERE call_id=? AND action=? AND status='confirmed' "
                        "ORDER BY issued_at DESC LIMIT 1", (call_id, action)).fetchone()
        if not row:
            raise ToolError("not_confirmed", f"Refused: {action} needs a read-back with prepare_confirmation and a "
                                             f"clear yes from the caller first.")
        target = json.loads(row["target"])
        for k, v in match.items():
            if v is not None and target.get(k) != v:
                raise ToolError("not_confirmed", f"Refused: the confirmed read-back was for a different {k}. "
                                                 f"Read back the new details and get a yes.")
        c.execute("UPDATE readbacks SET status='used', used_at=? WHERE id=?", (time.time(), row["id"]))
        return row["id"]

    # ---------- commits ----------
    def book(self, call_id: str, hold_id: str, patient_name: str | None = None,
             patient_phone: str | None = None, idempotency_key: str | None = None) -> dict:
        with self.tx() as c:
            hold = c.execute("SELECT * FROM holds WHERE id=?", (hold_id,)).fetchone()
            key = idempotency_key or f"{call_id}:book:{hold['slot_id'] if hold else hold_id}"
            alias = f"{call_id}:book-hold:{hold_id}"  # retry after the hold was consumed
            prior = self._idem_get(c, key) or self._idem_get(c, alias)
            if prior:
                return {**prior, "idempotent_replay": True}
            call = self._call(c, call_id)
            if call["emergency"]:
                raise ToolError("emergency_call", "No booking in an emergency call.")
            hold = self._live_hold(c, hold_id, call_id)
            match = {"hold_id": hold_id}
            if patient_phone:
                match["patient_phone"] = normalize_phone(patient_phone)
            rb_row = c.execute("SELECT target FROM readbacks WHERE call_id=? AND action='book' AND status='confirmed' "
                               "ORDER BY issued_at DESC LIMIT 1", (call_id,)).fetchone()
            rb_id = self._take_confirmed(c, call_id, "book", match)
            target = json.loads(rb_row["target"])
            name, phone, age = target["patient_name"], target["patient_phone"], target.get("patient_age")
            # one booking per call unless for a second named person
            prev = c.execute("SELECT a.patient_name FROM call_actions ca JOIN appointments a ON a.id=ca.appointment_id "
                             "WHERE ca.call_id=? AND ca.action='book'", (call_id,)).fetchall()
            if any(p["patient_name"].lower() == name.lower() for p in prev) or len(prev) >= MAX_BOOKINGS_PER_CALL:
                raise ToolError("booking_limit", "One booking per patient per call. A further booking is only for a "
                                                 "different named family member.")
            doc_id, day, hhmm = d.parse_slot_id(hold["slot_id"])
            pkey = patient_key(name, phone)
            existing = c.execute("SELECT * FROM appointments WHERE patient_key=? AND doctor_id=? AND status='active'",
                                 (pkey, doc_id)).fetchone()
            if existing:
                c.execute("UPDATE readbacks SET status='confirmed', used_at=NULL WHERE id=?", (rb_id,))
                raise ToolError("existing_appointment", "This patient already has an active appointment with this "
                                                        "doctor. Offer to reschedule it instead.",
                                appointment=self._appt_view(existing))
            appt_id = self._next_appt_id(c, day)
            now = time.time()
            c.execute("UPDATE slot_locks SET kind='appointment', ref=?, expires_at=NULL WHERE slot_id=? AND ref=?",
                      (appt_id, hold["slot_id"], hold_id))
            c.execute("DELETE FROM holds WHERE id=?", (hold_id,))
            c.execute("INSERT INTO appointments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (appt_id, hold["slot_id"], doc_id, day.isoformat(), hhmm, name, pkey, phone, age, "active",
                       call_id, now, now))
            sv = slot_view(hold["slot_id"])
            summary = f"Booked {sv['doctor']} ({sv['department'].lower()}), {d.spoken_date(day)}, {hhmm} for {d.first_name(name)}."
            c.execute("INSERT INTO call_actions VALUES(?,?,?,?,?,?,?)", (call_id, now, "book", appt_id, rb_id, 1, summary))
            row = c.execute("SELECT * FROM appointments WHERE id=?", (appt_id,)).fetchone()
            resp = {"ok": True, "appointment": self._appt_view(row),
                    "say": f"Done. Your appointment ID is {' '.join(appt_id.split('-')[-1])}. Please arrive 15 minutes early."}
            self._idem_put(c, key, resp)
            self._idem_put(c, alias, resp)
        return resp

    def reschedule(self, call_id: str, appointment_id: str, new_hold_id: str,
                   idempotency_key: str | None = None) -> dict:
        with self.tx() as c:
            hold = c.execute("SELECT * FROM holds WHERE id=?", (new_hold_id,)).fetchone()
            key = idempotency_key or f"{call_id}:reschedule:{appointment_id}:{hold['slot_id'] if hold else new_hold_id}"
            alias = f"{call_id}:reschedule-hold:{new_hold_id}"
            prior = self._idem_get(c, key) or self._idem_get(c, alias)
            if prior:
                return {**prior, "idempotent_replay": True}
            self._call(c, call_id)
            hold = self._live_hold(c, new_hold_id, call_id)
            appt = c.execute("SELECT * FROM appointments WHERE id=?", (appointment_id,)).fetchone()
            if not appt or appt["status"] != "active":
                raise ToolError("unknown_appointment", "That appointment is not active.")
            rb_id = self._take_confirmed(c, call_id, "reschedule",
                                         {"appointment_id": appointment_id, "hold_id": new_hold_id})
            doc_id, day, hhmm = d.parse_slot_id(hold["slot_id"])
            old_slot = appt["slot_id"]
            now = time.time()
            c.execute("DELETE FROM slot_locks WHERE slot_id=? AND kind='appointment' AND ref=?", (old_slot, appointment_id))
            c.execute("UPDATE slot_locks SET kind='appointment', ref=?, expires_at=NULL WHERE slot_id=? AND ref=?",
                      (appointment_id, hold["slot_id"], new_hold_id))
            c.execute("DELETE FROM holds WHERE id=?", (new_hold_id,))
            c.execute("UPDATE appointments SET slot_id=?, appt_date=?, appt_time=?, updated_at=? WHERE id=?",
                      (hold["slot_id"], day.isoformat(), hhmm, now, appointment_id))
            summary = (f"Rescheduled {d.BY_ID[doc_id].name} for {d.first_name(appt['patient_name'])} from "
                       f"{appt['appt_date']} {appt['appt_time']} to {d.spoken_date(day)}, {hhmm}.")
            c.execute("INSERT INTO call_actions VALUES(?,?,?,?,?,?,?)",
                      (call_id, now, "reschedule", appointment_id, rb_id, 1, summary))
            row = c.execute("SELECT * FROM appointments WHERE id=?", (appointment_id,)).fetchone()
            resp = {"ok": True, "appointment": self._appt_view(row), "old_slot_released": old_slot}
            self._idem_put(c, key, resp)
            self._idem_put(c, alias, resp)
        return resp

    def cancel(self, call_id: str, appointment_id: str, idempotency_key: str | None = None) -> dict:
        key = idempotency_key or f"{call_id}:cancel:{appointment_id}"
        with self.tx() as c:
            prior = self._idem_get(c, key)
            if prior:
                return {**prior, "idempotent_replay": True,
                        "message": "Already cancelled earlier in this call. Nothing more was changed."}
            self._call(c, call_id)
            appt = c.execute("SELECT * FROM appointments WHERE id=?", (appointment_id,)).fetchone()
            if not appt:
                raise ToolError("unknown_appointment", "Appointment not found.")
            if appt["status"] != "active":
                return {"ok": True, "already_cancelled": True, "appointment": self._appt_view(appt),
                        "message": "This appointment was already cancelled. Nothing changed."}
            rb_id = self._take_confirmed(c, call_id, "cancel", {"appointment_id": appointment_id})
            now = time.time()
            c.execute("UPDATE appointments SET status='cancelled', updated_at=? WHERE id=?", (now, appointment_id))
            c.execute("DELETE FROM slot_locks WHERE slot_id=? AND ref=?", (appt["slot_id"], appointment_id))
            summary = (f"Cancelled {d.BY_ID[appt['doctor_id']].name} on {appt['appt_date']} {appt['appt_time']} "
                       f"for {d.first_name(appt['patient_name'])}.")
            c.execute("INSERT INTO call_actions VALUES(?,?,?,?,?,?,?)",
                      (call_id, now, "cancel", appointment_id, rb_id, 1, summary))
            row = c.execute("SELECT * FROM appointments WHERE id=?", (appointment_id,)).fetchone()
            resp = {"ok": True, "appointment": self._appt_view(row)}
            self._idem_put(c, key, resp)
        return resp

    def transfer_to_human(self, call_id: str, reason: str, note: str | None = None) -> dict:
        if reason not in TRANSFER_REASONS:
            reason = "agent_stuck"
        with self.tx() as c:
            self._call(c, call_id)
            existing = c.execute("SELECT id FROM transfers WHERE call_id=? AND reason=?", (call_id, reason)).fetchone()
            if existing:
                return {"ok": True, "transfer_id": existing["id"], "already_queued": True}
            tid = "T-" + uuid.uuid4().hex[:6].upper()
            c.execute("INSERT INTO transfers(id, call_id, reason, note, created_at) VALUES(?,?,?,?,?)",
                      (tid, call_id, reason, mask_text(note or "")[:300], time.time()))
            if reason == "emergency":
                c.execute("UPDATE calls SET emergency=1 WHERE id=?", (call_id,))
                # release anything held so no booking can complete in this call
                c.execute("DELETE FROM slot_locks WHERE kind='hold' AND call_id=?", (call_id,))
                c.execute("DELETE FROM holds WHERE call_id=?", (call_id,))
                c.execute("UPDATE readbacks SET status='superseded' WHERE call_id=? AND status IN ('issued','confirmed')",
                          (call_id,))
        return {"ok": True, "transfer_id": tid}

    def resolve_transfer(self, transfer_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE transfers SET status='handled', handled_at=? WHERE id=?", (time.time(), transfer_id))

    # ---------------- call record ----------------
    def end_call(self, call_id: str, end_reason: str = "caller_hangup") -> dict:
        with self.tx() as c:
            call = self._call(c, call_id)
            if call["record"]:
                return json.loads(call["record"])
            c.execute("DELETE FROM slot_locks WHERE kind='hold' AND call_id=?", (call_id,))
            c.execute("DELETE FROM holds WHERE call_id=?", (call_id,))
            record = self._build_record(c, call, end_reason)
            c.execute("UPDATE calls SET ended_at=?, status='ended', end_reason=?, record=? WHERE id=?",
                      (time.time(), end_reason, json.dumps(record, ensure_ascii=False), call_id))
        return record

    def _build_record(self, c, call: sqlite3.Row, end_reason: str) -> dict:
        call_id = call["id"]
        actions = c.execute("SELECT * FROM call_actions WHERE call_id=? ORDER BY ts", (call_id,)).fetchall()
        transfers = c.execute("SELECT * FROM transfers WHERE call_id=? ORDER BY created_at", (call_id,)).fetchall()
        tools = [r["name"] for r in c.execute("SELECT name FROM tool_log WHERE call_id=? ORDER BY ts", (call_id,))]
        turns = [json.loads(r["data"]) for r in c.execute("SELECT data FROM turns WHERE call_id=? ORDER BY idx", (call_id,))]
        n_user = c.execute("SELECT COUNT(*) FROM transcript WHERE call_id=? AND role='caller'", (call_id,)).fetchone()[0]
        totals = [t["total_ms"] for t in turns if t.get("total_ms")]
        usage = json.loads(call["usage"])

        if transfers:
            outcome = "transferred"
        elif actions:
            outcome = {"book": "booked", "reschedule": "rescheduled", "cancel": "cancelled"}[actions[-1]["action"]]
        elif end_reason in {"silence_timeout", "caller_hangup"} and any(
                t in tools for t in ("get_slots", "hold_slot", "prepare_confirmation", "find_appointments")):
            outcome = "abandoned"
        else:
            outcome = "no_action"

        if any(t["reason"] == "emergency" for t in transfers):
            intent = "emergency"
        elif actions:
            intent = actions[-1]["action"]
        elif "find_appointments" in tools:
            intent = "reschedule_or_cancel"
        elif "get_slots" in tools or "hold_slot" in tools:
            intent = "book"
        elif transfers:
            intent = transfers[0]["reason"]
        else:
            intent = "enquiry"

        if transfers:
            summary = f"Transferred to front desk ({transfers[0]['reason'].replace('_', ' ')})."
            if actions:
                summary = " ".join(a["summary"] for a in actions) + " " + summary
        elif actions:
            summary = "Caller " + " ".join(a["summary"][0].lower() + a["summary"][1:] for a in actions)
        else:
            summary = {"abandoned": "Caller left before completing the request.",
                       "no_action": "Enquiry only; no appointment changed."}[outcome]

        return {
            "call_id": call_id,
            "language": json.loads(call["languages"]) or ["en"],
            "intent": intent,
            "outcome": outcome,
            "appointment_id": actions[-1]["appointment_id"] if actions else None,
            "transfer_reason": transfers[0]["reason"] if transfers else None,
            "confirmed_before_commit": all(a["confirmed"] and a["readback_id"] for a in actions) if actions else None,
            "turns": n_user,
            "latency_ms": {"p50": _pct(totals, 50), "p95": _pct(totals, 95)},
            "barge_ins": call["barge_ins"],
            "cost_inr": estimate_cost_inr(usage),
            "summary": summary,
            "end_reason": end_reason,
            "duration_s": round(time.time() - call["started_at"], 1),
            "transport": call["transport"],
        }

    # ---------------- staff view ----------------
    def dashboard(self) -> dict:
        today_start = datetime.combine(datetime.now(d.IST).date(), datetime.min.time(), d.IST).timestamp()
        with self._conn() as c:
            calls = c.execute("SELECT * FROM calls WHERE started_at>=? ORDER BY started_at DESC LIMIT 200",
                              (today_start,)).fetchall()
            transfers = c.execute("SELECT * FROM transfers WHERE status='waiting' ORDER BY created_at").fetchall()
            turn_rows = c.execute("SELECT t.call_id, t.data FROM turns t JOIN calls ca ON ca.id=t.call_id "
                                  "WHERE ca.started_at>=? ORDER BY t.created_at DESC LIMIT 400", (today_start,)).fetchall()
        out_calls, counts = [], {}
        for r in calls:
            rec = json.loads(r["record"]) if r["record"] else None
            outcome = rec["outcome"] if rec else "live"
            counts[outcome] = counts.get(outcome, 0) + 1
            out_calls.append({"call_id": r["id"], "started_at": r["started_at"], "status": r["status"],
                              "transport": r["transport"], "emergency": bool(r["emergency"]), "record": rec})
        turn_data = [json.loads(t["data"]) for t in turn_rows]
        def series(k):
            return [t[k] for t in turn_data if isinstance(t.get(k), (int, float))]
        totals = series("total_ms")
        barge = series("barge_in_ms")
        return {
            "calls": out_calls,
            "outcome_counts": counts,
            "transfers_waiting": [dict(t) for t in transfers],
            "latency": {
                "turns": len(totals),
                "total": {"p50": _pct(totals, 50), "p95": _pct(totals, 95)},
                "stt": {"p50": _pct(series("stt_ms"), 50), "p95": _pct(series("stt_ms"), 95)},
                "llm_ttft": {"p50": _pct(series("llm_ttft_ms"), 50), "p95": _pct(series("llm_ttft_ms"), 95)},
                "tts_ttfb": {"p50": _pct(series("tts_ttfb_ms"), 50), "p95": _pct(series("tts_ttfb_ms"), 95)},
                "barge_in": {"p50": _pct(barge, 50), "p95": _pct(barge, 95), "n": len(barge)},
                "recent": turn_data[:20],
            },
        }

    def call_detail(self, call_id: str) -> dict:
        with self._conn() as c:
            call = self._call(c, call_id)
            transcript = [dict(r) for r in c.execute("SELECT ts, role, text FROM transcript WHERE call_id=? ORDER BY ts",
                                                     (call_id,))]
            tools = [dict(r) for r in c.execute("SELECT ts, name, args, ok, duration_ms FROM tool_log WHERE call_id=? "
                                                "ORDER BY ts", (call_id,))]
            turns = [json.loads(r["data"]) for r in c.execute("SELECT data FROM turns WHERE call_id=? ORDER BY idx",
                                                              (call_id,))]
        return {"call_id": call_id, "status": call["status"],
                "record": json.loads(call["record"]) if call["record"] else None,
                "transcript": transcript, "tools": tools, "turns": turns}

    def appointments(self, active_only: bool = True) -> list[dict]:
        with self._conn() as c:
            q = "SELECT * FROM appointments" + (" WHERE status='active'" if active_only else "") + \
                " ORDER BY appt_date, appt_time"
            return [self._appt_view(r) for r in c.execute(q)]

    # ---------------- demo seed ----------------
    DEMO_PATIENTS = [
        ("Rahul Sharma", "9876543210", None, "D09"),
        ("Sunita Kulkarni", "9822012345", None, "D07"),
        ("Aarav Kulkarni", "9822012345", 8, "D05"),
        ("Imran Shaikh", "9123443120", None, "D08"),
    ]

    def seed_demo_patients(self) -> None:
        """Existing appointments for reschedule / cancel scripts (fictional patients)."""
        today = d.today_ist()
        with self.tx() as c:
            for name, phone, age, doc_id in self.DEMO_PATIENTS:
                pkey = patient_key(name, phone)
                if c.execute("SELECT 1 FROM appointments WHERE patient_key=? AND status='active'", (pkey,)).fetchone():
                    continue
                doc = d.BY_ID[doc_id]
                placed = False
                for n in range(3, d.BOOKING_WINDOW_DAYS + 1):
                    day = today + timedelta(days=n)
                    if d.closure_reason(doc, day, today):
                        continue
                    for hhmm in d.slot_times(doc):
                        sid = d.slot_id(doc_id, day, hhmm)
                        if d.is_prebooked(sid) or c.execute("SELECT 1 FROM slot_locks WHERE slot_id=?", (sid,)).fetchone():
                            continue
                        appt_id = self._next_appt_id(c, day)
                        now = time.time()
                        c.execute("INSERT INTO slot_locks VALUES(?,?,?,?,?)", (sid, "appointment", appt_id, None, None))
                        c.execute("INSERT INTO appointments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                  (appt_id, sid, doc_id, day.isoformat(), hhmm, name, pkey, phone, age, "active",
                                   "seed", now, now))
                        placed = True
                        break
                    if placed:
                        break


def _pct(values: list[float], p: int) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return int(values[0])
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return int(v[lo] + (v[hi] - v[lo]) * (k - lo))


# Estimated list prices converted at ~₹85/USD; override in .env. See README cost table.
DEFAULT_RATES = {
    "stt_audio_seconds": 0.0092 / 60 * 85,      # Deepgram Nova-3 multilingual streaming ≈ $0.0092/min
    "tts_characters": 0.000038 * 85,             # Cartesia Sonic ≈ $0.038 / 1k chars (startup plan)
    "llm_input_tokens": 0.15 / 1e6 * 85,         # Groq gpt-oss-120b ≈ $0.15 / M input
    "llm_output_tokens": 0.60 / 1e6 * 85,        # ≈ $0.60 / M output
}


def estimate_cost_inr(usage: dict) -> float:
    total = 0.0
    for k, default in DEFAULT_RATES.items():
        rate = os.getenv("RATE_INR_" + k.upper())
        rate_v = float(rate) if rate else default
        total += float(usage.get(k, 0)) * rate_v
    return round(total, 2)
