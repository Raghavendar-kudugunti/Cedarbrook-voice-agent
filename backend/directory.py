"""Cedarbrook doctor directory and calendar rules (from the Team Beta brief)."""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
SLOT_MINUTES = 15
BOOKING_WINDOW_DAYS = 14
HOLIDAYS = {date(2026, 12, 25): "Christmas Day"}
PREBOOKED_PERCENT = 60
PREBOOKED_SEED = "cedarbrook-2026"

MON, TUE, WED, THU, FRI, SAT, SUN = range(7)


@dataclass(frozen=True)
class Doctor:
    id: str
    name: str
    department: str
    days: tuple[int, ...]
    start: str
    end: str
    languages: tuple[str, ...]
    notes: str = ""
    leave: tuple[tuple[date, date], ...] = ()
    max_age: int | None = None

    @property
    def fee_inr(self) -> int:
        return 600 if self.department == "General Medicine" else 900

    def public(self) -> dict:
        d = asdict(self)
        d["days"] = [DAY_NAMES[x] for x in self.days]
        d["languages"] = list(self.languages)
        d["leave"] = [{"from": a.isoformat(), "to": b.isoformat()} for a, b in self.leave]
        d["opd_hours"] = f"{self.start}-{self.end}"
        d["fee_inr"] = self.fee_inr
        return d


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAYS = (MON, TUE, WED, THU, FRI)
MON_SAT = (MON, TUE, WED, THU, FRI, SAT)

DOCTORS: list[Doctor] = [
    Doctor("D01", "Dr. Anil Mehta", "Cardiology", (MON, WED, FRI), "09:00", "13:00", ("English", "Hindi")),
    Doctor("D02", "Dr. Rekha Mehta", "Dermatology", (TUE, THU, FRI), "16:00", "19:00", ("English", "Hindi", "Marathi")),
    Doctor("D03", "Dr. Suresh Kulkarni", "Orthopaedics", MON_SAT, "10:00", "13:00", ("English", "Marathi"),
           "On leave 17-21 Nov 2026", leave=((date(2026, 11, 17), date(2026, 11, 21)),)),
    Doctor("D04", "Dr. Farah Siddiqui", "Orthopaedics", (TUE, THU, SAT), "17:00", "20:00", ("English", "Hindi")),
    Doctor("D05", "Dr. Priya Raman", "Paediatrics", WEEKDAYS, "09:30", "12:30", ("English", "Tamil"),
           "Patients under 16 only", max_age=15),
    Doctor("D06", "Dr. Joseph Mathew", "Paediatrics", (MON, WED, SAT), "16:00", "19:00", ("English", "Malayalam"),
           "Patients under 16 only", max_age=15),
    Doctor("D07", "Dr. Kavita Deshpande", "Gynaecology", WEEKDAYS, "11:00", "14:00", ("English", "Hindi", "Marathi")),
    Doctor("D08", "Dr. Rohit Bansal", "ENT", (WED, THU, FRI), "10:00", "13:00", ("English", "Hindi")),
    Doctor("D09", "Dr. Meenal Joshi", "General Medicine", MON_SAT, "09:00", "13:00", ("English", "Hindi", "Marathi")),
    Doctor("D10", "Dr. Arjun Nair", "General Medicine", MON_SAT, "17:00", "20:00", ("English", "Malayalam")),
    Doctor("D11", "Dr. Sanjay Gupta", "Cardiology", (TUE, THU), "17:00", "20:00", ("English", "Hindi")),
    Doctor("D12", "Dr. Neha Patil", "Dermatology", (MON, WED), "10:00", "13:00", ("English", "Marathi")),
    Doctor("D13", "Dr. Vikram Sethi", "ENT", (MON, TUE), "17:00", "19:00", ("English", "Hindi", "Punjabi")),
    Doctor("D14", "Dr. Lata Iyer", "Gynaecology", (SAT,), "09:00", "13:00", ("English", "Tamil")),
]
BY_ID = {d.id: d for d in DOCTORS}

# Lay words callers use for departments.
DEPARTMENT_ALIASES = {
    "cardiology": "Cardiology", "heart": "Cardiology", "cardiologist": "Cardiology", "dil": "Cardiology",
    "dermatology": "Dermatology", "skin": "Dermatology", "dermatologist": "Dermatology", "twacha": "Dermatology",
    "orthopaedics": "Orthopaedics", "orthopedics": "Orthopaedics", "ortho": "Orthopaedics", "bone": "Orthopaedics",
    "haddi": "Orthopaedics", "joint": "Orthopaedics", "knee": "Orthopaedics",
    "paediatrics": "Paediatrics", "pediatrics": "Paediatrics", "child": "Paediatrics", "children": "Paediatrics",
    "bachcha": "Paediatrics", "baccha": "Paediatrics", "kids": "Paediatrics",
    "gynaecology": "Gynaecology", "gynecology": "Gynaecology", "gynaec": "Gynaecology", "women": "Gynaecology",
    "ent": "ENT", "ear": "ENT", "nose": "ENT", "throat": "ENT", "kaan": "ENT", "naak": "ENT", "gala": "ENT",
    "general medicine": "General Medicine", "general": "General Medicine", "physician": "General Medicine",
    "fever": "General Medicine", "bukhar": "General Medicine", "medicine": "General Medicine",
}


def normalize_department(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().lower()
    if t in DEPARTMENT_ALIASES:
        return DEPARTMENT_ALIASES[t]
    for key, dept in DEPARTMENT_ALIASES.items():
        if key in t:
            return dept
    return text.strip()


def today_ist() -> date:
    """Clinic 'today'. CLINIC_TODAY=YYYY-MM-DD pins it for tests and demos."""
    pinned = os.getenv("CLINIC_TODAY", "").strip()
    if pinned:
        return date.fromisoformat(pinned)
    return datetime.now(IST).date()


def closure_reason(doc: Doctor, day: date, today: date | None = None) -> str | None:
    """Why a doctor has no OPD on `day`, or None if the day is bookable."""
    today = today or today_ist()
    if day <= today:
        return "same_day_walk_in" if day == today else "past_date"
    if day > today + timedelta(days=BOOKING_WINDOW_DAYS):
        return "beyond_14_days"
    if day.weekday() == SUN:
        return "opd_closed_sunday"
    if day in HOLIDAYS:
        return f"opd_closed_holiday:{HOLIDAYS[day]}"
    for a, b in doc.leave:
        if a <= day <= b:
            return f"doctor_on_leave:{a.isoformat()}..{b.isoformat()}"
    if day.weekday() not in doc.days:
        return "doctor_not_in_opd_this_day"
    return None


def slot_times(doc: Doctor) -> list[str]:
    h, m = map(int, doc.start.split(":"))
    eh, em = map(int, doc.end.split(":"))
    return [f"{x // 60:02d}:{x % 60:02d}" for x in range(h * 60 + m, eh * 60 + em, SLOT_MINUTES)]


def slot_id(doctor_id: str, day: date, hhmm: str) -> str:
    return f"{doctor_id}-{day.strftime('%Y%m%d')}-{hhmm.replace(':', '')}"


def parse_slot_id(sid: str) -> tuple[str, date, str]:
    doc_id, ymd, hm = sid.split("-")
    return doc_id, datetime.strptime(ymd, "%Y%m%d").date(), f"{hm[:2]}:{hm[2:]}"


def is_prebooked(sid: str) -> bool:
    """~60% of slots are taken by other patients, chosen by a fixed seed (stable across runs)."""
    import hashlib
    h = hashlib.sha256(f"{PREBOOKED_SEED}:{sid}".encode()).digest()
    return int.from_bytes(h[:4], "big") % 100 < PREBOOKED_PERCENT


# ---------- spoken formats ----------
ONES = ["twelve", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]


def spoken_time(hhmm: str) -> str:
    """'17:30' -> 'half past five in the evening'."""
    h, m = map(int, hhmm.split(":"))
    part = "in the morning" if h < 12 else ("in the afternoon" if h < 16 else "in the evening")
    if h == 12 and m == 0:
        return "twelve noon"
    hour = ONES[h % 12]
    nxt = ONES[(h + 1) % 12]
    if m == 0:
        return f"{hour} o'clock {part}"
    if m == 15:
        return f"quarter past {hour} {part}"
    if m == 30:
        return f"half past {hour} {part}"
    if m == 45:
        return f"quarter to {nxt} {part}"
    return f"{hour} {m:02d} {part}"


def spoken_date(day: date) -> str:
    """'Friday, 20 November'."""
    return f"{day.strftime('%A')}, {day.day} {day.strftime('%B')}"


def spoken_phone(phone: str) -> str:
    """'9876543210' -> '98765 · 43210' read as two groups of digits."""
    digits = "".join(c for c in phone if c.isdigit())
    if len(digits) == 10:
        return f"{' '.join(digits[:5])}, {' '.join(digits[5:])}"
    return " ".join(digits)


def display_phone(phone: str) -> str:
    digits = "".join(c for c in phone if c.isdigit())
    return f"{digits[:5]} · {digits[5:]}" if len(digits) == 10 else digits


def mask_phone(phone: str | None) -> str:
    digits = "".join(c for c in (phone or "") if c.isdigit())
    return f"******{digits[-4:]}" if digits else ""


def first_name(name: str | None) -> str:
    return (name or "").strip().split(" ")[0] if name else ""


def booking_calendar(today: date | None = None) -> str:
    """Compact calendar for the system prompt so the LLM resolves relative dates correctly."""
    today = today or today_ist()
    lines = [f"Today is {today.strftime('%A')}, {today.day} {today.strftime('%B %Y')} (IST)."]
    days = []
    for n in range(1, BOOKING_WINDOW_DAYS + 1):
        d = today + timedelta(days=n)
        tag = ""
        if d.weekday() == SUN:
            tag = " CLOSED"
        elif d in HOLIDAYS:
            tag = " CLOSED-holiday"
        days.append(f"{d.strftime('%a')} {d.day} {d.strftime('%b')} = {d.isoformat()}{tag}")
    lines.append("Bookable days (tomorrow to +14): " + "; ".join(days) + ".")
    return " ".join(lines)


# ---------- relative dates, resolved in code (IST) ----------
_WEEKDAYS = {
    "monday": MON, "mon": MON, "somvar": MON, "somwar": MON, "सोमवार": MON,
    "tuesday": TUE, "tue": TUE, "mangalvar": TUE, "mangalwar": TUE, "मंगलवार": TUE, "मंगळवार": TUE,
    "wednesday": WED, "wed": WED, "budhvar": WED, "budhwar": WED, "बुधवार": WED,
    "thursday": THU, "thu": THU, "guruvar": THU, "guruwar": THU, "brihaspativar": THU, "गुरुवार": THU,
    "friday": FRI, "fri": FRI, "shukravar": FRI, "shukrawar": FRI, "शुक्रवार": FRI,
    "saturday": SAT, "sat": SAT, "shanivar": SAT, "shaniwar": SAT, "शनिवार": SAT,
    "sunday": SUN, "sun": SUN, "ravivar": SUN, "raviwar": SUN, "itvar": SUN, "रविवार": SUN,
}
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def resolve_day(expr: str | None, today: date | None = None) -> date | None:
    """'2026-11-20', 'friday', 'next friday', 'this sunday', 'tomorrow', 'kal', 'parso', 'day after tomorrow',
    'this weekend', '19 November', 'Nov 19' -> a date. None if not understood."""
    import re
    if not expr:
        return None
    today = today or today_ist()
    t = " ".join(str(expr).strip().lower().replace(",", " ").split())
    try:
        return date.fromisoformat(t)
    except ValueError:
        pass
    if t in {"today", "aaj", "आज"}:
        return today
    if t in {"tomorrow", "kal", "कल", "udya", "उद्या"}:
        return today + timedelta(days=1)
    if t in {"day after tomorrow", "parso", "परसों", "parva", "परवा"}:
        return today + timedelta(days=2)
    if "weekend" in t:  # Sunday is closed, so the weekend means Saturday
        return today + timedelta(days=(SAT - today.weekday()) % 7 or 7)
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3})[a-z]*$", t) or None
    m2 = re.match(r"^([a-z]{3})[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?$", t) or None
    if m or m2:
        day_n, mon = (int(m.group(1)), m.group(2)) if m else (int(m2.group(2)), m2.group(1))
        if mon in _MONTHS:
            year = today.year if (_MONTHS[mon], day_n) >= (today.month, today.day) else today.year + 1
            try:
                return date(year, _MONTHS[mon], day_n)
            except ValueError:
                return None
    words = t.split()
    wd = next((_WEEKDAYS[w] for w in words if w in _WEEKDAYS), None)
    if wd is not None:
        ahead = (wd - today.weekday()) % 7 or 7  # the coming one, never today
        if "next" in words or "agle" in words or "agla" in words or "pudhcha" in words:
            # "next Friday" = Friday of next week (Mon-start weeks)
            this_week_monday = today - timedelta(days=today.weekday())
            return this_week_monday + timedelta(days=7 + wd)
        return today + timedelta(days=ahead)
    return None
