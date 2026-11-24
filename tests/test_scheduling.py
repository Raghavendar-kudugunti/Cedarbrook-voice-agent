"""Deterministic tests for the scheduling service and safety gates (no network, no LLM)."""
import threading
from datetime import date, timedelta

import pytest

from backend import directory as d
from backend.scheduling import Scheduler, ToolError, classify_confirmation, mask_text, normalize_phone

TODAY = "2026-11-16"  # Monday: makes B07 (Kulkarni leave 17-21 Nov) and B08 (Sunday) testable


@pytest.fixture()
def s(tmp_path, monkeypatch):
    monkeypatch.setenv("CLINIC_TODAY", TODAY)
    return Scheduler(str(tmp_path / "t.db"))


def first_slot(s, call, doc="D01", frm="2026-11-17", to="2026-11-30", part=None):
    res = s.get_slots(call, doc, frm, to, part)
    assert res["slots"], res
    return res["slots"][0]


def confirmed_booking(s, call, doc="D01", name="Asha Verma", phone="9876501234"):
    slot = first_slot(s, call, doc)
    hold = s.hold_slot(call, slot["slot_id"])
    s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name=name, patient_phone=phone)
    assert s.record_caller_reply(call, "yes") == "yes"
    return s.book(call, hold["hold_id"]), slot


def test_directory_has_14_doctors_in_7_departments():
    assert len(d.DOCTORS) == 14
    assert len({x.department for x in d.DOCTORS}) == 7


def test_slots_respect_hours_days_and_prebooked_seed(s):
    call = s.start_call("text")
    res = s.get_slots(call, "D01", "2026-11-17", "2026-11-30", limit=3)
    for sl in res["slots"]:
        day = date.fromisoformat(sl["date"])
        assert day.weekday() in (0, 2, 4)
        assert "09:00" <= sl["time"] < "13:00"
        assert not d.is_prebooked(sl["slot_id"])
    assert len(res["slots"]) <= 3
    # roughly 60% prebooked across the whole directory
    total = taken = 0
    for doc in d.DOCTORS:
        for n in range(1, 15):
            day = date(2026, 11, 16) + timedelta(days=n)
            if d.closure_reason(doc, day, date(2026, 11, 16)):
                continue
            for t in d.slot_times(doc):
                total += 1
                taken += d.is_prebooked(d.slot_id(doc.id, day, t))
    assert 0.55 < taken / total < 0.65


def test_kulkarni_leave_sunday_holiday_same_day_window(s):
    call = s.start_call("text")
    r = s.get_slots(call, "D03", "2026-11-19")
    assert not r["slots"] and r["closed_days"][0]["reason"].startswith("doctor_on_leave")
    r = s.get_slots(call, "D09", "2026-11-22")  # Sunday
    assert not r["slots"] and r["closed_days"][0]["reason"] == "opd_closed_sunday"
    r = s.get_slots(call, "D09", "2026-11-16")  # today -> walk-in only, then searches from tomorrow
    assert r["closed_days"][0]["reason"] == "same_day_walk_in"
    assert all(x["date"] >= "2026-11-17" for x in r["slots"])
    r = s.get_slots(call, "D09", "2026-12-20")
    assert any(n["reason"] == "beyond_14_days" for n in r["closed_days"])
    assert d.closure_reason(d.BY_ID["D09"], date(2026, 12, 25), date(2026, 12, 20)).startswith("opd_closed_holiday")


def test_never_invent_a_slot(s):
    call = s.start_call("text")
    with pytest.raises(ToolError) as e:
        s.hold_slot(call, "D01-20261118-0900")
    assert e.value.code == "slot_not_offered"


def test_commit_refused_without_confirmation(s):
    call = s.start_call("text")
    slot = first_slot(s, call)
    hold = s.hold_slot(call, slot["slot_id"])
    with pytest.raises(ToolError) as e:
        s.book(call, hold["hold_id"])
    assert e.value.code == "not_confirmed"
    s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Asha", patient_phone="98765 01234")
    with pytest.raises(ToolError):
        s.book(call, hold["hold_id"])  # read-back given but no yes yet
    assert s.record_caller_reply(call, "No, 43120 not 43210") == "no"
    with pytest.raises(ToolError):
        s.book(call, hold["hold_id"])
    s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Asha", patient_phone="9876501234")
    assert s.record_caller_reply(call, "haan ji") == "yes"
    res = s.book(call, hold["hold_id"])
    assert res["ok"] and res["appointment"]["appointment_id"].startswith("AP-")
    rec = s.end_call(call, "agent_end")
    assert rec["outcome"] == "booked" and rec["confirmed_before_commit"] is True


def test_idempotent_book_and_double_cancel(s):
    call = s.start_call("text")
    first, slot = confirmed_booking(s, call)
    appt_id = first["appointment"]["appointment_id"]
    # cancel: yes said twice on a bad line
    call2 = s.start_call("text")
    s.prepare_confirmation(call2, "cancel", appointment_id=appt_id)
    s.record_caller_reply(call2, "yes")
    s.record_caller_reply(call2, "yes")
    c1 = s.cancel(call2, appt_id)
    c2 = s.cancel(call2, appt_id)
    assert c1["appointment"]["status"] == "cancelled"
    assert c2.get("idempotent_replay")
    rows = [a for a in s.appointments(active_only=False) if a["appointment_id"] == appt_id]
    assert len(rows) == 1 and rows[0]["status"] == "cancelled"
    with s._conn() as c:  # slot released
        assert not c.execute("SELECT 1 FROM slot_locks WHERE slot_id=?", (slot["slot_id"],)).fetchone()


def test_book_retry_returns_same_appointment(s):
    call = s.start_call("text")
    slot = first_slot(s, call)
    hold = s.hold_slot(call, slot["slot_id"])
    s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Ravi Rao", patient_phone="9000011111")
    s.record_caller_reply(call, "yes please")
    a = s.book(call, hold["hold_id"])
    b = s.book(call, hold["hold_id"])
    assert b["idempotent_replay"] and a["appointment"]["appointment_id"] == b["appointment"]["appointment_id"]
    assert len([x for x in s.appointments() if x["phone_last4"] == "1111"]) == 1


def test_concurrent_holds_only_one_wins(s):
    calls = [s.start_call("text") for _ in range(12)]
    slot = first_slot(s, calls[0])
    for c in calls[1:]:
        s.get_slots(c, "D01", "2026-11-17", "2026-11-30")
    wins, errors = [], []

    def grab(c):
        try:
            wins.append(s.hold_slot(c, slot["slot_id"]))
        except ToolError as e:
            errors.append(e.code)

    th = [threading.Thread(target=grab, args=(c,)) for c in calls]
    [t.start() for t in th]
    [t.join() for t in th]
    assert len(wins) == 1 and set(errors) == {"slot_taken"}


def test_slot_taken_between_offer_and_booking(s):
    a, b = s.start_call("text"), s.start_call("text")
    slot = first_slot(s, a)
    s.get_slots(b, "D01", "2026-11-17", "2026-11-30")
    s.hold_slot(b, slot["slot_id"])
    with pytest.raises(ToolError) as e:
        s.hold_slot(a, slot["slot_id"])
    assert e.value.code == "slot_taken"
    nxt = s.get_slots(a, "D01", "2026-11-17", "2026-11-30")["slots"]
    assert slot["slot_id"] not in {x["slot_id"] for x in nxt}


def test_reschedule_frees_old_slot(s):
    call = s.start_call("text")
    appts = s.find_appointments("98765 43210")
    assert appts, "demo patient seeded"
    old = appts[0]
    res = s.get_slots(call, old["doctor_id"], "2026-11-17", "2026-11-30")
    new = next(x for x in res["slots"] if x["slot_id"] != f"{old['doctor_id']}")
    hold = s.hold_slot(call, new["slot_id"])
    s.prepare_confirmation(call, "reschedule", hold_id=hold["hold_id"], appointment_id=old["appointment_id"])
    s.record_caller_reply(call, "theek hai")
    r = s.reschedule(call, old["appointment_id"], hold["hold_id"])
    assert r["appointment"]["date"] == new["date"] and r["appointment"]["time"] == new["time"]
    call2 = s.start_call("text")
    freed = s.get_slots(call2, old["doctor_id"], old["date"], old["date"], limit=3)
    # the old slot is free again (if it falls within the first three of that day it is returned)
    with s._conn() as c:
        assert not c.execute("SELECT 1 FROM slot_locks WHERE ref=? AND slot_id=?",
                             (old["appointment_id"], r["old_slot_released"])).fetchone()
    assert s.end_call(call, "agent_end")["outcome"] == "rescheduled"


def test_one_active_appointment_per_patient_doctor_and_family(s):
    call = s.start_call("text")
    confirmed_booking(s, call, name="Meera Shah", phone="9811111111")
    call2 = s.start_call("text")
    slot = s.get_slots(call2, "D01", "2026-11-17", "2026-11-30")["slots"][0]
    hold = s.hold_slot(call2, slot["slot_id"])
    s.prepare_confirmation(call2, "book", hold_id=hold["hold_id"], patient_name="Meera Shah", patient_phone="9811111111")
    s.record_caller_reply(call2, "yes")
    with pytest.raises(ToolError) as e:
        s.book(call2, hold["hold_id"])
    assert e.value.code == "existing_appointment"
    # same phone, different family member is fine
    s.prepare_confirmation(call2, "book", hold_id=hold["hold_id"], patient_name="Kiran Shah", patient_phone="9811111111")
    s.record_caller_reply(call2, "yes")
    assert s.book(call2, hold["hold_id"])["ok"]


def test_paediatrics_age_rule(s):
    call = s.start_call("text")
    slot = first_slot(s, call, "D05")
    hold = s.hold_slot(call, slot["slot_id"])
    with pytest.raises(ToolError) as e:
        s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Tia", patient_phone="9822200000")
    assert e.value.code == "need_age"
    with pytest.raises(ToolError) as e:
        s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Tia", patient_phone="9822200000",
                               patient_age=16)
    assert e.value.code == "too_old_for_paediatrics"
    assert s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Tia",
                                  patient_phone="9822200000", patient_age=15)["ok"]


def test_emergency_blocks_booking(s):
    call = s.start_call("text")
    slot = first_slot(s, call)
    hold = s.hold_slot(call, slot["slot_id"])
    s.transfer_to_human(call, "emergency", "chest pain")
    with pytest.raises(ToolError) as e:
        s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="A B", patient_phone="9876501234")
    assert e.value.code == "emergency_call"
    rec = s.end_call(call, "agent_end")
    assert rec["outcome"] == "transferred" and rec["transfer_reason"] == "emergency"
    assert s.dashboard()["transfers_waiting"][0]["reason"] == "emergency"


def test_one_booking_per_call_prompt_injection(s):
    call = s.start_call("text")
    confirmed_booking(s, call, doc="D01", name="Asha Verma")
    slot = first_slot(s, call, "D09")
    hold = s.hold_slot(call, slot["slot_id"])
    s.prepare_confirmation(call, "book", hold_id=hold["hold_id"], patient_name="Asha Verma", patient_phone="9876501234")
    s.record_caller_reply(call, "yes")
    with pytest.raises(ToolError) as e:
        s.book(call, hold["hold_id"])
    assert e.value.code == "booking_limit"


@pytest.mark.parametrize("text,verdict", [
    ("yes", "yes"), ("Yes, yes.", "yes"), ("haan ji bilkul", "yes"), ("हाँ", "yes"), ("theek hai", "yes"),
    ("yes please go ahead", "yes"), ("Yes, that’s correct. Please book it.", "yes"),
    ("Yes, please book it. All set. Thank you!", "yes"), ("haan ji, kar dijiye", "yes"), ("ok what time is it", "unclear"),
    ("Yes but make it Friday instead", "no"), ("no", "no"), ("No, 43120, not 43210", "no"), ("nahi", "no"),
    ("wait, Friday instead", "no"), ("what is the fee", "unclear"), ("", "unclear"),
])
def test_confirmation_classifier(text, verdict):
    assert classify_confirmation(text) == verdict


def test_privacy_helpers():
    assert mask_text("my number is 98765 43210") == "my number is ******3210"
    assert normalize_phone("+91 98765-43210") == "9876543210"
    assert normalize_phone("९८७६५४३२१०") == "9876543210"
    assert d.spoken_phone("9876543210") == "9 8 7 6 5, 4 3 2 1 0"
    assert d.spoken_time("17:30") == "half past five in the evening"
    assert d.spoken_time("09:45") == "quarter to ten in the morning"
    assert d.spoken_date(date(2026, 11, 20)) == "Friday, 20 November"


def test_closed_day_returns_next_available_and_alternatives(s):
    call = s.start_call("text")
    r = s.get_slots(call, "D09", "this sunday")
    assert r["closed_days"][0]["reason"] == "opd_closed_sunday"
    assert r["next_available"] and r["next_available"][0]["date"] == "2026-11-23"  # Monday
    s.hold_slot(call, r["next_available"][0]["slot_id"])  # offered, so it can be held
    r = s.get_slots(call, "D03", "19 November")
    assert any(o["doctor_id"] == "D04" for o in r["same_department_doctors"])
    assert all(x["date"] > "2026-11-21" for x in r["next_available"])


def test_relative_dates_in_get_slots(s):
    call = s.start_call("text")
    assert all(x["date"] == "2026-11-20" for x in s.get_slots(call, "D02", "friday", part_of_day="evening")["slots"])
    assert all(x["date"] == "2026-11-18" for x in s.get_slots(call, "D09", "parso")["slots"])


def test_lapsed_hold_is_retaken_if_free_else_slot_taken(s):
    a, b = s.start_call("text"), s.start_call("text")
    slot = first_slot(s, a)
    hold = s.hold_slot(a, slot["slot_id"])
    with s.tx() as c:  # simulate 120 s passing
        c.execute("UPDATE slot_locks SET expires_at=0 WHERE slot_id=?", (slot["slot_id"],))
    assert s.prepare_confirmation(a, "book", hold_id=hold["hold_id"], patient_name="Asha Rao",
                                  patient_phone="9876501234")["ok"]  # still free: re-taken
    with s.tx() as c:
        c.execute("UPDATE slot_locks SET expires_at=0 WHERE slot_id=?", (slot["slot_id"],))
    s.get_slots(b, "D01", "2026-11-17", "2026-11-30")
    s.hold_slot(b, slot["slot_id"])  # another caller takes it after the lapse
    s.record_caller_reply(a, "yes")
    with pytest.raises(ToolError) as e:
        s.book(a, hold["hold_id"])
    assert e.value.code == "slot_taken"
