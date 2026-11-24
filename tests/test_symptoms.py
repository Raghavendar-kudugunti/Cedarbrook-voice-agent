"""Callers describe problems, not departments: symptom words must reach the right OPD doctor."""
import pytest

from agent import guardrails as g
from backend.api import search_doctors_tool
from backend.scheduling import Scheduler
from backend.symptoms import triage


@pytest.mark.parametrize("text, dept", [
    # English
    ("Hi, I've been having back pain for the last two days.", "Orthopaedics"),
    ("I have pain in my legs and knee", "Orthopaedics"),
    ("my shoulder hurts a lot", "Orthopaedics"),
    ("there is a rash and itching on my arm", "Dermatology"),
    ("my hair is falling out", "Dermatology"),
    ("my heartbeat feels very fast sometimes", "Cardiology"),
    ("my ear is paining", "ENT"),
    ("my periods are irregular", "Gynaecology"),
    ("white discharge problem", "Gynaecology"),
    ("I have fever and cough", "General Medicine"),
    ("stomach upset and vomiting", "General Medicine"),
    # Hinglish
    ("Beta, mere ghutnon mein dard hai. Doctor ko dikhana hai.", "Orthopaedics"),
    ("kamar dukhti hai", "Orthopaedics"),
    ("pair mein moch aa gayi", "Orthopaedics"),
    ("skin pe bahut khujli ho rahi hai", "Dermatology"),
    ("baal bahut jhad rahe hain", "Dermatology"),
    ("dil ki dhadkan tez rehti hai", "Cardiology"),
    ("kaan se paani aa raha hai", "ENT"),
    ("gala kharab hai", "ENT"),
    ("mahina time pe nahi aa raha", "Gynaecology"),
    ("pet mein dard aur ulti", "General Medicine"),
    ("sir dard aur chakkar", "General Medicine"),
    ("sugar check karwana hai", "General Medicine"),
    # Marathi (roman) and Devanagari
    ("majha gudgha dukhto", "Orthopaedics"),
    ("माझी कंबर दुखते", "Orthopaedics"),
    ("मेरे घुटनों में दर्द है", "Orthopaedics"),
    ("त्वचा पर खुजली है", "Dermatology"),
    ("बुखार और खांसी है", "General Medicine"),
])
def test_symptom_routes_to_department(text, dept):
    assert triage(text)["departments"][:1] == [dept]


@pytest.mark.parametrize("text", [
    "I need an appointment, early morning please", "consulting", "मुझे appointment दिला दो",
    "दुकान पर हूँ", "I'll come early tomorrow",
])
def test_no_false_symptom(text):
    assert triage(text)["departments"] == []


def test_plain_names_never_use_jargon():
    r = triage("ghutnon mein dard")
    assert "bone and joint" in r["plain"]["Orthopaedics"]["en"]
    assert "हड्डी" in r["plain"]["Orthopaedics"]["hi"]


def test_eye_and_teeth_are_not_on_this_line():
    assert triage("aankh mein dard hai")["not_available"] == ["eyes"]
    assert triage("I have a toothache")["not_available"] == ["teeth"]


def test_child_flags_age_question():
    r = triage("Mere bete ke kaan mein dard hai")
    assert r["maybe_child"] and r["departments"] == ["ENT"]
    assert not triage("Beta, mere ghutnon mein dard hai")["maybe_child"]  # 'beta' as a form of address


def test_search_doctors_by_symptom(tmp_path):
    s = Scheduler(str(tmp_path / "t.db"))
    res = search_doctors_tool(s, {"symptom": "kamar mein dard hai"})
    assert {doc["id"] for doc in res["doctors"]} == {"D03", "D04"}
    assert res["departments"][0]["department"] == "Orthopaedics"
    child = search_doctors_tool(s, {"symptom": "mere bete ke kaan mein dard"})
    assert "ask_age_first" in child and {"D05", "D06"} <= {doc["id"] for doc in child["doctors"]}
    vague = search_doctors_tool(s, {"symptom": "tabiyat theek nahi"})
    assert {doc["department"] for doc in vague["doctors"]} == {"General Medicine"} and "guide" in vague
    eyes = search_doctors_tool(s, {"symptom": "aankh mein dard"})
    assert eyes["doctors"] == [] and eyes["not_available"] == ["eyes"]
    # naming a department still works the old way
    assert {doc["id"] for doc in search_doctors_tool(s, {"department": "skin"})["doctors"]} == {"D02", "D12"}


@pytest.mark.parametrize("text", [
    "white discharge problem", "Hi, I've been having back pain for the last two days.",
    "Beta, mere ghutnon mein dard hai. Doctor ko dikhana hai.", "skin pe khujli", "pet mein dard",
])
def test_symptoms_do_not_trigger_transfer_or_advice(text):
    assert not g.detect_complaint_or_billing(text)
    assert not g.detect_medical_advice(text)
    assert not g.detect_human_request(text)
    assert not g.detect_emergency(text)


def test_hospital_discharge_still_transfers():
    assert g.detect_complaint_or_billing("I need my discharge summary")
