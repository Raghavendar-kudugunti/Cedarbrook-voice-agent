import pytest

from agent.guardrails import (detect_complaint_or_billing, detect_emergency, detect_human_request, detect_language,
                              detect_medical_advice, fixed)

EMERGENCIES = [
    "Mere papa ko abhi chest pain ho raha hai",
    "my father has chest pain right now",
    "meri maa ko saans lene mein taklif ho rahi hai",
    "he is not breathing properly, can't breathe",
    "she fainted and is unconscious",
    "papa behosh ho gaye",
    "there is heavy bleeding from his head",
    "bahut khoon nikal raha hai",
    "I think she is having a stroke, her face is drooping",
    "bacche ko mirgi ka daura pad raha hai",
    "my son is having a seizure",
    "this is an emergency",
    "मेरे पापा के सीने में दर्द हो रहा है",
    "seene mein dard ho raha hai",
    "chhatit dukhat aahe",
    "heart attack aaya hai lagta hai",
]
NOT_EMERGENCIES = [
    "I want to book Dr. Anil Mehta next available morning",
    "Mujhe skin doctor chahiye, Friday shaam ko",
    "Dr. Mehta ke saath appointment chahiye",
    "No no, Friday instead",
    "I want to cancel my appointment",
    "my phone number is 98765 43210",
    "Does this outfit matter for the appointment",
    "follow-up for my chest pain from last year",
]


@pytest.mark.parametrize("text", EMERGENCIES)
def test_emergencies_detected(text):
    assert detect_emergency(text)


@pytest.mark.parametrize("text", NOT_EMERGENCIES)
def test_routine_not_emergency(text):
    assert not detect_emergency(text)


@pytest.mark.parametrize("text", [
    "What medicine should I take for fever?", "bukhar ke liye kaun si dawai lu?", "Is this serious?",
    "How many tablets of paracetamol can I take", "कौन सी दवा लूं",
])
def test_medical_advice(text):
    assert detect_medical_advice(text)


def test_medical_advice_not_triggered_by_booking():
    assert not detect_medical_advice("I need a General Medicine appointment for fever")


@pytest.mark.parametrize("text", ["I want to talk to a human", "connect me to a real person",
                                  "kisi insaan se baat karao", "front desk please", "transfer me to staff"])
def test_human_request(text):
    assert detect_human_request(text)


def test_complaint_billing():
    assert detect_complaint_or_billing("I have a complaint about the doctor")
    assert detect_complaint_or_billing("question about my bill")
    assert not detect_complaint_or_billing("book a cardiologist")


@pytest.mark.parametrize("text,lang", [
    ("Mujhe skin doctor chahiye, Friday shaam ko", "hi"), ("I want to book a doctor", "en"),
    ("मला उद्या डॉक्टर पाहिजे", "mr"), ("मुझे डॉक्टर चाहिए", "hi"), ("mala udya appointment pahije", "mr"),
])
def test_language(text, lang):
    assert detect_language(text) == lang


def test_fixed_messages_exact():
    assert fixed("emergency", "en") == ("This sounds like an emergency. Please call 108 or come to Cedarbrook "
                                        "Emergency right away. I'm connecting you to our staff now.")
    assert fixed("transfer", "en") == "I'll connect you to our front-desk team. Please stay on the line."
    assert "108" in fixed("emergency", "hi")


from agent.guardrails import phone_was_heard, spoken_digits  # noqa: E402


def test_spoken_digits_and_phone_heard():
    assert spoken_digits("my number is 98765 43210") == "9876543210"
    assert spoken_digits("nine eight seven six five, four three two one zero") == "9876543210"
    assert spoken_digits("nau aath saat chhe paanch double four") == "98765" + "44"
    assert spoken_digits("९८७६५४३२१०") == "9876543210"
    assert phone_was_heard("9876543210", " 98765 " + " 43210")       # given in two turns
    assert phone_was_heard("9876543210", "98765 do you need the rest? 43210")
    assert phone_was_heard("9876543210", "nine eight seven six five four three two one zero")
    assert not phone_was_heard("9876543210", "my number is 9876501234")  # invented number refused
    assert not phone_was_heard("9876543210", "")
