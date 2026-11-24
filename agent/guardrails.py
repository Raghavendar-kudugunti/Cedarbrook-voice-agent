"""Deterministic safety checks that run on every caller utterance *before* the LLM.

Emergencies must never depend on the model noticing them, so detection is keyword-based over
English, romanised Hindi/Marathi (Hinglish) and Devanagari. False positives are acceptable
(a transfer to a human); false negatives are not. The LLM also has transfer_to_human as a backstop.
"""
from __future__ import annotations

import re

FIXED = {
    "greeting": {
        "en": "Namaste, Cedarbrook Hospital appointments. I'm an automated assistant. How can I help you?",
    },
    "emergency": {
        "en": "This sounds like an emergency. Please call 108 or come to Cedarbrook Emergency right away. "
              "I'm connecting you to our staff now.",
        "hi": "Yeh emergency lag rahi hai. Kripya turant 108 par call karein ya Cedarbrook Emergency aa jaiye. "
              "Main aapko abhi hamare staff se jod rahi hoon.",
        "mr": "Hi emergency vaatte. Krupaya lagech 108 la call kara kiva Cedarbrook Emergency madhye ya. "
              "Mi tumhala aata aamchya staff shi jodte.",
    },
    "advice": {
        "en": "I can't give medical advice, but I can book you an appointment with a doctor. Would you like that?",
        "hi": "Main medical salah nahi de sakti, lekin main aapke liye doctor ka appointment book kar sakti hoon. "
              "Kya aap chahenge?",
        "mr": "Mi vaidyakiya salla deu shakat nahi, pan mi tumchi doctoranchi appointment book karu shakte. "
              "Tumhala chalel ka?",
    },
    "transfer": {
        "en": "I'll connect you to our front-desk team. Please stay on the line.",
        "hi": "Main aapko hamari front-desk team se jod rahi hoon. Kripya line par bane rahiye.",
        "mr": "Mi tumhala aamchya front-desk team shi jodte. Krupaya line var raha.",
    },
    "technical_problem": {
        "en": "Sorry, I'm having a technical problem on my side. I'll connect you to our front-desk team. "
              "Please stay on the line.",
        "hi": "Maaf kijiye, mere taraf technical dikkat aa rahi hai. Main aapko front-desk team se jod rahi hoon. "
              "Kripya line par bane rahiye.",
        "mr": "Maaf kara, majhyakade technical adchan yet aahe. Mi tumhala front-desk team shi jodte. "
              "Krupaya line var raha.",
    },
    "busy_retry": {
        "en": "Sorry, the line is a little busy. Could you please say that once more?",
        "hi": "Maaf kijiye, line thodi busy hai. Kya aap ek baar phir se bata sakte hain?",
        "mr": "Maaf kara, line thodi busy aahe. Krupaya punha ekda sangal ka?",
    },
    "still_there": {
        "en": "Are you still there?",
        "hi": "Kya aap line par hain?",
        "mr": "Tumhi line var aahat ka?",
    },
    "goodbye_silence": {
        "en": "I haven't heard anything, so I'll end the call now. Please call back any time. Goodbye.",
        "hi": "Mujhe kuch sunai nahi diya, isliye main call samaapt kar rahi hoon. Kabhi bhi dobara call kijiye. Namaste.",
        "mr": "Mala kahi aikoo aale nahi, mhanun mi call band karte. Kadhihi parat call kara. Namaskar.",
    },
    "max_duration": {
        "en": "We've been on the line for a while. Would you like me to connect you to our front-desk team?",
        "hi": "Hum kaafi der se baat kar rahe hain. Kya main aapko front-desk team se jod doon?",
        "mr": "Aapan barach vel bolat aahot. Mi tumhala front-desk team shi jodu ka?",
    },
}


def fixed(key: str, lang: str) -> str:
    msgs = FIXED[key]
    if lang.startswith("mr"):
        return msgs.get("mr", msgs["en"])
    if lang.startswith("hi"):
        return msgs.get("hi", msgs["en"])
    return msgs["en"]


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


EMERGENCY = _rx(
    # chest pain / heart attack
    r"chest\s*pain", r"pain\s+in\s+(?:my|his|her|the)?\s*chest", r"heart\s*attack", r"cardiac\s*arrest",
    r"(?:seene|seena|sine|chhati|chhaati|chaati|chati|chhatit|chatit)\s*(?:me|mein|mai|madhe|madhye|t)?\s*"
    r"(?:dard|dukh|dukhat|pain|jalan|bhaari|bhari)",
    r"सीने\s*में\s*दर्द", r"छाती\s*(?:में|त)?\s*(?:दर्द|दुखत)", r"दिल\s*का\s*दौरा", r"हार्ट\s*अटैक",
    # breathing
    r"(?:can'?t|cannot|can not|not able to|unable to|difficulty|trouble|hard to)\s*breath", r"short(?:ness)?\s+of\s+breath",
    r"not\s+breathing", r"stopped\s+breathing", r"breathless", r"choking",
    r"saa?ns?\s*(?:nahi|nahin|nai|lene\s*me|lena|phool|ruk|band|ki\s*taklif|ki\s*takleef|ghet\s*nahi|gheta\s*yet\s*nahi)",
    r"सांस\s*(?:नहीं|लेने\s*में|फूल|रुक)", r"श्वास\s*(?:घेता\s*येत\s*नाही|घेण्यास)", r"दम\s*लाग",
    # unconscious
    r"unconscious", r"passed\s*out", r"fainted", r"not\s+responding", r"won'?t\s+wake", r"collapsed",
    r"behosh", r"be\s*hosh", r"hosh\s*(?:nahi|nahin|me\s*nahi)", r"shuddh\s*(?:nahi|harpali)", r"beshuddh",
    r"बेहोश", r"होश\s*नहीं", r"बेशुद्ध",
    # bleeding
    r"heavy\s+bleeding", r"bleeding\s+(?:a\s+lot|heavily|badly|won'?t\s+stop|nonstop)", r"lot\s+of\s+blood",
    r"khoon\s*(?:bahut|ruk\s*nahi|beh|nikal\s*raha|band\s*nahi)", r"bahut\s*khoon", r"rakt\s*(?:stav|jaat)",
    r"खून\s*(?:बह|रुक\s*नहीं|बहुत)", r"बहुत\s*खून", r"रक्तस्त्राव",
    # stroke
    r"stroke", r"face\s+(?:is\s+)?droop", r"slurred\s+speech", r"can'?t\s+(?:move|feel)\s+(?:his|her|my)\s+(?:arm|leg|side)",
    r"paralys", r"lakwa", r"laqwa", r"लकवा", r"पक्षाघात",
    # seizure
    r"seizure", r"convulsion", r"\bfits?\b(?!\s+(?:me|you|in|my|your))", r"mirgi", r"daura\s*(?:pad|aa)", r"jhatke",
    r"मिर्गी", r"दौरा\s*पड़", r"झटके", r"फिट\s*आ",
    # explicit / other acute
    r"emergency", r"इमरजेंसी", r"आपातकाल", r"ambulance", r"एम्बुलेंस", r"\baccident\b", r"एक्सीडेंट", r"दुर्घटना",
    r"poison", r"overdose", r"zeher", r"जहर", r"suicid", r"kill\s+(?:my|him|her)self", r"burn(?:ed|t)\s+badly",
    r"severe\s+(?:allergic|burn)", r"anaphyla",
)

# "Book an appointment for my chest pain next week" is still treated as an emergency (safe side),
# except when the caller explicitly frames it as old / routine follow-up.
ROUTINE_FRAMING = _rx(r"follow[\s-]*up", r"check[\s-]*up\s+(?:after|for)\s+(?:my|his|her)\s+(?:old|past|last)",
                      r"last\s+(?:year|month)", r"history\s+of", r"pichhle\s+saal")

MEDICAL_ADVICE = _rx(
    r"what\s+(?:medicine|medication|tablet|pill|drug|dose|dosage)", r"which\s+(?:medicine|medication|tablet|pill|drug)",
    r"(?:should|can)\s+i\s+(?:take|give|have)\s+(?:a\s+|an\s+|some\s+)?(?:paracetamol|crocin|dolo|ibuprofen|aspirin|"
    r"antibiotic|medicine|tablet|syrup|pill)",
    r"how\s+(?:much|many)\s+(?:mg|tablets?|pills?|ml|doses?)", r"is\s+(?:it|this|that)\s+serious",
    r"what\s+(?:is\s+wrong|could\s+it\s+be|do\s+i\s+have)", r"diagnos", r"home\s+remed",
    r"kaun\s*(?:si|sa)\s*(?:dawa|dawai|davai|goli|medicine|tablet)", r"(?:dawa|dawai|davai|goli|medicine)\s*"
    r"(?:kya|konsi|kaunsi|bata|leni|lu|loon|du|doon|de\s*sakte)", r"kya\s+(?:lena|khana|du|doon)\s+chahiye",
    r"(?:dawa|dawai|davai|goli|aushadh)\s*(?:konti|kuthli|ghyavi|ghyaychi)", r"kitni\s+(?:goli|dose)",
    r"कौन\s*सी\s*दवा", r"दवा\s*(?:क्या|बताइए|बताओ)", r"कोणते\s*औषध", r"कोणती\s*गोळी", r"serious\s+(?:hai|toh\s+nahi)",
)

HUMAN_REQUEST = _rx(
    r"(?:talk|speak|connect|transfer)\s+(?:me\s+)?(?:to|with)\s+(?:a\s+|an\s+|the\s+|some\s*)?(?:human|person|someone|"
    r"somebody|staff|receptionist|operator|agent|real|actual|front\s*desk|manager)",
    r"(?:real|actual|live)\s+(?:person|human)", r"\bhuman\b", r"\boperator\b", r"front\s*desk",
    r"(?:insaan|insan|aadmi|manushya|staff|receptionist|kisi\s+(?:se|vyakti))\s*(?:se|shi|la)?\s*(?:baat|bolna|bol|"
    r"jodo|connect|transfer)",
    r"इंसान\s*से\s*बात", r"किसी\s*व्यक्ति\s*से", r"स्टाफ\s*से\s*बात", r"माणसाशी\s*बोल",
)

COMPLAINT_OR_BILLING = _rx(
    r"complain", r"complaint", r"shikayat", r"शिकायत", r"तक्रार", r"takrar", r"refund", r"bill(?:ing)?\b", r"invoice",
    r"insurance", r"payment\s+(?:issue|problem)", r"overcharg", r"बिल", r"रिफंड", r"report\s+(?:results?|collect)",
    # 'discharge' alone is also a symptom word ("white discharge"): only hospital discharge counts here
    r"lab\s+report", r"discharge\s+(?:summary|papers?|card|form|process|date)", r"discharged", r"get\s+discharge",
    r"admission", r"admit",
)


def detect_emergency(text: str) -> bool:
    if not text:
        return False
    if not EMERGENCY.search(text):
        return False
    # Routine framing only suppresses symptom words, never an explicit 'emergency'/'ambulance'.
    if ROUTINE_FRAMING.search(text) and not re.search(r"emergency|ambulance|right\s+now|abhi|अभी|now", text, re.I):
        return False
    return True


def detect_medical_advice(text: str) -> bool:
    return bool(text and MEDICAL_ADVICE.search(text))


def detect_human_request(text: str) -> bool:
    return bool(text and HUMAN_REQUEST.search(text))


def detect_complaint_or_billing(text: str) -> bool:
    return bool(text and COMPLAINT_OR_BILLING.search(text))


HINDI_ROMAN = set("""
mujhe muje mujhko mera meri mere hai hain ho tha thi kya kyu kyun kaise kab kahan nahi nahin haan han ji chahiye chaiye
karna karo kariye kar dijiye dijiye aap aapka aapke hum hamara ko ke ki ka se mein me par wala wali abhi kal parso
subah shaam sham raat dopahar doctor ke saath sath appointment lena hoga theek thik accha acha bhai papa mummy beta beti
batao bataiye dikhana dikhaana wahi woh yeh ye isko usko lekin aur ya bas
""".split())
MARATHI_ROMAN = set("""
mala majha majhi tumhi tumcha aahe ahe aahet nahi hoy kara karaycha pahije havay hava udya parva sakali sandhyakali
kiti kuthe kasa kashi ka mhanje aani pan
""".split())


def detect_language(text: str, stt_language: str | None = None) -> str:
    """Return 'en', 'hi' or 'mr' (Hinglish counts as 'hi'). Used to pick the fixed-message language."""
    if not text:
        return (stt_language or "en")[:2]
    if re.search(r"[ऀ-ॿ]", text):
        if re.search(r"(आहे|नाही|मला|तुम्ही|करायचे|पाहिजे|हवी|हवा|उद्या|परवा|कृपया\s*करा)", text):
            return "mr"
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return "en"
    mr = sum(w in MARATHI_ROMAN for w in words)
    hi = sum(w in HINDI_ROMAN for w in words)
    if mr >= 2 and mr >= hi:
        return "mr"
    if hi >= 2 or (hi >= 1 and len(words) <= 3):
        return "hi"
    if stt_language and stt_language[:2] in {"hi", "mr"}:
        return stt_language[:2]
    return "en"


DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9",
    "shunya": "0", "shoonya": "0", "ek": "1", "do": "2", "teen": "3", "char": "4", "chaar": "4", "paanch": "5",
    "panch": "5", "chhe": "6", "che": "6", "chah": "6", "saat": "7", "sat": "7", "aath": "8", "ath": "8", "nau": "9",
    "शून्य": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5", "छह": "6", "छः": "6",
    "सात": "7", "आठ": "8", "नौ": "9", "शून्य़": "0", "दोन": "2", "सहा": "6", "नऊ": "9",
}
_MULT = {"double": 2, "triple": 3}


def spoken_digits(text: str) -> str:
    """Digits a caller spoke, as a digit string: '98765 43210', 'nine eight seven...', 'double five' all work."""
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    out, mult = [], 1
    for tok in re.findall(r"\d+|[^\W\d_]+", text.lower()):
        if tok.isdigit():
            out.append(tok[0] * mult + tok[1:])
            mult = 1
        elif tok in _MULT:
            mult = _MULT[tok]
        elif tok in DIGIT_WORDS:
            out.append(DIGIT_WORDS[tok] * mult)
            mult = 1
    return "".join(out)


def phone_was_heard(phone: str, heard: str) -> bool:
    """True if the phone's 10 digits appear in what the caller said. `heard` is the raw text of every caller
    utterance in this call (kept in memory only), so a number given in two halves on separate turns still counts."""
    digits = "".join(c for c in str(phone).translate(str.maketrans("०१२३४५६७८९", "0123456789")) if c.isdigit())
    digits = digits[-10:]
    if len(digits) != 10:
        return False
    numerals = re.sub(r"\D", "", heard.translate(str.maketrans("०१२३४५६७८९", "0123456789")))
    words = spoken_digits(heard)  # also counts 'nine eight ...'; common words like 'do'/'one' may add noise
    return digits in numerals or digits in words


PERSON_WORDS = _rx(r"\bhuman\b", r"\bperson\b", r"\bsomeone\b", r"\bsomebody\b", r"\bstaff\b", r"\bdesk\b",
                   r"receptionist", r"\boperator\b", r"\bmanager\b", r"\bagent\b", r"insaan", r"insan", r"aadmi",
                   r"manushya", r"vyakti", r"baat\s+kar", r"इंसान", r"व्यक्ति", r"स्टाफ", r"माणूस", r"माणसा")


def mentions_person(text: str) -> bool:
    """Broad check that the caller asked for a person at some point (backstop for the LLM's transfer choice)."""
    return bool(text and PERSON_WORDS.search(text))
