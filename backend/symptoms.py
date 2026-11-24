"""Symptom words -> which OPD doctor to see, for callers who don't know department names.

Callers say "ghutnon mein dard", "kamar dukhti hai", "skin pe khujli", not "orthopaedics". This module
routes those words to a department and gives the plain-language name to say back ("haddi aur jodon ke
doctor"). It is routing only, never a diagnosis: no condition names, no advice. Emergencies (chest pain,
breathing, unconscious...) are caught earlier by agent/guardrails.py and never reach this.
"""
from __future__ import annotations

import re

# Plain names to say to the caller (en / hi in Devanagari / mr in Devanagari).
PLAIN = {
    "Orthopaedics": {"en": "a bone and joint doctor", "hi": "हड्डी और जोड़ों के डॉक्टर", "mr": "हाडांचे आणि सांध्यांचे डॉक्टर"},
    "Dermatology": {"en": "a skin doctor", "hi": "त्वचा यानी स्किन के डॉक्टर", "mr": "त्वचेचे डॉक्टर"},
    "Cardiology": {"en": "a heart doctor", "hi": "दिल के डॉक्टर", "mr": "हृदयाचे डॉक्टर"},
    "ENT": {"en": "an ear, nose and throat doctor", "hi": "कान, नाक और गले के डॉक्टर", "mr": "कान, नाक, घसा डॉक्टर"},
    "Gynaecology": {"en": "a women's doctor", "hi": "महिलाओं की डॉक्टर", "mr": "स्त्री रोग डॉक्टर"},
    "Paediatrics": {"en": "a children's doctor", "hi": "बच्चों के डॉक्टर", "mr": "लहान मुलांचे डॉक्टर"},
    "General Medicine": {"en": "a general physician, our everyday doctor", "hi": "जनरल फिजिशियन, यानी आम बीमारियों के डॉक्टर",
                         "mr": "जनरल फिजिशियन, म्हणजे नेहमीच्या आजारांचे डॉक्टर"},
}

# (department, patterns). Order matters only for ties at the same position in the sentence.
_RULES: list[tuple[str, list[str]]] = [
    ("Orthopaedics", [
        r"knee", r"back\s*(?:pain|ache)", r"backache", r"lower\s+back", r"(?:my|his|her)\s+back\s+(?:hurts|is)",
        r"waist", r"spine", r"slip(?:ped)?\s*disc", r"joint", r"shoulder", r"neck\s*(?:pain|ache|stiff)", r"\bhips?\b",
        r"ankle", r"wrist", r"elbow", r"heel", r"sprain", r"fracture", r"broken\s+(?:arm|leg|bone)", r"bone",
        r"arthritis", r"leg\s*pain", r"(?:legs?|feet|foot)\s+(?:hurt|pain)", r"ortho",
        # Hinglish
        r"ghut(?:na|ne|no|non|nu)", r"kamar", r"kamr", r"jod(?:o|on|on\s+me|\s+me)?\s*(?:me|mein)?\s*dard", r"jodon",
        r"haddi", r"haddiyo", r"kandh(?:a|e)", r"gardan\s*(?:me|mein)?\s*dard", r"reedh", r"ridh", r"peeth\s*(?:me|mein)?\s*dard",
        r"pith\s*dard", r"moch", r"gathiya", r"pair\s*(?:me|mein)?\s*dard", r"pairon", r"\bedi\b",
        # Marathi (roman)
        r"gudgh", r"kambar", r"sandh(?:e|yat|yan)", r"paath\s*dukh", r"haad",
        # Devanagari
        r"घुटन", r"घुटने", r"कमर", r"जोड़ों", r"जोडों", r"जोड़\s*में", r"हड्डी", r"पीठ", r"कंधे", r"कंधा", r"गर्दन",
        r"रीढ़", r"मोच", r"गठिया", r"पैर\s*में\s*दर्द", r"पैरों", r"गुडघ", r"कंबर", r"सांध", r"(?<![ऀ-ॿ])हाड", r"पाठ\s*दुख",
    ]),
    ("Dermatology", [
        r"skin", r"rash", r"itch", r"pimple", r"acne", r"eczema", r"psoriasis", r"white\s+patch", r"fungal",
        r"ringworm", r"hair.{0,15}(?:fall|loss|thinning)", r"dandruff", r"bald", r"nail", r"mole", r"wart", r"pigment",
        r"dark\s+spots?", r"derma",
        r"khujl", r"kharish", r"khaj", r"daane", r"dane\s+(?:nikal|ho)", r"muh[aā]se", r"chamdi", r"chamadi",
        r"safed\s*da+g", r"baal.{0,15}(?:jhad|gir|toot)", r"daad", r"twacha", r"tvacha", r"chakatte",
        r"खुजली", r"खाज", r"दाने", r"मुंहासे", r"मुहांसे", r"त्वचा", r"चमड़ी", r"बाल\s*(?:झड़|गिर)", r"दाद", r"चकत्ते",
        r"सफेद\s*दाग", r"त्वचे", r"केस\s*गळ",
    ]),
    ("Cardiology", [
        r"heart", r"palpitation", r"heartbeat", r"pulse\s+(?:fast|high)", r"cardio",
        r"dil\s*(?:ki|ka|ke)?\s*(?:dhadkan|bimari|problem|doctor)", r"dhadkan", r"dhad\s*dhad",
        r"दिल\s", r"धड़कन", r"धडकन", r"हृदय", r"छातीत\s*धडधड",
    ]),
    ("ENT", [
        r"\bears?\b", r"hearing", r"deaf", r"tinnitus", r"\bnose", r"sinus", r"nosebleed", r"throat", r"tonsil",
        r"voice\s+(?:gone|hoarse|lost)", r"hoarse", r"snor",
        r"kaan", r"kan\s+(?:me|mein|dukh|dard|beh)", r"sunai\s*(?:nahi|kam)", r"naak", r"gala\s*(?:kharab|dard|kharash|baith)",
        r"gale\s*(?:me|mein)", r"awaaz\s*(?:baith|nahi|kharab)", r"tonsil", r"ghasa",
        r"(?<![ऀ-ॿ])कान", r"सुनाई", r"नाक", r"गला", r"गले", r"टॉन्सिल", r"आवाज़\s*बैठ", r"घसा",
    ]),
    ("Gynaecology", [
        r"period", r"menstru", r"pregnan", r"white\s+discharge", r"vaginal", r"pcos", r"pcod", r"uterus", r"womb",
        r"menopause", r"gyna", r"gyne", r"lady\s+doctor", r"female\s+doctor",
        r"mahin(?:a|e)\s*(?:nahi|time|der|jyada|kam|ki|ka)", r"masik", r"maasik", r"garbh", r"pet\s+se\s+hai",
        r"bachcha\s+(?:hone|honewala)", r"safed\s+paani", r"\bpali\b", r"garbhavati",
        r"पीरियड", r"मासिक", r"माहवारी", r"गर्भ", r"प्रेगनेंसी", r"सफेद\s*पानी", r"पाळी", r"गरोदर",
    ]),
    ("General Medicine", [
        r"fever", r"temperature", r"cold\b", r"cough", r"flu\b", r"weak(?:ness)?", r"tired", r"fatigue", r"stomach",
        r"tummy", r"acidity", r"gas\b", r"vomit", r"nausea", r"loose\s+motion", r"diarr?h", r"constipat", r"headache",
        r"head\s+(?:hurts|pain)", r"migraine", r"dizz", r"giddy", r"diabet", r"sugar", r"thyroid", r"blood\s+pressure",
        r"\bbp\b", r"body\s*(?:ache|pain)", r"check[\s-]*up", r"general", r"physician", r"infection", r"urine", r"burning",
        r"weight", r"appetite", r"sleep",
        r"bukh?aa?r", r"taap", r"sardi", r"zukam", r"jukam", r"khans?i", r"khokla", r"kamzor", r"thakan", r"thakaan",
        r"pet\s*(?:me|mein|t)?\s*(?:dard|dukh|kharab|gas|jalan)", r"pet\s+dukh", r"acidity", r"\bulti", r"\bdast\b",
        r"sir\s*(?:me|mein)?\s*dard", r"sar\s*dard", r"dokedukh", r"chakkar", r"badan\s*dard", r"sharir\s*dard",
        r"peshab\s*(?:me|mein)?\s*jalan", r"neend",
        r"बुखार", r"(?<![ऀ-ॿ])ताप", r"सर्दी", r"जुकाम", r"खांसी", r"खोकला", r"कमज़ोरी", r"कमजोरी", r"थकान", r"पेट", r"उल्टी",
        r"दस्त", r"सिर\s*(?:में\s*)?दर्द", r"सर\s*दर्द", r"डोकेदुख", r"चक्कर", r"शुगर", r"मधुमेह", r"थायरॉइड", r"बीपी",
        r"बदन\s*दर्द", r"पेशाब", r"नींद",
    ]),
]

# Things this OPD line does not have a doctor for: say so plainly and offer the front desk.
_NOT_HERE = {
    "eyes": [r"\beyes?\b", r"vision", r"spectacles", r"specs\b", r"aankh", r"ankh\b", r"nazar", r"dolya", r"आंख", r"आँख",
             r"नज़र", r"डोळ"],
    "teeth": [r"teeth", r"tooth", r"dental", r"dentist", r"daant", r"dant\b", r"daat", r"दांत", r"दाँत", r"दात"],
    "mental health": [r"depress", r"anxiety", r"stress", r"tension\s+(?:bahut|hai)", r"psychiatr", r"mental"],
}

# The patient may be a child: ask the age (under 16 -> children's doctor, 16+ -> the problem's department).
_CHILD = re.compile("|".join(f"(?:{p})" for p in [
    r"\bbaby\b", r"infant", r"newborn", r"toddler", r"\bchild\b", r"\bkids?\b", r"my\s+(?:son|daughter)",
    r"bachch?(?:a|e|i|on)\b", r"\bbete\s+(?:ko|ka|ki|ke)", r"\bbeti\s+(?:ko|ka|ki|ke)", r"mer(?:e|a)\s+bet(?:e|a)\b",
    r"meri\s+beti", r"navjaat", r"mulga", r"mulgi", r"बच्च", r"शिशु", r"नवजात", r"बेटे\s*(?:को|का)", r"बेटी\s*(?:को|का)",
    r"मुलग", r"बाळ",
]), re.IGNORECASE)

_COMPILED = [(dept, re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE)) for dept, pats in _RULES]
_NOT_HERE_RX = {k: re.compile("|".join(f"(?:{p})" for p in v), re.IGNORECASE) for k, v in _NOT_HERE.items()}


def triage(text: str | None) -> dict:
    """Route what the caller said to OPD departments.

    Returns {"departments", "plain", "heard", "not_available", "maybe_child"}. More than one department
    means: ask the caller one simple question (which problem matters most today). None means: General
    Medicine is the safe first stop. maybe_child: ask the age first (under 16 -> Paediatrics).
    """
    if not text:
        return {"departments": [], "plain": {}, "heard": [], "not_available": [], "maybe_child": False}
    hits: list[tuple[int, str, str]] = []
    for dept, rx in _COMPILED:
        m = rx.search(text)
        if m:
            hits.append((m.start(), dept, m.group(0)))
    hits.sort()
    depts = [h[1] for h in hits]
    # "check-up"/"general" alongside a specific body part: the specific one wins.
    if len(depts) > 1 and "General Medicine" in depts:
        specific = [h for h in hits if h[1] != "General Medicine"]
        general_words = [h for h in hits if h[1] == "General Medicine"]
        if all(re.fullmatch(r"(?i)check[\s-]*up|general|physician|weak(?:ness)?|tired", g[2]) for g in general_words):
            depts = [h[1] for h in specific]
    depts = list(dict.fromkeys(depts))
    not_here = [k for k, rx in _NOT_HERE_RX.items() if rx.search(text)]
    return {
        "departments": depts,
        "plain": {dep: PLAIN[dep] for dep in depts},
        "heard": [h[2] for h in hits],
        "not_available": not_here,
        "maybe_child": bool(_CHILD.search(text)),
    }


def plain_name(department: str, lang: str = "en") -> str:
    return PLAIN.get(department, {}).get(lang) or PLAIN.get(department, {}).get("en", department)
