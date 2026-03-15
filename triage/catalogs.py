from __future__ import annotations

QUESTION_CATALOG = {
    "Q_DURATION": {
        "family_tags": ["RESP", "GI", "UTI", "ABDOMINAL", "HEADACHE", "RASH"],
        "prompt": "How long has this been going on?",
    },
    "Q_TEMPERATURE": {
        "family_tags": ["RESP", "GI", "UTI", "ABDOMINAL"],
        "prompt": "Do you know how high the temperature has been?",
    },
    "Q_SHORTNESS_OF_BREATH": {
        "family_tags": ["RESP", "RASH"],
        "prompt": "Are you short of breath?",
    },
    "Q_CHEST_PAIN": {
        "family_tags": ["RESP"],
        "prompt": "Do you have any chest pain?",
    },
    "Q_VOMITING_FREQUENCY": {
        "family_tags": ["GI", "ABDOMINAL"],
        "prompt": "How often are you vomiting?",
    },
    "Q_FLUID_INTAKE": {
        "family_tags": ["GI"],
        "prompt": "Are you able to keep fluids down?",
    },
    "Q_DEHYDRATION_SIGNS": {
        "family_tags": ["GI"],
        "prompt": "Are you feeling dizzy or very dry?",
    },
    "Q_BLOOD_PRESENT": {
        "family_tags": ["GI", "UTI"],
        "prompt": "Have you noticed any blood?",
    },
    "Q_PAIN_SEVERITY": {
        "family_tags": ["GI", "UTI", "HEADACHE", "ABDOMINAL"],
        "prompt": "How severe is the pain?",
    },
    "Q_FLANK_PAIN": {
        "family_tags": ["UTI"],
        "prompt": "Do you have pain in your side or back?",
    },
    "Q_NECK_STIFFNESS": {
        "family_tags": ["HEADACHE"],
        "prompt": "Do you have neck stiffness?",
    },
    "Q_CONFUSION": {
        "family_tags": ["HEADACHE", "RESP", "GI"],
        "prompt": "Any confusion or unusual drowsiness?",
    },
    "Q_WEAKNESS": {
        "family_tags": ["HEADACHE"],
        "prompt": "Any new weakness or numbness?",
    },
    "Q_RASH_SPREAD": {
        "family_tags": ["RASH"],
        "prompt": "Is the rash spreading quickly?",
    },
    "Q_SWELLING": {
        "family_tags": ["RASH"],
        "prompt": "Any facial or throat swelling?",
    },
    "Q_COMORBIDITY": {
        "family_tags": ["RESP", "GI", "UTI", "HEADACHE", "RASH", "ABDOMINAL"],
        "prompt": "Do you have any major medical conditions?",
    },
    "Q_URINARY_FREQUENCY": {
        "family_tags": ["UTI"],
        "prompt": "Are you urinating more often than usual?",
    },
    "Q_BURNING_URINATION": {
        "family_tags": ["UTI"],
        "prompt": "Does it burn when you urinate?",
    },
    "Q_HEADACHE_SUDDEN": {
        "family_tags": ["HEADACHE"],
        "prompt": "Did the headache start suddenly?",
    },
    "Q_LOCALIZATION": {
        "family_tags": ["ABDOMINAL"],
        "prompt": "Where exactly is the abdominal pain?",
    },
}

PROTOCOL_CATALOG = {
    "RESP": {
        "protocol_id": "PROTO_RESP_BASIC",
        "red_flags": ["SYM_DYSPNEA", "SYM_CHEST_PAIN", "ANS_CONFUSION_YES"],
        "recommended_questions": [
            "Q_SHORTNESS_OF_BREATH",
            "Q_CHEST_PAIN",
            "Q_TEMPERATURE",
            "Q_DURATION",
            "Q_CONFUSION",
        ],
        "allowed_advice_packs": ["PACK_RESP_SUPPORT"],
    },
    "GI": {
        "protocol_id": "PROTO_GI_BASIC",
        "red_flags": ["ANS_BLOOD_YES", "ANS_DEHYDRATION_SEVERE", "ANS_PAIN_SEVERE"],
        "recommended_questions": [
            "Q_VOMITING_FREQUENCY",
            "Q_FLUID_INTAKE",
            "Q_DEHYDRATION_SIGNS",
            "Q_BLOOD_PRESENT",
            "Q_PAIN_SEVERITY",
            "Q_DURATION",
        ],
        "allowed_advice_packs": ["PACK_GI_SUPPORT"],
    },
    "UTI": {
        "protocol_id": "PROTO_UTI_BASIC",
        "red_flags": ["ANS_FLANK_PAIN_YES", "ANS_FEVER_HIGH", "ANS_BLOOD_YES"],
        "recommended_questions": [
            "Q_BURNING_URINATION",
            "Q_URINARY_FREQUENCY",
            "Q_FLANK_PAIN",
            "Q_TEMPERATURE",
            "Q_BLOOD_PRESENT",
        ],
        "allowed_advice_packs": ["PACK_UTI_SUPPORT"],
    },
    "HEADACHE": {
        "protocol_id": "PROTO_HEADACHE_BASIC",
        "red_flags": [
            "ANS_NECK_STIFFNESS_YES",
            "ANS_WEAKNESS_YES",
            "ANS_HEADACHE_SUDDEN_YES",
            "ANS_CONFUSION_YES",
        ],
        "recommended_questions": [
            "Q_HEADACHE_SUDDEN",
            "Q_NECK_STIFFNESS",
            "Q_WEAKNESS",
            "Q_CONFUSION",
            "Q_PAIN_SEVERITY",
            "Q_DURATION",
        ],
        "allowed_advice_packs": ["PACK_HEADACHE_SUPPORT"],
    },
    "RASH": {
        "protocol_id": "PROTO_RASH_BASIC",
        "red_flags": ["ANS_SWELLING_YES", "SYM_DYSPNEA"],
        "recommended_questions": [
            "Q_SWELLING",
            "Q_SHORTNESS_OF_BREATH",
            "Q_RASH_SPREAD",
            "Q_DURATION",
        ],
        "allowed_advice_packs": ["PACK_RASH_SUPPORT"],
    },
    "ABDOMINAL": {
        "protocol_id": "PROTO_ABDOMINAL_BASIC",
        "red_flags": ["ANS_PAIN_SEVERE", "ANS_VOMIT_FREQ_HIGH", "ANS_FEVER_HIGH"],
        "recommended_questions": [
            "Q_LOCALIZATION",
            "Q_PAIN_SEVERITY",
            "Q_VOMITING_FREQUENCY",
            "Q_TEMPERATURE",
            "Q_DURATION",
        ],
        "allowed_advice_packs": ["PACK_ABDOMINAL_SUPPORT"],
    },
}

ADVICE_PACKS = {
    "PACK_RESP_SUPPORT": {
        "text": "Rest, fluids, monitor breathing, seek care if worsening.",
        "families": ["RESP"],
        "allowed_for": ["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY"],
    },
    "PACK_GI_SUPPORT": {
        "text": "Small sips of fluids, rest, seek care if unable to keep fluids down.",
        "families": ["GI"],
        "allowed_for": ["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY"],
    },
    "PACK_UTI_SUPPORT": {
        "text": "Hydrate and monitor urinary symptoms until the visit.",
        "families": ["UTI"],
        "allowed_for": ["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY"],
    },
    "PACK_HEADACHE_SUPPORT": {
        "text": "Rest, hydrate, monitor symptoms, and seek urgent care if red flags appear.",
        "families": ["HEADACHE"],
        "allowed_for": ["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY"],
    },
    "PACK_RASH_SUPPORT": {
        "text": "Avoid triggers, monitor spread, seek care if swelling or breathing issues develop.",
        "families": ["RASH"],
        "allowed_for": ["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY"],
    },
    "PACK_ABDOMINAL_SUPPORT": {
        "text": "Rest, sip fluids, avoid heavy meals, seek care if pain worsens.",
        "families": ["ABDOMINAL"],
        "allowed_for": ["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY"],
    },
}

FAMILY_BASE_INITIAL_ENTITIES = {
    "RESP": ["SYM_COUGH"],
    "GI": ["SYM_NAUSEA"],
    "UTI": ["SYM_DYSURIA"],
    "HEADACHE": ["SYM_HEADACHE"],
    "RASH": ["SYM_RASH"],
    "ABDOMINAL": ["SYM_ABDOMINAL_PAIN"],
}

FAMILY_DISTRACTOR_ENTITIES = {
    "RESP": ["SYM_SORE_THROAT", "SYM_RUNNY_NOSE", "SYM_FATIGUE"],
    "GI": ["SYM_HEADACHE", "SYM_MILD_FEVERISH", "SYM_FATIGUE"],
    "UTI": ["SYM_MILD_NAUSEA", "SYM_FATIGUE", "SYM_LOW_APPETITE"],
    "HEADACHE": ["SYM_NAUSEA", "SYM_FATIGUE", "SYM_LIGHT_SORE_THROAT"],
    "RASH": ["SYM_ITCHING", "SYM_MILD_FEVERISH", "SYM_FATIGUE"],
    "ABDOMINAL": ["SYM_BLOATING", "SYM_LOW_APPETITE", "SYM_FATIGUE"],
}
