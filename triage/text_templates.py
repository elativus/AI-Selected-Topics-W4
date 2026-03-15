from __future__ import annotations

import random
from typing import Dict


def _choose(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def render_initial_message(family: str, facts: Dict[str, bool], profile: Dict[str, object], rng: random.Random) -> str:
    age_group = profile.get("age_group", "adult")
    age_prefix = {
        "child": "My child",
        "adult": "I",
        "senior": "I'm older and I",
    }.get(age_group, "I")

    if family == "RESP":
        if facts.get("SYM_FEVER"):
            return _choose(
                rng,
                [
                    f"{age_prefix} have had a cough and fever since yesterday.",
                    f"{age_prefix} started coughing and feeling feverish recently.",
                ],
            )
        return _choose(rng, [f"{age_prefix} have had a cough for a few days."])

    if family == "GI":
        return _choose(
            rng,
            [
                f"{age_prefix} have nausea and vomiting.",
                f"{age_prefix} feel sick to my stomach and keep throwing up.",
            ],
        )

    if family == "UTI":
        return _choose(
            rng,
            [
                f"{age_prefix} have burning when urinating.",
                f"{age_prefix} think this may be a urine infection because it burns to pee.",
            ],
        )

    if family == "HEADACHE":
        return _choose(
            rng,
            [
                f"{age_prefix} have a headache.",
                f"{age_prefix} have been dealing with a bad headache.",
            ],
        )

    if family == "RASH":
        return _choose(
            rng,
            [
                f"{age_prefix} have a rash.",
                f"{age_prefix} noticed a rash on my skin.",
            ],
        )

    if family == "ABDOMINAL":
        return _choose(
            rng,
            [
                f"{age_prefix} have abdominal pain.",
                f"{age_prefix} stomach hurts.",
            ],
        )

    return "I need help with symptoms."


def answer_for_question(question_id: str, facts: Dict[str, bool], rng: random.Random) -> str:
    if question_id == "Q_DURATION":
        if facts.get("ANS_DURATION_1D"):
            return _choose(rng, ["Since yesterday.", "About a day."])
        if facts.get("ANS_DURATION_1_3D"):
            return _choose(rng, ["A couple of days.", "About two days."])
        if facts.get("ANS_DURATION_GT5D"):
            return _choose(rng, ["More than five days.", "Nearly a week."])

    if question_id == "Q_TEMPERATURE":
        if facts.get("ANS_TEMP_39_PLUS") or facts.get("ANS_FEVER_HIGH"):
            return _choose(rng, ["Around 39.1 C.", "It went above 39 C."])
        if facts.get("ANS_TEMP_SUB39") or facts.get("ANS_FEVER_LOW"):
            return _choose(rng, ["Mild, under 39 C.", "Low-grade fever."])
        return _choose(rng, ["No measured fever."])

    if question_id == "Q_SHORTNESS_OF_BREATH":
        if facts.get("SYM_DYSPNEA"):
            return _choose(rng, ["Yes, breathing feels harder.", "Yes, I'm short of breath."])
        return _choose(rng, ["No, breathing is fine.", "No shortness of breath."])

    if question_id == "Q_CHEST_PAIN":
        if facts.get("SYM_CHEST_PAIN"):
            return _choose(rng, ["Yes, there is chest pain.", "Yes, my chest hurts."])
        return _choose(rng, ["No chest pain.", "No, none."])

    if question_id == "Q_CONFUSION":
        if facts.get("ANS_CONFUSION_YES"):
            return _choose(rng, ["Yes, I feel confused.", "Yes, I'm unusually foggy."])
        return _choose(rng, ["No confusion.", "No, mentally clear."])

    if question_id == "Q_VOMITING_FREQUENCY":
        if facts.get("ANS_VOMIT_FREQ_HIGH"):
            return _choose(rng, ["Many times today.", "I keep vomiting repeatedly."])
        return _choose(rng, ["Only once or twice.", "A small number of times."])

    if question_id == "Q_FLUID_INTAKE":
        if facts.get("ANS_FLUIDS_POOR"):
            return _choose(rng, ["I can't keep fluids down well.", "Not really, I vomit after drinking."])
        return _choose(rng, ["Yes, I can sip fluids.", "Fluids are staying down."])

    if question_id == "Q_DEHYDRATION_SIGNS":
        if facts.get("ANS_DEHYDRATION_SEVERE"):
            return _choose(rng, ["Yes, very dizzy and dry.", "I feel very dehydrated."])
        if facts.get("ANS_DEHYDRATION_MILD"):
            return _choose(rng, ["A little dry, but manageable.", "Slightly dehydrated."])
        return _choose(rng, ["No dehydration symptoms."])

    if question_id == "Q_BLOOD_PRESENT":
        if facts.get("ANS_BLOOD_YES"):
            return _choose(rng, ["Yes, there was some blood.", "I noticed blood."])
        return _choose(rng, ["No blood."])

    if question_id == "Q_PAIN_SEVERITY":
        if facts.get("ANS_PAIN_SEVERE"):
            return _choose(rng, ["Severe.", "Very strong pain."])
        return _choose(rng, ["Mild to moderate.", "Manageable."])

    if question_id == "Q_FLANK_PAIN":
        if facts.get("ANS_FLANK_PAIN_YES"):
            return _choose(rng, ["Yes, pain in my side/back.", "Yes, flank pain."])
        return _choose(rng, ["No flank pain."])

    if question_id == "Q_NECK_STIFFNESS":
        if facts.get("ANS_NECK_STIFFNESS_YES"):
            return _choose(rng, ["Yes, my neck feels stiff.", "Yes, neck stiffness is there."])
        return _choose(rng, ["No neck stiffness."])

    if question_id == "Q_WEAKNESS":
        if facts.get("ANS_WEAKNESS_YES"):
            return _choose(rng, ["Yes, new weakness.", "Yes, one side feels weak."])
        return _choose(rng, ["No weakness."])

    if question_id == "Q_RASH_SPREAD":
        if facts.get("ANS_RASH_SPREAD_FAST"):
            return _choose(rng, ["Yes, it is spreading quickly.", "It is spreading fast."])
        return _choose(rng, ["No, it is not spreading fast."])

    if question_id == "Q_SWELLING":
        if facts.get("ANS_SWELLING_YES"):
            return _choose(rng, ["Yes, there is swelling.", "Yes, facial or throat swelling."])
        return _choose(rng, ["No swelling."])

    if question_id == "Q_COMORBIDITY":
        if facts.get("RISK_ASTHMA"):
            return _choose(rng, ["I have asthma.", "Yes, asthma."])
        if facts.get("RISK_IMMUNOCOMP"):
            return _choose(rng, ["I'm immunocompromised.", "Yes, my immune system is weak."])
        return _choose(rng, ["No major conditions."])

    if question_id == "Q_URINARY_FREQUENCY":
        if facts.get("ANS_URINARY_FREQ_YES"):
            return _choose(rng, ["Yes, much more often.", "Yes, frequent urination."])
        return _choose(rng, ["No, not especially."])

    if question_id == "Q_BURNING_URINATION":
        if facts.get("SYM_DYSURIA"):
            return _choose(rng, ["Yes, it burns.", "Yes, definitely burning."])
        return _choose(rng, ["No burning."])

    if question_id == "Q_HEADACHE_SUDDEN":
        if facts.get("ANS_HEADACHE_SUDDEN_YES"):
            return _choose(rng, ["Yes, it started suddenly.", "It came on all at once."])
        return _choose(rng, ["No, it came on gradually."])

    if question_id == "Q_LOCALIZATION":
        if facts.get("ANS_RLQ_PAIN"):
            return _choose(rng, ["Lower right side.", "Right lower abdomen."])
        return _choose(rng, ["More central or diffuse."])

    return "I'm not sure."
