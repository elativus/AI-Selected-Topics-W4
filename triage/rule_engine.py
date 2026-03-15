from __future__ import annotations

from typing import Dict, List, Optional

from triage.catalogs import ADVICE_PACKS, PROTOCOL_CATALOG
from triage.constants import DIFFICULTY_CFG
from triage.schema import RequiredWorldAction


def infer_target_disposition(family: str, facts: Dict[str, bool]) -> str:
    if family == "RESP":
        return _infer_resp(facts)
    if family == "GI":
        return _infer_gi(facts)
    if family == "UTI":
        return _infer_uti(facts)
    if family == "HEADACHE":
        return _infer_headache(facts)
    if family == "RASH":
        return _infer_rash(facts)
    if family == "ABDOMINAL":
        return _infer_abdominal(facts)
    raise ValueError(f"Unknown family: {family}")


def build_required_world_action(disposition: str) -> Optional[RequiredWorldAction]:
    if disposition == "BOOK_ROUTINE":
        return RequiredWorldAction(tool="book_visit", urgency="ROUTINE")
    if disposition == "BOOK_SAME_DAY":
        return RequiredWorldAction(tool="book_visit", urgency="SAME_DAY")
    if disposition == "ESCALATE_NOW":
        return RequiredWorldAction(tool="create_escalation", level="ESCALATE_NOW")
    return None


def get_allowed_advice_packs(family: str, disposition: str) -> List[str]:
    return [
        pack_id
        for pack_id, spec in ADVICE_PACKS.items()
        if family in spec["families"] and disposition in spec["allowed_for"]
    ]


def get_relevant_question_ids(family: str, facts: Dict[str, bool], difficulty: int) -> List[str]:
    del facts  # reserved for future family-specific refinements
    protocol = PROTOCOL_CATALOG[family]
    qids = list(protocol["recommended_questions"])
    cfg = DIFFICULTY_CFG[difficulty]
    if cfg["num_required_groups"] >= 4:
        if "Q_COMORBIDITY" not in qids:
            qids.append("Q_COMORBIDITY")
    return qids


def build_required_evidence_groups(family: str, facts: Dict[str, bool], difficulty: int) -> Dict[str, List[str]]:
    del facts  # difficulty controls how many groups are required, not their semantics.
    all_groups = _family_group_catalog(family)
    k = DIFFICULTY_CFG[difficulty]["num_required_groups"]
    selected_items = list(all_groups.items())[:k]
    return {group_id: list(entity_ids) for group_id, entity_ids in selected_items}


def build_acceptable_dispositions(target_disposition: str, difficulty: int) -> List[str]:
    # v1 keeps the world mostly single-label to simplify verifier and RL.
    # A small relaxation is allowed on the hardest same-day cases.
    if difficulty >= 9 and target_disposition == "BOOK_SAME_DAY":
        return ["BOOK_SAME_DAY", "ESCALATE_NOW"]
    return [target_disposition]


def _family_group_catalog(family: str) -> Dict[str, List[str]]:
    if family == "RESP":
        return {
            "resp_red_flag": ["SYM_DYSPNEA", "SYM_CHEST_PAIN", "ANS_SOB_NO", "ANS_CHEST_PAIN_NO", "ANS_CONFUSION_YES", "ANS_CONFUSION_NO"],
            "severity": ["ANS_TEMP_39_PLUS", "ANS_TEMP_SUB39"],
            "course": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
            "risk_modifier": ["RISK_ASTHMA", "RISK_IMMUNOCOMP", "ANS_COMORBIDITY_NO", "AGE_SENIOR", "AGE_ADULT", "AGE_CHILD"],
        }
    if family == "GI":
        return {
            "gi_red_flag": ["ANS_BLOOD_YES", "ANS_BLOOD_NO", "ANS_DEHYDRATION_SEVERE", "ANS_DEHYDRATION_MILD", "ANS_DEHYDRATION_NONE"],
            "severity": ["ANS_PAIN_SEVERE", "ANS_PAIN_MILD"],
            "course": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
            "intake": ["ANS_FLUIDS_POOR", "ANS_FLUIDS_OK", "ANS_VOMIT_FREQ_HIGH", "ANS_VOMIT_FREQ_LOW"],
        }
    if family == "UTI":
        return {
            "uti_red_flag": ["ANS_FLANK_PAIN_YES", "ANS_FLANK_PAIN_NO", "ANS_BLOOD_YES", "ANS_BLOOD_NO"],
            "severity": ["ANS_FEVER_HIGH", "ANS_FEVER_LOW"],
            "core_symptoms": ["SYM_DYSURIA", "ANS_URINARY_FREQ_YES", "ANS_URINARY_FREQ_NO"],
            "course": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
        }
    if family == "HEADACHE":
        return {
            "headache_red_flag": ["ANS_HEADACHE_SUDDEN_YES", "ANS_HEADACHE_SUDDEN_NO", "ANS_NECK_STIFFNESS_YES", "ANS_NECK_STIFFNESS_NO", "ANS_WEAKNESS_YES", "ANS_WEAKNESS_NO", "ANS_CONFUSION_YES", "ANS_CONFUSION_NO"],
            "severity": ["ANS_PAIN_SEVERE", "ANS_PAIN_MILD"],
            "course": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
            "risk_modifier": ["AGE_SENIOR", "AGE_ADULT", "AGE_CHILD", "RISK_IMMUNOCOMP", "ANS_COMORBIDITY_NO"],
        }
    if family == "RASH":
        return {
            "rash_red_flag": ["ANS_SWELLING_YES", "ANS_SWELLING_NO", "SYM_DYSPNEA", "ANS_SOB_NO"],
            "spread": ["ANS_RASH_SPREAD_FAST", "ANS_RASH_SPREAD_SLOW"],
            "course": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
            "risk_modifier": ["RISK_IMMUNOCOMP", "ANS_COMORBIDITY_NO", "AGE_CHILD", "AGE_ADULT", "AGE_SENIOR"],
        }
    if family == "ABDOMINAL":
        return {
            "abdominal_red_flag": ["ANS_PAIN_SEVERE", "ANS_PAIN_MILD", "ANS_FEVER_HIGH", "ANS_FEVER_LOW"],
            "localization": ["ANS_RLQ_PAIN", "ANS_DIFFUSE_PAIN"],
            "vomiting": ["ANS_VOMIT_FREQ_HIGH", "ANS_VOMIT_FREQ_LOW"],
            "course": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
        }
    raise ValueError(f"Unknown family: {family}")


def _infer_resp(facts: Dict[str, bool]) -> str:
    if facts.get("SYM_DYSPNEA") or facts.get("SYM_CHEST_PAIN") or facts.get("ANS_CONFUSION_YES"):
        return "ESCALATE_NOW"
    if facts.get("ANS_TEMP_39_PLUS") and (facts.get("AGE_SENIOR") or facts.get("RISK_ASTHMA") or facts.get("RISK_IMMUNOCOMP")):
        return "BOOK_SAME_DAY"
    if facts.get("ANS_DURATION_GT5D"):
        return "BOOK_ROUTINE"
    return "SELF_CARE"


def _infer_gi(facts: Dict[str, bool]) -> str:
    if facts.get("ANS_BLOOD_YES") or facts.get("ANS_DEHYDRATION_SEVERE") or facts.get("ANS_PAIN_SEVERE"):
        return "ESCALATE_NOW"
    if facts.get("ANS_VOMIT_FREQ_HIGH") or facts.get("ANS_FLUIDS_POOR"):
        return "BOOK_SAME_DAY"
    if facts.get("ANS_DURATION_GT5D"):
        return "BOOK_ROUTINE"
    return "SELF_CARE"


def _infer_uti(facts: Dict[str, bool]) -> str:
    if facts.get("ANS_FLANK_PAIN_YES") or facts.get("ANS_BLOOD_YES"):
        return "ESCALATE_NOW"
    if facts.get("ANS_FEVER_HIGH"):
        return "BOOK_SAME_DAY"
    if facts.get("ANS_DURATION_GT5D") or facts.get("RISK_IMMUNOCOMP"):
        return "BOOK_ROUTINE"
    return "SELF_CARE"


def _infer_headache(facts: Dict[str, bool]) -> str:
    if facts.get("ANS_HEADACHE_SUDDEN_YES") or facts.get("ANS_NECK_STIFFNESS_YES") or facts.get("ANS_WEAKNESS_YES") or facts.get("ANS_CONFUSION_YES"):
        return "ESCALATE_NOW"
    if facts.get("ANS_PAIN_SEVERE") and facts.get("AGE_SENIOR"):
        return "BOOK_SAME_DAY"
    if facts.get("ANS_DURATION_GT5D"):
        return "BOOK_ROUTINE"
    return "SELF_CARE"


def _infer_rash(facts: Dict[str, bool]) -> str:
    if facts.get("ANS_SWELLING_YES") or facts.get("SYM_DYSPNEA"):
        return "ESCALATE_NOW"
    if facts.get("ANS_RASH_SPREAD_FAST") or facts.get("RISK_IMMUNOCOMP"):
        return "BOOK_SAME_DAY"
    if facts.get("ANS_DURATION_GT5D"):
        return "BOOK_ROUTINE"
    return "SELF_CARE"


def _infer_abdominal(facts: Dict[str, bool]) -> str:
    if facts.get("ANS_PAIN_SEVERE") or facts.get("ANS_VOMIT_FREQ_HIGH") and facts.get("ANS_FEVER_HIGH"):
        return "ESCALATE_NOW"
    if facts.get("ANS_RLQ_PAIN") or facts.get("ANS_FEVER_HIGH"):
        return "BOOK_SAME_DAY"
    if facts.get("ANS_DURATION_GT5D"):
        return "BOOK_ROUTINE"
    return "SELF_CARE"
