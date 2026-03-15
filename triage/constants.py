from __future__ import annotations

DISPOSITIONS = [
    "SELF_CARE",
    "BOOK_ROUTINE",
    "BOOK_SAME_DAY",
    "ESCALATE_NOW",
]

URGENCY_VALUES = ["ROUTINE", "SAME_DAY"]
ESCALATION_LEVELS = ["ESCALATE_NOW"]
FAMILIES = ["RESP", "GI", "UTI", "HEADACHE", "RASH", "ABDOMINAL"]

TOOL_NAMES = [
    "ask_question",
    "lookup_protocol",
    "list_slots",
    "book_visit",
    "create_escalation",
    "finish",
]

WORLD_MUTATING_TOOLS = {"book_visit", "create_escalation"}

URGENCY_RANK = {
    "SELF_CARE": 0,
    "BOOK_ROUTINE": 1,
    "BOOK_SAME_DAY": 2,
    "ESCALATE_NOW": 3,
}

GLOBAL_ENUM_IDS = set(DISPOSITIONS) | set(URGENCY_VALUES) | set(ESCALATION_LEVELS)

DIFFICULTY_CFG = {
    1: {
        "num_required_groups": 2,
        "num_distractors": 0,
        "noise": 0.00,
        "num_risk_modifiers": 0,
        "protocol_lookup_expected": False,
        "max_steps": 6,
    },
    2: {
        "num_required_groups": 2,
        "num_distractors": 0,
        "noise": 0.10,
        "num_risk_modifiers": 0,
        "protocol_lookup_expected": False,
        "max_steps": 6,
    },
    3: {
        "num_required_groups": 3,
        "num_distractors": 1,
        "noise": 0.10,
        "num_risk_modifiers": 1,
        "protocol_lookup_expected": False,
        "max_steps": 7,
    },
    4: {
        "num_required_groups": 3,
        "num_distractors": 1,
        "noise": 0.20,
        "num_risk_modifiers": 1,
        "protocol_lookup_expected": True,
        "max_steps": 8,
    },
    5: {
        "num_required_groups": 3,
        "num_distractors": 2,
        "noise": 0.30,
        "num_risk_modifiers": 1,
        "protocol_lookup_expected": True,
        "max_steps": 8,
    },
    6: {
        "num_required_groups": 3,
        "num_distractors": 2,
        "noise": 0.40,
        "num_risk_modifiers": 1,
        "protocol_lookup_expected": True,
        "max_steps": 9,
    },
    7: {
        "num_required_groups": 4,
        "num_distractors": 2,
        "noise": 0.50,
        "num_risk_modifiers": 2,
        "protocol_lookup_expected": True,
        "max_steps": 10,
    },
    8: {
        "num_required_groups": 4,
        "num_distractors": 3,
        "noise": 0.60,
        "num_risk_modifiers": 2,
        "protocol_lookup_expected": True,
        "max_steps": 10,
    },
    9: {
        "num_required_groups": 4,
        "num_distractors": 3,
        "noise": 0.70,
        "num_risk_modifiers": 2,
        "protocol_lookup_expected": True,
        "max_steps": 11,
    },
    10: {
        "num_required_groups": 4,
        "num_distractors": 3,
        "noise": 0.80,
        "num_risk_modifiers": 2,
        "protocol_lookup_expected": True,
        "max_steps": 12,
    },
}

REWARD_CFG = {
    "step_penalty": -0.02,
    "tool_penalty": -0.03,
    "invalid_action": -0.10,
    "duplicate_question": -0.05,
    "irrelevant_question": -0.04,
    "hallucination": -0.20,
    "premature_finish": -0.40,
    "mutation_without_confirmation": -0.50,
    "mismatched_confirmation": -0.35,
    "conflicting_world_action": -0.35,
    "success": +1.00,
    "fail": -1.00,
    "undertriage_extra": -0.50,
    "over_escalation_extra": -0.15,
    "unnecessary_world_action": -0.20,
}
