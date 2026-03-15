from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from base.data import Data

Disposition = Literal["SELF_CARE", "BOOK_ROUTINE", "BOOK_SAME_DAY", "ESCALATE_NOW"]
Family = Literal["RESP", "GI", "UTI", "HEADACHE", "RASH", "ABDOMINAL"]
Urgency = Literal["ROUTINE", "SAME_DAY"]
ToolName = Literal[
    "ask_question",
    "lookup_protocol",
    "list_slots",
    "book_visit",
    "create_escalation",
    "finish",
]


@dataclass
class QuestionResponse:
    answer_text: str
    new_entities: List[str]
    evidence_groups: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "QuestionResponse":
        return cls(
            answer_text=obj["answer_text"],
            new_entities=list(obj.get("new_entities", [])),
            evidence_groups=list(obj.get("evidence_groups", [])),
        )


@dataclass
class SlotSpec:
    slot_id: str
    urgency: Urgency
    label: str

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "SlotSpec":
        return cls(slot_id=obj["slot_id"], urgency=obj["urgency"], label=obj["label"])


@dataclass
class RequiredWorldAction:
    tool: Literal["book_visit", "create_escalation"]
    urgency: Optional[Urgency] = None
    level: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional["RequiredWorldAction"]:
        if obj is None:
            return None
        return cls(tool=obj["tool"], urgency=obj.get("urgency"), level=obj.get("level"))


@dataclass
class TriageData(Data):
    case_id: str = ""
    difficulty: int = 1
    family: Family = "RESP"

    initial_message: str = ""
    patient_profile: Dict[str, Any] = field(default_factory=dict)
    initial_entities: List[str] = field(default_factory=list)

    hidden_facts: Dict[str, Any] = field(default_factory=dict)
    qa_map: Dict[str, QuestionResponse] = field(default_factory=dict)
    relevant_question_ids: List[str] = field(default_factory=list)
    required_evidence_groups: Dict[str, List[str]] = field(default_factory=dict)

    target_disposition: Disposition = "SELF_CARE"
    acceptable_dispositions: List[Disposition] = field(default_factory=lambda: ["SELF_CARE"])
    required_world_action: Optional[RequiredWorldAction] = None

    allowed_advice_packs: List[str] = field(default_factory=list)
    slot_inventory: Dict[Urgency, List[SlotSpec]] = field(default_factory=dict)

    max_steps: int = 8
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep base.Data uid aligned with case_id for convenience.
        self.uid = self.case_id

    def to_dict(self) -> Dict[str, Any]:
        obj = asdict(self)
        # base.Data duplicates metadata; keep case-centric layout.
        obj["uid"] = self.uid
        return obj

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "TriageData":
        qa_map = {k: QuestionResponse.from_dict(v) for k, v in obj.get("qa_map", {}).items()}
        slot_inventory = {
            urgency: [SlotSpec.from_dict(x) for x in slots]
            for urgency, slots in obj.get("slot_inventory", {}).items()
        }
        return cls(
            uid=obj.get("uid", obj.get("case_id", "")),
            case_id=obj["case_id"],
            difficulty=int(obj.get("difficulty", 1)),
            family=obj.get("family", "RESP"),
            initial_message=obj.get("initial_message", ""),
            patient_profile=dict(obj.get("patient_profile", {})),
            initial_entities=list(obj.get("initial_entities", [])),
            hidden_facts=dict(obj.get("hidden_facts", {})),
            qa_map=qa_map,
            relevant_question_ids=list(obj.get("relevant_question_ids", [])),
            required_evidence_groups={
                k: list(v) for k, v in obj.get("required_evidence_groups", {}).items()
            },
            target_disposition=obj.get("target_disposition", "SELF_CARE"),
            acceptable_dispositions=list(obj.get("acceptable_dispositions", [obj.get("target_disposition", "SELF_CARE")])),
            required_world_action=RequiredWorldAction.from_dict(obj.get("required_world_action")),
            allowed_advice_packs=list(obj.get("allowed_advice_packs", [])),
            slot_inventory=slot_inventory,
            max_steps=int(obj.get("max_steps", 8)),
            seed=obj.get("seed"),
            metadata=dict(obj.get("metadata", {})),
        )


@dataclass
class ConfirmedWorldAction:
    tool: Literal["book_visit", "create_escalation"]
    args: Dict[str, Any]
    used: bool = False


@dataclass
class EpisodeState:
    case_id: str
    step_idx: int
    max_steps: int

    known_entities: set[str] = field(default_factory=set)
    asked_questions: List[str] = field(default_factory=list)
    covered_groups: set[str] = field(default_factory=set)

    confirmed_action: Optional[ConfirmedWorldAction] = None

    booked_slot_id: Optional[str] = None
    booked_urgency: Optional[Urgency] = None
    escalation_level: Optional[str] = None

    final_disposition: Optional[Disposition] = None
    final_advice_pack_id: Optional[str] = None

    done: bool = False
    done_reason: Optional[str] = None
    failure_reason: Optional[str] = None

    tool_calls: int = 0
    invalid_actions: int = 0
    policy_violations: int = 0
    critical_policy_violations: int = 0
    duplicate_questions: int = 0
    irrelevant_questions: int = 0
    hallucination_violations: int = 0
    confirmation_violations: int = 0

    messages: List[str] = field(default_factory=list)
    info_trace: List[Dict[str, Any]] = field(default_factory=list)
