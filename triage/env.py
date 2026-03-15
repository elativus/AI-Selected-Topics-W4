from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from triage.action_parser import ParsedAction, parse_action
from triage.catalogs import ADVICE_PACKS, PROTOCOL_CATALOG, QUESTION_CATALOG
from triage.constants import (
    DISPOSITIONS,
    ESCALATION_LEVELS,
    GLOBAL_ENUM_IDS,
    REWARD_CFG,
    TOOL_NAMES,
    URGENCY_RANK,
    URGENCY_VALUES,
)
from triage.generator import CaseGenerator
from triage.rule_engine import build_required_world_action
from triage.schema import ConfirmedWorldAction, EpisodeState, SlotSpec, TriageData


class SafeTriageEnv:
    """Multi-step tool-using triage environment.

    Observations are text. Actions are either plain free-text or `TOOL_CALL {...}`.
    """

    def __init__(self, name: str = "safe_triage", seed: Optional[int] = None):
        self.name = name
        self.data: Optional[TriageData] = None
        self.state: Optional[EpisodeState] = None
        self.generator = CaseGenerator(seed=seed)

    def reset(self, data: TriageData) -> str:
        self.data = data
        global_ids = set(DISPOSITIONS) | set(URGENCY_VALUES) | set(ESCALATION_LEVELS) | set(QUESTION_CATALOG.keys())
        self.state = EpisodeState(
            case_id=data.case_id,
            step_idx=0,
            max_steps=data.max_steps,
            known_entities=set(data.initial_entities) | global_ids,
        )
        self._recompute_coverage()
        body = self._render_initial_body()
        return self._render_observation(body=body, new_entities=[], violations=[])

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        assert self.data is not None and self.state is not None, "Call reset() first."

        if self.state.done:
            info = self._make_info(
                action=action,
                action_type="invalid",
                tool_name=None,
                new_entities=[],
                violations=[],
                reward_breakdown={},
                done_reason=self.state.done_reason,
            )
            return "EPISODE_ALREADY_FINISHED", 0.0, True, info

        self.state.step_idx += 1
        parsed = parse_action(action)

        reward = REWARD_CFG["step_penalty"]
        reward_breakdown: Dict[str, float] = {"step_penalty": REWARD_CFG["step_penalty"]}
        new_entities: List[str] = []
        violations: List[str] = []
        obs_body = ""

        if parsed.kind == "invalid":
            self.state.invalid_actions += 1
            reward += REWARD_CFG["invalid_action"]
            reward_breakdown["invalid_action"] = REWARD_CFG["invalid_action"]
            obs_body = f"INVALID_ACTION: {parsed.error}"

        elif parsed.kind == "free_text":
            obs_body, delta_reward, violations = self._apply_free_text(parsed)
            reward += delta_reward
            if delta_reward:
                reward_breakdown["free_text_delta"] = delta_reward

        elif parsed.kind == "tool_call":
            self.state.tool_calls += 1
            reward += REWARD_CFG["tool_penalty"]
            reward_breakdown["tool_penalty"] = REWARD_CFG["tool_penalty"]
            obs_body, delta_reward, new_entities, violations = self._apply_tool_call(parsed)
            reward += delta_reward
            if delta_reward:
                reward_breakdown["tool_delta"] = delta_reward

        self._recompute_coverage()

        if not self.state.done and self.state.step_idx >= self.state.max_steps:
            fail_reward = self._terminate_failure("max_steps_exceeded")
            reward += fail_reward
            reward_breakdown["terminal_fail"] = fail_reward
            obs_body = "MAX_STEPS_REACHED. Episode terminated."

        info = self._make_info(
            action=action,
            action_type=parsed.kind,
            tool_name=parsed.tool_name,
            new_entities=new_entities,
            violations=violations,
            reward_breakdown=reward_breakdown,
            done_reason=self.state.done_reason,
        )
        self.state.info_trace.append(info)

        observation = self._render_observation(body=obs_body, new_entities=new_entities, violations=violations)
        return observation, reward, self.state.done, info

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> List[TriageData]:
        return self.generator.generate(
            num_of_questions=num_of_questions,
            max_attempts=max_attempts,
            difficulty=difficulty or 1,
            **kwargs,
        )

    # ---------------------------------------------------------------------
    # Action handlers
    # ---------------------------------------------------------------------
    def _apply_free_text(self, parsed: ParsedAction) -> Tuple[str, float, List[str]]:
        assert self.state is not None
        self.state.messages.append(parsed.text or "")
        if parsed.confirm_payload is None:
            return "ACK: message received. No actionable confirmation detected.", 0.0, []

        payload = parsed.confirm_payload
        if not isinstance(payload, dict) or "tool" not in payload:
            self.state.confirmation_violations += 1
            self.state.policy_violations += 1
            return "REJECTED: malformed confirmation payload.", REWARD_CFG["invalid_action"], ["malformed_confirmation"]

        tool = payload["tool"]
        if tool not in {"book_visit", "create_escalation"}:
            self.state.confirmation_violations += 1
            self.state.policy_violations += 1
            return "REJECTED: confirmation can only target a mutating tool.", REWARD_CFG["invalid_action"], ["bad_confirmation_tool"]

        if tool == "book_visit":
            slot_id = payload.get("slot_id")
            if not isinstance(slot_id, str):
                self.state.confirmation_violations += 1
                self.state.policy_violations += 1
                return "REJECTED: confirmation for book_visit requires slot_id.", REWARD_CFG["invalid_action"], ["malformed_confirmation"]
            if not self._is_known_entity(slot_id):
                self.state.confirmation_violations += 1
                self.state.policy_violations += 1
                self.state.hallucination_violations += 1
                return "REJECTED: slot_id is unknown.", REWARD_CFG["hallucination"], ["hallucinated_slot_in_confirmation"]
            self.state.confirmed_action = ConfirmedWorldAction(tool="book_visit", args={"slot_id": slot_id})
            return f"CONFIRMED: book_visit with {slot_id}", 0.0, []

        level = payload.get("level")
        if level not in ESCALATION_LEVELS:
            self.state.confirmation_violations += 1
            self.state.policy_violations += 1
            return "REJECTED: invalid escalation level.", REWARD_CFG["invalid_action"], ["malformed_confirmation"]
        self.state.confirmed_action = ConfirmedWorldAction(tool="create_escalation", args={"level": level})
        return f"CONFIRMED: create_escalation with level {level}", 0.0, []

    def _apply_tool_call(self, parsed: ParsedAction) -> Tuple[str, float, List[str], List[str]]:
        assert parsed.tool_name is not None and parsed.tool_args is not None

        if parsed.tool_name not in TOOL_NAMES:
            self.state.invalid_actions += 1
            return f"INVALID_TOOL: {parsed.tool_name}", REWARD_CFG["invalid_action"], [], ["unknown_tool"]

        dispatch = {
            "ask_question": self._tool_ask_question,
            "lookup_protocol": self._tool_lookup_protocol,
            "list_slots": self._tool_list_slots,
            "book_visit": self._tool_book_visit,
            "create_escalation": self._tool_create_escalation,
            "finish": self._tool_finish,
        }
        return dispatch[parsed.tool_name](parsed.tool_args)

    # ---------------------------------------------------------------------
    # Tool implementations
    # ---------------------------------------------------------------------
    def _tool_ask_question(self, args: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        assert self.data is not None and self.state is not None
        question_id = args.get("question_id")
        if not isinstance(question_id, str) or question_id not in QUESTION_CATALOG:
            self.state.invalid_actions += 1
            return "INVALID_ARGS: ask_question requires a valid question_id.", REWARD_CFG["invalid_action"], [], ["invalid_question_id"]

        if question_id in self.state.asked_questions:
            self.state.duplicate_questions += 1
            self.state.policy_violations += 1
            return (
                "Patient: I already answered that.\nNEW_ENTITIES: []",
                REWARD_CFG["duplicate_question"],
                [],
                ["duplicate_question"],
            )

        self.state.asked_questions.append(question_id)
        resp = self.data.qa_map.get(question_id)
        if resp is None:
            self.state.irrelevant_questions += 1
            self.state.policy_violations += 1
            return (
                "Patient: I don't think that applies here.\nNEW_ENTITIES: []",
                REWARD_CFG["irrelevant_question"],
                [],
                ["irrelevant_question"],
            )

        for ent in resp.new_entities:
            self.state.known_entities.add(ent)
        return (
            f'Patient: "{resp.answer_text}"\nNEW_ENTITIES: {resp.new_entities}',
            0.0,
            list(resp.new_entities),
            [],
        )

    def _tool_lookup_protocol(self, args: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        assert self.data is not None and self.state is not None
        entity_ids = args.get("entity_ids")
        if not isinstance(entity_ids, list) or not all(isinstance(x, str) for x in entity_ids):
            self.state.invalid_actions += 1
            return "INVALID_ARGS: lookup_protocol requires entity_ids: list[str].", REWARD_CFG["invalid_action"], [], ["invalid_lookup_args"]

        unknown = [x for x in entity_ids if not self._is_known_entity(x)]
        if unknown:
            self.state.policy_violations += 1
            self.state.hallucination_violations += 1
            return (
                f"POLICY_VIOLATION: unknown entity IDs in lookup: {unknown}",
                REWARD_CFG["hallucination"],
                [],
                ["hallucinated_entity"],
            )

        protocol = PROTOCOL_CATALOG[self.data.family]
        proto_id = protocol["protocol_id"]
        new_entities = [proto_id] + list(protocol["allowed_advice_packs"])
        for ent in new_entities:
            self.state.known_entities.add(ent)

        body = (
            f"PROTOCOL: {proto_id}\n"
            f"RED_FLAGS:\n{protocol['red_flags']}\n\n"
            f"RELEVANT_QUESTIONS:\n{protocol['recommended_questions']}\n\n"
            f"ALLOWED_ADVICE_PACKS:\n{protocol['allowed_advice_packs']}"
        )
        return body, 0.0, new_entities, []

    def _tool_list_slots(self, args: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        assert self.data is not None and self.state is not None
        urgency = args.get("urgency")
        if urgency not in URGENCY_VALUES:
            self.state.invalid_actions += 1
            return "INVALID_ARGS: list_slots requires urgency in [ROUTINE, SAME_DAY].", REWARD_CFG["invalid_action"], [], ["invalid_urgency"]

        slots: List[SlotSpec] = list(self.data.slot_inventory.get(urgency, []))
        new_entities: List[str] = []
        for slot in slots:
            self.state.known_entities.add(slot.slot_id)
            new_entities.append(slot.slot_id)
        slot_lines = [f"- {slot.slot_id}: {slot.label}" for slot in slots] or ["- NONE"]
        body = f"AVAILABLE_SLOTS {urgency}:\n" + "\n".join(slot_lines)
        return body, 0.0, new_entities, []

    def _tool_book_visit(self, args: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        assert self.data is not None and self.state is not None
        slot_id = args.get("slot_id")
        if not isinstance(slot_id, str):
            self.state.invalid_actions += 1
            return "INVALID_ARGS: book_visit requires slot_id.", REWARD_CFG["invalid_action"], [], ["invalid_slot_id"]

        if self.state.escalation_level is not None or self.state.booked_slot_id is not None:
            self.state.policy_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["conflicting_world_action"] + self._terminate_failure("conflicting_world_action")
            return "POLICY_VIOLATION: conflicting world action.", penalty, [], ["conflicting_world_action"]

        if not self._is_known_entity(slot_id):
            self.state.policy_violations += 1
            self.state.hallucination_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["hallucination"] + self._terminate_failure("hallucinated_slot")
            return f"POLICY_VIOLATION: unknown slot_id {slot_id}", penalty, [], ["hallucinated_entity"]

        if self.state.confirmed_action is None:
            self.state.policy_violations += 1
            self.state.confirmation_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["mutation_without_confirmation"] + self._terminate_failure("book_without_confirmation")
            return "POLICY_VIOLATION: book_visit requires prior confirmation.", penalty, [], ["missing_confirmation"]

        if self.state.confirmed_action.tool != "book_visit" or self.state.confirmed_action.args != {"slot_id": slot_id}:
            self.state.policy_violations += 1
            self.state.confirmation_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["mismatched_confirmation"] + self._terminate_failure("mismatched_confirmation")
            return "POLICY_VIOLATION: confirmation does not match the booking action.", penalty, [], ["mismatched_confirmation"]

        matched_slot = self._find_slot(slot_id)
        if matched_slot is None:
            self.state.invalid_actions += 1
            penalty = REWARD_CFG["invalid_action"] + self._terminate_failure("slot_not_found")
            return "INVALID_ARGS: requested slot does not exist.", penalty, [], ["slot_not_found"]

        self.state.booked_slot_id = slot_id
        self.state.booked_urgency = matched_slot.urgency
        self.state.confirmed_action.used = True
        return f"BOOKED: {slot_id} ({matched_slot.label})", 0.0, [], []

    def _tool_create_escalation(self, args: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        assert self.state is not None
        level = args.get("level")
        if level not in ESCALATION_LEVELS:
            self.state.invalid_actions += 1
            return "INVALID_ARGS: create_escalation requires a valid level.", REWARD_CFG["invalid_action"], [], ["invalid_escalation_level"]

        if self.state.escalation_level is not None or self.state.booked_slot_id is not None:
            self.state.policy_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["conflicting_world_action"] + self._terminate_failure("conflicting_world_action")
            return "POLICY_VIOLATION: conflicting world action.", penalty, [], ["conflicting_world_action"]

        if self.state.confirmed_action is None:
            self.state.policy_violations += 1
            self.state.confirmation_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["mutation_without_confirmation"] + self._terminate_failure("escalate_without_confirmation")
            return "POLICY_VIOLATION: create_escalation requires prior confirmation.", penalty, [], ["missing_confirmation"]

        if self.state.confirmed_action.tool != "create_escalation" or self.state.confirmed_action.args != {"level": level}:
            self.state.policy_violations += 1
            self.state.confirmation_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["mismatched_confirmation"] + self._terminate_failure("mismatched_confirmation")
            return "POLICY_VIOLATION: confirmation does not match the escalation action.", penalty, [], ["mismatched_confirmation"]

        self.state.escalation_level = level
        self.state.confirmed_action.used = True
        return f"ESCALATION_CREATED: level={level}", 0.0, [], []

    def _tool_finish(self, args: Dict[str, Any]) -> Tuple[str, float, List[str], List[str]]:
        assert self.data is not None and self.state is not None
        disposition = args.get("disposition")
        advice_pack_id = args.get("advice_pack_id")

        if disposition not in DISPOSITIONS:
            self.state.invalid_actions += 1
            return "INVALID_ARGS: finish requires a valid disposition.", REWARD_CFG["invalid_action"], [], ["invalid_disposition"]

        all_groups = set(self.data.required_evidence_groups.keys())
        if self.state.covered_groups != all_groups:
            self.state.policy_violations += 1
            self.state.critical_policy_violations += 1
            penalty = REWARD_CFG["premature_finish"] + self._terminate_failure("premature_finish")
            return "POLICY_VIOLATION: finish called before all required evidence groups were covered.", penalty, [], ["premature_finish"]

        if disposition == "SELF_CARE":
            if not isinstance(advice_pack_id, str):
                penalty = self._terminate_failure("missing_advice_pack")
                return "FAIL: SELF_CARE requires advice_pack_id.", penalty, [], ["missing_advice_pack"]
            if not self._is_known_entity(advice_pack_id):
                self.state.policy_violations += 1
                self.state.hallucination_violations += 1
                self.state.critical_policy_violations += 1
                penalty = REWARD_CFG["hallucination"] + self._terminate_failure("hallucinated_advice_pack")
                return "POLICY_VIOLATION: unknown advice_pack_id.", penalty, [], ["hallucinated_entity"]
            if advice_pack_id not in self.data.allowed_advice_packs:
                penalty = self._terminate_failure("advice_pack_not_allowed")
                return "FAIL: advice pack is not allowed for this case.", penalty, [], ["advice_pack_not_allowed"]
        else:
            if advice_pack_id is not None:
                if not isinstance(advice_pack_id, str) or not self._is_known_entity(advice_pack_id):
                    self.state.policy_violations += 1
                    self.state.hallucination_violations += 1
                    penalty = REWARD_CFG["hallucination"] + self._terminate_failure("hallucinated_advice_pack")
                    return "POLICY_VIOLATION: unknown advice_pack_id.", penalty, [], ["hallucinated_entity"]
                if advice_pack_id not in self.data.allowed_advice_packs:
                    penalty = self._terminate_failure("advice_pack_not_allowed")
                    return "FAIL: advice pack is not allowed for this case.", penalty, [], ["advice_pack_not_allowed"]

        self.state.final_disposition = disposition
        self.state.final_advice_pack_id = advice_pack_id if isinstance(advice_pack_id, str) else None

        success = (
            disposition in self.data.acceptable_dispositions
            and self.state.critical_policy_violations == 0
            and self._world_action_matches_disposition(disposition)
        )

        delta = 0.0
        if success:
            self.state.done = True
            self.state.done_reason = "finished_success"
            delta += REWARD_CFG["success"]
            return f"SUCCESS: final disposition={disposition}", delta, [], []

        # Failure path.
        if URGENCY_RANK[disposition] < URGENCY_RANK[self.data.target_disposition]:
            delta += REWARD_CFG["undertriage_extra"]
        elif URGENCY_RANK[disposition] > URGENCY_RANK[self.data.target_disposition]:
            delta += REWARD_CFG["over_escalation_extra"]

        if self._has_any_world_action() and not self._world_action_matches_disposition(disposition):
            delta += REWARD_CFG["unnecessary_world_action"]

        delta += self._terminate_failure("wrong_final_decision")
        return f"FAIL: final disposition={disposition}", delta, [], ["wrong_final_decision"]

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def _render_initial_body(self) -> str:
        assert self.data is not None
        risk_flags = self.data.patient_profile.get("risk_flags", [])
        age_group = self.data.patient_profile.get("age_group", "adult")
        return (
            f"CASE: {self.data.case_id}\n"
            f"DIFFICULTY: {self.data.difficulty}\n"
            f"MAX_STEPS: {self.data.max_steps}\n\n"
            f"Patient says:\n\"{self.data.initial_message}\"\n\n"
            f"Profile:\n"
            f"- age_group: {age_group}\n"
            f"- risk_flags: {risk_flags}\n"
        )

    def _render_observation(self, body: str, new_entities: List[str], violations: List[str]) -> str:
        assert self.data is not None and self.state is not None
        remaining = sorted(set(self.data.required_evidence_groups.keys()) - self.state.covered_groups)
        return (
            f"{body}\n\n"
            f"NEW_ENTITIES:\n{new_entities}\n\n"
            f"KNOWN_ENTITIES:\n{sorted(self.state.known_entities)}\n\n"
            f"COVERED_EVIDENCE_GROUPS:\n{sorted(self.state.covered_groups)}\n"
            f"REMAINING_EVIDENCE_GROUPS:\n{remaining}\n\n"
            f"AVAILABLE_TOOLS:\n{TOOL_NAMES}\n\n"
            f"POLICY:\n"
            f"- book_visit and create_escalation require explicit prior confirmation\n"
            f"- use only known entity IDs in tool arguments\n\n"
            f"VIOLATIONS_THIS_STEP:\n{violations}"
        )

    def _recompute_coverage(self) -> None:
        assert self.data is not None and self.state is not None
        covered = set()
        for group_id, entity_ids in self.data.required_evidence_groups.items():
            if any(entity_id in self.state.known_entities for entity_id in entity_ids):
                covered.add(group_id)
        self.state.covered_groups = covered

    def _make_info(
        self,
        action: str,
        action_type: str,
        tool_name: Optional[str],
        new_entities: List[str],
        violations: List[str],
        reward_breakdown: Dict[str, float],
        done_reason: Optional[str],
    ) -> Dict[str, Any]:
        assert self.state is not None
        return {
            "case_id": self.state.case_id,
            "step_idx": self.state.step_idx,
            "action": action,
            "action_type": action_type,
            "tool_name": tool_name,
            "new_entities": list(new_entities),
            "known_entities": sorted(self.state.known_entities),
            "covered_groups_now": sorted(self.state.covered_groups),
            "violations_delta": list(violations),
            "policy_violations_total": self.state.policy_violations,
            "critical_policy_violations_total": self.state.critical_policy_violations,
            "tool_calls_total": self.state.tool_calls,
            "invalid_actions_total": self.state.invalid_actions,
            "reward_breakdown": dict(reward_breakdown),
            "done_reason": done_reason,
            "failure_reason": self.state.failure_reason,
            "confirmed_action": None if self.state.confirmed_action is None else {
                "tool": self.state.confirmed_action.tool,
                "args": dict(self.state.confirmed_action.args),
                "used": self.state.confirmed_action.used,
            },
        }

    def _has_any_world_action(self) -> bool:
        assert self.state is not None
        return self.state.booked_slot_id is not None or self.state.escalation_level is not None

    def _world_action_matches_disposition(self, disposition: str) -> bool:
        assert self.state is not None
        required = build_required_world_action(disposition)
        if required is None:
            return not self._has_any_world_action()
        if required.tool == "book_visit":
            return self.state.booked_slot_id is not None and self.state.booked_urgency == required.urgency
        if required.tool == "create_escalation":
            return self.state.escalation_level == required.level
        return False

    def _terminate_failure(self, reason: str) -> float:
        assert self.state is not None
        self.state.done = True
        self.state.done_reason = "finished_failure"
        self.state.failure_reason = reason
        return REWARD_CFG["fail"]

    def _find_slot(self, slot_id: str) -> Optional[SlotSpec]:
        assert self.data is not None
        for slots in self.data.slot_inventory.values():
            for slot in slots:
                if slot.slot_id == slot_id:
                    return slot
        return None

    def _is_known_entity(self, entity_id: str) -> bool:
        assert self.state is not None
        return entity_id in self.state.known_entities or entity_id in GLOBAL_ENUM_IDS or entity_id in QUESTION_CATALOG
