from __future__ import annotations

from typing import Dict, List, Set

from triage.action_parser import render_tool_call
from triage.schema import TriageData


class OracleSolver:
    """Deterministic, environment-aware policy for sanity checks and validation."""

    def solve(self, data: TriageData) -> List[str]:
        actions: List[str] = []
        known: Set[str] = set(data.initial_entities)
        covered = self._covered_groups(data, known)

        need_protocol = data.target_disposition == "SELF_CARE"
        if need_protocol:
            lookup_entities = sorted([ent for ent in known if ent.startswith(("SYM_", "AGE_", "RISK_", "ANS_"))])
            actions.append(render_tool_call("lookup_protocol", {"entity_ids": lookup_entities}))
            known.update(data.allowed_advice_packs)

        while covered != set(data.required_evidence_groups.keys()):
            qid = self._choose_best_question(data, known, covered)
            if qid is None:
                break
            actions.append(render_tool_call("ask_question", {"question_id": qid}))
            resp = data.qa_map[qid]
            known.update(resp.new_entities)
            covered = self._covered_groups(data, known)

        disposition = data.target_disposition
        if disposition in {"BOOK_ROUTINE", "BOOK_SAME_DAY"}:
            urgency = "ROUTINE" if disposition == "BOOK_ROUTINE" else "SAME_DAY"
            actions.append(render_tool_call("list_slots", {"urgency": urgency}))
            slots = data.slot_inventory.get(urgency, [])
            if not slots:
                raise RuntimeError(f"Oracle cannot solve case {data.case_id}: no slots for {urgency}")
            slot_id = slots[0].slot_id
            actions.append(
                f'I recommend a {urgency.lower().replace("_", "-")} visit.\n<CONFIRM>{{"tool":"book_visit","slot_id":"{slot_id}"}}</CONFIRM>'
            )
            actions.append(render_tool_call("book_visit", {"slot_id": slot_id}))
            actions.append(render_tool_call("finish", {"disposition": disposition}))
            return actions

        if disposition == "ESCALATE_NOW":
            actions.append(
                'This case should be escalated urgently.\n<CONFIRM>{"tool":"create_escalation","level":"ESCALATE_NOW"}</CONFIRM>'
            )
            actions.append(render_tool_call("create_escalation", {"level": "ESCALATE_NOW"}))
            actions.append(render_tool_call("finish", {"disposition": "ESCALATE_NOW"}))
            return actions

        # SELF_CARE
        if not data.allowed_advice_packs:
            raise RuntimeError(f"Oracle cannot solve case {data.case_id}: no advice pack available")
        advice_pack = data.allowed_advice_packs[0]
        actions.append(render_tool_call("finish", {"disposition": "SELF_CARE", "advice_pack_id": advice_pack}))
        return actions

    def _covered_groups(self, data: TriageData, known: Set[str]) -> Set[str]:
        return {
            gid
            for gid, entities in data.required_evidence_groups.items()
            if any(ent in known for ent in entities)
        }

    def _choose_best_question(self, data: TriageData, known: Set[str], covered: Set[str]) -> str | None:
        remaining = set(data.required_evidence_groups.keys()) - covered
        best_qid = None
        best_gain = -1
        for qid, resp in data.qa_map.items():
            if all(ent in known for ent in resp.new_entities):
                continue
            gain = len(set(resp.evidence_groups) & remaining)
            if gain > best_gain:
                best_gain = gain
                best_qid = qid
        if best_gain <= 0:
            # fall back to any unanswered question that introduces something new
            for qid, resp in data.qa_map.items():
                if not all(ent in known for ent in resp.new_entities):
                    return qid
            return None
        return best_qid
