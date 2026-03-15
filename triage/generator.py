from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from triage.catalogs import (
    FAMILY_BASE_INITIAL_ENTITIES,
    FAMILY_DISTRACTOR_ENTITIES,
    PROTOCOL_CATALOG,
)
from triage.constants import DIFFICULTY_CFG, DISPOSITIONS, FAMILIES
from triage.rule_engine import (
    build_acceptable_dispositions,
    build_required_evidence_groups,
    build_required_world_action,
    get_allowed_advice_packs,
    get_relevant_question_ids,
    infer_target_disposition,
)
from triage.schema import QuestionResponse, SlotSpec, TriageData
from triage.text_templates import answer_for_question, render_initial_message


class CaseGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.seed = seed

    def generate(
        self,
        num_of_questions: int = 100,
        max_attempts: int = 100,
        difficulty: Optional[int] = 1,
        **kwargs: Any,
    ) -> List[TriageData]:
        difficulty = int(difficulty or 1)
        if difficulty not in DIFFICULTY_CFG:
            raise ValueError(f"difficulty must be in 1..10, got {difficulty}")

        rows: List[TriageData] = []
        for idx in range(num_of_questions):
            case = None
            for _ in range(max_attempts):
                case = self._generate_one(index=idx, difficulty=difficulty, **kwargs)
                if self._validate_case(case):
                    break
            if case is None or not self._validate_case(case):
                raise RuntimeError(f"Could not generate a valid case after {max_attempts} attempts")
            rows.append(case)
        return rows

    def _generate_one(self, index: int, difficulty: int, **kwargs: Any) -> TriageData:
        cfg = DIFFICULTY_CFG[difficulty]
        local_seed = kwargs.get("seed")
        rng = random.Random(local_seed + index) if isinstance(local_seed, int) else self.rng

        family = kwargs.get("family") or self._sample_family(rng, index)
        target = kwargs.get("target_disposition") or self._sample_target_disposition(rng, index)
        facts = self._sample_hidden_facts(family=family, target=target, difficulty=difficulty, rng=rng)
        derived_target = infer_target_disposition(family, facts)
        if derived_target != target:
            # Keep world deterministic: trust the rule engine and align target.
            target = derived_target

        profile = self._build_patient_profile(family=family, facts=facts, difficulty=difficulty, rng=rng)
        initial_entities = self._build_initial_entities(
            family=family,
            facts=facts,
            profile=profile,
            difficulty=difficulty,
            rng=rng,
            num_distractors=kwargs.get("num_distractors", cfg["num_distractors"]),
        )
        initial_message = render_initial_message(family, facts, profile, rng)

        required_groups = build_required_evidence_groups(family, facts, difficulty)
        relevant_question_ids = get_relevant_question_ids(family, facts, difficulty)
        qa_map = self._build_qa_map(
            family=family,
            facts=facts,
            relevant_question_ids=relevant_question_ids,
            required_groups=required_groups,
            rng=rng,
        )

        allowed_advice_packs = get_allowed_advice_packs(family, target)
        slot_inventory = self._build_slot_inventory(target=target, rng=rng)
        max_steps = kwargs.get("max_steps", cfg["max_steps"])

        case_id = kwargs.get("case_id") or f"TRIAGE_{family}_{difficulty:02d}_{index:05d}"
        metadata = {
            "noise_level": kwargs.get("noise", cfg["noise"]),
            "num_distractors": kwargs.get("num_distractors", cfg["num_distractors"]),
            "protocol_lookup_expected": cfg["protocol_lookup_expected"],
            "question_protocol": PROTOCOL_CATALOG[family]["recommended_questions"],
            "generator_seed": local_seed,
        }

        return TriageData(
            case_id=case_id,
            difficulty=difficulty,
            family=family,
            initial_message=initial_message,
            patient_profile=profile,
            initial_entities=sorted(set(initial_entities)),
            hidden_facts=facts,
            qa_map=qa_map,
            relevant_question_ids=relevant_question_ids,
            required_evidence_groups=required_groups,
            target_disposition=target,
            acceptable_dispositions=build_acceptable_dispositions(target, difficulty),
            required_world_action=build_required_world_action(target),
            allowed_advice_packs=allowed_advice_packs,
            slot_inventory=slot_inventory,
            max_steps=max_steps,
            seed=local_seed,
            metadata=metadata,
        )

    def _sample_family(self, rng: random.Random, index: int) -> str:
        return FAMILIES[index % len(FAMILIES)] if self.seed is None else rng.choice(FAMILIES)

    def _sample_target_disposition(self, rng: random.Random, index: int) -> str:
        return DISPOSITIONS[index % len(DISPOSITIONS)] if self.seed is None else rng.choice(DISPOSITIONS)

    def _build_patient_profile(self, family: str, facts: Dict[str, bool], difficulty: int, rng: random.Random) -> Dict[str, Any]:
        del family
        if facts.get("AGE_SENIOR"):
            age_group = "senior"
        elif facts.get("AGE_CHILD"):
            age_group = "child"
        else:
            age_group = "adult"

        reveal_risks = difficulty <= 3
        risk_flags: List[str] = []
        for risk in ["RISK_ASTHMA", "RISK_IMMUNOCOMP"]:
            if facts.get(risk) and reveal_risks:
                risk_flags.append(risk)

        return {
            "age_group": age_group,
            "risk_flags": risk_flags,
            "style": rng.choice(["terse", "neutral", "concise"]),
        }

    def _build_initial_entities(
        self,
        family: str,
        facts: Dict[str, bool],
        profile: Dict[str, Any],
        difficulty: int,
        rng: random.Random,
        num_distractors: int,
    ) -> List[str]:
        entities = list(FAMILY_BASE_INITIAL_ENTITIES[family])

        age_group = profile["age_group"]
        entities.append(f"AGE_{age_group.upper()}")
        entities.extend(profile.get("risk_flags", []))

        # reveal some family-specific surface symptoms up front
        for candidate in [
            "SYM_FEVER",
            "SYM_COUGH",
            "SYM_HEADACHE",
            "SYM_RASH",
            "SYM_ABDOMINAL_PAIN",
            "SYM_DYSURIA",
            "SYM_NAUSEA",
        ]:
            if facts.get(candidate):
                entities.append(candidate)

        if difficulty <= 2:
            for duration in ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"]:
                if facts.get(duration):
                    entities.append(duration)
                    break

        if difficulty <= 3 and family == "RESP":
            for fever in ["ANS_TEMP_39_PLUS", "ANS_TEMP_SUB39"]:
                if facts.get(fever):
                    entities.append(fever)
                    break

        distractors = rng.sample(FAMILY_DISTRACTOR_ENTITIES[family], k=min(num_distractors, len(FAMILY_DISTRACTOR_ENTITIES[family])))
        entities.extend(distractors)
        return entities

    def _build_qa_map(
        self,
        family: str,
        facts: Dict[str, bool],
        relevant_question_ids: List[str],
        required_groups: Dict[str, List[str]],
        rng: random.Random,
    ) -> Dict[str, QuestionResponse]:
        qa_map: Dict[str, QuestionResponse] = {}
        for qid in relevant_question_ids:
            new_entities = self._entities_for_question(family, qid, facts)
            evidence_groups = [
                group_id
                for group_id, entity_ids in required_groups.items()
                if any(ent in entity_ids for ent in new_entities)
            ]
            answer_text = answer_for_question(qid, facts, rng)
            qa_map[qid] = QuestionResponse(
                answer_text=answer_text,
                new_entities=new_entities,
                evidence_groups=evidence_groups,
            )
        return qa_map

    def _build_slot_inventory(self, target: str, rng: random.Random) -> Dict[str, List[SlotSpec]]:
        def mk(prefix: str, label1: str, label2: str) -> List[SlotSpec]:
            return [
                SlotSpec(slot_id=f"{prefix}_1", urgency=prefix.split("_")[0], label=label1),
                SlotSpec(slot_id=f"{prefix}_2", urgency=prefix.split("_")[0], label=label2),
            ]

        routine = [
            SlotSpec(slot_id="ROUTINE_1", urgency="ROUTINE", label="Tomorrow 10:30"),
            SlotSpec(slot_id="ROUTINE_2", urgency="ROUTINE", label="Tomorrow 14:10"),
        ]
        same_day = [
            SlotSpec(slot_id="SAME_DAY_1", urgency="SAME_DAY", label="Today 16:10"),
            SlotSpec(slot_id="SAME_DAY_2", urgency="SAME_DAY", label="Today 18:20"),
        ]
        if target == "BOOK_ROUTINE":
            return {"ROUTINE": routine, "SAME_DAY": same_day[:1]}
        if target == "BOOK_SAME_DAY":
            return {"ROUTINE": routine[:1], "SAME_DAY": same_day}
        return {"ROUTINE": routine[:1], "SAME_DAY": same_day[:1]} if rng.random() < 0.25 else {"ROUTINE": [], "SAME_DAY": []}

    def _sample_hidden_facts(self, family: str, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        facts: Dict[str, bool] = {}
        if family == "RESP":
            facts = self._facts_resp(target, difficulty, rng)
        elif family == "GI":
            facts = self._facts_gi(target, difficulty, rng)
        elif family == "UTI":
            facts = self._facts_uti(target, difficulty, rng)
        elif family == "HEADACHE":
            facts = self._facts_headache(target, difficulty, rng)
        elif family == "RASH":
            facts = self._facts_rash(target, difficulty, rng)
        elif family == "ABDOMINAL":
            facts = self._facts_abdominal(target, difficulty, rng)
        else:
            raise ValueError(f"Unknown family: {family}")
        return facts

    def _facts_resp(self, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        facts = {"SYM_COUGH": True, "SYM_FEVER": True}
        self._set_duration(facts, target)
        facts["ANS_CHEST_PAIN_NO"] = True
        facts["ANS_CONFUSION_NO"] = True
        facts["ANS_SOB_NO"] = True
        facts["AGE_ADULT"] = True

        if target == "SELF_CARE":
            facts["ANS_TEMP_SUB39"] = True
        elif target == "BOOK_ROUTINE":
            facts["ANS_TEMP_SUB39"] = True
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_SAME_DAY":
            facts["ANS_TEMP_39_PLUS"] = True
            facts.pop("AGE_ADULT", None)
            if rng.random() < 0.5:
                facts["AGE_SENIOR"] = True
            else:
                facts["RISK_ASTHMA"] = True
        elif target == "ESCALATE_NOW":
            facts["ANS_TEMP_39_PLUS"] = True
            facts.pop("ANS_SOB_NO", None)
            if rng.random() < 0.5:
                facts["SYM_DYSPNEA"] = True
            else:
                facts.pop("ANS_CHEST_PAIN_NO", None)
                facts["SYM_CHEST_PAIN"] = True
        if difficulty >= 7 and target in {"BOOK_SAME_DAY", "ESCALATE_NOW"} and rng.random() < 0.3:
            facts["RISK_IMMUNOCOMP"] = True
        return facts

    def _facts_gi(self, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        del difficulty
        facts = {"SYM_NAUSEA": True, "ANS_BLOOD_NO": True, "ANS_PAIN_MILD": True, "ANS_FLUIDS_OK": True, "ANS_VOMIT_FREQ_LOW": True, "ANS_DEHYDRATION_NONE": True, "AGE_ADULT": True}
        self._set_duration(facts, target)
        if target == "BOOK_ROUTINE":
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_SAME_DAY":
            facts.pop("ANS_VOMIT_FREQ_LOW", None)
            facts["ANS_VOMIT_FREQ_HIGH"] = True
            facts.pop("ANS_DEHYDRATION_NONE", None)
            facts["ANS_DEHYDRATION_MILD"] = True
            facts["ANS_DURATION_1D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "ESCALATE_NOW":
            facts.pop("ANS_BLOOD_NO", None)
            facts["ANS_BLOOD_YES"] = True
            if rng.random() < 0.5:
                facts.pop("ANS_PAIN_MILD", None)
                facts["ANS_PAIN_SEVERE"] = True
            else:
                facts.pop("ANS_DEHYDRATION_NONE", None)
                facts["ANS_DEHYDRATION_SEVERE"] = True
        return facts

    def _facts_uti(self, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        del difficulty
        facts = {"SYM_DYSURIA": True, "ANS_URINARY_FREQ_YES": True, "ANS_FLANK_PAIN_NO": True, "ANS_FEVER_LOW": True, "ANS_BLOOD_NO": True, "AGE_ADULT": True}
        self._set_duration(facts, target)
        if target == "BOOK_ROUTINE":
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_SAME_DAY":
            facts.pop("ANS_FEVER_LOW", None)
            facts["ANS_FEVER_HIGH"] = True
            facts["ANS_DURATION_1_3D"] = True
        elif target == "ESCALATE_NOW":
            facts.pop("ANS_FLANK_PAIN_NO", None)
            if rng.random() < 0.5:
                facts["ANS_FLANK_PAIN_YES"] = True
            else:
                facts.pop("ANS_BLOOD_NO", None)
                facts["ANS_BLOOD_YES"] = True
        return facts

    def _facts_headache(self, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        del difficulty
        facts = {"SYM_HEADACHE": True, "ANS_HEADACHE_SUDDEN_NO": True, "ANS_NECK_STIFFNESS_NO": True, "ANS_WEAKNESS_NO": True, "ANS_CONFUSION_NO": True, "ANS_PAIN_MILD": True, "AGE_ADULT": True}
        self._set_duration(facts, target)
        if target == "BOOK_ROUTINE":
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_SAME_DAY":
            facts.pop("ANS_PAIN_MILD", None)
            facts["ANS_PAIN_SEVERE"] = True
            facts.pop("AGE_ADULT", None)
            facts["AGE_SENIOR"] = True
        elif target == "ESCALATE_NOW":
            choice = rng.choice(["ANS_HEADACHE_SUDDEN_YES", "ANS_NECK_STIFFNESS_YES", "ANS_WEAKNESS_YES", "ANS_CONFUSION_YES"])
            for negative in ["ANS_HEADACHE_SUDDEN_NO", "ANS_NECK_STIFFNESS_NO", "ANS_WEAKNESS_NO", "ANS_CONFUSION_NO"]:
                facts.pop(negative, None)
            facts[choice] = True
        return facts

    def _facts_rash(self, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        del difficulty
        facts = {"SYM_RASH": True, "ANS_SWELLING_NO": True, "ANS_SOB_NO": True, "ANS_RASH_SPREAD_SLOW": True, "AGE_ADULT": True}
        self._set_duration(facts, target)
        if target == "BOOK_ROUTINE":
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_SAME_DAY":
            if rng.random() < 0.5:
                facts.pop("ANS_RASH_SPREAD_SLOW", None)
                facts["ANS_RASH_SPREAD_FAST"] = True
            else:
                facts["RISK_IMMUNOCOMP"] = True
        elif target == "ESCALATE_NOW":
            if rng.random() < 0.5:
                facts.pop("ANS_SWELLING_NO", None)
                facts["ANS_SWELLING_YES"] = True
            else:
                facts.pop("ANS_SOB_NO", None)
                facts["SYM_DYSPNEA"] = True
        return facts

    def _facts_abdominal(self, target: str, difficulty: int, rng: random.Random) -> Dict[str, bool]:
        del difficulty
        facts = {"SYM_ABDOMINAL_PAIN": True, "ANS_PAIN_MILD": True, "ANS_DIFFUSE_PAIN": True, "ANS_VOMIT_FREQ_LOW": True, "ANS_FEVER_LOW": True, "AGE_ADULT": True}
        self._set_duration(facts, target)
        if target == "BOOK_ROUTINE":
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_SAME_DAY":
            if rng.random() < 0.5:
                facts.pop("ANS_DIFFUSE_PAIN", None)
                facts["ANS_RLQ_PAIN"] = True
            else:
                facts.pop("ANS_FEVER_LOW", None)
                facts["ANS_FEVER_HIGH"] = True
        elif target == "ESCALATE_NOW":
            if rng.random() < 0.5:
                facts.pop("ANS_PAIN_MILD", None)
                facts["ANS_PAIN_SEVERE"] = True
            else:
                facts.pop("ANS_VOMIT_FREQ_LOW", None)
                facts.pop("ANS_FEVER_LOW", None)
                facts["ANS_VOMIT_FREQ_HIGH"] = True
                facts["ANS_FEVER_HIGH"] = True
        return facts

    def _set_duration(self, facts: Dict[str, bool], target: str) -> None:
        facts["ANS_DURATION_1_3D"] = True
        if target == "SELF_CARE":
            facts["ANS_DURATION_1D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        elif target == "BOOK_ROUTINE":
            facts["ANS_DURATION_GT5D"] = True
            facts.pop("ANS_DURATION_1_3D", None)
        else:
            facts["ANS_DURATION_1_3D"] = True

    def _entities_for_question(self, family: str, question_id: str, facts: Dict[str, bool]) -> List[str]:
        del family
        mapping = {
            "Q_DURATION": ["ANS_DURATION_1D", "ANS_DURATION_1_3D", "ANS_DURATION_GT5D"],
            "Q_TEMPERATURE": ["ANS_TEMP_39_PLUS", "ANS_TEMP_SUB39", "ANS_FEVER_HIGH", "ANS_FEVER_LOW"],
            "Q_SHORTNESS_OF_BREATH": ["SYM_DYSPNEA", "ANS_SOB_NO"],
            "Q_CHEST_PAIN": ["SYM_CHEST_PAIN", "ANS_CHEST_PAIN_NO"],
            "Q_CONFUSION": ["ANS_CONFUSION_YES", "ANS_CONFUSION_NO"],
            "Q_VOMITING_FREQUENCY": ["ANS_VOMIT_FREQ_HIGH", "ANS_VOMIT_FREQ_LOW"],
            "Q_FLUID_INTAKE": ["ANS_FLUIDS_POOR", "ANS_FLUIDS_OK"],
            "Q_DEHYDRATION_SIGNS": ["ANS_DEHYDRATION_SEVERE", "ANS_DEHYDRATION_MILD", "ANS_DEHYDRATION_NONE"],
            "Q_BLOOD_PRESENT": ["ANS_BLOOD_YES", "ANS_BLOOD_NO"],
            "Q_PAIN_SEVERITY": ["ANS_PAIN_SEVERE", "ANS_PAIN_MILD"],
            "Q_FLANK_PAIN": ["ANS_FLANK_PAIN_YES", "ANS_FLANK_PAIN_NO"],
            "Q_NECK_STIFFNESS": ["ANS_NECK_STIFFNESS_YES", "ANS_NECK_STIFFNESS_NO"],
            "Q_WEAKNESS": ["ANS_WEAKNESS_YES", "ANS_WEAKNESS_NO"],
            "Q_RASH_SPREAD": ["ANS_RASH_SPREAD_FAST", "ANS_RASH_SPREAD_SLOW"],
            "Q_SWELLING": ["ANS_SWELLING_YES", "ANS_SWELLING_NO"],
            "Q_COMORBIDITY": ["RISK_ASTHMA", "RISK_IMMUNOCOMP", "ANS_COMORBIDITY_NO"],
            "Q_URINARY_FREQUENCY": ["ANS_URINARY_FREQ_YES", "ANS_URINARY_FREQ_NO"],
            "Q_BURNING_URINATION": ["SYM_DYSURIA"],
            "Q_HEADACHE_SUDDEN": ["ANS_HEADACHE_SUDDEN_YES", "ANS_HEADACHE_SUDDEN_NO"],
            "Q_LOCALIZATION": ["ANS_RLQ_PAIN", "ANS_DIFFUSE_PAIN"],
        }
        out = [entity for entity in mapping.get(question_id, []) if facts.get(entity)]
        if question_id == "Q_COMORBIDITY" and not out:
            out = ["ANS_COMORBIDITY_NO"]
        if question_id == "Q_URINARY_FREQUENCY" and not out:
            out = ["ANS_URINARY_FREQ_NO"]
        return out

    def _validate_case(self, case: TriageData) -> bool:
        if infer_target_disposition(case.family, case.hidden_facts) != case.target_disposition:
            return False
        if case.target_disposition not in case.acceptable_dispositions:
            return False
        if not case.required_evidence_groups:
            return False
        group_ids = set(case.required_evidence_groups.keys())
        covered_initially = {
            gid for gid, entities in case.required_evidence_groups.items() if any(ent in case.initial_entities for ent in entities)
        }
        revealed_groups = set(covered_initially)
        for resp in case.qa_map.values():
            for gid in resp.evidence_groups:
                revealed_groups.add(gid)
        if not group_ids.issubset(revealed_groups):
            return False
        if case.target_disposition == "SELF_CARE" and not case.allowed_advice_packs:
            return False
        return True
