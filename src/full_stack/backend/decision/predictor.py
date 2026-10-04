"""
COMPASS Decision Predictor

The Predictor role served by a structured decision model (for example TypeSafe
Jev) instead of a conventional LLM.

It reads the same executor output as the LLM Predictor (direct route: the raw
participant record; orchestrated route: tool syntheses, chunk evidence and the
unprocessed raw features), and it returns the same PredictionResult, so the
critic, the reports, the dashboard and the validation layer need no special
path. What differs:

- no generated rationale: the model answers typed questions and returns
  calibrated probabilities;
- every judgement is asked in more than one presentation order and averaged;
- regression outputs are estimated as a continuous density (coarse Score, then a
  zoomed Score) so the point estimate keeps full resolution;
- a `decision_report` records the question book, the questions, the packed
  state, the per-request usage and the quality signals the decision critic reads.

One run makes one request (classification-only and small integer scales) or two
(any continuous regression output: coarse round, then refinement round).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..agents.base_agent import BaseAgent
from ..data.models.prediction_result import (
    KeyFinding,
    NodePrediction,
    PredictionResult,
)
from ..data.models.prediction_task import PredictionTaskNode, PredictionTaskSpec, build_binary_task_spec
from ..utils.json_parser import parse_json_response
from ..utils.token_packer import count_tokens
from .aggregate import (
    ClassificationAggregate,
    RegressionAggregate,
    aggregate_classification,
    answer_usable,
    noul_value,
    aggregate_regression_output,
    build_node_prediction,
    coarse_ensemble,
    is_classification,
    refinement_window,
)
from .client import DecisionClient, DecisionRequestError, DecisionResponse
from .questions import (
    QuestionBook,
    QuestionSet,
    add_fine_regression_questions,
    build_question_book,
    round_one_questions,
    task_hash,
)
from .registry import DecisionModelSpec, get_decision_model_spec
from .scales import default_scale
from .state import PackedState, build_sections, measurement_rows, pack_state

logger = logging.getLogger("compass.decision.predictor")

_TOKEN_HINT = "cl100k_base"
_MARGIN_TOKENS = 256


class DecisionPredictor(BaseAgent):
    """Predictor role on a structured decision model."""

    AGENT_NAME = "Predictor"
    PROMPT_FILE = ""
    predictor_kind = "decision"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = str(self.settings.models.predictor_model)
        spec = get_decision_model_spec(self.model)
        if spec is None:
            raise ValueError(f"{self.model} is not a structured decision model.")
        self.spec: DecisionModelSpec = spec
        self.config = self.settings.decision
        self.client = DecisionClient(settings=self.settings)
        self.revision_feedback = ""
        self._book_cache: Dict[str, QuestionBook] = {}

    # ----------------------------------------------------------------- helpers
    def _count(self, text: str) -> int:
        return count_tokens(text or "", model_hint=_TOKEN_HINT)

    def task_context(self) -> str:
        return str(self.runtime_instruction or "").strip()

    def set_revision_feedback(self, text: Optional[str]) -> None:
        # A decision model cannot read feedback; the critic acts by changing the
        # evidence route instead (see main.py). Kept for interface parity.
        self.revision_feedback = str(text or "").strip()

    def _companion_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        model = str(self.config.compiler_model or self.settings.models.orchestrator_model)
        response = self.llm_client.call(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            model=model,
            max_tokens=6000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if self.token_manager:
            self.token_manager.record_usage(
                component="predictor",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                step_tool="QuestionBookCompiler",
            )
        # The reply is requested as one JSON object; read it whole first. The
        # tolerant parser can settle on a fragment (for example the first node of
        # the "nodes" map), so it is only the fallback for malformed replies.
        parsed: Any = None
        try:
            parsed = json.loads(str(response.content or "").strip())
        except (TypeError, ValueError):
            parsed = parse_json_response(response.content, expected_keys=["nodes"])
        if not isinstance(parsed, dict):
            raise ValueError("Question book compiler returned no JSON object.")
        return parsed

    def question_book(self, task_spec: PredictionTaskSpec) -> QuestionBook:
        context = self.task_context()
        key = task_hash(task_spec, context)
        if key not in self._book_cache:
            compiler = str(self.config.compiler_model or self.settings.models.orchestrator_model)
            book = build_question_book(
                task_spec,
                context=context,
                llm_json=self._companion_json,
                compiler_label=compiler,
            )
            if any(note.startswith("Compilation incomplete") for note in book.notes):
                return book  # not memoized: the next attempt compiles again
            self._book_cache[key] = book
        return self._book_cache[key]

    def _question_tokens(self, questions: QuestionSet) -> List[int]:
        return [self._count(json.dumps(q.payload, ensure_ascii=False)) for q in questions.items]

    def _request_limit(self) -> int:
        return int(self.spec.request_context_tokens / float(self.config.tokenizer_ratio or 1.2)) - _MARGIN_TOKENS

    def _state_budget(self, questions: QuestionSet) -> int:
        """Largest state that fits with this question set (state + longest question, state + all questions)."""
        ratio = float(self.config.tokenizer_ratio or 1.2)
        sizes = self._question_tokens(questions)
        state_limit = self.spec.state_budget_cl100k(ratio=ratio) - max(sizes, default=0) - _MARGIN_TOKENS
        request_limit = self._request_limit() - sum(sizes)
        return min(state_limit, request_limit)

    def _split(self, questions: QuestionSet) -> List[QuestionSet]:
        """
        Split a question set so every request leaves room for the full state. The
        state is billed once per request, so one request is preferred; large
        hierarchies (many outputs, many levels) need more.
        """
        ratio = float(self.config.tokenizer_ratio or 1.2)
        state_room = self.spec.state_budget_cl100k(ratio=ratio)
        per_request = max(2000, self._request_limit() - state_room)
        sizes = self._question_tokens(questions)
        if sum(sizes) <= per_request:
            return [questions]
        parts: List[QuestionSet] = []
        current = QuestionSet(prefix=questions.prefix)
        used = 0
        for q, size in zip(questions.items, sizes):
            if current.items and used + size > per_request:
                parts.append(current)
                current = QuestionSet(prefix=questions.prefix)
                used = 0
            current.items.append(q)
            used += size
        if current.items:
            parts.append(current)
        return parts

    # ------------------------------------------------------------- routing
    def measure_direct(
        self,
        executor_output: Dict[str, Any],
        *,
        prediction_task_spec: PredictionTaskSpec,
        **_: Any,
    ) -> Dict[str, Any]:
        """Tokens of the complete (unpacked) state versus the state budget."""
        book = self.question_book(prediction_task_spec)
        qs = round_one_questions(
            prediction_task_spec,
            book,
            choice_orders=int(self.config.choice_orders),
            score_levels=int(self.config.score_levels),
        )
        sections, _total = build_sections(executor_output=executor_output, task_context=self.task_context(), count=self._count)
        full_state = {s.key: s.variants[0] for s in sections if s.variants and s.variants[0].strip()}
        tokens = self._count(json.dumps(full_state, ensure_ascii=False))
        return {
            "input_tokens": int(tokens),
            "budget_tokens": int(min(self._state_budget(part) for part in self._split(qs))),
            "unit": "cl100k tokens of the decision state",
            "predictor_kind": "decision",
            "model": self.model,
        }

    # ------------------------------------------------------------- requests
    def _ask(self, packed: PackedState, questions: QuestionSet, ledger: List[Dict[str, Any]], round_name: str) -> DecisionResponse:
        response = self.client.ask(model=self.model, state=packed.state, questions=questions.payload())
        if self.token_manager:
            self.token_manager.record_usage(
                component="predictor",
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                step_tool=f"DecisionModel:{self.spec.model_id}:{round_name}",
            )
        try:
            self.llm_client.token_tracker.add(response.usage.input_tokens, response.usage.output_tokens, self.spec.model_id)
        except Exception:
            pass
        ledger.append(
            {
                "round": round_name,
                "questions": len(questions),
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "cost_usd": response.usage.cost_usd,
                "latency_ms": response.latency_ms,
                "served_by": response.model,
                "request_id": response.request_id,
            }
        )
        missing = [q.qid for q in questions.items if q.qid not in response.answers]
        if missing:
            logger.warning("Decision model left %s questions unanswered: %s", len(missing), missing[:6])
        return response

    def _pack_and_ask(
        self,
        executor_output: Dict[str, Any],
        questions: QuestionSet,
        ledger: List[Dict[str, Any]],
        round_name: str,
        packed: Optional[PackedState] = None,
    ) -> Tuple[PackedState, Dict[str, Dict[str, Any]]]:
        """Ask every question of a round (split over requests when needed) on one packed state."""
        parts = self._split(questions)
        budget = min(self._state_budget(part) for part in parts)
        if budget < 1000:
            raise ValueError(
                f"The question set leaves no room for a state on {self.spec.label}; reduce score levels or choice orders."
            )

        def repack(limit: int) -> PackedState:
            sections, total = build_sections(executor_output=executor_output, task_context=self.task_context(), count=self._count)
            return pack_state(sections, budget_tokens=limit, count=self._count, features_total=total)

        if packed is None or packed.tokens > budget:
            packed = repack(budget)
        answers: Dict[str, Dict[str, Any]] = {}
        for index, part in enumerate(parts):
            name = round_name if len(parts) == 1 else f"{round_name}.{index + 1}"
            for attempt in range(3):
                try:
                    answers.update(self._ask(packed, part, ledger, name).answers)
                    break
                except DecisionRequestError as exc:
                    if not exc.looks_like_length_error or attempt == 2:
                        raise
                    # Shrink the state actually sent, not only the budget it was under.
                    budget = int(min(budget, packed.tokens) * 0.8)
                    if budget < 1000:
                        raise
                    logger.warning("Decision state rejected as too long; repacking to %s tokens.", budget)
                    packed = repack(budget)

        # One re-ask for questions the model left unanswered.
        missing = QuestionSet(prefix=questions.prefix)
        missing.items = [q for q in questions.items if not answer_usable(q, answers.get(q.qid))]
        if missing.items:
            for part in self._split(missing):
                try:
                    answers.update(self._ask(packed, part, ledger, f"{round_name}.reask").answers)
                except DecisionRequestError as exc:
                    logger.warning("Re-asking %s unanswered questions failed: %s", len(part), exc)
        return packed, answers

    # ---------------------------------------------------------------- execute
    def execute(
        self,
        executor_output: Dict[str, Any],
        target_condition: str,
        control_condition: str,
        prediction_task_spec: Optional[PredictionTaskSpec] = None,
        iteration: int = 1,
    ) -> PredictionResult:
        if prediction_task_spec is None:
            prediction_task_spec = build_binary_task_spec(target_label=target_condition, control_label=control_condition)
        self._log_start(f"structured decision on {self.model}")
        participant_id = str(executor_output.get("participant_id") or "unknown")
        route = str(executor_output.get("route") or "orchestrated")
        book = self.question_book(prediction_task_spec)
        ledger: List[Dict[str, Any]] = []

        # Round 1: classification, coarse regression and evidence sufficiency.
        round_one = round_one_questions(
            prediction_task_spec,
            book,
            choice_orders=int(self.config.choice_orders),
            score_levels=int(self.config.score_levels),
        )
        packed, first_answers = self._pack_and_ask(executor_output, round_one, ledger, "coarse")
        answers: Dict[str, Dict[str, Any]] = dict(first_answers)
        print(
            f"[DecisionPredictor] State {packed.tokens:,}/{packed.budget:,} tokens, "
            f"{packed.features_included}/{packed.features_total} feature leaves, route={route}"
        )

        # Round 2: zoom into the most likely region of each continuous output.
        round_two = QuestionSet(prefix="r")
        windows: Dict[Tuple[str, str], Tuple[int, int]] = {}
        if bool(self.config.regression_refine):
            for node in prediction_task_spec.root.walk():
                if is_classification(node):
                    continue
                node_book = book.nodes.get(node.node_id)
                for out in node.regression_outputs:
                    scale = (node_book.scales.get(out) if node_book else None) or default_scale(out)
                    coarse_qs = [q for q in round_one.items if q.node_id == node.node_id and q.output == out and q.kind == "score_coarse"]
                    bins, probs, _runs, _conf = coarse_ensemble(coarse_qs, answers)
                    window = refinement_window(scale, bins, probs)
                    if window is None:
                        continue
                    lo, hi = bins[window[0]].lo, bins[window[1] - 1].hi
                    fine_bins = scale.fine_grid(lo, hi, int(self.config.score_levels))
                    add_fine_regression_questions(round_two, node, out, scale, fine_bins, lo, hi)
                    windows[(node.node_id, out)] = window
        if len(round_two):
            packed, second_answers = self._pack_and_ask(executor_output, round_two, ledger, "refine", packed=packed)
            answers.update(second_answers)

        sufficiency = None
        for q in round_one.items:
            if q.kind == "sufficiency":
                sufficiency = noul_value(answers.get(q.qid))

        all_questions = round_one.items + round_two.items
        node_stats: Dict[str, Dict[str, Any]] = {}
        class_threshold = float(self.config.stability_threshold)
        reg_threshold = float(self.config.regression_stability_threshold)

        def build(node: PredictionTaskNode, path: str) -> NodePrediction:
            children = [build(child, f"{path}/{child.node_id}") for child in node.children]
            node_questions = [q for q in all_questions if q.node_id == node.node_id and q.kind != "sufficiency"]
            classification: Optional[ClassificationAggregate] = None
            regression: Dict[str, RegressionAggregate] = {}
            if is_classification(node):
                classification = aggregate_classification(node, node_questions, answers)
                node_stats[node.node_id] = {
                    "kind": "classification",
                    "instability": classification.stability,
                    "threshold": class_threshold,
                    "confidence": classification.confidence,
                    "unassessed": classification.unassessed,
                }
            else:
                node_book = book.nodes.get(node.node_id)
                for out in node.regression_outputs:
                    scale = (node_book.scales.get(out) if node_book else None) or default_scale(out)
                    coarse_qs = [q for q in node_questions if q.output == out and q.kind == "score_coarse"]
                    fine_qs = [q for q in node_questions if q.output == out and q.kind == "score_fine"]
                    regression[out] = aggregate_regression_output(scale, coarse_qs, fine_qs, answers, windows.get((node.node_id, out)))
                stabilities = [a.stability for a in regression.values()]
                confidences = [a.confidence for a in regression.values()]
                node_stats[node.node_id] = {
                    "kind": "regression",
                    "instability": max(stabilities) if stabilities else 0.0,
                    "threshold": reg_threshold,
                    "confidence": sum(confidences) / len(confidences) if confidences else 0.5,
                    "unassessed": any(a.unassessed for a in regression.values()),
                }
            return build_node_prediction(
                node,
                path=path,
                classification=classification,
                regression=regression,
                children=children,
                sufficiency=sufficiency if node.node_id == prediction_task_spec.root.node_id else None,
                stability_threshold=class_threshold,
                regression_stability_threshold=reg_threshold,
            )

        root = build(prediction_task_spec.root, prediction_task_spec.root.node_id)
        report = self._report(
            book=book,
            questions=all_questions,
            packed=packed,
            ledger=ledger,
            route=route,
            sufficiency=sufficiency,
            node_stats=node_stats,
        )
        result = self._result(
            root=root,
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            task_spec=prediction_task_spec,
            executor_output=executor_output,
            iteration=iteration,
            report=report,
            packed=packed,
        )
        self._log_complete(
            f"{len(all_questions)} questions in {len(ledger)} request(s), "
            f"instability {report['quality']['max_instability']:.2f} of threshold, "
            f"evidence sufficiency {sufficiency if sufficiency is not None else 'n/a'}"
        )
        return result

    # --------------------------------------------------------------- outputs
    def _report(
        self,
        *,
        book: QuestionBook,
        questions: List[Any],
        packed: PackedState,
        ledger: List[Dict[str, Any]],
        route: str,
        sufficiency: Optional[float],
        node_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        for stats in node_stats.values():
            ratio = float(stats["instability"]) / max(1e-9, float(stats["threshold"]))
            # Unanswered presentation orders: unstable whatever the threshold.
            stats["instability_ratio"] = max(ratio, 2.0) if stats.get("unassessed") else ratio
        instabilities = [float(s["instability_ratio"]) for s in node_stats.values()]
        confidences = [float(s["confidence"]) for s in node_stats.values()]
        return {
            "model": self.model,
            "model_label": self.spec.label,
            "route": route,
            "question_book": book.to_dict(),
            "question_book_hash": book.content_hash(),
            "questions": [q.summary() for q in questions],
            "state": packed.report(),
            "requests": ledger,
            "cost_usd": float(sum(float(r.get("cost_usd") or 0.0) for r in ledger)),
            "input_tokens": int(sum(int(r.get("input_tokens") or 0) for r in ledger)),
            "quality": {
                "evidence_sufficiency": sufficiency,
                # Instability as a multiple of each node's threshold (1.0 = at the threshold).
                "mean_instability": (sum(instabilities) / len(instabilities)) if instabilities else 0.0,
                "max_instability": max(instabilities) if instabilities else 0.0,
                "default_scale_outputs": book.default_scale_outputs(),
                "placeholder_items": book.placeholder_items(),
                "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
                "feature_coverage": packed.feature_coverage,
                "per_node": node_stats,
            },
            "settings": {
                "choice_orders": int(self.config.choice_orders),
                "stability_threshold": float(self.config.stability_threshold),
                "regression_stability_threshold": float(self.config.regression_stability_threshold),
                "score_levels": int(self.config.score_levels),
                "regression_refine": bool(self.config.regression_refine),
                "tokenizer_ratio": float(self.config.tokenizer_ratio),
            },
        }

    def _key_findings(self, executor_output: Dict[str, Any], limit: int = 8) -> List[KeyFinding]:
        """Largest deviations in the record, as context (not model rationale)."""
        predictor_input = executor_output.get("predictor_input") or {}
        multimodal = predictor_input.get("multimodal_unprocessed_raw") or executor_output.get("multimodal_data") or {}
        rows = [(g, line, z) for g, line, z in measurement_rows(multimodal) if z is not None]
        rows.sort(key=lambda r: -abs(r[2]))
        findings: List[KeyFinding] = []
        for group, line, z in rows[:limit]:
            direction = "ABNORMAL_HIGH" if z >= 1.5 else ("ABNORMAL_LOW" if z <= -1.5 else "NORMAL")
            parts = group.split(" > ") if group else ["record"]
            within = " > ".join(parts[1:])
            findings.append(
                KeyFinding(
                    domain=parts[0],
                    finding=f"{within}: {line}" if within else line,
                    direction=direction,
                    z_score=round(float(z), 3),
                    relevance_to_prediction=(
                        "One of the largest deviations in the participant record. Listed as context; "
                        "a decision model returns probabilities, not a rationale."
                    ),
                )
            )
        return findings

    def _summary(self, root: NodePrediction, report: Dict[str, Any]) -> str:
        quality = report["quality"]
        parts: List[str] = []
        for node in root.walk():
            if node.classification is not None:
                p = node.classification.probabilities.get(node.classification.predicted_label, 0.0)
                parts.append(f"{node.node_id}: {node.classification.predicted_label} (p={p:.2f})")
            elif node.regression is not None:
                vals = ", ".join(
                    f"{k}={v:.2f} (sd {node.regression.uncertainty.get(k, {}).get('sd', 0.0):.2f})"
                    for k, v in node.regression.values.items()
                )
                parts.append(f"{node.node_id}: {vals}")
        lead = (
            f"Structured decision by {self.spec.label} on the {report['route']} evidence route "
            f"({report['state']['features_included']}/{report['state']['features_total']} feature leaves in the state)."
        )
        tail = (
            f" Mean confidence {quality['mean_confidence']:.2f}; instability across presentation orders "
            f"{quality['max_instability']:.2f} of the threshold"
        )
        if quality.get("evidence_sufficiency") is not None:
            tail += f"; evidence sufficiency {quality['evidence_sufficiency']:.2f}"
        return lead + " " + "; ".join(parts[:12]) + ("." if parts else "") + tail + "."

    def _result(
        self,
        *,
        root: NodePrediction,
        participant_id: str,
        target_condition: str,
        control_condition: str,
        task_spec: PredictionTaskSpec,
        executor_output: Dict[str, Any],
        iteration: int,
        report: Dict[str, Any],
        packed: PackedState,
    ) -> PredictionResult:
        uncertainty: List[str] = []
        for node in root.walk():
            uncertainty.extend(node.uncertainty_factors)
        if packed.feature_coverage < 0.999:
            uncertainty.append(
                f"State packing kept {packed.features_included} of {packed.features_total} feature leaves "
                f"to fit the {self.spec.label} limit."
            )
        quality = report["quality"]
        reasoning = [
            f"Evidence route: {report['route']}; state of {packed.tokens:,} tokens (budget {packed.budget:,}).",
            f"Question book compiled by {report['question_book'].get('compiled_by')} and frozen for this task.",
            f"{len(report['questions'])} typed questions in {len(report['requests'])} request(s); each judgement asked in "
            f"several presentation orders and averaged.",
            f"Instability across presentation orders (multiple of the threshold): mean "
            f"{quality['mean_instability']:.2f}, max {quality['max_instability']:.2f}.",
        ]
        if quality.get("evidence_sufficiency") is not None:
            reasoning.append(f"Evidence sufficiency (Noul): {quality['evidence_sufficiency']:.2f}.")
        return PredictionResult(
            prediction_id=f"decision_{uuid.uuid4().hex[:10]}",
            participant_id=participant_id,
            target_condition=target_condition,
            control_condition=control_condition,
            created_at=datetime.now(),
            prediction_task_spec=task_spec,
            root_prediction=root,
            confidence_level=root.confidence_level,
            key_findings=self._key_findings(executor_output),
            reasoning_chain=reasoning,
            uncertainty_factors=uncertainty,
            clinical_summary=self._summary(root, report),
            domains_processed=list(executor_output.get("domains_processed") or []),
            total_tokens_used=int(report["input_tokens"]),
            iteration=int(iteration),
            predictor_kind="decision",
            predictor_model=self.model,
            input_route=str(report["route"]),
            decision_report=report,
        )

