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
        parsed = parse_json_response(response.content, expected_keys=["nodes"])
        if not isinstance(parsed, dict):
            raise ValueError("Question book compiler returned no JSON object.")
        return parsed

    def question_book(self, task_spec: PredictionTaskSpec) -> QuestionBook:
        context = self.task_context()
        key = f"{id(task_spec)}:{hash(context)}"
        if key not in self._book_cache:
            compiler = str(self.config.compiler_model or self.settings.models.orchestrator_model)
            self._book_cache[key] = build_question_book(
                task_spec,
                context=context,
                llm_json=self._companion_json,
                compiler_label=compiler,
            )
        return self._book_cache[key]

    def _state_budget(self, questions: QuestionSet) -> int:
        ratio = float(self.config.tokenizer_ratio or 1.2)
        sizes = [self._count(json.dumps(q.payload, ensure_ascii=False)) for q in questions.items]
        state_limit = self.spec.state_budget_cl100k(ratio=ratio) - max(sizes, default=0) - _MARGIN_TOKENS
        request_limit = int(self.spec.request_context_tokens / ratio) - sum(sizes) - _MARGIN_TOKENS
        return max(1000, min(state_limit, request_limit))

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
            "budget_tokens": int(self._state_budget(qs)),
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
    ) -> Tuple[PackedState, DecisionResponse]:
        budget = self._state_budget(questions)
        if packed is None or packed.tokens > budget:
            sections, total = build_sections(executor_output=executor_output, task_context=self.task_context(), count=self._count)
            packed = pack_state(sections, budget_tokens=budget, count=self._count, features_total=total)
        for attempt in range(3):
            try:
                return packed, self._ask(packed, questions, ledger, round_name)
            except DecisionRequestError as exc:
                if not exc.looks_like_length_error or attempt == 2:
                    raise
                budget = int(budget * 0.8)
                logger.warning("Decision state rejected as too long; repacking to %s tokens.", budget)
                sections, total = build_sections(executor_output=executor_output, task_context=self.task_context(), count=self._count)
                packed = pack_state(sections, budget_tokens=budget, count=self._count, features_total=total)
        raise RuntimeError("unreachable")

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
        packed, first = self._pack_and_ask(executor_output, round_one, ledger, "coarse")
        answers: Dict[str, Dict[str, Any]] = dict(first.answers)
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
            packed, second = self._pack_and_ask(executor_output, round_two, ledger, "refine", packed=packed)
            answers.update(second.answers)

        sufficiency = None
        for q in round_one.items:
            if q.kind == "sufficiency" and isinstance(answers.get(q.qid), dict):
                try:
                    sufficiency = max(0.0, min(1.0, float(answers[q.qid].get("noul"))))
                except (TypeError, ValueError):
                    sufficiency = None

        all_questions = round_one.items + round_two.items
        node_stats: Dict[str, Dict[str, Any]] = {}

        def build(node: PredictionTaskNode, path: str) -> NodePrediction:
            children = [build(child, f"{path}/{child.node_id}") for child in node.children]
            node_questions = [q for q in all_questions if q.node_id == node.node_id and q.kind != "sufficiency"]
            classification: Optional[ClassificationAggregate] = None
            regression: Dict[str, RegressionAggregate] = {}
            if is_classification(node):
                classification = aggregate_classification(node, node_questions, answers)
                node_stats[node.node_id] = {"stability": classification.stability, "confidence": classification.confidence}
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
                    "stability": sum(stabilities) / len(stabilities) if stabilities else 0.0,
                    "confidence": sum(confidences) / len(confidences) if confidences else 0.5,
                }
            return build_node_prediction(
                node,
                path=path,
                classification=classification,
                regression=regression,
                children=children,
                sufficiency=sufficiency if node.node_id == prediction_task_spec.root.node_id else None,
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
            f"stability {report['quality']['mean_instability']:.3f}, "
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
        instabilities = [float(s["stability"]) for s in node_stats.values()]
        confidences = [float(s["confidence"]) for s in node_stats.values()]
        return {
            "model": self.model,
            "model_label": self.spec.label,
            "route": route,
            "question_book": book.to_dict(),
            "questions": [q.summary() for q in questions],
            "state": packed.report(),
            "requests": ledger,
            "cost_usd": float(sum(float(r.get("cost_usd") or 0.0) for r in ledger)),
            "input_tokens": int(sum(int(r.get("input_tokens") or 0) for r in ledger)),
            "quality": {
                "evidence_sufficiency": sufficiency,
                "mean_instability": (sum(instabilities) / len(instabilities)) if instabilities else 0.0,
                "max_instability": max(instabilities) if instabilities else 0.0,
                "mean_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
                "feature_coverage": packed.feature_coverage,
                "per_node": node_stats,
            },
            "settings": {
                "choice_orders": int(self.config.choice_orders),
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
            findings.append(
                KeyFinding(
                    domain=group.split(" > ")[0] if group else "record",
                    finding=f"{group}: {line}" if group else line,
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
            f"{quality['mean_instability']:.3f}"
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
            f"Instability across orders: mean {quality['mean_instability']:.3f}, max {quality['max_instability']:.3f}.",
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

