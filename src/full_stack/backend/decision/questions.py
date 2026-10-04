"""
From a prediction task to typed decision-model questions.

The instrument is built in two steps.

1. A question book (once per task, then frozen). For every task node it holds a
   plain-language question, an operational definition per class label and a
   measurement scale per regression output. Values given in the task
   specification (`class_definitions`, `output_scales`) are used as given; the
   rest is written by the companion LLM from the task specification and the
   study context only (never from participant data) and cached on disk. Every
   participant in a cohort is therefore scored with the same questions, which
   is what makes decision-model outputs comparable across participants.

2. A question set per request, compiled from the book:
   - binary node:      one Noul (P of the first label) plus the two-option Choice
                       asked in both option orders;
   - multiclass node:  the Choice asked in up to `choice_orders` rotated orders;
   - regression output: a coarse Score asked in ascending and descending level
                       order, then (second request) a zoomed Score over the
                       most likely region, again in both orders;
   - every request:    one Noul on whether the record holds enough evidence.

Asking every judgement in more than one presentation order and averaging is the
documented remedy for the model's preference for the first option, and the
spread between orders is the stability signal the decision critic reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data.models.prediction_task import PredictionMode, PredictionTaskNode, PredictionTaskSpec
from .scales import OutputScale, ScaleBin, default_scale, scale_from_mapping

logger = logging.getLogger("compass.decision.questions")

BOOK_VERSION = "2026-10-04.1"


# ----------------------------------------------------------------------------
# Question book
# ----------------------------------------------------------------------------
@dataclass
class NodeBook:
    node_id: str
    question: str
    label_definitions: Dict[str, str] = field(default_factory=dict)
    scales: Dict[str, OutputScale] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "question": self.question,
            "label_definitions": dict(self.label_definitions),
            "scales": {k: v.to_dict() for k, v in self.scales.items()},
        }


@dataclass
class QuestionBook:
    version: str
    task_hash: str
    nodes: Dict[str, NodeBook]
    compiled_by: str = "deterministic"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "task_hash": self.task_hash,
            "compiled_by": self.compiled_by,
            "notes": list(self.notes),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionBook":
        nodes: Dict[str, NodeBook] = {}
        for node_id, raw in dict(data.get("nodes") or {}).items():
            scales = {}
            for out, sdata in dict(raw.get("scales") or {}).items():
                scales[out] = OutputScale(**sdata)
            nodes[node_id] = NodeBook(
                node_id=node_id,
                question=str(raw.get("question") or ""),
                label_definitions=dict(raw.get("label_definitions") or {}),
                scales=scales,
            )
        return cls(
            version=str(data.get("version") or ""),
            task_hash=str(data.get("task_hash") or ""),
            nodes=nodes,
            compiled_by=str(data.get("compiled_by") or "deterministic"),
            notes=list(data.get("notes") or []),
        )


def task_hash(task_spec: PredictionTaskSpec, context: str) -> str:
    payload = task_spec.model_dump() if hasattr(task_spec, "model_dump") else task_spec.dict()
    blob = json.dumps({"task": payload, "context": context, "version": BOOK_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _default_question(node: PredictionTaskNode) -> str:
    name = str(node.display_name or node.node_id).strip()
    if len(name) > 90:
        name = str(node.node_id).replace("_", " ")
    if node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
        return f"Which group does this participant belong to for: {name}?"
    return f"What is this participant's {name}?"


def _deterministic_node_book(node: PredictionTaskNode) -> NodeBook:
    labels = {}
    for label in node.class_labels:
        given = str((node.class_definitions or {}).get(label) or "").strip()
        labels[label] = given or f"The participant belongs to the '{label}' group."
    scales: Dict[str, OutputScale] = {}
    for out in node.regression_outputs:
        unit = str((node.unit_by_output or {}).get(out) or "")
        given = scale_from_mapping(out, (node.output_scales or {}).get(out) or {}, source="task_spec", unit=unit)
        scales[out] = given or default_scale(out, unit=unit)
    return NodeBook(node_id=node.node_id, question=_default_question(node), label_definitions=labels, scales=scales)


def deterministic_book(task_spec: PredictionTaskSpec, context: str = "") -> QuestionBook:
    nodes = {n.node_id: _deterministic_node_book(n) for n in task_spec.root.walk()}
    return QuestionBook(version=BOOK_VERSION, task_hash=task_hash(task_spec, context), nodes=nodes)


def _needs_compilation(task_spec: PredictionTaskSpec) -> bool:
    for node in task_spec.root.walk():
        for label in node.class_labels:
            if not str((node.class_definitions or {}).get(label) or "").strip():
                return True
        for out in node.regression_outputs:
            if scale_from_mapping(out, (node.output_scales or {}).get(out) or {}, source="task_spec") is None:
                return True
    return False


COMPILER_SYSTEM_PROMPT = (
    "You prepare the question instrument for a structured decision model that will rate clinical "
    "research participants. You never see participant data. From the task specification and the "
    "study context, write precise, literal questions, operational class definitions, and measurement "
    "scales. The decision model reads instructions literally and is weak with arithmetic, so state "
    "conditions explicitly and describe scale positions in words. Return only one JSON object."
)


def _compiler_prompt(task_spec: PredictionTaskSpec, context: str) -> str:
    spec_json = task_spec.model_dump_json(indent=1) if hasattr(task_spec, "model_dump_json") else json.dumps(task_spec.dict())
    rows = [
        "## Task specification",
        spec_json,
        "",
        "## Study context (may describe the cohort and how targets are defined)",
        context.strip() or "Not provided.",
        "",
        "## What to return",
        "For EVERY node_id in the specification return an entry:",
        "{",
        '  "nodes": {',
        '    "<node_id>": {',
        '      "question": "one literal question about the participant for this node",',
        '      "label_definitions": {"<class label>": "operational definition, 1 sentence"},',
        '      "scales": {',
        '        "<regression output>": {',
        '          "description": "what is measured, naming the instrument",',
        '          "min": 0, "max": 60, "integer": true, "unit": "points",',
        '          "low_meaning": "what a minimum value means", "high_meaning": "what a maximum value means",',
        '          "reference_mean": null, "reference_sd": null',
        "        }",
        "      }",
        "    }",
        "  }",
        "}",
        "Rules:",
        "- question: one short, literal question about this participant for this node only (at most 25 words),",
        "  in plain clinical language; never list the other nodes in it.",
        "- label_definitions only for classification nodes, using the exact labels given. When the specification",
        "  already defines a label (class_definitions), keep its meaning.",
        "- scales only for regression outputs, using the exact output names given.",
        "- min and max must be the instrument's real possible range (not a guess about this cohort).",
        "  For change or difference scores use a symmetric plausible range around 0.",
        "- reference_mean and reference_sd: typical value and spread in the population the study context",
        "  describes, when you know them; otherwise null.",
        "- Keep definitions consistent with the study context (for example responder thresholds).",
    ]
    return "\n".join(rows)


def _merge_compiled(task_spec: PredictionTaskSpec, base: QuestionBook, compiled: Dict[str, Any]) -> QuestionBook:
    nodes_raw = dict((compiled or {}).get("nodes") or {})
    for node in task_spec.root.walk():
        entry = nodes_raw.get(node.node_id) or {}
        book = base.nodes[node.node_id]
        question = str(entry.get("question") or "").strip()
        if question:
            book.question = question
        defs = dict(entry.get("label_definitions") or {})
        for label in node.class_labels:
            if str((node.class_definitions or {}).get(label) or "").strip():
                continue  # the task specification wins
            text = str(defs.get(label) or "").strip()
            if text:
                book.label_definitions[label] = text
        scales = dict(entry.get("scales") or {})
        for out in node.regression_outputs:
            if book.scales.get(out) is not None and book.scales[out].source == "task_spec":
                continue
            unit = str((node.unit_by_output or {}).get(out) or "")
            compiled_scale = scale_from_mapping(out, scales.get(out) or {}, source="compiler", unit=unit)
            if compiled_scale is not None:
                book.scales[out] = compiled_scale
            else:
                base.notes.append(f"No usable compiled scale for '{node.node_id}.{out}'; default scale kept.")
    return base


def cache_dir() -> Path:
    root = os.getenv("COMPASS_CACHE_DIR", "").strip()
    if root:
        return Path(root).expanduser() / "decision_question_books"
    return Path(__file__).resolve().parents[4] / ".compass_cache" / "decision_question_books"


def build_question_book(
    task_spec: PredictionTaskSpec,
    *,
    context: str,
    llm_json: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    compiler_label: str = "",
    use_cache: bool = True,
) -> QuestionBook:
    """
    Return the frozen question book for this task and context.

    `llm_json(system_prompt, user_prompt)` must return a parsed JSON object; it
    is only called when the task specification leaves something undefined and
    no cached book exists.
    """
    base = deterministic_book(task_spec, context)
    if llm_json is None and not _needs_compilation(task_spec):
        base.compiled_by = "task_spec"
        return base
    # With an LLM available the book is always compiled: the question wording
    # matters to a model that reads literally. Definitions and scales given in
    # the task specification are kept as given.
    path = cache_dir() / f"{base.task_hash}.json"
    if use_cache and path.exists():
        try:
            return QuestionBook.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            logger.warning("Ignoring unreadable question book cache %s: %s", path, exc)
    if llm_json is None:
        base.notes.append("No companion LLM available; generic label definitions and default scales used.")
        return base
    try:
        compiled = llm_json(COMPILER_SYSTEM_PROMPT, _compiler_prompt(task_spec, context))
        book = _merge_compiled(task_spec, base, compiled)
        book.compiled_by = compiler_label or "companion_llm"
    except Exception as exc:
        logger.warning("Question book compilation failed, using deterministic book: %s", exc)
        base.notes.append(f"Compilation failed ({type(exc).__name__}); generic definitions used.")
        return base
    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(book.to_dict(), indent=1))
        except Exception as exc:
            logger.warning("Could not cache question book: %s", exc)
    return book


# ----------------------------------------------------------------------------
# Question sets
# ----------------------------------------------------------------------------
@dataclass
class AskedQuestion:
    qid: str
    node_id: str
    kind: str  # binary_noul | choice | score_coarse | score_fine | sufficiency
    payload: Dict[str, Any]
    output: str = ""
    variant: str = ""
    # Choice: labels in presented order. Score: index of the bin shown at each level.
    order: List[Any] = field(default_factory=list)
    bins: List[ScaleBin] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "qid": self.qid,
            "node_id": self.node_id,
            "kind": self.kind,
            "output": self.output,
            "variant": self.variant,
            "order": list(self.order),
        }


class QuestionSet:
    """Questions of one request. The prefix keeps ids unique across rounds."""

    def __init__(self, prefix: str = "q") -> None:
        self.prefix = prefix
        self.items: List[AskedQuestion] = []

    def add(self, **kwargs: Any) -> AskedQuestion:
        q = AskedQuestion(qid=f"{self.prefix}{len(self.items) + 1:03d}", **kwargs)
        self.items.append(q)
        return q

    def payload(self) -> Dict[str, Dict[str, Any]]:
        return {q.qid: q.payload for q in self.items}

    def longest_question_chars(self) -> int:
        return max((len(json.dumps(q.payload)) for q in self.items), default=0)

    def all_chars(self) -> int:
        return sum(len(json.dumps(q.payload)) for q in self.items)

    def __len__(self) -> int:
        return len(self.items)


def _rotations(labels: Sequence[str], count: int) -> List[List[str]]:
    labels = list(labels)
    n = len(labels)
    if n <= 1:
        return [labels]
    count = max(1, min(int(count), n))
    if n == 2:
        return [labels, list(reversed(labels))][:max(2, count)]
    shifts = sorted({int(round(i * n / count)) % n for i in range(count)})
    out = [labels[s:] + labels[:s] for s in shifts]
    if count >= 2 and list(reversed(labels)) not in out:
        out[-1] = list(reversed(labels))
    return out


def _record_ref(node_book: NodeBook) -> str:
    return f"{node_book.question} Use `task_context` for study definitions and the participant record for evidence."


def add_classification_questions(qs: QuestionSet, node: PredictionTaskNode, book: NodeBook, *, choice_orders: int) -> None:
    labels = list(node.class_labels)
    defs = {label: book.label_definitions.get(label) or f"The participant belongs to the '{label}' group." for label in labels}
    if node.mode == PredictionMode.BINARY_CLASSIFICATION and len(labels) == 2:
        first, second = labels
        qs.add(
            node_id=node.node_id,
            kind="binary_noul",
            variant="noul",
            order=[first, second],
            payload={
                "type": "noul",
                "instructions": (
                    f"{book.question} Is it more likely that the participant belongs to '{first}' "
                    f"than to '{second}'? Answer for this participant using `task_context` and the participant record."
                ),
                "criteria": {"true": f"{first}: {defs[first]}", "false": f"{second}: {defs[second]}"},
            },
        )
        orders = _rotations(labels, 2)
    else:
        orders = _rotations(labels, choice_orders)
    for index, order in enumerate(orders):
        qs.add(
            node_id=node.node_id,
            kind="choice",
            variant=f"order{index + 1}",
            order=list(order),
            payload={
                "type": "choice",
                "instructions": _record_ref(book),
                "criteria": {label: defs[label] for label in order},
            },
        )


def _score_payload(scale: OutputScale, bins: Sequence[ScaleBin], order: Sequence[int], instructions: str) -> Dict[str, Any]:
    total = len(bins)
    levels = [scale.level_label(bins[i], index=i, total=total) for i in order]
    return {"type": "score", "instructions": instructions, "criteria": levels}


def _scale_instruction(node: PredictionTaskNode, book: NodeBook, scale: OutputScale) -> str:
    what = scale.description or scale.output.replace("_", " ")
    unit = f" {scale.unit}" if scale.unit else ""
    lo = scale.format_value(scale.minimum, width=scale.span / 10.0)
    hi = scale.format_value(scale.maximum, width=scale.span / 10.0)
    meaning = ""
    if scale.low_meaning or scale.high_meaning:
        meaning = f" Low values mean {scale.low_meaning or 'less'}; high values mean {scale.high_meaning or 'more'}."
    return (
        f"Estimate this participant's {what} ({scale.output}), which ranges from {lo} to {hi}{unit}.{meaning} "
        f"Choose the level that contains the participant's most likely value, using `task_context` and the participant record."
    )


def add_coarse_regression_questions(
    qs: QuestionSet, node: PredictionTaskNode, book: NodeBook, *, levels: int
) -> None:
    for out in node.regression_outputs:
        scale = book.scales.get(out) or default_scale(out)
        bins = scale.coarse_grid(levels)
        instructions = _scale_instruction(node, book, scale)
        ascending = list(range(len(bins)))
        for variant, order in (("ascending", ascending), ("descending", list(reversed(ascending)))):
            qs.add(
                node_id=node.node_id,
                kind="score_coarse",
                output=out,
                variant=variant,
                order=order,
                bins=list(bins),
                payload=_score_payload(scale, bins, order, instructions),
            )


def add_fine_regression_questions(
    qs: QuestionSet,
    node: PredictionTaskNode,
    output: str,
    scale: OutputScale,
    fine_bins: Sequence[ScaleBin],
    window_lo: float,
    window_hi: float,
) -> None:
    unit = f" {scale.unit}" if scale.unit else ""
    width = (window_hi - window_lo) / 10.0
    what = scale.description or scale.output.replace("_", " ")
    instructions = (
        f"The participant's {what} ({scale.output}) most likely lies between "
        f"{scale.format_value(window_lo, width=width)} and {scale.format_value(window_hi, width=width)}{unit}. "
        f"Within that range, choose the level that contains the participant's most likely value, "
        f"using `task_context` and the participant record."
    )
    ascending = list(range(len(fine_bins)))
    for variant, order in (("ascending", ascending), ("descending", list(reversed(ascending)))):
        qs.add(
            node_id=node.node_id,
            kind="score_fine",
            output=output,
            variant=variant,
            order=order,
            bins=list(fine_bins),
            payload=_score_payload(scale, fine_bins, order, instructions),
        )


def add_sufficiency_question(qs: QuestionSet, task_spec: PredictionTaskSpec, book: QuestionBook) -> None:
    root = task_spec.root
    root_book = book.nodes.get(root.node_id)
    focus = (root_book.question if root_book and root_book.question else _default_question(root)).rstrip("?")
    extra = len(root.walk()) - 1
    if extra:
        focus = f"{focus} (and {extra} related outputs)"
    qs.add(
        node_id=root.node_id,
        kind="sufficiency",
        variant="evidence",
        payload={
            "type": "noul",
            "instructions": (
                f"Does the participant record contain enough relevant evidence to make an informed judgement on this question: {focus}?"
            ),
            "criteria": {
                "true": "The record holds direct or closely related evidence (scores, symptoms, history, measurements) for this judgement.",
                "false": "The record holds little or no relevant evidence; any judgement would be close to a guess.",
            },
        },
    )


def round_one_questions(task_spec: PredictionTaskSpec, book: QuestionBook, *, choice_orders: int, score_levels: int) -> QuestionSet:
    qs = QuestionSet()
    for node in task_spec.root.walk():
        node_book = book.nodes.get(node.node_id) or _deterministic_node_book(node)
        if node.mode in (PredictionMode.BINARY_CLASSIFICATION, PredictionMode.MULTICLASS_CLASSIFICATION):
            add_classification_questions(qs, node, node_book, choice_orders=choice_orders)
        else:
            add_coarse_regression_questions(qs, node, node_book, levels=score_levels)
    add_sufficiency_question(qs, task_spec, book)
    return qs


def asked_summary(qs: QuestionSet) -> List[Dict[str, Any]]:
    return [q.summary() for q in qs.items]

