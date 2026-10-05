"""Normalize COMPASS predictions and annotations into one tidy long table.

The tidy table has one row per prediction instance x task node x output:

- a *prediction instance* is one engine run for one participant in one
  configuration (for example one tier, one predictor, one repeat);
- a *node* is a node of the COMPASS ``PredictionTaskSpec`` tree;
- an *output* is a regression output name, the node id for a classification
  node, or a label name for a multi-label group.

Every expected (instance, node, output) cell gets a row, so a missing
prediction is a row with ``status == "missing_prediction"`` and is counted by
the metric functions instead of silently disappearing.

Columns (see ``TIDY_COLUMNS``): ``instance_id``, ``participant_id``,
``node_id``, ``parent_node_id``, ``depth``, ``output``, ``kind`` (one of
``regression``, ``binary``, ``multiclass``, ``multilabel``), ``status``,
``true_value``, ``pred_value``, ``error`` (pred minus true), ``true_label``,
``pred_label``, ``correct``, ``positive_label``, ``p_positive``, ``p_true``,
``probs`` (dict of class probabilities), plus any grouping or metadata
columns (``tier``, ``predictor``, ``provider``, ``repeat``, ``route`` and so
on) carried over from the prediction records and the annotations.

Only numpy and pandas are needed; the engine package is never imported, so a
``PredictionTaskSpec`` object, its ``model_dump()`` dict or its JSON file all
work as task structure.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .io_utils import (
    extract_participant_id_candidates,
    resolve_identifier,
)

TIDY_COLUMNS = [
    "instance_id",
    "participant_id",
    "node_id",
    "parent_node_id",
    "depth",
    "output",
    "kind",
    "status",
    "true_value",
    "pred_value",
    "error",
    "true_label",
    "pred_label",
    "correct",
    "positive_label",
    "p_positive",
    "p_true",
    "probs",
]

KINDS = ("regression", "binary", "multiclass", "multilabel")
STATUSES = ("ok", "missing_prediction", "missing_truth", "missing_both")

REGRESSION_MODES = {"univariate_regression", "multivariate_regression", "regression"}
CLASSIFICATION_MODES = {"binary_classification", "multiclass_classification", "classification"}

_ID_KEYS = ("participant_id", "eid", "subject", "subject_id", "participant", "id", "sub")
_FAILED_STATUSES = {"failed", "error", "fail", "timeout", "crashed", "skipped"}
_NEGATIVE_LABELS = {
    "control", "controls", "healthy", "healthy control", "healthy controls", "hc",
    "negative", "absent", "no", "none", "false", "0", "non_case", "non-case", "noncase",
    "normal", "typical", "typically developing", "td", "unaffected",
}
_LABEL_KEYS = ("label", "predicted_label", "classification", "class", "class_label", "target", "diagnosis")
_PROB_KEYS = ("probs", "probabilities", "probability_by_label", "class_probabilities")
# Record fields that are payload, not metadata.
_RECORD_PAYLOAD_KEYS = {
    "prediction", "predictions", "report", "root_prediction", "flat_predictions", "classifications",
    "regression", "values", "nodes", "label", "predicted_label", "probs", "probabilities",
    "ground_truth", "labels_truth", "truth", "annotation", "annotations", "multilabel",
    "prediction_task_spec", "task_spec", "outputs", "diagnosis", "prediction_result",
}
_ANNOTATION_RESERVED = {
    "label", "classification", "class", "target", "class_label", "diagnosis", "regression",
    "values", "value", "output_name", "nodes", "labels_truth", "ground_truth", "multilabel",
}
_DEFAULT_META_NAMES = {"disorder", "group", "cohort", "phenotype_group", "site", "sex", "split", "fold"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _norm_text(value: Any) -> str:
    return " ".join(str(value).strip().split()).casefold()


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, np.integer, np.floating))


def _clean_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def _read_json_any(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                out.append(item)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, na_values=["", "NA", "NaN", "nan", "null", "None"])


def _plain(obj: Any) -> Any:
    """Turn a pydantic model (v1 or v2) into plain dicts and lists."""
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")) and not isinstance(obj, dict):
        try:
            return obj.dict()
        except TypeError:
            pass
    return obj


# ---------------------------------------------------------------------------
# Task structure
# ---------------------------------------------------------------------------


@dataclass
class NodeInfo:
    node_id: str
    mode: str
    class_labels: List[str] = field(default_factory=list)
    regression_outputs: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    depth: int = 0
    positive_label: Optional[str] = None
    display_name: str = ""

    @property
    def is_regression(self) -> bool:
        return self.mode in REGRESSION_MODES

    @property
    def kind(self) -> str:
        if self.mode == "multilabel":
            return "multilabel"
        if self.is_regression:
            return "regression"
        if self.mode == "binary_classification" or len(self.class_labels) == 2:
            return "binary"
        return "multiclass"


def guess_positive_label(labels: Sequence[str]) -> Optional[str]:
    """Positive label of a binary node: the label that does not look negative,
    else the first label (COMPASS convention: ``class_labels[0]`` is the case)."""
    labels = [str(x) for x in labels if x is not None]
    if len(labels) != 2:
        return None
    neg = [_norm_text(x) in _NEGATIVE_LABELS for x in labels]
    if neg[0] and not neg[1]:
        return labels[1]
    if neg[1] and not neg[0]:
        return labels[0]
    if {_norm_text(x) for x in labels} == {"1", "0"}:
        return [x for x in labels if _norm_text(x) == "1"][0]
    return labels[0]


class TaskStructure:
    """Node tree of a COMPASS prediction task (from a spec or inferred)."""

    def __init__(self, nodes: Sequence[NodeInfo], source: str = "spec") -> None:
        self.nodes: Dict[str, NodeInfo] = {}
        for node in nodes:
            self.nodes[node.node_id] = node
        self.source = source

    # -- construction -------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Any, positive_labels: Optional[Mapping[str, str]] = None) -> "TaskStructure":
        """Build from a PredictionTaskSpec object, its dict, a node dict, or a JSON path."""
        if isinstance(spec, (str, Path)) and Path(str(spec)).exists():
            spec = _read_json_any(Path(str(spec)))
        elif isinstance(spec, str):
            spec = json.loads(spec)
        spec = _plain(spec)
        if not isinstance(spec, dict):
            raise ValueError("task spec must be a PredictionTaskSpec, a dict, or a JSON file")
        root = spec.get("root") if isinstance(spec.get("root"), dict) else spec
        if "node_id" not in root and "mode" not in root:
            raise ValueError("task spec has no 'root' node with 'node_id' and 'mode'")
        nodes: List[NodeInfo] = []
        positive_labels = dict(positive_labels or {})

        def _walk(node: Dict[str, Any], parent: Optional[str], depth: int) -> None:
            node = _plain(node)
            mode = str(getattr(node.get("mode"), "value", node.get("mode")) or "").strip().lower()
            node_id = str(node.get("node_id") or ("root" if parent is None else f"node_{len(nodes)}")).strip()
            labels = [str(x) for x in (node.get("class_labels") or [])]
            outputs = [str(x) for x in (node.get("regression_outputs") or [])]
            if not mode:
                mode = "multivariate_regression" if len(outputs) > 1 else (
                    "univariate_regression" if outputs else (
                        "binary_classification" if len(labels) == 2 else "multiclass_classification"))
            info = NodeInfo(
                node_id=node_id,
                mode=mode,
                class_labels=labels,
                regression_outputs=outputs,
                parent=parent,
                depth=depth,
                display_name=str(node.get("display_name") or node_id),
            )
            if info.kind == "binary":
                info.positive_label = positive_labels.get(node_id) or guess_positive_label(labels)
            nodes.append(info)
            for child in node.get("children") or []:
                _walk(child, node_id, depth + 1)

        _walk(root, None, 0)
        return cls(nodes, source="spec")

    # -- queries ------------------------------------------------------------
    @property
    def root_id(self) -> Optional[str]:
        for node in self.nodes.values():
            if node.parent is None:
                return node.node_id
        return None

    def output_to_node(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for node in self.nodes.values():
            for name in node.regression_outputs:
                out.setdefault(name, node.node_id)
        return out

    def classification_nodes(self) -> List[str]:
        return [n.node_id for n in self.nodes.values() if n.kind in {"binary", "multiclass"}]

    def expected_keys(self) -> List[Tuple[str, str]]:
        keys: List[Tuple[str, str]] = []
        for node in self.nodes.values():
            if node.is_regression:
                keys.extend((node.node_id, out) for out in node.regression_outputs)
            elif node.kind in {"binary", "multiclass"}:
                keys.append((node.node_id, node.node_id))
            else:
                keys.extend((node.node_id, label) for label in node.class_labels)
        return keys

    def ancestors(self, node_id: str) -> List[str]:
        out: List[str] = []
        cur = self.nodes.get(node_id)
        while cur is not None and cur.parent is not None:
            out.append(cur.parent)
            cur = self.nodes.get(cur.parent)
        return out

    def paths(self) -> List[List[str]]:
        """Root-to-leaf node id paths."""
        children: Dict[Optional[str], List[str]] = {}
        for node in self.nodes.values():
            children.setdefault(node.parent, []).append(node.node_id)
        leaves = [nid for nid in self.nodes if nid not in children]
        return [list(reversed([leaf] + self.ancestors(leaf))) for leaf in leaves]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "mode": n.mode,
                    "kind": n.kind,
                    "parent": n.parent,
                    "depth": n.depth,
                    "class_labels": n.class_labels,
                    "regression_outputs": n.regression_outputs,
                    "positive_label": n.positive_label,
                }
                for n in self.nodes.values()
            ],
        }


def load_task_spec(spec: Any, positive_labels: Optional[Mapping[str, str]] = None) -> TaskStructure:
    """Public alias of ``TaskStructure.from_spec``."""
    if isinstance(spec, TaskStructure):
        return spec
    return TaskStructure.from_spec(spec, positive_labels=positive_labels)


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


@dataclass
class PredictionSet:
    """Predictions in an intermediate long form.

    ``instances``: one row per prediction instance (``instance_id``,
    ``participant_id``, ``ok`` and metadata columns). ``rows``: one row per
    predicted value (``instance_id``, ``node_id`` (may be None), ``output``,
    ``kind_hint``, ``pred_value``, ``pred_label``, ``probs``). ``truth``:
    ground truth embedded in the records (same layout as ``load_annotations``).
    ``specs``: task specs found inside COMPASS reports.
    """

    instances: pd.DataFrame
    rows: pd.DataFrame
    truth: pd.DataFrame
    specs: List[Dict[str, Any]] = field(default_factory=list)


_PRED_ROW_COLUMNS = ["instance_id", "node_id", "output", "kind_hint", "pred_value", "pred_label", "probs"]
_TRUTH_COLUMNS = ["participant_id", "node_id", "output", "kind_hint", "true_value", "true_label", "instance_id"]


class _Collector:
    def __init__(self) -> None:
        self.instances: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []
        self.truth: List[Dict[str, Any]] = []
        self.specs: List[Dict[str, Any]] = []
        # Truth emitted while this is set is embedded in that prediction instance.
        self.current_iid: Optional[int] = None

    def new_instance(self, participant_id: Any, meta: Mapping[str, Any], ok: bool) -> int:
        iid = len(self.instances)
        rec = {"instance_id": iid, "participant_id": None if participant_id is None else str(participant_id), "ok": bool(ok)}
        for key, value in meta.items():
            if key in rec:
                continue
            rec[key] = value
        self.instances.append(rec)
        return iid

    def pred(self, iid: int, node_id: Optional[str], output: str, *, value: Any = None, label: Any = None,
             probs: Optional[Mapping[str, Any]] = None, kind_hint: Optional[str] = None) -> None:
        self.rows.append(
            {
                "instance_id": iid,
                "node_id": node_id,
                "output": str(output),
                "kind_hint": kind_hint,
                "pred_value": _finite(value),
                "pred_label": _clean_label(label),
                "probs": _probs_dict(probs),
            }
        )

    def true(self, pid: Any, node_id: Optional[str], output: str, *, value: Any = None, label: Any = None,
             kind_hint: Optional[str] = None) -> None:
        self.truth.append(
            {
                "participant_id": str(pid),
                "node_id": node_id,
                "output": str(output),
                "kind_hint": kind_hint,
                "true_value": _finite(value),
                "true_label": _clean_label(label),
                "instance_id": self.current_iid,
            }
        )

    def result(self) -> PredictionSet:
        inst = pd.DataFrame(self.instances) if self.instances else pd.DataFrame(columns=["instance_id", "participant_id", "ok"])
        rows = pd.DataFrame(self.rows, columns=_PRED_ROW_COLUMNS)
        truth = pd.DataFrame(self.truth, columns=_TRUTH_COLUMNS)
        return PredictionSet(instances=inst, rows=rows, truth=truth, specs=self.specs)


def _probs_dict(probs: Any) -> Optional[Dict[str, float]]:
    if probs is None:
        return None
    if isinstance(probs, str):
        text = probs.strip()
        if not text:
            return None
        try:
            probs = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(probs, Mapping):
        return None
    out: Dict[str, float] = {}
    for key, value in probs.items():
        fv = _finite(value)
        if fv is not None:
            out[str(key)] = fv
    return out or None


def _first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = mapping[key]
            if isinstance(value, str) and not value.strip():
                continue
            if isinstance(value, float) and not math.isfinite(value):
                continue
            return value
    return None


def _participant_of(rec: Mapping[str, Any]) -> Optional[str]:
    value = _first(rec, _ID_KEYS)
    return None if value is None else str(value)


def _emit_node_prediction(col: _Collector, iid: int, node: Mapping[str, Any]) -> None:
    """One COMPASS NodePrediction dict (children walked recursively)."""
    node = _plain(node)
    if not isinstance(node, Mapping):
        return
    node_id = str(node.get("node_id") or "root")
    mode = str(getattr(node.get("mode"), "value", node.get("mode")) or "").lower()
    cls = node.get("classification") if isinstance(node.get("classification"), Mapping) else None
    reg = node.get("regression") if isinstance(node.get("regression"), Mapping) else None
    if cls is not None or mode in CLASSIFICATION_MODES:
        cls = cls or {}
        col.pred(iid, node_id, node_id, label=_first(cls, _LABEL_KEYS), probs=_first(cls, _PROB_KEYS), kind_hint="classification")
    if reg is not None:
        values = reg.get("values") if isinstance(reg.get("values"), Mapping) else {}
        for name, value in values.items():
            col.pred(iid, node_id, name, value=value, kind_hint="regression")
    for child in node.get("children") or []:
        _emit_node_prediction(col, iid, child)


def _prediction_block(rec: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    """(prediction block, task spec) of a record, or (None, spec) when absent."""
    spec = None
    for holder in (rec, rec.get("prediction") if isinstance(rec.get("prediction"), Mapping) else {}):
        if isinstance(holder.get("prediction_task_spec"), Mapping):
            spec = holder["prediction_task_spec"]
            break
    if isinstance(rec.get("report"), Mapping):
        inner_pred, inner_spec = _prediction_block(rec["report"])
        return inner_pred, spec or inner_spec
    for key in ("prediction", "prediction_result", "predictions"):
        block = rec.get(key)
        if isinstance(block, Mapping):
            return block, spec
    execution = rec.get("execution_summary")
    if isinstance(execution, Mapping) and isinstance(execution.get("final_prediction"), Mapping):
        return execution["final_prediction"], spec
    if any(k in rec for k in ("root_prediction", "flat_predictions", "classifications", "regression", "values", "nodes")):
        return rec, spec
    return None, spec


def _emit_prediction_block(col: _Collector, iid: int, block: Mapping[str, Any]) -> int:
    """Emit prediction rows from any supported block shape; returns rows emitted."""
    before = len(col.rows)
    seen_nodes: set = set()
    root = block.get("root_prediction")
    if isinstance(root, Mapping) or hasattr(root, "model_dump"):
        _emit_node_prediction(col, iid, root)
        seen_nodes = {r["node_id"] for r in col.rows[before:]}
    flat = block.get("flat_predictions")
    if isinstance(flat, list):
        for node in flat:
            node = _plain(node)
            if isinstance(node, Mapping) and str(node.get("node_id") or "") not in seen_nodes:
                child_free = dict(node)
                child_free["children"] = []
                _emit_node_prediction(col, iid, child_free)
                seen_nodes.add(str(node.get("node_id") or ""))
    if len(col.rows) > before:
        return len(col.rows) - before
    # Legacy extracted form: {"nodes": {node_id: {"predicted_label", "probabilities", "values"}}}
    nodes = block.get("nodes")
    if isinstance(nodes, Mapping):
        for node_id, payload in nodes.items():
            if not isinstance(payload, Mapping):
                continue
            label = _first(payload, _LABEL_KEYS)
            if label is not None or _first(payload, _PROB_KEYS) is not None:
                col.pred(iid, str(node_id), str(node_id), label=label, probs=_first(payload, _PROB_KEYS), kind_hint="classification")
            values = payload.get("values") if isinstance(payload.get("values"), Mapping) else {}
            for name, value in values.items():
                col.pred(iid, str(node_id), name, value=value, kind_hint="regression")
    classifications = block.get("classifications")
    if isinstance(classifications, Mapping):
        for node_id, payload in classifications.items():
            if isinstance(payload, Mapping):
                col.pred(iid, str(node_id), str(node_id), label=_first(payload, _LABEL_KEYS),
                         probs=_first(payload, _PROB_KEYS), kind_hint="classification")
            else:
                col.pred(iid, str(node_id), str(node_id), label=payload, kind_hint="classification")
    has_node_cls = isinstance(classifications, Mapping) and len(classifications) > 0
    label = _first(block, _LABEL_KEYS)
    probs = _first(block, _PROB_KEYS)
    if not has_node_cls and (label is not None or isinstance(probs, Mapping)) and not isinstance(label, (Mapping, list)):
        if not isinstance(probs, Mapping):
            p_pos = _finite(_first(block, ("p_positive", "probability", "binary_probability", "probability_score")))
            probs = {"__p_positive__": p_pos} if p_pos is not None else None
        col.pred(iid, None, "__label__", label=label, probs=probs, kind_hint="classification")
    for key in ("regression", "values", "regression_values"):
        values = block.get(key)
        if isinstance(values, Mapping):
            if isinstance(values.get("values"), Mapping):
                values = values["values"]
            for name, value in values.items():
                col.pred(iid, None, name, value=value, kind_hint="regression")
            break
    multilabel = block.get("multilabel")
    if isinstance(multilabel, Mapping):
        for group, payload in multilabel.items():
            if isinstance(payload, Mapping):
                for lab, prob in payload.items():
                    fv = _finite(prob)
                    col.pred(iid, str(group), str(lab), value=None if fv is None else float(fv >= 0.5),
                             probs={"p": fv} if fv is not None else None, kind_hint="multilabel")
            elif isinstance(payload, (list, tuple, set)):
                for lab in payload:
                    col.pred(iid, str(group), str(lab), value=1.0, kind_hint="multilabel_set")
    return len(col.rows) - before


def _emit_embedded_truth(col: _Collector, pid: str, rec: Mapping[str, Any]) -> None:
    gt = rec.get("ground_truth")
    if isinstance(gt, Mapping):
        for name, value in gt.items():
            if isinstance(value, (Mapping, list)):
                continue
            if _finite(value) is not None:
                col.true(pid, None, name, value=value, kind_hint="regression")
            elif _clean_label(value) is not None:
                col.true(pid, None, name, label=value, kind_hint="classification")
    labels_truth = rec.get("labels_truth")
    if isinstance(labels_truth, Mapping):
        for node_id, label in labels_truth.items():
            if _clean_label(label) is not None:
                col.true(pid, str(node_id), str(node_id), label=label, kind_hint="classification")
    diagnosis = rec.get("diagnosis")
    if _clean_label(diagnosis) is not None and not isinstance(diagnosis, (Mapping, list)):
        if not (isinstance(labels_truth, Mapping) and labels_truth):
            col.true(pid, None, "__label__", label=diagnosis, kind_hint="classification")
    truth = rec.get("truth") if isinstance(rec.get("truth"), Mapping) else rec.get("annotation")
    if isinstance(truth, Mapping):
        _emit_annotation_payload(col, pid, truth)


_COMPASS_REPORT_MARKERS = ("report_id", "decision_trace", "key_findings", "clinical_summary")


def _record_meta(rec: Mapping[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if any(key in rec for key in _COMPASS_REPORT_MARKERS):
        return meta
    for key, value in rec.items():
        if key in _RECORD_PAYLOAD_KEYS or key in _ID_KEYS:
            continue
        if _is_scalar(value):
            meta[str(key)] = value
    return meta


def _record_ok(rec: Mapping[str, Any]) -> bool:
    if rec.get("ok") is False or rec.get("success") is False:
        return False
    status = str(rec.get("status") or "").strip().lower()
    return status not in _FAILED_STATUSES


def _add_record(col: _Collector, rec: Mapping[str, Any], extra_meta: Optional[Mapping[str, Any]] = None,
                participant_id: Optional[str] = None) -> None:
    rec = _plain(rec)
    if not isinstance(rec, Mapping):
        return
    pid = participant_id or _participant_of(rec)
    if pid is None and isinstance(rec.get("report"), Mapping):
        pid = _participant_of(rec["report"])
    if pid is None:
        return
    meta = dict(extra_meta or {})
    meta.update(_record_meta(rec))
    block, spec = _prediction_block(rec)
    if isinstance(spec, Mapping):
        col.specs.append(dict(spec))
    ok = _record_ok(rec) and block is not None
    iid = col.new_instance(pid, meta, ok)
    if ok and block is not None:
        if _emit_prediction_block(col, iid, block) == 0:
            col.instances[iid]["ok"] = False
    col.current_iid = iid
    try:
        _emit_embedded_truth(col, pid, rec)
    finally:
        col.current_iid = None


def _compass_folder_meta(report: Mapping[str, Any], perf: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    evaluation = report.get("evaluation") if isinstance(report.get("evaluation"), Mapping) else {}
    execution = report.get("execution") if isinstance(report.get("execution"), Mapping) else {}
    if evaluation.get("verdict") is not None:
        meta["critic_verdict"] = evaluation.get("verdict")
    if execution.get("iterations") is not None:
        meta["iterations"] = execution.get("iterations")
    if isinstance(perf, Mapping):
        predictor = perf.get("predictor") if isinstance(perf.get("predictor"), Mapping) else {}
        routing = perf.get("routing") if isinstance(perf.get("routing"), Mapping) else {}
        tokens = perf.get("token_usage") if isinstance(perf.get("token_usage"), Mapping) else {}
        if predictor.get("kind") is not None:
            meta["predictor_kind"] = predictor.get("kind")
        if predictor.get("model") is not None:
            meta["predictor_model"] = predictor.get("model")
        if routing.get("selected_route") is not None:
            meta["route"] = routing.get("selected_route")
        if tokens.get("total_tokens") is not None:
            meta["total_tokens"] = tokens.get("total_tokens")
        duration = perf.get("total_duration_seconds")
        if duration is not None:
            meta["duration_seconds"] = duration
    return meta


def _load_compass_folders(col: _Collector, root: Path, path_groups: Sequence[str]) -> int:
    """COMPASS participant result folders (``report_<id>.json`` written by main.py)."""
    n = 0
    report_dirs = sorted({p.parent for p in root.rglob("report_*.json")})
    seen_dirs: set = set()
    for folder in report_dirs:
        seen_dirs.add(folder)
        rel = folder.relative_to(root).parts
        meta: Dict[str, Any] = {}
        for i, name in enumerate(path_groups):
            if i < len(rel) - 1:
                meta[name] = rel[i]
        reports = sorted(folder.glob("report_*.json"))
        try:
            report = _read_json_any(reports[0])
        except (OSError, json.JSONDecodeError):
            report = None
        perf = None
        perf_files = sorted(folder.glob("performance_report_*.json"))
        if perf_files:
            try:
                perf = _read_json_any(perf_files[0])
            except (OSError, json.JSONDecodeError):
                perf = None
        pid = None
        if isinstance(report, Mapping):
            pid = _participant_of(report)
        if pid is None:
            match = re.match(r"report_(.+)\.json$", reports[0].name)
            pid = match.group(1) if match else folder.name
        if not isinstance(report, Mapping):
            col.new_instance(pid, meta, False)
            n += 1
            continue
        meta.update(_compass_folder_meta(report, perf))
        record = dict(report)
        _add_record(col, record, extra_meta=meta, participant_id=pid)
        n += 1
    # Participant folders without any report are failed runs: counted, not dropped.
    for folder in sorted(root.rglob("participant_*")):
        if folder.is_dir() and folder not in seen_dirs and not any(folder.glob("report_*.json")):
            if any(p.is_dir() and any(p.glob("report_*.json")) for p in folder.iterdir()):
                continue
            rel = folder.relative_to(root).parts
            meta = {name: rel[i] for i, name in enumerate(path_groups) if i < len(rel) - 1}
            cands = extract_participant_id_candidates(folder.name)
            pid = cands[1] if len(cands) > 1 else folder.name
            col.new_instance(pid, meta, False)
            n += 1
    return n


def _load_prediction_table(col: _Collector, df: pd.DataFrame, id_column: Optional[str],
                           prediction_columns: Optional[Mapping[str, str]], group_columns: Sequence[str]) -> None:
    df = df.copy()
    id_col = id_column or next((c for c in _ID_KEYS if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"prediction table needs a participant id column (one of {_ID_KEYS}) or id_column=")
    pred_col = next((c for c in ("pred_value", "pred_label", "prediction", "predicted") if c in df.columns), None)
    if "output" in df.columns and pred_col is not None:
        # Long (tidy) layout.
        payload_cols = {"output", "node_id", "pred_value", "pred_label", "prediction", "predicted", "probs",
                        "p_positive", "true_value", "true_label", "truth", "kind", id_col, "status", "error",
                        "correct", "p_true", "positive_label", "parent_node_id", "depth", "instance_id"}
        prob_cols = [c for c in df.columns if c.startswith("prob_") or c.startswith("p_") and c not in {"p_positive", "p_true"}]
        meta_cols = [c for c in df.columns if c not in payload_cols and c not in prob_cols]
        key_cols = [id_col] + meta_cols
        if "instance_id" in df.columns:
            key_cols = ["instance_id"]
        for _, inst in df.groupby(key_cols, dropna=False, sort=False):
            first = inst.iloc[0]
            meta = {c: first[c] for c in meta_cols}
            pid = first[id_col]
            has_pred = inst.apply(lambda r: any(_clean_label(r.get(c)) is not None for c in ("pred_value", "pred_label", "prediction", "predicted")), axis=1)
            iid = col.new_instance(pid, meta, bool(has_pred.any()))
            col.current_iid = iid
            for _, row in inst.iterrows():
                node_id = _clean_label(row.get("node_id")) if "node_id" in inst.columns else None
                output = str(row["output"])
                probs = _probs_dict(row.get("probs")) if "probs" in inst.columns else None
                if probs is None and prob_cols:
                    probs = {}
                    for c in prob_cols:
                        fv = _finite(row.get(c))
                        if fv is not None:
                            probs[c.split("_", 1)[1]] = fv
                    probs = probs or None
                kind = (_clean_label(row.get("kind")) or "").lower()
                pv = _finite(row.get("pred_value"))
                pl = _clean_label(row.get("pred_label"))
                generic = _first(row, ("prediction", "predicted"))
                if kind == "regression" or (not kind and pl is None and (pv is not None or _finite(generic) is not None) and probs is None):
                    col.pred(iid, node_id, output, value=pv if pv is not None else generic, kind_hint="regression")
                elif kind == "multilabel":
                    col.pred(iid, node_id, output, value=pv if pv is not None else generic, probs=probs, kind_hint="multilabel")
                else:
                    label = pl if pl is not None else _clean_label(generic)
                    p_pos = _finite(row.get("p_positive")) if "p_positive" in inst.columns else None
                    if probs is None and p_pos is not None:
                        probs = {"__p_positive__": p_pos}
                    if label is not None or probs is not None:
                        col.pred(iid, node_id, output, label=label, probs=probs, kind_hint="classification")
                tv = _finite(row.get("true_value"))
                tl = _clean_label(row.get("true_label"))
                tgen = row.get("truth") if "truth" in inst.columns else None
                if kind == "regression" or (not kind and tl is None and (tv is not None or _finite(tgen) is not None)):
                    value = tv if tv is not None else _finite(tgen)
                    if value is not None:
                        col.true(pid, node_id, output, value=value, kind_hint="regression")
                elif kind == "multilabel":
                    value = tv if tv is not None else _finite(tgen)
                    if value is not None:
                        col.true(pid, node_id, output, value=value, kind_hint="multilabel")
                else:
                    label = tl if tl is not None else _clean_label(tgen)
                    if label is not None:
                        col.true(pid, node_id, output, label=label, kind_hint="classification")
            col.current_iid = None
        return
    # Wide layout: one row per instance, one column per output.
    mapping = dict(prediction_columns or {})
    if not mapping:
        skip = {id_col, *group_columns, "status", "ok"}
        mapping = {c: c for c in df.columns if c not in skip and not c.startswith("prob_")}
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    meta_cols = [c for c in df.columns if c != id_col and c not in mapping.values() and c not in prob_cols]
    for _, row in df.iterrows():
        meta = {c: row[c] for c in meta_cols}
        ok = str(row.get("status") or "").strip().lower() not in _FAILED_STATUSES
        iid = col.new_instance(row[id_col], meta, ok)
        if not ok:
            continue
        for target, column in mapping.items():
            raw = row.get(column)
            if _clean_label(raw) is None:
                continue
            if _finite(raw) is not None:
                col.pred(iid, None, target, value=raw, kind_hint="regression")
            else:
                probs = {c[len("prob_"):]: _finite(row.get(c)) for c in prob_cols if _finite(row.get(c)) is not None}
                col.pred(iid, None if target in {"label", "__label__"} else target,
                         "__label__" if target in {"label", "__label__"} else target,
                         label=raw, probs=probs or None, kind_hint="classification")


def load_predictions(
    source: Any,
    *,
    id_column: Optional[str] = None,
    path_groups: Sequence[str] = (),
    prediction_columns: Optional[Mapping[str, str]] = None,
    group_columns: Sequence[str] = (),
    meta: Optional[Mapping[str, Any]] = None,
) -> PredictionSet:
    """Load predictions from any supported source into a ``PredictionSet``.

    ``source`` may be: a directory of COMPASS participant result folders
    (searched recursively for ``report_<id>.json``; ``path_groups`` names the
    folder levels below the root, for example ``("tier", "predictor")`` for
    ``root/<tier>/<predictor>/participant_<id>/``); a ``.jsonl`` file of job
    records; a ``.json`` file holding one record, a list of records or
    ``{"records": [...]}``; a ``.csv``/``.tsv`` table (long tidy layout with
    an ``output`` column, or wide with one column per output); a pandas
    DataFrame; a dict or list of dicts; or a list of any of these (each item's
    rows are concatenated). ``meta`` adds constant columns (for example
    ``{"predictor": "llm"}``).
    """
    col = _Collector()
    _load_into(col, source, id_column=id_column, path_groups=tuple(path_groups),
               prediction_columns=prediction_columns, group_columns=tuple(group_columns), meta=dict(meta or {}))
    return col.result()


def _load_into(col: _Collector, source: Any, **kw: Any) -> None:
    meta = kw.get("meta") or {}
    start = len(col.instances)
    if isinstance(source, PredictionSet):
        offset = len(col.instances)
        inst = source.instances.copy()
        inst["instance_id"] = inst["instance_id"] + offset
        col.instances.extend(inst.to_dict("records"))
        rows = source.rows.copy()
        rows["instance_id"] = rows["instance_id"] + offset
        col.rows.extend(rows.to_dict("records"))
        col.truth.extend(source.truth.to_dict("records"))
        col.specs.extend(source.specs)
    elif isinstance(source, pd.DataFrame):
        _load_prediction_table(col, source, kw.get("id_column"), kw.get("prediction_columns"), kw.get("group_columns") or ())
    elif isinstance(source, Mapping):
        records = None
        for key in ("records", "predictions", "results", "rows", "jobs"):
            if isinstance(source.get(key), list):
                records = source[key]
                break
        if records is not None:
            for rec in records:
                _add_record(col, rec)
        elif _participant_of(source) is not None or isinstance(source.get("report"), Mapping):
            _add_record(col, source)
        else:
            # {participant_id: record}
            for pid, rec in source.items():
                if isinstance(rec, Mapping):
                    _add_record(col, rec, participant_id=str(pid))
    elif isinstance(source, (list, tuple)):
        if source and all(isinstance(x, Mapping) or hasattr(x, "model_dump") for x in source):
            for rec in source:
                _add_record(col, rec)
        else:
            for item in source:
                _load_into(col, item, **kw)
    elif isinstance(source, (str, Path)):
        path = Path(str(source)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"prediction source not found: {path}")
        if path.is_dir():
            n = _load_compass_folders(col, path, kw.get("path_groups") or ())
            if n == 0:
                for child in sorted(path.iterdir()):
                    if child.suffix.lower() in {".jsonl", ".json", ".csv", ".tsv"}:
                        _load_into(col, child, **{**kw, "meta": {}})
        elif path.suffix.lower() == ".jsonl":
            for rec in _read_jsonl(path):
                _add_record(col, rec)
        elif path.suffix.lower() == ".json":
            payload = _read_json_any(path)
            if isinstance(payload, Mapping) and _participant_of(payload) is None and "prediction" in payload and not any(
                k in payload for k in ("records", "predictions", "results")
            ):
                match = re.match(r"report_(.+)\.json$", path.name)
                _add_record(col, payload, participant_id=match.group(1) if match else path.stem)
            else:
                _load_into(col, payload, **{**kw, "meta": {}})
        elif path.suffix.lower() in {".csv", ".tsv", ".tab", ".txt"}:
            _load_prediction_table(col, _read_table(path), kw.get("id_column"), kw.get("prediction_columns"), kw.get("group_columns") or ())
        else:
            raise ValueError(f"unsupported prediction file type: {path.suffix}")
    elif hasattr(source, "model_dump") or hasattr(source, "dict"):
        _add_record(col, _plain(source))
    else:
        raise TypeError(f"unsupported prediction source: {type(source).__name__}")
    for rec in col.instances[start:]:
        for key, value in meta.items():
            rec.setdefault(key, value)


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------


@dataclass
class AnnotationSet:
    """``truth``: long truth rows (``participant_id``, ``node_id`` (may be
    None), ``output``, ``kind_hint``, ``true_value``, ``true_label``).
    ``meta``: one row per participant with annotation covariates (for example
    ``disorder``, ``site``) usable as grouping columns."""

    truth: pd.DataFrame
    meta: pd.DataFrame


def _emit_annotation_payload(col: _Collector, pid: str, payload: Mapping[str, Any],
                             targets: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Truth rows of one participant payload; returns the leftover metadata."""
    meta: Dict[str, Any] = {}
    targets = set(targets or [])
    nodes = payload.get("nodes")
    if isinstance(nodes, Mapping):
        for node_id, node in nodes.items():
            if not isinstance(node, Mapping):
                if _finite(node) is not None:
                    col.true(pid, str(node_id), str(node_id), value=node, kind_hint="regression")
                elif _clean_label(node) is not None:
                    col.true(pid, str(node_id), str(node_id), label=node, kind_hint="classification")
                continue
            label = _first(node, _LABEL_KEYS)
            if label is not None and not isinstance(label, (Mapping, list)):
                col.true(pid, str(node_id), str(node_id), label=label, kind_hint="classification")
            values = node.get("values")
            if not isinstance(values, Mapping) and isinstance(node.get("regression"), Mapping):
                reg = node["regression"]
                values = reg.get("values") if isinstance(reg.get("values"), Mapping) else reg
            if isinstance(values, Mapping):
                for name, value in values.items():
                    col.true(pid, str(node_id), name, value=value, kind_hint="regression")
            labels = node.get("labels")
            if isinstance(labels, (list, tuple)):
                for lab in labels:
                    col.true(pid, str(node_id), str(lab), value=1.0, kind_hint="multilabel_set")
    label = _first(payload, ("label", "classification", "class", "class_label", "target"))
    if label is not None and not isinstance(label, (Mapping, list)):
        col.true(pid, None, "__label__", label=label, kind_hint="classification")
    for key in ("regression", "values", "ground_truth"):
        values = payload.get(key)
        if isinstance(values, Mapping):
            for name, value in values.items():
                if _finite(value) is not None:
                    col.true(pid, None, name, value=value, kind_hint="regression")
                elif _clean_label(value) is not None:
                    col.true(pid, None, name, label=value, kind_hint="classification")
    if "value" in payload and _finite(payload.get("value")) is not None:
        col.true(pid, None, str(payload.get("output_name") or "value"), value=payload.get("value"), kind_hint="regression")
    labels_truth = payload.get("labels_truth")
    if isinstance(labels_truth, Mapping):
        for node_id, lab in labels_truth.items():
            if _clean_label(lab) is not None:
                col.true(pid, str(node_id), str(node_id), label=lab, kind_hint="classification")
    multilabel = payload.get("multilabel")
    if isinstance(multilabel, Mapping):
        for group, labs in multilabel.items():
            if isinstance(labs, (list, tuple, set)):
                for lab in labs:
                    col.true(pid, str(group), str(lab), value=1.0, kind_hint="multilabel_set")
                if not labs:
                    col.true(pid, str(group), "__none__", value=0.0, kind_hint="multilabel_set")
    for key, value in payload.items():
        if key in _ANNOTATION_RESERVED or key in _ID_KEYS:
            continue
        if key == "diagnosis" and not isinstance(value, (Mapping, list)):
            if label is None and _clean_label(value) is not None:
                col.true(pid, None, "__label__", label=value, kind_hint="classification")
            continue
        if key in targets:
            if _finite(value) is not None:
                col.true(pid, None, key, value=value, kind_hint="regression")
            elif _clean_label(value) is not None:
                col.true(pid, None, key, label=value, kind_hint="classification")
            continue
        if isinstance(value, (list, tuple)) and value and all(_is_scalar(v) for v in value):
            for lab in value:
                col.true(pid, str(key), str(lab), value=1.0, kind_hint="multilabel_set")
            continue
        if _is_scalar(value):
            meta[str(key)] = value
    return meta


def _annotation_records(payload: Any) -> List[Tuple[str, Mapping[str, Any]]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("annotations"), list):
        payload = payload["annotations"]
    out: List[Tuple[str, Mapping[str, Any]]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            row = value if isinstance(value, Mapping) else {"label": value} if not _finite(value) else {"value": value}
            out.append((str(key), row))
    elif isinstance(payload, list):
        for row in payload:
            if isinstance(row, Mapping):
                pid = _participant_of(row)
                if pid is not None:
                    out.append((pid, row))
    else:
        raise ValueError("annotations must be a dict keyed by participant id, a list of rows, or {'annotations': [...]}")
    return out


def load_annotations(
    source: Any,
    *,
    id_column: Optional[str] = None,
    columns: Optional[Mapping[str, str]] = None,
    targets: Optional[Sequence[str]] = None,
    task: Optional[TaskStructure] = None,
) -> AnnotationSet:
    """Load ground truth from JSON templates, JSONL, CSV/TSV, a DataFrame or dicts.

    Tables: ``id_column`` names the participant column (auto-detected from
    ``participant_id``, ``eid``, ``subject`` ...). ``columns`` maps a target
    name (regression output or classification node id) to the table column
    holding it, for example ``{"total_score": "TOTAL", "root": "dx"}``.
    Without ``columns``, ``targets`` lists the target columns; without both,
    target columns are the task-spec outputs and node ids, or (no spec) every
    column except the id and common covariates (``disorder``, ``group``,
    ``cohort``, ``site``, ``sex``). A long table with an ``output`` column and
    ``true_value``/``true_label`` is read as is.
    """
    col = _Collector()
    metas: List[Dict[str, Any]] = []
    spec_targets: List[str] = []
    if task is not None:
        spec_targets = list(task.output_to_node().keys()) + task.classification_nodes()

    def _from_table(df: pd.DataFrame) -> None:
        id_col = id_column or next((c for c in _ID_KEYS if c in df.columns), None)
        if id_col is None:
            raise ValueError(f"annotation table needs a participant id column (one of {_ID_KEYS}) or id_column=")
        if "output" in df.columns and any(c in df.columns for c in ("true_value", "true_label", "truth")):
            for _, row in df.iterrows():
                pid = str(row[id_col])
                node = _clean_label(row.get("node_id")) if "node_id" in df.columns else None
                kind = (_clean_label(row.get("kind")) or "").lower()
                raw = _first(row, ("true_value", "true_label", "truth"))
                if raw is None:
                    continue
                if kind == "regression" or (kind == "" and _finite(raw) is not None):
                    col.true(pid, node, str(row["output"]), value=raw, kind_hint="regression")
                elif kind == "multilabel":
                    col.true(pid, node, str(row["output"]), value=raw, kind_hint="multilabel")
                else:
                    col.true(pid, node, str(row["output"]), label=raw, kind_hint="classification")
            return
        mapping = dict(columns or {})
        if not mapping:
            if targets:
                mapping = {t: t for t in targets}
            elif spec_targets:
                mapping = {t: t for t in spec_targets if t in df.columns}
                for alias in ("label", "diagnosis", "class", "target"):
                    if alias in df.columns and not any(t in df.columns for t in task.classification_nodes()):
                        mapping["__label__"] = alias
                        break
            else:
                mapping = {c: c for c in df.columns if c != id_col and c.lower() not in _DEFAULT_META_NAMES}
        used = set(mapping.values())
        for _, row in df.iterrows():
            pid = str(row[id_col])
            for target, column in mapping.items():
                raw = row.get(column)
                if _clean_label(raw) is None:
                    continue
                node = None
                output = target
                if target in {"label", "diagnosis", "class"}:
                    output = "__label__"
                if _finite(raw) is not None and not (task is not None and target in task.classification_nodes()):
                    col.true(pid, node, output, value=raw, kind_hint="regression")
                else:
                    col.true(pid, node, output, label=raw, kind_hint="classification")
            metas.append({"participant_id": pid, **{c: row[c] for c in df.columns if c != id_col and c not in used}})

    if isinstance(source, AnnotationSet):
        return source
    if isinstance(source, pd.DataFrame):
        _from_table(source)
    elif isinstance(source, (str, Path)) and Path(str(source)).suffix.lower() in {".csv", ".tsv", ".tab", ".txt"}:
        _from_table(_read_table(Path(str(source))))
    else:
        if isinstance(source, (str, Path)):
            path = Path(str(source))
            payload = _read_jsonl(path) if path.suffix.lower() == ".jsonl" else _read_json_any(path)
        else:
            payload = source
        for pid, row in _annotation_records(payload):
            row = {k: v for k, v in row.items() if k not in _ID_KEYS}
            leftover = _emit_annotation_payload(col, pid, row, targets=list(targets or []) + spec_targets)
            metas.append({"participant_id": pid, **leftover})
    truth = col.result().truth
    meta = pd.DataFrame(metas) if metas else pd.DataFrame(columns=["participant_id"])
    if not meta.empty:
        meta = meta.drop_duplicates("participant_id", keep="first")
    return AnnotationSet(truth=truth, meta=meta)


# ---------------------------------------------------------------------------
# Tidy table construction
# ---------------------------------------------------------------------------


def _infer_structure(pred_rows: pd.DataFrame, truth: pd.DataFrame, root_hint: Optional[str]) -> TaskStructure:
    """Infer a flat task structure when no spec is available."""
    nodes: Dict[str, NodeInfo] = {}
    frames = [f for f in (pred_rows, truth) if not f.empty]
    if not frames:
        return TaskStructure([], source="inferred")
    both = pd.concat([f[["node_id", "output", "kind_hint"]] for f in frames], ignore_index=True)
    labels_by_node: Dict[str, List[str]] = {}
    for frame, col in ((pred_rows, "pred_label"), (truth, "true_label")):
        if frame.empty:
            continue
        for (node, output), sub in frame.dropna(subset=[col]).groupby([frame["node_id"].fillna(""), "output"]):
            key = node or (root_hint if output == "__label__" else output) or "root"
            labels_by_node.setdefault(key, [])
            for lab in sub[col]:
                if lab not in labels_by_node[key]:
                    labels_by_node[key].append(lab)
    reg_by_node: Dict[str, List[str]] = {}
    ml_by_node: Dict[str, List[str]] = {}
    for _, row in both.drop_duplicates().iterrows():
        hint = str(row["kind_hint"] or "")
        node = row["node_id"] if isinstance(row["node_id"], str) and row["node_id"] else None
        if hint == "regression":
            reg_by_node.setdefault(node or "root", [])
            if row["output"] not in reg_by_node[node or "root"]:
                reg_by_node[node or "root"].append(row["output"])
        elif hint.startswith("multilabel"):
            ml_by_node.setdefault(node or "labels", [])
            if row["output"] not in ml_by_node[node or "labels"] and row["output"] != "__none__":
                ml_by_node[node or "labels"].append(row["output"])
    root = root_hint or "root"
    for node_id, labels in labels_by_node.items():
        node_id = node_id if node_id != "__label__" else root
        labels = sorted(labels, key=lambda x: (_norm_text(x) in _NEGATIVE_LABELS, x))
        mode = "binary_classification" if len(labels) <= 2 else "multiclass_classification"
        info = NodeInfo(node_id=node_id, mode=mode, class_labels=labels)
        if len(labels) == 2:
            info.positive_label = guess_positive_label(labels)
        nodes[node_id] = info
    for node_id, outputs in reg_by_node.items():
        if node_id in nodes:
            node_id = f"{node_id}_regression"
        nodes[node_id] = NodeInfo(node_id=node_id, mode="multivariate_regression" if len(outputs) > 1 else "univariate_regression",
                                  regression_outputs=outputs)
    for node_id, labels in ml_by_node.items():
        nodes[node_id] = NodeInfo(node_id=node_id, mode="multilabel", class_labels=sorted(labels))
    ids = list(nodes)
    root_id = root if root in nodes else (ids[0] if ids else root)
    for nid in ids:
        if nid != root_id:
            nodes[nid].parent = root_id
            nodes[nid].depth = 1
    ordered = [nodes[root_id]] + [nodes[n] for n in ids if n != root_id] if ids else []
    return TaskStructure(ordered, source="inferred")


def _augment_structure(task: TaskStructure, pred_rows: pd.DataFrame, truth: pd.DataFrame) -> None:
    """Add nodes the spec does not declare (multi-label groups, extra targets)."""
    known_outputs = set(task.output_to_node())
    known_nodes = set(task.nodes)

    def _unknown(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        node = frame["node_id"]
        hint = frame["kind_hint"].astype(str)
        explicit_unknown = node.notna() & ~node.isin(known_nodes)
        flat_unknown = node.isna() & (hint == "regression") & ~frame["output"].isin(known_outputs)
        return frame[explicit_unknown | flat_unknown]

    extra = _infer_structure(_unknown(pred_rows), _unknown(truth), None)
    root = task.root_id
    for node in extra.nodes.values():
        if node.node_id in task.nodes or node.node_id == "root":
            if node.is_regression:
                for out in node.regression_outputs:
                    if out not in known_outputs and node.node_id == "root" and root in task.nodes and task.nodes[root].is_regression:
                        task.nodes[root].regression_outputs.append(out)
            continue
        node.parent = root
        node.depth = 1
        task.nodes[node.node_id] = node


def _resolve_nodes(frame: pd.DataFrame, task: TaskStructure) -> pd.DataFrame:
    """Fill missing node ids (flat regression dicts, root labels) from the task tree."""
    if frame.empty:
        return frame
    frame = frame.copy()
    out2node = task.output_to_node()
    cls_nodes = task.classification_nodes()
    root = task.root_id or "root"
    label_node = root if root in cls_nodes else (cls_nodes[0] if cls_nodes else root)
    node_col = []
    out_col = []
    for node, output, hint in zip(frame["node_id"], frame["output"], frame["kind_hint"]):
        node = node if isinstance(node, str) and node else None
        if output == "__label__":
            node = node or label_node
            output = node
        elif node is None:
            if output in out2node:
                node = out2node[output]
            elif output in task.nodes and str(hint) == "classification":
                node = output
            elif str(hint).startswith("multilabel"):
                ml = [n.node_id for n in task.nodes.values() if n.kind == "multilabel" and output in n.class_labels]
                node = ml[0] if ml else "labels"
            elif str(hint) == "classification" and len(cls_nodes) == 1:
                node, output = cls_nodes[0], cls_nodes[0]
            else:
                reg_nodes = [n.node_id for n in task.nodes.values() if n.is_regression]
                node = reg_nodes[0] if len(reg_nodes) == 1 else root
        elif str(hint) == "classification" and output != node and node in task.nodes and not task.nodes[node].is_regression:
            output = node
        node_col.append(node)
        out_col.append(output)
    frame["node_id"] = node_col
    frame["output"] = out_col
    return frame


def _canonical_label(label: Optional[str], node: Optional[NodeInfo]) -> Optional[str]:
    if label is None or node is None or not node.class_labels:
        return label
    if label in node.class_labels:
        return label
    folded = {_norm_text(x): x for x in node.class_labels}
    hit = folded.get(_norm_text(label))
    if hit is not None:
        return hit
    if node.kind == "binary" and node.positive_label:
        negative = [x for x in node.class_labels if x != node.positive_label][0]
        key = _norm_text(label)
        if key in {"case", "positive", "1", "true", "yes", "present"}:
            return node.positive_label
        if key in _NEGATIVE_LABELS:
            return negative
        # A long label that starts with a class label (COMPASS target strings with codes).
        for lab in node.class_labels:
            if key.startswith(_norm_text(lab)) or _norm_text(lab).startswith(key):
                return lab
    return label


def _canonical_probs(probs: Optional[Dict[str, float]], node: Optional[NodeInfo]) -> Optional[Dict[str, float]]:
    if not probs:
        return None
    if node is None or not node.class_labels:
        return probs
    if "__p_positive__" in probs and node.kind == "binary" and node.positive_label:
        p = min(1.0, max(0.0, probs["__p_positive__"]))
        neg = [x for x in node.class_labels if x != node.positive_label][0]
        return {node.positive_label: p, neg: 1.0 - p}
    out: Dict[str, float] = {lab: 0.0 for lab in node.class_labels}
    matched = False
    for key, value in probs.items():
        lab = _canonical_label(key, node)
        if lab in out and value >= 0:
            out[lab] += float(value)
            matched = True
    total = sum(out.values())
    if not matched or total <= 0:
        return None
    return {k: v / total for k, v in out.items()}


def _match_ids(pred_ids: Iterable[str], truth_ids: Iterable[str], lenient: bool) -> Dict[str, str]:
    truth_ids = [str(x) for x in truth_ids]
    truth_set = set(truth_ids)
    mapping: Dict[str, str] = {}
    for pid in {str(x) for x in pred_ids}:
        if pid in truth_set:
            mapping[pid] = pid
            continue
        if not lenient:
            mapping[pid] = pid
            continue
        cands = extract_participant_id_candidates(pid) or [pid]
        hit = resolve_identifier(cands, truth_ids)
        mapping[pid] = hit if hit is not None else pid
    return mapping


def _numeric_truth_as_label(truth: pd.DataFrame, targets: set) -> pd.DataFrame:
    if truth.empty or not targets:
        return truth
    key = truth["node_id"].where(truth["node_id"].notna(), truth["output"])
    mask = (truth["kind_hint"] == "regression") & (truth["output"].isin(targets) | key.isin(targets)) & truth["true_label"].isna()
    if not mask.any():
        return truth
    truth = truth.copy()

    def _as_label(value: Any) -> Optional[str]:
        fv = _finite(value)
        if fv is None:
            return None
        return str(int(fv)) if float(fv).is_integer() else str(fv)

    truth.loc[mask, "true_label"] = truth.loc[mask, "true_value"].map(_as_label)
    truth.loc[mask, "true_value"] = np.nan
    truth.loc[mask, "kind_hint"] = "classification"
    return truth


def _merge_truth(tidy: pd.DataFrame, embedded: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Attach truth: external annotations (by participant) win; then truth
    embedded in the same prediction record (by instance); then truth embedded
    in other records of the same participant, only when those agree."""
    keys = ["node_id", "output"]
    parts = []
    if not external.empty:
        ext = external.drop_duplicates(["participant_id", *keys], keep="last")
        parts.append(("e", ext[["participant_id", *keys, "true_value", "true_label"]], ["participant_id", *keys]))
    if not embedded.empty:
        inst = embedded.dropna(subset=["instance_id"]).copy()
        if not inst.empty:
            inst["instance_id"] = inst["instance_id"].astype(int)
            inst = inst.drop_duplicates(["instance_id", *keys], keep="last")
            parts.append(("i", inst[["instance_id", *keys, "true_value", "true_label"]], ["instance_id", *keys]))
        grouped = embedded.groupby(["participant_id", *keys], dropna=False)
        agree = grouped.agg(_nv=("true_value", "nunique"), _nl=("true_label", "nunique"),
                            true_value=("true_value", "first"), true_label=("true_label", "first")).reset_index()
        agree = agree[(agree["_nv"] <= 1) & (agree["_nl"] <= 1)]
        parts.append(("p", agree[["participant_id", *keys, "true_value", "true_label"]], ["participant_id", *keys]))
    tidy = tidy.copy()
    tidy["true_value"] = np.nan
    tidy["true_label"] = None
    for tag, frame, on in parts:
        renamed = frame.rename(columns={"true_value": f"_tv_{tag}", "true_label": f"_tl_{tag}"})
        tidy = tidy.merge(renamed, on=on, how="left")
        tidy["true_value"] = tidy["true_value"].where(tidy["true_value"].notna(), tidy[f"_tv_{tag}"])
        tidy["true_label"] = tidy["true_label"].where(tidy["true_label"].notna(), tidy[f"_tl_{tag}"])
        tidy = tidy.drop(columns=[f"_tv_{tag}", f"_tl_{tag}"])
    return tidy


def build_tidy(
    predictions: Any,
    annotations: Any = None,
    *,
    task_spec: Any = None,
    group_by: Sequence[str] = (),
    expected_participants: str = "observed",
    positive_labels: Optional[Mapping[str, str]] = None,
    id_matching: str = "lenient",
    prediction_kwargs: Optional[Mapping[str, Any]] = None,
    annotation_kwargs: Optional[Mapping[str, Any]] = None,
    return_structure: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, TaskStructure]]:
    """Build the tidy long table from predictions and annotations.

    ``predictions``: anything ``load_predictions`` accepts (or a
    ``PredictionSet``). ``annotations``: anything ``load_annotations`` accepts,
    or None when the truth is embedded in the prediction records. External
    annotations override embedded truth for the same participant and output.

    ``expected_participants``: ``"observed"`` scores the instances present in
    the predictions (failed runs included); ``"annotations"`` also adds, for
    every combination of ``group_by`` values, one missing-prediction instance
    per annotated participant that has no prediction there.
    """
    pred = predictions if isinstance(predictions, PredictionSet) else load_predictions(predictions, **dict(prediction_kwargs or {}))
    task: Optional[TaskStructure] = None
    if task_spec is not None:
        task = load_task_spec(task_spec, positive_labels=positive_labels)
    elif pred.specs:
        try:
            task = TaskStructure.from_spec(pred.specs[0], positive_labels=positive_labels)
        except ValueError:
            task = None
    ann: Optional[AnnotationSet] = None
    if annotations is not None:
        ann = load_annotations(annotations, task=task, **dict(annotation_kwargs or {}))

    embedded = pred.truth
    external = ann.truth if ann is not None else pd.DataFrame(columns=_TRUTH_COLUMNS)
    # Numeric truth of a target predicted as a label (for example 0/1 codes) is a label.
    cls_rows = pred.rows[pred.rows["kind_hint"] == "classification"]
    cls_targets = set(cls_rows["output"]) | set(cls_rows["node_id"].dropna())
    if task is not None:
        cls_targets |= set(task.classification_nodes())
    if "__label__" in cls_targets or (task is not None and task.classification_nodes()):
        cls_targets.add("__label__")
    embedded = _numeric_truth_as_label(embedded, cls_targets)
    external = _numeric_truth_as_label(external, cls_targets)
    nonempty = [f for f in (embedded, external) if not f.empty]
    truth = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=_TRUTH_COLUMNS)

    if task is None:
        task = _infer_structure(pred.rows, truth, None)
    else:
        _augment_structure(task, pred.rows, truth)
    if positive_labels:
        for nid, lab in positive_labels.items():
            if nid in task.nodes:
                task.nodes[nid].positive_label = lab

    rows = _resolve_nodes(pred.rows, task)
    truth = _resolve_nodes(truth, task)
    embedded = _resolve_nodes(embedded, task)
    external = _resolve_nodes(external, task)
    # Multi-label sets ({label, ...}) become 0/1 rows over the group's label universe.
    ml_groups = {n.node_id: list(n.class_labels) for n in task.nodes.values() if n.kind == "multilabel"}

    instances = pred.instances.copy()
    if instances.empty:
        raise ValueError("no prediction instances found in the prediction source")
    # Participant id matching between predictions and annotations.
    truth_ids: List[str] = []
    if ann is not None:
        truth_ids = list(dict.fromkeys(list(external["participant_id"].dropna().astype(str))
                                       + list(ann.meta["participant_id"].astype(str) if not ann.meta.empty else [])))
    mapping: Dict[str, str] = {}
    if truth_ids:
        mapping = _match_ids(instances["participant_id"].dropna().astype(str), truth_ids, id_matching == "lenient")
        instances["participant_id"] = instances["participant_id"].astype(str).map(mapping)
    if not embedded.empty and mapping:
        embedded = embedded.assign(participant_id=embedded["participant_id"].astype(str).map(lambda x: mapping.get(x, x)))
    if not truth_ids and not embedded.empty:
        truth_ids = list(dict.fromkeys(embedded["participant_id"].astype(str)))

    group_by = [g for g in group_by if g]
    if expected_participants == "annotations" and truth_ids:
        present_groups = [g for g in group_by if g in instances.columns]
        combos = instances[present_groups].drop_duplicates() if present_groups else pd.DataFrame(index=[0])
        extra: List[Dict[str, Any]] = []
        next_id = int(instances["instance_id"].max()) + 1
        for _, combo in combos.iterrows():
            mask = np.ones(len(instances), dtype=bool)
            for g in present_groups:
                mask &= (instances[g].astype(str) == str(combo[g])).to_numpy()
            have = set(instances.loc[mask, "participant_id"].astype(str))
            for pid in truth_ids:
                if pid not in have:
                    rec = {"instance_id": next_id, "participant_id": pid, "ok": False, "absent": True}
                    rec.update({g: combo[g] for g in present_groups})
                    extra.append(rec)
                    next_id += 1
        if extra:
            instances = pd.concat([instances, pd.DataFrame(extra)], ignore_index=True)

    # Expected (node, output) keys.
    keys = list(task.expected_keys())
    seen = set(keys)
    for frame in (rows, truth):
        for node, output, hint in frame[["node_id", "output", "kind_hint"]].drop_duplicates().itertuples(index=False):
            if node in ml_groups:
                continue
            if (node, output) not in seen:
                keys.append((node, output))
                seen.add((node, output))
    for gid, labels in ml_groups.items():
        for lab in labels:
            if (gid, lab) not in seen:
                keys.append((gid, lab))
                seen.add((gid, lab))
    key_frame = pd.DataFrame(keys, columns=["node_id", "output"])

    base = instances.merge(key_frame, how="cross")
    pr = rows.drop(columns=["kind_hint"]).drop_duplicates(["instance_id", "node_id", "output"], keep="last")
    tidy = base.merge(pr, on=["instance_id", "node_id", "output"], how="left")
    tidy = _merge_truth(tidy, embedded, external)

    # Multi-label: a set-valued prediction or truth implies 0 for unlisted labels.
    if ml_groups:
        is_ml = tidy["node_id"].isin(list(ml_groups))
        ml_rows = rows[rows["node_id"].isin(list(ml_groups))]
        pred_groups = set(zip(ml_rows["instance_id"], ml_rows["node_id"]))
        ext_ml = external[external["node_id"].isin(list(ml_groups))]
        emb_ml = embedded[embedded["node_id"].isin(list(ml_groups))]
        truth_by_pid = set(zip(ext_ml["participant_id"], ext_ml["node_id"]))
        truth_by_inst = set(zip(emb_ml["instance_id"], emb_ml["node_id"]))
        inst_key = list(zip(tidy["instance_id"], tidy["node_id"]))
        pid_key = list(zip(tidy["participant_id"], tidy["node_id"]))
        ok_inst = tidy["ok"].fillna(False).astype(bool).to_numpy()
        fill_pred = is_ml.to_numpy() & tidy["pred_value"].isna().to_numpy() & ok_inst & np.array([k in pred_groups for k in inst_key], dtype=bool)
        has_truth = np.array([(p in truth_by_pid) or (i in truth_by_inst) for p, i in zip(pid_key, inst_key)], dtype=bool)
        fill_truth = is_ml.to_numpy() & tidy["true_value"].isna().to_numpy() & has_truth
        tidy.loc[fill_pred, "pred_value"] = 0.0
        tidy.loc[fill_truth, "true_value"] = 0.0
        tidy = tidy[~(is_ml & (tidy["output"] == "__none__"))].reset_index(drop=True)

    # Annotation covariates.
    if ann is not None and not ann.meta.empty:
        meta = ann.meta.copy()
        meta["participant_id"] = meta["participant_id"].astype(str)
        clash = [c for c in meta.columns if c != "participant_id" and c in tidy.columns]
        meta = meta.drop(columns=clash)
        tidy = tidy.merge(meta, on="participant_id", how="left")

    # Kinds, labels, probabilities, status.
    node_info = task.nodes
    kinds, parents, depths, pos_labels = [], [], [], []
    for node_id, output in zip(tidy["node_id"], tidy["output"]):
        info = node_info.get(node_id)
        if info is None:
            kinds.append("regression")
            parents.append(None)
            depths.append(0)
            pos_labels.append(None)
            continue
        kind = info.kind
        if kind == "multilabel":
            pos_labels.append("1")
        else:
            pos_labels.append(info.positive_label if kind == "binary" else None)
        kinds.append(kind)
        parents.append(info.parent)
        depths.append(info.depth)
    tidy["kind"] = kinds
    tidy["parent_node_id"] = parents
    tidy["depth"] = depths
    tidy["positive_label"] = pos_labels
    for colname in ("pred_label", "true_label", "probs", "pred_value", "true_value"):
        if colname not in tidy.columns:
            tidy[colname] = None
    is_cls = tidy["kind"].isin(["binary", "multiclass"]).to_numpy()
    pred_labels = tidy["pred_label"].to_numpy(dtype=object)
    true_labels = tidy["true_label"].to_numpy(dtype=object)
    probs_col = tidy["probs"].to_numpy(dtype=object)
    p_pos = np.full(len(tidy), np.nan)
    p_true = np.full(len(tidy), np.nan)
    for i in np.flatnonzero(is_cls):
        info = node_info.get(tidy["node_id"].iat[i])
        probs = probs_col[i] if isinstance(probs_col[i], dict) else None
        probs = _canonical_probs(probs, info)
        probs_col[i] = probs
        pl = _canonical_label(pred_labels[i] if isinstance(pred_labels[i], str) else None, info)
        if pl is None and probs:
            pl = max(probs.items(), key=lambda kv: kv[1])[0]
        pred_labels[i] = pl
        tl = _canonical_label(true_labels[i] if isinstance(true_labels[i], str) else None, info)
        true_labels[i] = tl
        if probs and info is not None and info.kind == "binary" and info.positive_label in probs:
            p_pos[i] = probs[info.positive_label]
        if probs and tl in (probs or {}):
            p_true[i] = probs[tl]
    is_ml = (tidy["kind"] == "multilabel").to_numpy()
    for i in np.flatnonzero(is_ml):
        probs = probs_col[i] if isinstance(probs_col[i], dict) else None
        if probs and "p" in probs:
            p_pos[i] = probs["p"]
        probs_col[i] = None
    tidy["pred_label"] = pred_labels
    tidy["true_label"] = true_labels
    tidy["probs"] = probs_col
    tidy["p_positive"] = p_pos
    tidy["p_true"] = p_true
    tidy["pred_value"] = pd.to_numeric(tidy["pred_value"], errors="coerce")
    tidy["true_value"] = pd.to_numeric(tidy["true_value"], errors="coerce")
    # A failed instance never has a prediction, even if a stray row slipped through.
    if "ok" in tidy.columns:
        failed = ~tidy["ok"].fillna(False).astype(bool)
        tidy.loc[failed, ["pred_value", "pred_label"]] = np.nan
        tidy.loc[failed, "p_positive"] = np.nan
    has_pred = np.where(is_cls, pd.notna(tidy["pred_label"]), pd.notna(tidy["pred_value"]))
    has_true = np.where(is_cls, pd.notna(tidy["true_label"]), pd.notna(tidy["true_value"]))
    status = np.where(has_pred & has_true, "ok", np.where(has_true, "missing_prediction", np.where(has_pred, "missing_truth", "missing_both")))
    tidy["status"] = status
    is_reg = (tidy["kind"] == "regression").to_numpy()
    tidy["error"] = np.where(is_reg & (status == "ok"), tidy["pred_value"] - tidy["true_value"], np.nan)
    correct = np.full(len(tidy), np.nan)
    ok_mask = status == "ok"
    correct[is_cls & ok_mask] = (tidy.loc[is_cls & ok_mask, "pred_label"].astype(str).to_numpy()
                                 == tidy.loc[is_cls & ok_mask, "true_label"].astype(str).to_numpy()).astype(float)
    correct[is_ml & ok_mask] = (tidy.loc[is_ml & ok_mask, "pred_value"].to_numpy() == tidy.loc[is_ml & ok_mask, "true_value"].to_numpy()).astype(float)
    tidy["correct"] = correct
    # Rows that carry nothing at all (neither truth nor prediction, and the
    # node was not part of the declared task) are noise from inference.
    tidy = tidy[~((tidy["status"] == "missing_both") & (task.source == "inferred"))]

    meta_cols = [c for c in tidy.columns if c not in TIDY_COLUMNS and c not in {"ok", "absent"}]
    front = [c for c in group_by if c in meta_cols]
    rest = [c for c in meta_cols if c not in front]
    ordered = front + TIDY_COLUMNS + rest + [c for c in ("ok", "absent") if c in tidy.columns]
    tidy = tidy[ordered].reset_index(drop=True)
    tidy["participant_id"] = tidy["participant_id"].astype(str)
    if return_structure:
        return tidy, task
    return tidy


def tidy_to_csv(tidy: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write a tidy table (``probs`` serialized as JSON) to CSV or TSV."""
    out = tidy.copy()
    if "probs" in out.columns:
        out["probs"] = [json.dumps(p, sort_keys=True) if isinstance(p, dict) else "" for p in out["probs"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, sep="\t" if path.suffix.lower() == ".tsv" else ",")


def read_tidy(path: Union[str, Path]) -> pd.DataFrame:
    """Read a tidy table written by ``tidy_to_csv`` (restores dtypes and ``probs``)."""
    path = Path(path)
    df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", keep_default_na=True)
    for colname in ("true_value", "pred_value", "error", "correct", "p_positive", "p_true", "depth"):
        if colname in df.columns:
            df[colname] = pd.to_numeric(df[colname], errors="coerce")
    for colname in ("participant_id", "node_id", "output"):
        if colname in df.columns:
            df[colname] = df[colname].astype(str)
    if "probs" in df.columns:
        df["probs"] = [_probs_dict(p) if isinstance(p, str) and p else None for p in df["probs"]]
    return df
