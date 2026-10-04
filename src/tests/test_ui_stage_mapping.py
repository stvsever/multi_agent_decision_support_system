import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.runtime.event_bus import EventStore


def test_step_start_stage_mapping_for_predictor_and_communicator():
    store = EventStore()
    store.add_event("STEP_START", {"id": 910, "tool": "Predictor Agent", "desc": "Predict"})
    assert store.state["current_stage"] == 4

    store.add_event("STEP_START", {"id": 930, "tool": "Communicator Agent", "desc": "Communicate"})
    assert store.state["current_stage"] == 6


def test_explicit_stage_on_step_start_overrides_inference():
    store = EventStore()
    store.add_event("STEP_START", {"id": 4, "tool": "X", "desc": "Y", "stage": 4})
    assert store.state["current_stage"] == 4


def test_deep_report_state_transitions():
    store = EventStore()

    store.add_event("DEEP_REPORT", {"status": "queued", "available": False, "error": None})
    assert store.state["deep_report_status"] == "queued"
    assert store.state["deep_report_available"] is False
    assert store.state["deep_report_error"] is None

    store.add_event("DEEP_REPORT", {"status": "running", "available": False, "error": None})
    assert store.state["deep_report_status"] == "running"

    store.add_event("DEEP_REPORT", {"status": "completed", "available": True, "error": None})
    assert store.state["deep_report_status"] == "completed"
    assert store.state["deep_report_available"] is True
    assert store.state["deep_report_last_generated_at"] is not None

    store.add_event("DEEP_REPORT", {"status": "failed", "available": False, "error": "boom"})
    assert store.state["deep_report_status"] == "failed"
    assert store.state["deep_report_available"] is False
    assert store.state["deep_report_error"] == "boom"


def test_event_sink_receives_every_event_with_state():
    """The dashboard streams a run by attaching a sink to this store."""
    store = EventStore()
    seen = []
    store.add_sink(lambda payload: seen.append(payload))

    store.add_event("INIT", {"participant_id": "P1", "target": "T", "max_iterations": 2})
    store.add_event("STEP_START", {"id": 1, "tool": "UnimodalCompressor", "desc": "d", "stage": 2})
    store.add_event("STEP_COMPLETE", {"id": 1, "tokens": 400, "duration_ms": 90, "preview": "p"})

    assert [item["event"]["type"] for item in seen] == ["INIT", "STEP_START", "STEP_COMPLETE"]
    assert seen[-1]["state"]["total_tokens"] == 400
    assert seen[-1]["state"]["current_stage"] == 2
    # Every payload must already be JSON-safe: it is written straight to stdout.
    import json

    json.dumps(seen[-1])


def test_events_are_coerced_to_plain_json():
    """Engine objects reach the sink as plain data, never as pydantic models."""
    from enum import Enum

    class Verdict(str, Enum):
        SATISFACTORY = "SATISFACTORY"

    store = EventStore()
    store.add_event("CRITIC", {"verdict": Verdict.SATISFACTORY, "score": 0.9})
    assert store.state["critic"]["verdict"] == "SATISFACTORY"


def _route(route, iteration, **extra):
    return {
        "route": route,
        "mode": "auto",
        "input_tokens": 9000,
        "budget_tokens": 120000,
        "threshold_source": "predictor_budget",
        "predictor_kind": "llm",
        "reason": "r",
        "escalated": False,
        "iteration": iteration,
        **extra,
    }


def test_a_direct_route_resets_the_attempt_and_jumps_to_prediction():
    store = EventStore()
    store.add_event("INIT", {"participant_id": "P1", "target": "T", "max_iterations": 2})
    store.add_event("STEP_START", {"id": 911, "tool": "Predictor Agent", "desc": "d", "stage": 4})
    store.add_event("PREDICTION", {"result": "CASE", "prob": 0.7})

    store.add_event("ROUTE", _route("direct", 2))
    assert store.state["iteration"] == 2
    assert store.state["route"]["route"] == "direct"
    assert store.state["current_stage"] == 4
    assert store.state["steps"] == []
    assert [s["id"] for s in store.state["history"]] == [911]
    assert store.state["prediction"] is None
    assert store.state["max_steps"] == 0
    assert "orchestration skipped" in store.state["status"]


def test_route_decisions_accumulate_and_an_orchestrated_one_leaves_the_steps_to_the_plan():
    store = EventStore()
    store.add_event("ROUTE", _route("direct", 1))
    store.add_event("STEP_START", {"id": 911, "tool": "Predictor Agent", "desc": "d", "stage": 4})
    store.add_event("ROUTE", _route("orchestrated", 2, escalated=True))

    assert [r["route"] for r in store.state["routes"]] == ["direct", "orchestrated"]
    assert store.state["route"]["escalated"] is True
    # The PLAN event that follows archives the steps; the route event does not.
    assert [s["id"] for s in store.state["steps"]] == [911]


def test_the_emitter_stamps_the_route_with_its_iteration():
    from src.full_stack.backend.runtime import event_bus

    event_bus.reset_ui()
    seen = []
    sink = event_bus.attach_sink(seen.append)
    try:
        event_bus.get_ui().on_route_decision({"route": "direct", "predictor_kind": "decision"}, iteration=3)
    finally:
        event_bus.detach_sink(sink)
        event_bus.reset_ui()
    event = seen[-1]["event"]
    assert event["type"] == "ROUTE"
    assert event["data"] == {"route": "direct", "predictor_kind": "decision", "iteration": 3}


def test_the_prediction_payload_says_what_kind_of_output_it_holds():
    """`primary_output_kind` is a property, so `model_dump()` drops it; the emitter adds it."""
    from src.full_stack.backend.data.models.prediction_task import PredictionMode
    from src.full_stack.backend.runtime import event_bus

    event_bus.reset_ui()
    seen = []
    sink = event_bus.attach_sink(seen.append)
    ui = event_bus.get_ui()
    try:
        ui.on_prediction("CASE", 0.8, "HIGH", {"root_prediction": {"mode": PredictionMode.BINARY_CLASSIFICATION}})
        ui.on_prediction("", None, "LOW", {"flat_predictions": [{"mode": "univariate_regression"}]})
        ui.on_prediction("", None, "LOW", {"primary_output_kind": "regression", "decision_report": {}})
        ui.on_prediction("", None, "LOW", None)
    finally:
        event_bus.detach_sink(sink)
        event_bus.reset_ui()
    payloads = [item["event"]["data"]["payload"] for item in seen if item["event"]["type"] == "PREDICTION"]
    assert payloads[0]["primary_output_kind"] == "classification"
    assert payloads[1]["primary_output_kind"] == "regression"
    assert payloads[2] == {"primary_output_kind": "regression", "decision_report": {}}
    assert payloads[3] is None
