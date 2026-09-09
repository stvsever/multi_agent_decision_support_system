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
