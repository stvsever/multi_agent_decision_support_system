from src.full_stack.backend.agents.orchestrator import Orchestrator


def test_plan_normalization_keeps_valid_steps_and_drops_stray_values():
    orchestrator = object.__new__(Orchestrator)
    raw = {
        "plan_id": "plan-1",
        "target_condition": "target",
        "steps": [
            {"step_id": 1, "tool_name": "PhenotypeRepresentation"},
            "a stray explanatory sentence",
            '{"step_id": 2, "tool_name": "FeatureSynthesizer"}',
            None,
        ],
    }

    normalized = orchestrator._normalize_plan_data(raw, "target")

    assert normalized["steps"] == [
        {"step_id": 1, "tool_name": "PhenotypeRepresentation"},
        {"step_id": 2, "tool_name": "FeatureSynthesizer"},
    ]
