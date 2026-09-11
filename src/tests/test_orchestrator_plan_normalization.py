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


def _plan(steps):
    return {"plan_id": "plan-1", "target_condition": "target", "steps": steps}


def test_an_invented_tool_and_the_steps_that_hang_on_it_are_reported():
    # The shape of a real run: the model blended UnimodalCompressor and
    # MultimodalNarrativeCreator into a tool that does not exist, the parser
    # dropped that step, and the global fusion step, and everything after it,
    # were left waiting on it for ever.
    orchestrator = object.__new__(Orchestrator)
    plan = _plan([
        {"step_id": 14, "tool_name": "UnimodalCompressor"},
        {"step_id": 15, "tool_name": "UnimodalNarrativeCreator"},
        {"step_id": 17, "tool_name": "MultimodalNarrativeCreator"},
        {"step_id": 18, "tool_name": "MultimodalNarrativeCreator", "depends_on": [15, 17, 14]},
        {"step_id": 20, "tool_name": "HypothesisGenerator", "depends_on": [18]},
    ])

    defects = orchestrator._plan_defects(plan)

    assert defects == [
        "Step 15 uses the tool 'UnimodalNarrativeCreator', which does not exist.",
        "Step 18 depends on step 15, which is not a runnable step in this plan.",
    ]


def test_a_reference_to_a_step_that_was_never_written_is_reported():
    orchestrator = object.__new__(Orchestrator)
    plan = _plan([
        {"step_id": 1, "tool_name": "FeatureSynthesizer"},
        {"step_id": 2, "tool_name": "ClinicalRelevanceRanker", "depends_on": [1, 9]},
    ])

    assert orchestrator._plan_defects(plan) == [
        "Step 2 depends on step 9, which is not a runnable step in this plan.",
    ]


def test_a_sound_plan_has_no_defects():
    orchestrator = object.__new__(Orchestrator)
    plan = _plan([
        {"step_id": 1, "tool_name": "UnimodalCompressor"},
        {"step_id": 2, "tool_name": "unimodal_compressor"},
        {"step_id": 3, "tool_name": "MultimodalNarrativeCreator", "depends_on": [1, 2]},
        {"step_id": 4, "tool_name": "DifferentialDiagnosis", "depends_on": "3"},
    ])

    assert orchestrator._plan_defects(plan) == []


def test_repair_feedback_names_the_defects_and_every_real_tool():
    orchestrator = object.__new__(Orchestrator)
    feedback = orchestrator._repair_feedback(
        ["Step 15 uses the tool 'UnimodalNarrativeCreator', which does not exist."],
        previous_feedback="Cover the cognition domain.",
    )

    assert "Step 15 uses the tool 'UnimodalNarrativeCreator'" in feedback
    assert "MultimodalNarrativeCreator" in feedback and "UnimodalCompressor" in feedback
    assert "Cover the cognition domain." in feedback
