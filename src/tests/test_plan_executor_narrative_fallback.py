from src.full_stack.backend.utils.core.plan_executor import PlanExecutor
from src.full_stack.backend.data.models.execution_plan import PlanStep, ToolName


def test_narrative_fallback_injects_unimodal_outputs():
    pe = PlanExecutor()
    step = PlanStep(
        step_id=2,
        tool_name=ToolName.MULTIMODAL_NARRATIVE,
        description="Fuse modalities",
        reasoning="Test",
        input_domains=["BRAIN_MRI"],
        parameters={},
        expected_output="narrative",
        estimated_tokens=0,
        depends_on=[]
    )

    context = {
        "hierarchical_deviation": {},
        "non_numerical_data": "",
        "target_condition": "test",
        "participant_id": "P1",
        "multimodal_data": {"BRAIN_MRI": []},
    }

    previous_outputs = {
        1: {
            "tool_name": "UnimodalCompressor",
            "domain": "BRAIN_MRI:structural",
            "_step_meta": {
                "tool_name": "UnimodalCompressor",
                "input_domains": ["BRAIN_MRI"],
            },
        }
    }

    tool_input = pe._build_tool_input(step, context, previous_outputs)
    assert "dependency_outputs" in tool_input
    assert tool_input["dependency_outputs"]
