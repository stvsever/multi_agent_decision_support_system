from src.full_stack.backend.utils.core.plan_executor import PlanExecutor
from src.full_stack.backend.data.models.execution_plan import PlanStep, ToolName


def test_inline_input_domain_promotes_node_paths():
    pe = PlanExecutor()
    step = PlanStep(
        step_id=1,
        tool_name=ToolName.UNIMODAL_COMPRESSOR,
        description="Compress subtree",
        reasoning="Test",
        input_domains=["BRAIN_MRI|structural|morphology"],
        parameters={},
        expected_output="summary",
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

    tool_input = pe._build_tool_input(step, context, {})
    assert tool_input["input_domains"] == ["BRAIN_MRI"]
    assert "node_paths" in tool_input["parameters"]
    assert "BRAIN_MRI|structural|morphology" in tool_input["parameters"]["node_paths"]
