"""The plan graph must recover the real sequential and parallel structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api.flow import plan_to_graph


def _plan(steps, **extra):
    return {"plan_id": "P", "iteration": 1, "steps": steps, **extra}


def _step(step_id, tool="UnimodalCompressor", depends_on=()):
    return {
        "step_id": step_id,
        "tool_name": tool,
        "description": f"step {step_id}",
        "depends_on": list(depends_on),
    }


def _edge(graph, source, target):
    return next(e for e in graph["edges"] if e["source"] == source and e["target"] == target)


def test_dependency_depth_becomes_the_rank_and_shared_ranks_run_in_parallel():
    graph = plan_to_graph(
        _plan([_step(1), _step(2), _step(3, depends_on=[1, 2]), _step(4, depends_on=[3])])
    )
    ranks = {n["id"]: n["rank"] for n in graph["nodes"]}
    assert ranks["step:1"] == ranks["step:2"] == 0
    assert ranks["step:3"] == 1
    assert ranks["step:4"] == 2
    # Steps sharing a rank get distinct lanes so the layout can spread them.
    lanes = sorted(n["lane"] for n in graph["nodes"] if n["rank"] == 0 and n["type"] == "tool")
    assert lanes == [0, 1]
    assert graph["meta"]["max_parallel"] == 2
    assert graph["meta"]["depth"] == 3


def test_edges_are_classified_by_their_branching_shape():
    graph = plan_to_graph(
        _plan([_step(1), _step(2), _step(3, depends_on=[1, 2]), _step(4, depends_on=[3]), _step(5, depends_on=[3])])
    )
    assert _edge(graph, "step:1", "step:3")["kind"] == "fan_in"
    assert _edge(graph, "step:3", "step:4")["kind"] == "fan_out"
    # Several root steps are dispatched together by the orchestrator.
    assert _edge(graph, "agent:orchestrator", "step:1")["kind"] == "fan_out"


def test_a_single_root_step_is_a_plain_sequential_dispatch():
    graph = plan_to_graph(_plan([_step(1), _step(2, depends_on=[1])]))
    assert _edge(graph, "agent:orchestrator", "step:1")["kind"] == "sequential"
    assert _edge(graph, "step:1", "step:2")["kind"] == "sequential"


def test_the_actor_critic_scaffold_is_always_present():
    graph = plan_to_graph(_plan([_step(1)]))
    ids = {n["id"] for n in graph["nodes"]}
    assert {"agent:orchestrator", "agent:fusion", "agent:predictor", "agent:critic", "agent:communicator"} <= ids
    feedback = _edge(graph, "agent:critic", "agent:orchestrator")
    assert feedback["kind"] == "feedback"
    assert feedback["label"] == "revise"


def test_terminal_steps_converge_on_the_fusion_layer():
    graph = plan_to_graph(_plan([_step(1), _step(2), _step(3, depends_on=[1])]))
    into_fusion = [e for e in graph["edges"] if e["target"] == "agent:fusion"]
    # Steps 2 and 3 have no dependents, so both feed fusion.
    assert {e["source"] for e in into_fusion} == {"step:2", "step:3"}
    assert all(e["kind"] == "fan_in" for e in into_fusion)


def test_an_empty_plan_still_produces_a_renderable_scaffold():
    graph = plan_to_graph({})
    assert graph["meta"]["step_count"] == 0
    assert any(n["id"] == "agent:orchestrator" for n in graph["nodes"])
    assert graph["nodes"]


def test_a_cyclic_plan_does_not_hang_the_layout():
    """A malformed plan must degrade, never loop."""
    graph = plan_to_graph(_plan([_step(1, depends_on=[2]), _step(2, depends_on=[1])]))
    assert {n["rank"] for n in graph["nodes"] if n["type"] == "tool"}


def test_unknown_dependencies_are_dropped_rather_than_dangling():
    graph = plan_to_graph(_plan([_step(1, depends_on=[99])]))
    assert not any(e["source"] == "step:99" for e in graph["edges"])
    assert _edge(graph, "agent:orchestrator", "step:1")


def test_tool_family_and_plan_metadata_reach_the_client():
    graph = plan_to_graph(
        _plan(
            [_step(1, tool="DifferentialDiagnosis")],
            priority_domains=["BRAIN_MRI"],
            user_facing_explanation="why",
            total_estimated_tokens=1234,
        )
    )
    node = next(n for n in graph["nodes"] if n["id"] == "step:1")
    assert node["family"] == "reasoning"
    assert graph["meta"]["priority_domains"] == ["BRAIN_MRI"]
    assert graph["meta"]["explanation"] == "why"
    assert graph["meta"]["estimated_tokens"] == 1234
