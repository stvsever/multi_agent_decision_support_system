"""
Execution plan to renderable graph.

The orchestrator emits steps with `depends_on` edges; the plan executor then
runs every currently-unblocked step concurrently. That means dependency depth
is exactly the parallel structure, so ranking by depth recovers the real
sequential/parallel shape rather than an invented layout.

The graph also carries the fixed actor-critic scaffold around the tool steps so
the interface can show the whole workflow, including the critic feedback loop.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

#: Every tool the orchestrator may schedule, with the family it belongs to.
TOOL_FAMILIES: Dict[str, str] = {
    "UnimodalCompressor": "compression",
    "MultimodalNarrativeCreator": "narrative",
    "HypothesisGenerator": "reasoning",
    "CodeExecutor": "computation",
    "FeatureSynthesizer": "synthesis",
    "ClinicalRelevanceRanker": "ranking",
    "AnomalyNarrativeBuilder": "narrative",
    "PhenotypeRepresentation": "synthesis",
    "DifferentialDiagnosis": "reasoning",
    "ChunkEvidenceExtractor": "synthesis",
}

#: The agents that bracket the tool layer, in pipeline order.
AGENT_NODES = [
    {"id": "agent:orchestrator", "label": "Orchestrator", "role": "orchestrator", "stage": 1},
    {"id": "agent:fusion", "label": "Fusion Layer", "role": "integrator", "stage": 3},
    {"id": "agent:predictor", "label": "Predictor", "role": "predictor", "stage": 4},
    {"id": "agent:critic", "label": "Critic", "role": "critic", "stage": 5},
    {"id": "agent:communicator", "label": "Communicator", "role": "communicator", "stage": 6},
]


def _ranks(steps: List[Dict[str, Any]]) -> Dict[int, int]:
    """Longest-path depth per step; steps sharing a depth execute together."""
    by_id = {int(s.get("step_id")): s for s in steps if s.get("step_id") is not None}
    memo: Dict[int, int] = {}
    visiting: Set[int] = set()

    def depth(step_id: int) -> int:
        if step_id in memo:
            return memo[step_id]
        if step_id in visiting:  # a malformed plan must not hang the layout
            return 0
        visiting.add(step_id)
        deps = [int(d) for d in (by_id.get(step_id, {}).get("depends_on") or []) if int(d) in by_id]
        value = 0 if not deps else 1 + max(depth(d) for d in deps)
        visiting.discard(step_id)
        memo[step_id] = value
        return value

    return {sid: depth(sid) for sid in by_id}


def _edge_kind(source: int, target: int, out_degree: Dict[int, int], in_degree: Dict[int, int]) -> str:
    if out_degree.get(source, 0) > 1 and in_degree.get(target, 0) > 1:
        return "mesh"
    if out_degree.get(source, 0) > 1:
        return "fan_out"
    if in_degree.get(target, 0) > 1:
        return "fan_in"
    return "sequential"


def plan_to_graph(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build ``{"nodes": [...], "edges": [...], "lanes": [...], "meta": {...}}``.

    Node ids are stable across iterations so a live run can update a node in
    place instead of re-mounting the graph.
    """
    if not isinstance(plan, dict):
        plan = {}
    steps = [s for s in (plan.get("steps") or []) if isinstance(s, dict)]
    ranks = _ranks(steps)
    by_id = {int(s["step_id"]): s for s in steps if s.get("step_id") is not None}

    out_degree: Dict[int, int] = {}
    in_degree: Dict[int, int] = {}
    for step in steps:
        target = int(step["step_id"])
        deps = [int(d) for d in (step.get("depends_on") or []) if int(d) in by_id]
        in_degree[target] = len(deps)
        for dep in deps:
            out_degree[dep] = out_degree.get(dep, 0) + 1

    max_rank = max(ranks.values()) if ranks else -1
    # Steps with no dependencies are dispatched together by the orchestrator.
    root_step_count = sum(1 for s in steps if not [d for d in (s.get("depends_on") or []) if int(d) in by_id])
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    nodes.append(
        {
            "id": "agent:orchestrator",
            "type": "agent",
            "label": "Orchestrator",
            "role": "orchestrator",
            "stage": 1,
            "rank": -1,
            "lane": 0,
            "detail": plan.get("user_facing_explanation") or plan.get("reasoning") or "",
            "meta": {
                "total_steps": len(steps),
                "priority_domains": plan.get("priority_domains") or [],
                "estimated_tokens": plan.get("total_estimated_tokens"),
                "fusion_strategy": plan.get("fusion_strategy") or "",
            },
        }
    )

    lane_counter: Dict[int, int] = {}
    for step in sorted(steps, key=lambda s: (ranks.get(int(s["step_id"]), 0), int(s["step_id"]))):
        step_id = int(step["step_id"])
        rank = ranks.get(step_id, 0)
        lane = lane_counter.get(rank, 0)
        lane_counter[rank] = lane + 1
        tool = str(step.get("tool_name") or "")
        nodes.append(
            {
                "id": f"step:{step_id}",
                "type": "tool",
                "step_id": step_id,
                "label": tool,
                "family": TOOL_FAMILIES.get(tool, "other"),
                "stage": 2,
                "rank": rank,
                "lane": lane,
                "detail": step.get("description") or "",
                "reasoning": step.get("reasoning") or "",
                "status": step.get("status") or "PENDING",
                "meta": {
                    "input_domains": step.get("input_domains") or [],
                    "expected_output": step.get("expected_output") or "",
                    "estimated_tokens": step.get("estimated_tokens") or 0,
                    "actual_tokens": step.get("actual_tokens"),
                    "execution_time_ms": step.get("execution_time_ms"),
                    "parameters": step.get("parameters") or {},
                },
            }
        )

        deps = [int(d) for d in (step.get("depends_on") or []) if int(d) in by_id]
        if deps:
            for dep in deps:
                edges.append(
                    {
                        "id": f"e:{dep}->{step_id}",
                        "source": f"step:{dep}",
                        "target": f"step:{step_id}",
                        "kind": _edge_kind(dep, step_id, out_degree, in_degree),
                    }
                )
        else:
            edges.append(
                {
                    "id": f"e:orch->{step_id}",
                    "source": "agent:orchestrator",
                    "target": f"step:{step_id}",
                    "kind": "fan_out" if root_step_count > 1 else "sequential",
                    "dispatch": True,
                }
            )

    terminal = [
        int(s["step_id"])
        for s in steps
        if out_degree.get(int(s["step_id"]), 0) == 0 and s.get("step_id") is not None
    ]

    scaffold_rank = max_rank + 1
    for index, agent in enumerate(AGENT_NODES[1:]):  # orchestrator already added
        nodes.append(
            {
                **agent,
                "type": "agent",
                "rank": scaffold_rank + index,
                "lane": 0,
                "detail": "",
                "meta": {},
            }
        )

    for step_id in terminal:
        edges.append(
            {
                "id": f"e:{step_id}->fusion",
                "source": f"step:{step_id}",
                "target": "agent:fusion",
                "kind": "fan_in" if len(terminal) > 1 else "sequential",
            }
        )
    if not terminal and steps:
        edges.append(
            {"id": "e:orch->fusion", "source": "agent:orchestrator", "target": "agent:fusion", "kind": "sequential"}
        )

    chain = ["agent:fusion", "agent:predictor", "agent:critic", "agent:communicator"]
    for source, target in zip(chain, chain[1:]):
        edges.append(
            {"id": f"e:{source}->{target}", "source": source, "target": target, "kind": "sequential"}
        )

    edges.append(
        {
            "id": "e:critic->orchestrator",
            "source": "agent:critic",
            "target": "agent:orchestrator",
            "kind": "feedback",
            "label": "revise",
        }
    )

    lanes = sorted({int(n["rank"]) for n in nodes})
    return {
        "plan_id": plan.get("plan_id") or "",
        "iteration": plan.get("iteration") or 1,
        "nodes": nodes,
        "edges": edges,
        "lanes": [
            {
                "rank": rank,
                "size": sum(1 for n in nodes if n["rank"] == rank),
                "parallel": sum(1 for n in nodes if n["rank"] == rank and n["type"] == "tool") > 1,
            }
            for rank in lanes
        ],
        "meta": {
            "step_count": len(steps),
            "depth": max_rank + 1,
            "max_parallel": max(lane_counter.values()) if lane_counter else 0,
            "priority_domains": plan.get("priority_domains") or [],
            "explanation": plan.get("user_facing_explanation") or "",
            "reasoning": plan.get("reasoning") or "",
            "fusion_strategy": plan.get("fusion_strategy") or "",
            "estimated_tokens": plan.get("total_estimated_tokens") or 0,
        },
    }
