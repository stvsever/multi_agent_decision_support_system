"""The cost projection must track real run ledgers, not a flat guess."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.api import cost as cost_module
from src.full_stack.backend.api.schemas import DashboardConfig

REPO = Path(__file__).resolve().parent.parent.parent
PSEUDO_INPUTS = REPO / "src/full_stack/backend/data/pseudo_data/inputs"
PSEUDO_OUTPUTS = REPO / "src/full_stack/backend/data/pseudo_data/outputs"

PRICE = 0.065
COMPLETION_PRICE = 0.18


@pytest.fixture
def priced(monkeypatch):
    """A deterministic price index so assertions do not depend on the network."""
    monkeypatch.setattr(
        cost_module,
        "pricing_index",
        lambda cached_only=False: {
            "deepseek/deepseek-v4-flash-0731": {
                "prompt_usd_per_mtok": PRICE,
                "completion_usd_per_mtok": COMPLETION_PRICE,
                "context_length": 1_310_720,
            }
        },
    )
    return DashboardConfig()


def test_plan_size_grows_with_the_number_of_domains():
    assert cost_module.expected_plan_steps(0) == cost_module.DEFAULT_PLAN_STEPS
    assert cost_module.expected_plan_steps(1) < cost_module.expected_plan_steps(5)
    assert cost_module.expected_plan_steps(5) == 13
    # Bounded at both ends so a pathological overview cannot distort the estimate.
    assert cost_module.expected_plan_steps(200) <= 28


def test_input_tokens_come_from_the_participant_overview():
    directory = PSEUDO_INPUTS / "SUBJ_001_PSEUDO"
    assert cost_module.participant_input_tokens(directory) == 8680
    assert cost_module.participant_domain_count(directory) == 5


def test_missing_participant_yields_zero_rather_than_raising(tmp_path):
    assert cost_module.participant_input_tokens(tmp_path) == 0
    assert cost_module.participant_domain_count(tmp_path) == 0


def test_projection_lands_within_the_stated_spread_of_recorded_runs(priced):
    """Every recorded run must fall inside the range the interface advertises."""
    checked = 0
    for report_path in sorted(PSEUDO_OUTPUTS.glob("*/performance_report_*.json")):
        report = json.loads(report_path.read_text())
        directory = PSEUDO_INPUTS / report["participant_id"]
        if not directory.is_dir():
            continue
        deep = bool((report.get("deep_phenotype") or {}).get("generated"))
        if deep:
            # The ledger in the report excludes the communicator pass, so that
            # run is not a like-for-like comparison here.
            continue
        estimate = cost_module.estimate_run(
            config=priced,
            input_tokens=cost_module.participant_input_tokens(directory),
            iterations=report["iterations"],
            plan_steps=report["plan_summary"]["total_steps"],
            include_deep_report=False,
        )
        actual = report["token_usage"]["total_tokens"]
        ratio = estimate["total_tokens"] / actual
        assert 1 - cost_module.ESTIMATE_SPREAD <= ratio <= 1 + cost_module.ESTIMATE_SPREAD, (
            f"{report['participant_id']}: projected {estimate['total_tokens']} against {actual}"
        )
        checked += 1
    assert checked >= 4


def test_projection_scales_with_plan_size_and_iterations(priced):
    small = cost_module.estimate_run(config=priced, input_tokens=8680, iterations=1, plan_steps=9)
    large = cost_module.estimate_run(config=priced, input_tokens=8680, iterations=1, plan_steps=16)
    assert large["total_tokens"] > small["total_tokens"]

    once = cost_module.estimate_run(config=priced, input_tokens=8680, iterations=1, plan_steps=9)
    thrice = cost_module.estimate_run(config=priced, input_tokens=8680, iterations=3, plan_steps=9)
    assert thrice["total_tokens"] > 2 * once["total_tokens"]


def test_deep_report_is_the_largest_single_addition(priced):
    without = cost_module.estimate_run(
        config=priced, input_tokens=8680, iterations=1, include_deep_report=False
    )
    with_report = cost_module.estimate_run(
        config=priced, input_tokens=8680, iterations=1, include_deep_report=True
    )
    assert with_report["total_tokens"] > without["total_tokens"]
    assert any(line["role"] == "communicator" for line in with_report["lines"])


def test_usd_uses_the_per_million_prices(priced):
    estimate = cost_module.estimate_run(config=priced, input_tokens=0, iterations=1, plan_steps=1)
    expected = sum(
        line["prompt_tokens"] / 1_000_000 * PRICE + line["completion_tokens"] / 1_000_000 * COMPLETION_PRICE
        for line in estimate["lines"]
    )
    assert estimate["usd"] == pytest.approx(expected, rel=1e-6)
    assert estimate["usd_low"] < estimate["usd"] < estimate["usd_high"]


def test_an_unpriced_model_reports_no_total_rather_than_zero(monkeypatch):
    monkeypatch.setattr(cost_module, "pricing_index", lambda cached_only=False: {})
    estimate = cost_module.estimate_run(config=DashboardConfig(), input_tokens=1000, iterations=1)
    assert estimate["fully_priced"] is False
    assert estimate["usd"] is None


def test_actual_cost_from_observed_usage(priced):
    actual = cost_module.actual_cost_from_usage(
        {"deepseek/deepseek-v4-flash-0731": {"prompt": 1_000_000, "completion": 1_000_000}}
    )
    assert actual["usd"] == pytest.approx(PRICE + COMPLETION_PRICE)
    assert actual["total_tokens"] == 2_000_000
    assert actual["fully_priced"] is True


def test_actual_cost_flags_an_unknown_model(priced):
    actual = cost_module.actual_cost_from_usage({"someone/unlisted": {"prompt": 10, "completion": 10}})
    assert actual["fully_priced"] is False
    assert actual["usd"] is None
    assert actual["total_tokens"] == 20


def test_a_hot_path_can_price_without_touching_the_network(monkeypatch):
    """The thread draining a live worker must never block on a catalog fetch."""
    calls: list[bool] = []

    def _index(cached_only=False):
        calls.append(cached_only)
        return {}

    monkeypatch.setattr(cost_module, "pricing_index", _index)
    cost_module.actual_cost_from_usage({"m": {"prompt": 1, "completion": 1}}, cached_only=True)
    assert calls == [True]
