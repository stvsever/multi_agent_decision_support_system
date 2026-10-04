"""
Structured decision models: the model registry, the settings context window for
decision models, and the measurement scales that turn a regression output into
ordered Score levels and back into a continuous estimate.

Everything here is pure computation; no request leaves the process.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.config.settings import reload_settings
from src.full_stack.backend.decision import registry
from src.full_stack.backend.decision.registry import (
    DecisionModelSpec,
    get_decision_model_spec,
    is_decision_model,
    list_decision_models,
    native_model_id,
    normalize_model_id,
    register_decision_model,
)
from src.full_stack.backend.decision.scales import (
    DensityPiece,
    OutputScale,
    ScaleBin,
    best_window,
    combine_density,
    default_scale,
    density_summary,
    equal_width_bins,
    scale_from_mapping,
)


@pytest.fixture
def settings():
    s = reload_settings()
    yield s
    reload_settings()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_jev_1_13_spec_has_documented_limits_and_prices():
    spec = get_decision_model_spec("typesafe/jev-1.13")
    assert spec is not None
    assert spec.model_id == "typesafe/jev-1.13"
    assert spec.state_context_tokens == 32_000
    assert spec.request_context_tokens == 64_000
    assert spec.input_price_per_million == pytest.approx(0.042)
    assert spec.output_price_per_million == 0.0
    assert spec.max_score_levels == 10
    assert spec.native_id == "jev-1.13"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("typesafe/jev-1.13", "typesafe/jev-1.13"),
        ("TypeSafe/JEV-1.13", "typesafe/jev-1.13"),
        ("  typesafe/jev-1.13  ", "typesafe/jev-1.13"),
        ("jev-1.13", "typesafe/jev-1.13"),
        ("typesafe/jev-latest", "typesafe/jev-latest"),
        ("~typesafe/jev-latest", "typesafe/jev-latest"),
        ("~jev-latest", "typesafe/jev-latest"),
    ],
)
def test_registered_jev_ids_resolve(model_id, expected):
    assert is_decision_model(model_id)
    assert get_decision_model_spec(model_id).model_id == expected


def test_unregistered_jev_release_is_treated_as_jev_class():
    spec = get_decision_model_spec("typesafe/jev-1.14")
    assert spec is not None
    assert spec.model_id == "typesafe/jev-1.14"
    assert spec.native_id == "jev-1.14"
    assert spec.state_context_tokens == 32_000
    assert spec.request_context_tokens == 64_000
    bare = get_decision_model_spec("jev-preview")
    assert bare is not None and bare.model_id == "typesafe/jev-preview"


@pytest.mark.parametrize("model_id", ["typesafe/jev-router", "jev-router", "~typesafe/jev-router", "typesafe/jev-router-2"])
def test_jev_router_is_not_a_decision_model(model_id):
    assert not is_decision_model(model_id)
    assert get_decision_model_spec(model_id) is None


@pytest.mark.parametrize(
    "model_id",
    [
        None,
        "",
        "   ",
        "deepseek/deepseek-v4-flash-0731",
        "openai/gpt-5",
        "typesafe/other-model",
        "anthropic/jev-1.13",
        "jev",
    ],
)
def test_unknown_ids_are_not_decision_models(model_id):
    assert not is_decision_model(model_id)
    assert get_decision_model_spec(model_id) is None


def test_normalize_and_native_model_id():
    assert normalize_model_id(" ~TypeSafe/Jev-1.13 ") == "typesafe/jev-1.13"
    assert normalize_model_id(None) == ""
    assert native_model_id("typesafe/jev-1.13") == "jev-1.13"
    assert native_model_id("~typesafe/jev-latest") == "jev-latest"
    assert native_model_id("typesafe/jev-1.14") == "jev-1.14"
    assert native_model_id("openai/gpt-5") == "gpt-5"


def test_state_budget_converts_to_cl100k_tokens():
    spec = get_decision_model_spec("typesafe/jev-1.13")
    assert spec.state_budget_cl100k() == int(32_000 / 1.2)
    assert spec.state_budget_cl100k(ratio=1.0) == 32_000
    assert spec.state_budget_cl100k(ratio=1.6) == 20_000
    # A ratio below 1 (or missing) never inflates the budget beyond the limit.
    assert spec.state_budget_cl100k(ratio=0.5) == 32_000
    assert spec.state_budget_cl100k(ratio=0) == 32_000


def test_spec_as_dict_marks_predictor_only():
    payload = get_decision_model_spec("typesafe/jev-1.13").as_dict()
    assert payload["kind"] == "decision"
    assert payload["roles"] == ["predictor"]
    assert payload["model_id"] == "typesafe/jev-1.13"
    assert payload["state_context_tokens"] == 32_000


def test_list_and_register_decision_models(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    ids = {spec.model_id for spec in list_decision_models()}
    assert {"typesafe/jev-1.13", "typesafe/jev-latest"} <= ids
    assert not is_decision_model("acme/oracle-2")

    register_decision_model(
        DecisionModelSpec(
            model_id="Acme/Oracle-2",
            label="Acme Oracle 2",
            state_context_tokens=8_000,
            request_context_tokens=16_000,
        )
    )
    spec = get_decision_model_spec("acme/oracle-2")
    assert spec is not None and spec.state_context_tokens == 8_000
    assert "Acme/Oracle-2" in {s.model_id for s in list_decision_models()}


# ---------------------------------------------------------------------------
# Settings: context window of a decision model
# ---------------------------------------------------------------------------
def test_effective_context_window_is_the_cl100k_state_budget(settings):
    assert settings.effective_context_window("typesafe/jev-1.13") == int(32_000 / 1.2)
    assert settings.effective_context_window("~typesafe/jev-latest") == int(32_000 / 1.2)
    settings.decision.tokenizer_ratio = 1.0
    assert settings.effective_context_window("typesafe/jev-1.13") == 32_000
    settings.decision.tokenizer_ratio = 1.6
    assert settings.effective_context_window("typesafe/jev-1.13") == 20_000


def test_effective_context_window_unchanged_for_llms(settings):
    before = {
        name: settings.effective_context_window(name)
        for name in (None, "deepseek/deepseek-v4-flash-0731", "openai/gpt-5", "openai/gpt-4o-mini", "typesafe/jev-router")
    }
    assert before["deepseek/deepseek-v4-flash-0731"] == 1_048_576
    assert before["openai/gpt-5"] == 128_000
    assert before["openai/gpt-4o-mini"] == 128_000
    # Not a decision model, so it gets the ordinary fallback window.
    assert before["typesafe/jev-router"] == max(8192, int(settings.models.public_max_context_tokens))

    settings.decision.tokenizer_ratio = 2.0
    after = {name: settings.effective_context_window(name) for name in before}
    assert after == before


def test_auto_output_limit_follows_the_decision_state_budget(settings):
    assert settings.auto_output_token_limit("typesafe/jev-1.13") == int(int(32_000 / 1.2) * 0.5)


# ---------------------------------------------------------------------------
# Scales: grids
# ---------------------------------------------------------------------------
def test_small_integer_scale_gets_one_exact_level_per_value():
    scale = OutputScale(output="item", minimum=0, maximum=6, integer=True)
    assert scale.exact_integer_levels(10)
    bins = scale.coarse_grid(10)
    assert len(bins) == 7
    assert all(b.exact for b in bins)
    assert [b.center for b in bins] == [float(v) for v in range(7)]
    assert [(b.lo, b.hi) for b in bins] == [(v - 0.5, v + 0.5) for v in range(7)]
    assert all(b.width == 0.0 for b in bins)


def test_integer_scale_with_ten_values_is_exact_eleven_is_not():
    ten = OutputScale(output="x", minimum=0, maximum=9, integer=True)
    assert ten.exact_integer_levels(10)
    assert len(ten.coarse_grid(10)) == 10 and all(b.exact for b in ten.coarse_grid(10))
    eleven = OutputScale(output="x", minimum=0, maximum=10, integer=True)
    assert not eleven.exact_integer_levels(10)
    bins = eleven.coarse_grid(10)
    assert len(bins) == 10 and not any(b.exact for b in bins)
    # With fewer levels allowed, the same 0..6 scale is no longer exact.
    assert not OutputScale(output="x", minimum=0, maximum=6, integer=True).exact_integer_levels(5)
    # A continuous scale is never exact.
    assert not OutputScale(output="x", minimum=0, maximum=6).exact_integer_levels(10)


def test_large_integer_scale_gets_equal_width_levels():
    scale = OutputScale(output="madrs_total", minimum=0, maximum=60, integer=True)
    bins = scale.coarse_grid(10)
    assert len(bins) == 10
    assert not any(b.exact for b in bins)
    assert [b.lo for b in bins] == pytest.approx([6.0 * i for i in range(10)])
    assert [b.hi for b in bins] == pytest.approx([6.0 * (i + 1) for i in range(10)])
    assert [b.center for b in bins] == pytest.approx([3.0 + 6.0 * i for i in range(10)])
    assert all(b.width == pytest.approx(6.0) for b in bins)


def test_coarse_grid_level_count_is_clamped_between_2_and_10():
    scale = OutputScale(output="x", minimum=-3, maximum=3)
    assert len(scale.coarse_grid(25)) == 10
    assert len(scale.coarse_grid(1)) == 2
    assert len(scale.coarse_grid(6)) == 6


def test_equal_width_bins_are_contiguous_and_end_exactly_at_hi():
    bins = equal_width_bins(0.0, 1.0, 3)
    assert len(bins) == 3
    assert bins[0].lo == 0.0
    assert bins[-1].hi == 1.0
    for left, right in zip(bins, bins[1:]):
        assert left.hi == pytest.approx(right.lo)
    for b in bins:
        assert b.center == pytest.approx((b.lo + b.hi) / 2.0)


def test_fine_grid_within_a_window():
    scale = OutputScale(output="madrs_total", minimum=0, maximum=60, integer=True)
    # Window of three coarse levels, 12 to 30: 19 integers do not fit 10 levels.
    wide = scale.fine_grid(12.0, 30.0, 10)
    assert len(wide) == 10
    assert not any(b.exact for b in wide)
    assert wide[0].lo == pytest.approx(12.0) and wide[-1].hi == pytest.approx(30.0)
    assert all(b.width == pytest.approx(1.8) for b in wide)
    # Window 12 to 18 holds 7 integers: one exact level per value.
    narrow = scale.fine_grid(12.0, 18.0, 10)
    assert [b.center for b in narrow] == [float(v) for v in range(12, 19)]
    assert all(b.exact for b in narrow)


def test_fine_grid_continuous_scale_is_equal_width():
    scale = OutputScale(output="z", minimum=-3, maximum=3)
    bins = scale.fine_grid(-0.6, 1.2, 9)
    assert len(bins) == 9
    assert bins[0].lo == pytest.approx(-0.6) and bins[-1].hi == pytest.approx(1.2)
    assert all(b.width == pytest.approx(0.2) for b in bins)


# ---------------------------------------------------------------------------
# Scales: density
# ---------------------------------------------------------------------------
def test_best_window_picks_the_heaviest_adjacent_levels():
    assert best_window([0.1, 0.2, 0.3, 0.25, 0.15], width=3) == (1, 4)
    assert best_window([0.6, 0.3, 0.05, 0.03, 0.02], width=3) == (0, 3)
    assert best_window([0.0, 0.0, 0.1, 0.4, 0.5], width=3) == (2, 5)
    # Ties keep the first window.
    assert best_window([0.25, 0.25, 0.25, 0.25], width=3) == (0, 3)
    # Fewer levels than the window width: the whole range.
    assert best_window([0.5, 0.5], width=3) == (0, 2)
    assert best_window([0.2, 0.3, 0.5], width=3) == (0, 3)


def test_combine_density_without_refinement_normalizes_coarse():
    bins = equal_width_bins(0.0, 4.0, 4)
    pieces = combine_density(bins, [1.0, 2.0, 3.0, 4.0])
    assert [p.mass for p in pieces] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert [(p.lo, p.hi) for p in pieces] == [(b.lo, b.hi) for b in bins]


def test_combine_density_replaces_the_window_and_conserves_mass():
    bins = equal_width_bins(0.0, 40.0, 4)  # 0-10, 10-20, 20-30, 30-40
    coarse = [1.0, 2.0, 3.0, 4.0]  # unnormalized on purpose
    fine_bins = equal_width_bins(10.0, 30.0, 4)  # 10-15, 15-20, 20-25, 25-30
    fine = [1.0, 1.0, 1.0, 1.0]
    pieces = combine_density(bins, coarse, window=(1, 3), fine_bins=fine_bins, fine_probs=fine)

    assert sum(p.mass for p in pieces) == pytest.approx(1.0)
    assert len(pieces) == 2 + len(fine_bins)
    assert [(p.lo, p.hi) for p in pieces] == [
        (0.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
        (20.0, 25.0),
        (25.0, 30.0),
        (30.0, 40.0),
    ]
    # The window held 0.2 + 0.3 of the coarse mass, spread over the fine levels.
    assert [p.mass for p in pieces] == pytest.approx([0.1, 0.125, 0.125, 0.125, 0.125, 0.4])


def test_combine_density_with_uneven_fine_answers():
    bins = equal_width_bins(0.0, 30.0, 3)
    fine_bins = equal_width_bins(0.0, 20.0, 2)
    pieces = combine_density(bins, [0.5, 0.3, 0.2], window=(0, 2), fine_bins=fine_bins, fine_probs=[3.0, 1.0])
    assert [p.mass for p in pieces] == pytest.approx([0.6, 0.2, 0.2])
    assert sum(p.mass for p in pieces) == pytest.approx(1.0)


def test_density_summary_uniform_on_zero_two():
    pieces = [DensityPiece(0.0, 1.0, 0.5, 0.5), DensityPiece(1.0, 2.0, 1.5, 0.5)]
    summary = density_summary(pieces)
    assert summary["mean"] == pytest.approx(1.0)
    assert summary["sd"] == pytest.approx(math.sqrt(1.0 / 3.0))
    assert summary["q05"] == pytest.approx(0.1)
    assert summary["q25"] == pytest.approx(0.5)
    assert summary["median"] == pytest.approx(1.0)
    assert summary["q75"] == pytest.approx(1.5)
    assert summary["q95"] == pytest.approx(1.9)


def test_density_summary_unnormalized_three_pieces():
    # Masses 2:6:2 on 0-10, 10-20, 20-30 (normalized 0.2, 0.6, 0.2).
    pieces = [
        DensityPiece(0.0, 10.0, 5.0, 2.0),
        DensityPiece(10.0, 20.0, 15.0, 6.0),
        DensityPiece(20.0, 30.0, 25.0, 2.0),
    ]
    summary = density_summary(pieces)
    # E[X^2] = sum p (c^2 + w^2/12) = 5 + 135 + 125 + 100/12; Var = E[X^2] - 15^2.
    expected_var = 5.0 + 135.0 + 125.0 + 100.0 / 12.0 - 225.0
    assert summary["mean"] == pytest.approx(15.0)
    assert summary["sd"] == pytest.approx(math.sqrt(expected_var))
    assert summary["q05"] == pytest.approx(2.5)
    assert summary["q25"] == pytest.approx(10.0 + 10.0 * 0.05 / 0.6)
    assert summary["median"] == pytest.approx(15.0)
    assert summary["q75"] == pytest.approx(10.0 + 10.0 * 0.55 / 0.6)
    assert summary["q95"] == pytest.approx(27.5)


def test_density_summary_exact_levels_have_no_within_level_spread():
    pieces = [
        DensityPiece(-0.5, 0.5, 0.0, 0.2, exact=True),
        DensityPiece(0.5, 1.5, 1.0, 0.5, exact=True),
        DensityPiece(1.5, 2.5, 2.0, 0.3, exact=True),
    ]
    summary = density_summary(pieces)
    assert summary["mean"] == pytest.approx(1.1)
    assert summary["sd"] == pytest.approx(0.7)
    assert summary["q05"] == 0.0
    assert summary["q25"] == 1.0
    assert summary["median"] == 1.0
    assert summary["q75"] == 2.0
    assert summary["q95"] == 2.0


def test_density_summary_skips_empty_levels_in_quantiles():
    pieces = [
        DensityPiece(0.0, 1.0, 0.5, 0.0),
        DensityPiece(1.0, 2.0, 1.5, 1.0),
        DensityPiece(2.0, 3.0, 2.5, 0.0),
    ]
    summary = density_summary(pieces)
    assert summary["mean"] == pytest.approx(1.5)
    assert summary["q05"] == pytest.approx(1.05)
    assert summary["q95"] == pytest.approx(1.95)
    assert summary["q05"] <= summary["q25"] <= summary["median"] <= summary["q75"] <= summary["q95"]


def test_point_estimate_is_continuous_not_snapped_to_a_level_centre():
    bins = OutputScale(output="madrs_total", minimum=0, maximum=60, integer=True).coarse_grid(10)
    probs = [0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    summary = density_summary(combine_density(bins, probs))
    assert summary["mean"] == pytest.approx(0.3 * 15.0 + 0.7 * 21.0)
    assert summary["mean"] not in [b.center for b in bins]


# ---------------------------------------------------------------------------
# Scales: labels
# ---------------------------------------------------------------------------
def test_level_labels_without_reference_use_the_range():
    scale = OutputScale(
        output="madrs_total",
        minimum=0,
        maximum=60,
        integer=True,
        unit="points",
        low_meaning="no depressive symptoms",
        high_meaning="most severe depression",
    )
    bins = scale.coarse_grid(10)
    total = len(bins)
    assert scale.level_label(bins[0], index=0, total=total) == (
        "0 to 6 points; at the very low end of the scale; low end: no depressive symptoms"
    )
    assert scale.level_label(bins[2], index=2, total=total) == "12 to 18 points; in the low part of the scale"
    assert scale.level_label(bins[4], index=4, total=total) == "24 to 30 points; in the middle of the scale"
    assert scale.level_label(bins[7], index=7, total=total) == "42 to 48 points; in the high part of the scale"
    assert scale.level_label(bins[9], index=9, total=total) == (
        "54 to 60 points; at the very high end of the scale; high end: most severe depression"
    )


def test_level_labels_for_exact_levels_show_the_value():
    scale = OutputScale(output="item_sadness", minimum=0, maximum=6, integer=True)
    bins = scale.coarse_grid(10)
    labels = [scale.level_label(b, index=i, total=len(bins)) for i, b in enumerate(bins)]
    assert labels[0] == "0; at the very low end of the scale"
    assert labels[3] == "3; in the middle of the scale"
    assert labels[6] == "6; at the very high end of the scale"
    with_unit = OutputScale(output="item_sadness", minimum=0, maximum=6, integer=True, unit="points")
    assert with_unit.level_label(bins[2], index=2, total=7).startswith("2 points; ")


def test_level_labels_with_reference_mean_and_sd():
    scale = OutputScale(output="iq", minimum=40, maximum=160, reference_mean=100.0, reference_sd=15.0)
    assert scale.band(100.0) == "close to the reference mean"
    assert scale.band(104.0) == "close to the reference mean"
    assert scale.band(108.0) == "about half a standard deviation above the reference mean"
    assert scale.band(115.0) == "about 1 standard deviation above the reference mean"
    assert scale.band(77.5) == "about 1.5 standard deviations below the reference mean"
    assert scale.band(70.0) == "about 2 standard deviations below the reference mean"
    assert scale.band(145.0) == "far above the reference mean (more than 2.5 standard deviations)"
    label = scale.level_label(ScaleBin(lo=110.0, hi=120.0, center=115.0), index=5, total=10)
    # A level 10 wide is printed without decimals.
    assert label == "110 to 120; about 1 standard deviation above the reference mean"
    narrow = scale.level_label(ScaleBin(lo=112.5, hi=117.5, center=115.0), index=5, total=10)
    assert narrow == "112.5 to 117.5; about 1 standard deviation above the reference mean"


def test_invalid_reference_sd_falls_back_to_range_bands():
    for bad in (0.0, -2.0, float("nan"), float("inf")):
        scale = OutputScale(output="x", minimum=0, maximum=10, reference_mean=5.0, reference_sd=bad)
        assert scale.reference_sd is None
        assert scale.band(5.0) == "in the middle of the scale"


def test_format_value_decimals_follow_level_width():
    continuous = OutputScale(output="x", minimum=-3, maximum=3)
    assert continuous.format_value(1.23456, width=12.0) == "1"
    assert continuous.format_value(1.23456, width=2.0) == "1.2"
    assert continuous.format_value(1.23456, width=0.6) == "1.23"
    assert continuous.format_value(1.23456, width=0.05) == "1.235"
    integer = OutputScale(output="x", minimum=0, maximum=60, integer=True)
    assert integer.format_value(13.8, width=1.8) == "14"


def test_output_scale_bounds_are_sanitized():
    swapped = OutputScale(output="x", minimum=10, maximum=0)
    assert (swapped.minimum, swapped.maximum) == (0.0, 10.0)
    flat = OutputScale(output="x", minimum=4, maximum=4)
    assert (flat.minimum, flat.maximum) == (4.0, 5.0)
    assert flat.span == 1.0
    with pytest.raises(ValueError):
        OutputScale(output="x", minimum=0, maximum=float("inf"))
    with pytest.raises(ValueError):
        OutputScale(output="x", minimum=float("nan"), maximum=1)


def test_output_scale_round_trips_through_to_dict():
    scale = OutputScale(
        output="madrs_total",
        minimum=0,
        maximum=60,
        integer=True,
        unit="points",
        reference_mean=20.0,
        reference_sd=8.0,
        source="task_spec",
        notes=["given"],
    )
    clone = OutputScale(**scale.to_dict())
    assert clone == scale


# ---------------------------------------------------------------------------
# Scales: parsing
# ---------------------------------------------------------------------------
def test_scale_from_mapping_parses_every_documented_key():
    scale = scale_from_mapping(
        "madrs_total",
        {
            "min": 0,
            "max": 60,
            "integer": True,
            "unit": "points",
            "description": "MADRS total score",
            "low_meaning": "no symptoms",
            "high_meaning": "most severe",
            "reference_mean": 22.5,
            "reference_sd": 7.5,
        },
        source="task_spec",
    )
    assert scale is not None
    assert (scale.minimum, scale.maximum, scale.integer) == (0.0, 60.0, True)
    assert scale.unit == "points"
    assert scale.description == "MADRS total score"
    assert scale.low_meaning == "no symptoms"
    assert scale.high_meaning == "most severe"
    assert scale.reference_mean == 22.5
    assert scale.reference_sd == 7.5
    assert scale.source == "task_spec"
    assert scale.output == "madrs_total"


def test_scale_from_mapping_accepts_aliases_and_strings():
    scale = scale_from_mapping("hamd", {"minimum": "0", "maximum": "52", "integer": "yes", "mean": "18", "sd": "6"}, source="compiler")
    assert (scale.minimum, scale.maximum, scale.integer) == (0.0, 52.0, True)
    assert (scale.reference_mean, scale.reference_sd) == (18.0, 6.0)
    alias = scale_from_mapping("z", {"lo": -5, "hi": 5, "integer": "false"}, source="compiler")
    assert (alias.minimum, alias.maximum, alias.integer) == (-5.0, 5.0, False)
    low_high = scale_from_mapping("z", {"low": 1, "high": 7}, source="compiler")
    assert (low_high.minimum, low_high.maximum) == (1.0, 7.0)


def test_scale_from_mapping_unit_fallback_and_swapped_bounds():
    scale = scale_from_mapping("score", {"min": 10, "max": 0}, source="compiler", unit="points")
    assert scale.unit == "points"
    assert (scale.minimum, scale.maximum) == (0.0, 10.0)
    own_unit = scale_from_mapping("score", {"min": 0, "max": 10, "unit": "mm"}, source="compiler", unit="points")
    assert own_unit.unit == "mm"


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        "0-60",
        {},
        {"min": 0},
        {"max": 60},
        {"min": 5, "max": 5},
        {"min": "zero", "max": 60},
        {"min": 0, "max": "lots"},
        {"min": float("nan"), "max": 60},
        {"min": 0, "max": float("inf")},
        {"min": "", "max": ""},
        {"min": None, "max": None},
    ],
)
def test_scale_from_mapping_rejects_unusable_input(data):
    assert scale_from_mapping("x", data, source="compiler") is None


def test_scale_from_mapping_drops_a_non_positive_reference_sd():
    scale = scale_from_mapping("x", {"min": 0, "max": 10, "reference_mean": 5, "reference_sd": 0}, source="compiler")
    assert scale is not None
    assert scale.reference_mean == 5.0
    assert scale.reference_sd is None


def test_default_scale_is_standardized():
    scale = default_scale("phq_total")
    assert (scale.minimum, scale.maximum) == (-3.0, 3.0)
    assert scale.integer is False
    assert scale.unit == "standardized units"
    assert scale.description == "phq total"
    assert (scale.reference_mean, scale.reference_sd) == (0.0, 1.0)
    assert scale.source == "default"
    assert scale.notes
    assert default_scale("phq_total", unit="points").unit == "points"
    bins = scale.coarse_grid(10)
    assert len(bins) == 10 and bins[0].lo == -3.0 and bins[-1].hi == 3.0
