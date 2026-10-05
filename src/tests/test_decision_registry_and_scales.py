"""
Structured decision models: the model registry, the settings context window for
decision models, and the measurement scales that turn a regression output into
ordered Score levels and back into a continuous estimate.

Everything here is pure computation; no request leaves the process.
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.full_stack.backend.config import settings as settings_module
from src.full_stack.backend.config.settings import LLMBackend, reload_settings
from src.full_stack.backend.agents.decision import registry
from src.full_stack.backend.agents.decision.registry import (
    DecisionModelSpec,
    get_decision_model_spec,
    is_decision_model,
    list_decision_models,
    native_model_id,
    normalize_model_id,
    register_decision_model,
)
from src.full_stack.backend.agents.decision.scales import (
    DensityPiece,
    OutputScale,
    ScaleBin,
    best_window,
    combine_density,
    default_scale,
    density_summary,
    equal_width_bins,
    integer_bins,
    scale_from_mapping,
)


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    """The dashboard's model catalog cache lives in COMPASS_HOME; never read the developer's."""
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "compass_home"))
    monkeypatch.setattr(settings_module, "_CATALOG_CONTEXT_CACHE", {})


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
    # Nothing configured: the public window only covers models nothing else knows.
    assert settings.models.public_max_context_tokens == 0
    before = {
        name: settings.effective_context_window(name)
        for name in (None, "deepseek/deepseek-v4-flash-0731", "openai/gpt-5", "openai/gpt-4o-mini", "typesafe/jev-router")
    }
    # Without a catalog the built-in table decides.
    assert before[None] == 1_048_576
    assert before["deepseek/deepseek-v4-flash-0731"] == 1_048_576
    assert before["openai/gpt-5"] == 272_000
    assert before["openai/gpt-4o-mini"] == 128_000
    # Not a decision model and not in the table or the catalog: the generic
    # fallback (a provider that rejects a direct prompt as too long makes the
    # pipeline orchestrate that attempt).
    assert before["typesafe/jev-router"] == 128_000

    settings.decision.tokenizer_ratio = 2.0
    after = {name: settings.effective_context_window(name) for name in before}
    assert after == before


def test_unknown_model_window_is_the_configured_public_window(settings):
    assert settings.effective_context_window("acme/unknown-model") == 128_000
    assert settings.context_window_resolution("acme/unknown-model").source == "fallback"
    settings.models.public_max_context_tokens = 64_000
    assert settings.effective_context_window("acme/unknown-model") == 64_000
    assert settings.context_window_resolution("acme/unknown-model").source == "configured"
    settings.models.public_max_context_tokens = 4_000
    # Never below 8192.
    assert settings.effective_context_window("acme/unknown-model") == 8192
    # A model the table knows keeps its own window.
    assert settings.effective_context_window("openai/gpt-4o") == 128_000


def _write_catalog(home: Path, rows) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "model_catalog.json").write_text(json.dumps({"models": rows}))


def test_unknown_model_window_comes_from_the_dashboard_catalog(settings, tmp_path):
    _write_catalog(
        tmp_path / "compass_home",
        [
            {"id": "acme/big-context", "context_length": 200_000},
            {"id": "Acme/Mixed-Case", "context_length": 65_536},
            {"id": "acme/no-length"},
            {"id": "openai/gpt-5", "context_length": 999},
            {"id": "deepseek/deepseek-v4-flash-0731", "context_length": 1_310_720},
            {"id": "~acme/latest", "context_length": 77_000},
            "not a row",
        ],
    )
    assert settings.effective_context_window("acme/big-context") == 200_000
    # Ids are matched case-insensitively and a leading "~" (latest alias) is ignored
    # unless the catalog lists the alias itself.
    assert settings.effective_context_window("~acme/big-context") == 200_000
    assert settings.effective_context_window("~acme/latest") == 77_000
    assert settings.effective_context_window("acme/mixed-case") == 65_536
    # A ":variant" suffix falls back to the base id.
    assert settings.effective_context_window("acme/big-context:free") == 200_000
    # A row without a context length, or a model the catalog does not list, falls
    # back to the built-in table, then to the generic fallback.
    assert settings.effective_context_window("acme/no-length") == 128_000
    assert settings.effective_context_window("acme/not-listed") == 128_000
    # The catalog is more current than the built-in table ...
    assert settings.effective_context_window("openai/gpt-5") == 999
    assert settings.effective_context_window("gpt-5") == 999  # sent as openai/gpt-5
    assert settings.effective_context_window("openai/gpt-4o-mini") == 128_000
    # ... and it decides the public model's window too.
    assert settings.effective_context_window(None) == 1_310_720
    assert settings.context_window_resolution(None).source == "catalog"
    # A configured public window does not override the catalog ...
    settings.models.public_max_context_tokens = 32_000
    assert settings.effective_context_window("deepseek/deepseek-v4-flash-0731") == 1_310_720
    # ... a role override does.
    settings.models.role_context_windows = {"predictor": 50_000}
    assert settings.effective_context_window("deepseek/deepseek-v4-flash-0731", role="predictor") == 50_000
    assert settings.effective_context_window("deepseek/deepseek-v4-flash-0731", role="critic") == 1_310_720


def test_the_catalog_does_not_describe_a_self_hosted_endpoint(settings, tmp_path):
    _write_catalog(tmp_path / "compass_home", [{"id": "qwen/qwen3-14b", "context_length": 131_072}])
    assert settings.effective_context_window("qwen/qwen3-14b") == 131_072
    settings.openrouter_base_url = "http://gpu042.cluster.local:8000/v1"
    # A vLLM server behind the OpenAI protocol: the table's native window, or the forced one.
    assert settings.effective_context_window("qwen/qwen3-14b") == 40_960
    settings.models.role_context_windows = {"all": 32_768}
    assert settings.effective_context_window("qwen/qwen3-14b", role="tool") == 32_768


def test_catalog_cache_follows_the_catalog_path_not_only_its_mtime(settings, tmp_path, monkeypatch):
    first, second = tmp_path / "home_a", tmp_path / "home_b"
    _write_catalog(first, [{"id": "acme/model", "context_length": 100_000}])
    _write_catalog(second, [{"id": "acme/model", "context_length": 300_000}])
    # Same modification time on both files: only the path tells them apart.
    stamp = (first / "model_catalog.json").stat().st_mtime
    os.utime(second / "model_catalog.json", (stamp, stamp))
    monkeypatch.setenv("COMPASS_HOME", str(first))
    assert settings.effective_context_window("acme/model") == 100_000
    monkeypatch.setenv("COMPASS_HOME", str(second))
    assert settings.effective_context_window("acme/model") == 300_000
    monkeypatch.setenv("COMPASS_HOME", str(first))
    assert settings.effective_context_window("acme/model") == 100_000


def test_unreadable_catalog_is_ignored(settings, tmp_path):
    home = tmp_path / "compass_home"
    home.mkdir(parents=True)
    (home / "model_catalog.json").write_text("{not json")
    assert settings.effective_context_window("acme/big-context") == 128_000
    assert settings.effective_context_window("openai/gpt-5") == 272_000


def test_decision_model_window_holds_on_the_local_backend(settings):
    settings.models.backend = LLMBackend.LOCAL
    settings.models.local_max_model_len = 16_384
    # A decision model's state limit applies on any backend.
    assert settings.effective_context_window("typesafe/jev-1.13") == int(32_000 / 1.2)
    # Other models on the local backend get the local window.
    assert settings.effective_context_window("openai/gpt-5") == 16_384
    settings.models.local_max_model_len = 0
    settings.models.local_max_tokens = 500
    assert settings.effective_context_window("openai/gpt-5") == 1024


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


def _integer_members(b: ScaleBin):
    return list(range(int(math.ceil(b.lo)), int(math.floor(b.hi)) + 1))


def test_large_integer_scale_gets_contiguous_integer_ranges():
    scale = OutputScale(output="madrs_total", minimum=0, maximum=60, integer=True)
    bins = scale.coarse_grid(10)
    assert len(bins) == 10
    assert not any(b.exact for b in bins)
    # 61 values over 10 levels: divmod(61, 10) = (6, 1), so the first level holds 7 values.
    members = [_integer_members(b) for b in bins]
    assert [len(m) for m in members] == [7, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    assert members[0] == [0, 1, 2, 3, 4, 5, 6]
    assert members[1] == [7, 8, 9, 10, 11, 12]
    assert members[-1] == [55, 56, 57, 58, 59, 60]
    # Each level is the uniform interval [a - 0.5, b + 0.5] with center (a + b) / 2.
    assert [b.lo for b in bins] == [-0.5, 6.5, 12.5, 18.5, 24.5, 30.5, 36.5, 42.5, 48.5, 54.5]
    assert [b.hi for b in bins] == [6.5, 12.5, 18.5, 24.5, 30.5, 36.5, 42.5, 48.5, 54.5, 60.5]
    assert [b.center for b in bins] == [3.0, 9.5, 15.5, 21.5, 27.5, 33.5, 39.5, 45.5, 51.5, 57.5]
    assert [b.width for b in bins] == [7.0] + [6.0] * 9
    # No overlap and no gap: every integer of the scale is in exactly one level.
    flat = [v for m in members for v in m]
    assert flat == list(range(0, 61))
    for left, right in zip(bins, bins[1:]):
        assert left.hi == right.lo


@pytest.mark.parametrize(
    "first, last, levels, sizes",
    [
        (0, 60, 10, [7, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
        (1, 23, 4, [6, 6, 6, 5]),  # divmod(23, 4) = (5, 3)
        (13, 30, 10, [2, 2, 2, 2, 2, 2, 2, 2, 1, 1]),  # divmod(18, 10) = (1, 8)
        (0, 10, 10, [2, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        (-5, 5, 3, [4, 4, 3]),
    ],
)
def test_integer_bins_are_contiguous_ranges_of_near_equal_size(first, last, levels, sizes):
    bins = integer_bins(first, last, levels)
    members = [_integer_members(b) for b in bins]
    assert [len(m) for m in members] == sizes
    assert [v for m in members for v in m] == list(range(first, last + 1))
    assert not any(b.exact for b in bins)
    for b, m in zip(bins, members):
        assert (b.lo, b.hi) == (m[0] - 0.5, m[-1] + 0.5)
        assert b.center == (m[0] + m[-1]) / 2.0


def test_integer_bins_with_few_values_are_exact():
    bins = integer_bins(3, 7, 10)
    assert [b.center for b in bins] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert all(b.exact for b in bins)
    assert [(b.lo, b.hi) for b in bins] == [(v - 0.5, v + 0.5) for v in range(3, 8)]
    # Exactly `levels` values is still one exact level per value.
    assert all(b.exact for b in integer_bins(0, 9, 10))
    single = integer_bins(5, 5, 3)
    assert len(single) == 1 and single[0].exact and single[0].center == 5.0


def test_integer_level_labels_name_their_integer_range():
    scale = OutputScale(output="x", minimum=0, maximum=10, integer=True, unit="points")
    bins = scale.coarse_grid(10)
    # 11 values over 10 levels: the first level is 0 to 1, every other level one value.
    assert scale.level_label(bins[0], index=0, total=10).startswith("0 to 1 points; ")
    assert scale.level_label(bins[1], index=1, total=10).startswith("2 points; ")
    assert scale.level_label(bins[9], index=9, total=10).startswith("10 points; ")
    no_unit = OutputScale(output="x", minimum=0, maximum=10, integer=True)
    assert no_unit.level_label(bins[1], index=1, total=10).startswith("2; ")


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
    coarse = scale.coarse_grid(10)
    # Window of coarse levels 2 to 4 (13 to 18, 19 to 24, 25 to 30): [12.5, 30.5].
    lo, hi = coarse[2].lo, coarse[4].hi
    assert (lo, hi) == (12.5, 30.5)
    wide = scale.fine_grid(lo, hi, 10)
    # 18 integers (13..30) over 10 levels: eight levels of two values, then two of one.
    members = [_integer_members(b) for b in wide]
    assert members == [[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29], [30]]
    assert not any(b.exact for b in wide)
    assert wide[0].lo == 12.5 and wide[-1].hi == 30.5
    assert [b.center for b in wide] == [13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5, 29.0, 30.0]
    # Window of one coarse level, 13 to 18: six integers, one exact level per value.
    narrow = scale.fine_grid(coarse[2].lo, coarse[2].hi, 10)
    assert [b.center for b in narrow] == [float(v) for v in range(13, 19)]
    assert all(b.exact for b in narrow)
    # A window holding no integer falls back to equal-width levels.
    empty = scale.fine_grid(12.2, 12.8, 4)
    assert [(b.lo, b.hi) for b in empty] == pytest.approx([(12.2, 12.35), (12.35, 12.5), (12.5, 12.65), (12.65, 12.8)])


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
    # A flat distribution: the peak is the first level (ties go to the lowest
    # index), so the window whose middle is closest to it, the first, wins.
    assert best_window([0.25, 0.25, 0.25, 0.25], width=3) == (0, 3)
    # Fewer levels than the window width: the whole range.
    assert best_window([0.5, 0.5], width=3) == (0, 2)
    assert best_window([0.2, 0.3, 0.5], width=3) == (0, 3)


def test_best_window_ties_center_the_most_likely_level():
    # A single peak at level 2 is in three windows of equal mass (starts 0, 1, 2,
    # middles 1, 2, 3); the one centred on it wins, not the first.
    assert best_window([0.0, 0.0, 1.0, 0.0, 0.0], width=3) == (1, 4)
    assert best_window([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], width=3) == (3, 6)
    # At the edge only one window holds it.
    assert best_window([1.0, 0.0, 0.0, 0.0], width=3) == (0, 3)
    assert best_window([0.0, 0.0, 0.0, 1.0], width=3) == (1, 4)
    # Floating point noise is not a reason to prefer a window: 0.1 + 0.2 + 0.3
    # is 0.6000000000000001 and 0.2 + 0.3 + 0.1 is 0.6; these are a tie, and the
    # window centred on the peak (level 2) wins.
    assert 0.1 + 0.2 + 0.3 > 0.2 + 0.3 + 0.1
    assert best_window([0.1, 0.2, 0.3, 0.1], width=3) == (1, 4)
    # A real mass difference still decides before the tie-break.
    assert best_window([0.3, 0.3, 0.29, 0.0, 0.31], width=3) == (0, 3)


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
    # Levels 2 and 3 are 13 to 18 (center 15.5) and 19 to 24 (center 21.5).
    assert summary["mean"] == pytest.approx(0.3 * 15.5 + 0.7 * 21.5)
    assert summary["mean"] == pytest.approx(19.7)
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
    # Centers 3 (fraction 0.05 of the range), 9.5 (0.158), 15.5 (0.258), 21.5
    # (0.358), 27.5 (0.458), 45.5 (0.758) and 57.5 (0.958).
    assert scale.level_label(bins[0], index=0, total=total) == (
        "0 to 6 points; at the very low end of the scale; low end: no depressive symptoms"
    )
    assert scale.level_label(bins[1], index=1, total=total) == "7 to 12 points; in the low part of the scale"
    assert scale.level_label(bins[2], index=2, total=total) == "13 to 18 points; in the low part of the scale"
    assert scale.level_label(bins[3], index=3, total=total) == "19 to 24 points; just below the middle of the scale"
    assert scale.level_label(bins[4], index=4, total=total) == "25 to 30 points; in the middle of the scale"
    assert scale.level_label(bins[7], index=7, total=total) == "43 to 48 points; in the high part of the scale"
    assert scale.level_label(bins[9], index=9, total=total) == (
        "55 to 60 points; at the very high end of the scale; high end: most severe depression"
    )


def _meaning_scale(**extra):
    return OutputScale(
        output="madrs_total",
        minimum=0,
        maximum=60,
        integer=True,
        unit="points",
        low_meaning="no depressive symptoms",
        high_meaning="most severe depression",
        **extra,
    )


def _labels(scale, bins):
    return [scale.level_label(b, index=i, total=len(bins)) for i, b in enumerate(bins)]


def test_end_meanings_follow_the_scale_ends_not_the_window_ends():
    scale = _meaning_scale()
    coarse = scale.coarse_grid(10)
    # A refinement window in the middle (13 to 30) touches neither end of the
    # scale, so neither its first nor its last level carries an end meaning.
    middle = _labels(scale, scale.fine_grid(coarse[2].lo, coarse[4].hi, 10))
    assert middle[0] == "13 to 14 points; in the low part of the scale"
    assert middle[-1] == "30 points; in the middle of the scale"
    assert not any("low end:" in label or "high end:" in label for label in middle)

    # A window at the top (43 to 60): only the level holding 60 is the high end.
    top = _labels(scale, scale.fine_grid(coarse[7].lo, coarse[9].hi, 10))
    assert top[0] == "43 to 44 points; in the high part of the scale"
    assert top[-2] == "59 points; at the very high end of the scale"
    assert top[-1] == "60 points; at the very high end of the scale; high end: most severe depression"
    assert sum("high end:" in label for label in top) == 1
    assert not any("low end:" in label for label in top)

    # A window at the bottom (0 to 18): only the level holding 0 is the low end.
    bottom = _labels(scale, scale.fine_grid(coarse[0].lo, coarse[2].hi, 10))
    assert bottom[0] == "0 to 1 points; at the very low end of the scale; low end: no depressive symptoms"
    assert bottom[-1] == "18 points; just below the middle of the scale"
    assert sum("low end:" in label for label in bottom) == 1
    assert not any("high end:" in label for label in bottom)


def test_end_meanings_on_exact_and_continuous_levels():
    item = OutputScale(output="item", minimum=0, maximum=6, integer=True, low_meaning="absent", high_meaning="extreme")
    labels = _labels(item, item.coarse_grid(10))
    assert labels[0] == "0; at the very low end of the scale; low end: absent"
    assert labels[6] == "6; at the very high end of the scale; high end: extreme"
    assert not any("end:" in label for label in labels[1:6])
    # A refinement window of exact values inside the scale has no end meaning.
    inner = _labels(item, item.fine_grid(1.5, 4.5, 10))
    assert inner == [
        "2; just below the middle of the scale",
        "3; in the middle of the scale",
        "4; just above the middle of the scale",
    ]

    z = OutputScale(output="z", minimum=-3, maximum=3, low_meaning="far below", high_meaning="far above")
    low = _labels(z, z.fine_grid(-3.0, -1.2, 9))
    assert low[0] == "-3.00 to -2.80; at the very low end of the scale; low end: far below"
    assert not any("end:" in label for label in low[1:])
    high = _labels(z, z.fine_grid(1.2, 3.0, 9))
    assert high[-1] == "2.80 to 3.00; at the very high end of the scale; high end: far above"
    assert not any("end:" in label for label in high[:-1])
    inside = _labels(z, z.fine_grid(-1.2, 0.6, 9))
    assert not any("end:" in label for label in inside)


def test_range_text_formats_integer_bounds_without_decimals():
    assert _meaning_scale().range_text() == "0 to 60"
    assert OutputScale(output="item", minimum=0, maximum=6, integer=True).range_text() == "0 to 6"
    # A continuous range of 6 is printed at a tenth of its span: 0.6 wide, two decimals.
    assert OutputScale(output="z", minimum=-3, maximum=3).range_text() == "-3.00 to 3.00"
    # A range of 120 is printed 12 wide: no decimals.
    assert OutputScale(output="iq", minimum=40, maximum=160).range_text() == "40 to 160"


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
    # Below 1 wide: max(2, ceil(-log10(width)) + 1) decimals.
    assert continuous.format_value(1.23456, width=0.1) == "1.23"
    assert continuous.format_value(1.23456, width=0.09) == "1.235"
    assert continuous.format_value(1.23456, width=0.005) == "1.2346"
    assert continuous.format_value(1.23456, width=0.0005) == "1.23456"


def test_narrow_levels_never_print_the_same_bounds():
    scale = OutputScale(output="ratio", minimum=0.0, maximum=0.01)
    bins = scale.coarse_grid(10)
    cores = [scale.level_label(b, index=i, total=len(bins)).split(";")[0] for i, b in enumerate(bins)]
    # Levels 0.001 wide print with at least four decimals (ceil(3) + 1).
    assert cores[0] == "0.0000 to 0.0010"
    assert len(set(cores)) == len(cores)
    for core, b in zip(cores, bins):
        lo_text, hi_text = core.split(" to ")
        assert len(lo_text.split(".")[1]) >= 4
        assert float(lo_text) == pytest.approx(b.lo, abs=1e-9) and float(hi_text) == pytest.approx(b.hi, abs=1e-9)
    assert scale.range_text() == "0.0000 to 0.0100"


def test_reference_statistics_that_contradict_the_scale_are_ignored():
    outside = OutputScale(output="madrs_total", minimum=0, maximum=60, reference_mean=75.0, reference_sd=8.0)
    assert outside.reference_mean is None
    assert outside.reference_sd == 8.0
    assert outside.notes == ["reference_mean 75.0 lies outside the scale and was ignored"]
    # Without a usable mean the bands describe the range.
    assert outside.band(30.0) == "in the middle of the scale"
    tiny = OutputScale(output="madrs_total", minimum=0, maximum=60, reference_mean=20.0, reference_sd=2.9)
    # 2.9 is below a twentieth of the range (3).
    assert tiny.reference_sd is None
    assert tiny.reference_mean == 20.0
    assert tiny.notes == ["reference_sd 2.9 is below a twentieth of the range and was ignored"]
    edge = OutputScale(output="madrs_total", minimum=0, maximum=60, reference_mean=60.0, reference_sd=3.0)
    # The bounds themselves are allowed.
    assert (edge.reference_mean, edge.reference_sd, edge.notes) == (60.0, 3.0, [])
    # The note is not repeated when the scale is rebuilt from its own dict.
    clone = OutputScale(**{**outside.to_dict(), "reference_mean": 75.0})
    assert clone.notes == ["reference_mean 75.0 lies outside the scale and was ignored"]


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
