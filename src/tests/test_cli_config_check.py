"""
The command line: every engine flag is documented, an inconsistent command line
is refused with a message naming the flag, and `--check_config` reports the
resolved configuration and the route offline. Network access is blocked in every
test, so a check that tried to call a provider would fail here.
"""

import io
import json
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import main as main_mod
from src.full_stack.backend.config import settings as settings_module
from src.full_stack.backend.config.settings import reload_settings

REPO = Path(__file__).resolve().parents[2]
SUBJ_001 = REPO / "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"
JEV = "typesafe/jev-1.13"
LLM = "deepseek/deepseek-v4-flash-0731"


@pytest.fixture(autouse=True)
def _offline(monkeypatch, tmp_path):
    def _blocked(*_args, **_kwargs):
        raise RuntimeError("network access is not allowed in these tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(settings_module, "_CATALOG_CONTEXT_CACHE", {})
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    reload_settings()
    yield
    reload_settings()


def _main(monkeypatch, argv):
    monkeypatch.setattr(main_mod, "run_compass_pipeline", lambda **_kwargs: pytest.fail("must not run"))
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    with pytest.raises(SystemExit) as info:
        main_mod.main()
    return info.value.code


# --- help ------------------------------------------------------------------------


def test_help_documents_every_engine_flag():
    text = main_mod.build_parser().format_help()
    for flag in (
        "--predictor_model",
        "--companion_model",
        "--orchestration",
        "--orchestration_threshold",
        "--decision_provider",
        "--decision_choice_orders",
        "--decision_score_levels",
        "--decision_refine",
        "--decision_compiler_model",
        "--decision_stability_threshold",
        "--decision_regression_stability_threshold",
        "--decision_sufficiency_threshold",
        "--public_max_context_tokens",
        "--context_window",
        "--role_context_window",
        "--max_tokens",
        "--local_max_model_len",
        "--check_config",
        "--check_format",
        "--refresh_catalog",
    ):
        assert flag in text, flag
    assert "--check_config" in main_mod.build_parser().epilog


# --- validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--decision_score_levels", "25"], "--decision_score_levels must be between 2 and 10"),
        (["--decision_choice_orders", "0"], "--decision_choice_orders must be between 1 and 6"),
        (["--decision_stability_threshold", "0.9"], "--decision_stability_threshold must be between"),
        (["--orchestration_threshold", "-1"], "--orchestration_threshold must be at least 0"),
        (["--context_window", "100"], "--context_window must be at least 1024"),
        (["--public_max_context_tokens", "500"], "must be 0 (not configured) or at least 1024"),
        (["--role_context_window", "planner=4096"], "Unknown role 'planner'"),
        (["--role_context_window", "tool"], "ROLE=TOKENS"),
        (["--companion_model", JEV], "--companion_model must be a conventional LLM"),
        (["--decision_compiler_model", JEV], "--decision_compiler_model must be a conventional LLM"),
        (["--backend", "local", "--model", JEV], "is a structured decision model"),
        (["--backend", "local", "--predictor_model", "openai/gpt-5"], "cannot run on --backend local"),
        (["--backend", "local", "--companion_model", LLM], "applies to hosted backends"),
        (["--public_model", JEV, "--predictor_model", "~typesafe/jev-latest"], "two different decision models"),
        (["--iterations", "0"], "--iterations must be between 1 and 20"),
        (["--max_agent_input", "-5"], "--max_agent_input must be at least 1"),
    ],
)
def test_an_inconsistent_command_line_is_refused(monkeypatch, capsys, argv, message):
    assert _main(monkeypatch, [str(SUBJ_001), *argv]) == 2
    assert message in capsys.readouterr().err


def test_the_check_needs_no_participant(monkeypatch, capsys):
    assert _main(monkeypatch, ["--check_config", "--check_format", "json"]) in (0, 1)
    report = json.loads(capsys.readouterr().out)
    assert report["record"] is None
    assert report["roles"]["predictor"]["model"] == LLM


# --- the offline check -------------------------------------------------------------


def test_check_config_reports_windows_budget_and_route(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-value")
    code = _main(
        monkeypatch,
        [str(SUBJ_001), "--check_config", "--check_format", "json", "--role_context_window", "predictor=32000"],
    )
    report = json.loads(capsys.readouterr().out)
    assert code == (1 if report["problems"] else 0)
    predictor = report["roles"]["predictor"]
    assert (predictor["context_window"], predictor["context_source"]) == (32_000, "role_override")
    assert report["roles"]["critic"]["context_source"] == "known_table"
    # The Predictor ceiling is clamped to half the forced window, so the reserve is the full 8,192.
    assert report["roles"]["predictor"]["max_output_tokens"] == 16_000
    assert report["predictor"]["budget_tokens"] == 32_000 - 8_192
    record = report["record"]
    assert record["predictor_kind"] == "llm"
    assert record["route"] == ("direct" if record["input_tokens"] <= record["budget_tokens"] else "orchestrated")
    assert record["budget_tokens"] == report["predictor"]["budget_tokens"]


def test_check_config_with_a_decision_predictor_and_a_small_threshold(monkeypatch, capsys):
    code = _main(
        monkeypatch,
        [
            str(SUBJ_001),
            "--predictor_model",
            JEV,
            "--companion_model",
            LLM,
            "--orchestration_threshold",
            "1000",
            "--check_config",
            "--check_format",
            "json",
        ],
    )
    report = json.loads(capsys.readouterr().out)
    assert report["predictor_kind"] == "decision"
    assert report["roles"]["predictor"]["context_source"] == "decision_state"
    record = report["record"]
    assert record["predictor_kind"] == "decision"
    assert record["threshold_source"] == "override"
    assert record["threshold_tokens"] == 1000
    assert record["route"] == "orchestrated"
    assert record["requests_round_one"] >= 1
    assert code in (0, 1)


def test_check_config_flags_a_missing_typesafe_key(monkeypatch, capsys):
    monkeypatch.setattr(settings_module, "_resolve_secret_from_env_or_dotenv", lambda name: "")
    code = _main(
        monkeypatch,
        [str(SUBJ_001), "--predictor_model", JEV, "--decision_provider", "typesafe", "--check_config"],
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "TYPESAFE_API_KEY MISSING" in out
    assert "TYPESAFE_API_KEY is not set" in out
    assert "no model was called" in out


def test_check_config_on_the_local_backend_uses_the_served_length(monkeypatch, capsys):
    code = _main(
        monkeypatch,
        [
            str(SUBJ_001),
            "--backend",
            "local",
            "--model",
            "Qwen/Qwen3-14B-AWQ",
            "--local_max_model_len",
            "32768",
            "--check_config",
            "--check_format",
            "json",
        ],
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert {row["context_source"] for row in report["roles"].values()} == {"local_served"}
    assert {row["context_window"] for row in report["roles"].values()} == {32_768}
    assert report["credentials"] == {}
    assert report["record"]["route"] == "direct"


def test_apply_cli_settings_matches_a_run(monkeypatch):
    args = main_mod.build_parser().parse_args(
        [str(SUBJ_001), "--context_window", "65536", "--role_context_window", "tool=32768", "--quiet"]
    )
    settings = main_mod.apply_cli_settings(reload_settings(), args, interactive=False)
    assert settings.models.role_context_windows == {"all": 65_536, "tool": 32_768}
    assert settings.effective_context_window(settings.models.tool_model, role="tool") == 32_768
    # Derived budgets follow the forced window.
    assert settings.token_budget.max_agent_input_tokens == int(65_536 * 0.95)
    assert settings.models.tool_max_tokens <= 32_768 // 2


def test_notes_name_flags_without_effect(monkeypatch, capsys):
    args = main_mod.build_parser().parse_args(
        [str(SUBJ_001), "--orchestration", "always", "--orchestration_threshold", "5000", "--decision_score_levels", "5"]
    )
    main_mod.apply_cli_settings(reload_settings(), args, interactive=False)
    err = capsys.readouterr().err
    assert "--orchestration_threshold has no effect" in err
    assert "--decision_score_levels only apply" in err


def test_json_check_output_stays_parseable_with_notes(monkeypatch, capsys):
    _main(
        monkeypatch,
        [str(SUBJ_001), "--orchestration", "never", "--orchestration_threshold", "5000", "--check_config", "--check_format", "json"],
    )
    report = json.loads(capsys.readouterr().out)
    assert any("--orchestration_threshold has no effect" in note for note in report["notes"])
    assert report["record"]["route"] == "direct"


def test_refresh_catalog_writes_the_cache_the_windows_come_from(monkeypatch, capsys, tmp_path):
    from src.full_stack.backend.api import catalog as catalog_module

    def _fetch(force=False):
        assert force is True
        path = settings_module.catalog_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"models": [{"id": LLM, "context_length": 1_310_720}], "fetched_at": 0}))
        return {"models": [{"id": LLM}], "stale": False}

    monkeypatch.setattr(catalog_module, "fetch_catalog", _fetch)
    _main(monkeypatch, [str(SUBJ_001), "--refresh_catalog", "--check_config", "--check_format", "json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert "Model catalog refreshed: 1 models" in captured.err
    assert report["roles"]["predictor"]["context_source"] == "catalog"
    assert report["roles"]["predictor"]["context_window"] == 1_310_720
    assert report["catalog_cache"]["present"] is True


def test_a_window_above_the_served_length_is_flagged(monkeypatch, capsys):
    args = main_mod.build_parser().parse_args(
        [str(SUBJ_001), "--backend", "local", "--local_max_model_len", "32768", "--role_context_window", "predictor=65536"]
    )
    settings = main_mod.apply_cli_settings(reload_settings(), args, interactive=False)
    assert settings.effective_context_window(settings.models.predictor_model, role="predictor") == 65_536
    assert "above the served length 32,768 (predictor=65,536)" in capsys.readouterr().err


def test_a_public_window_a_known_model_ignores_is_named(monkeypatch, capsys):
    # Older versions applied --public_max_context_tokens to the public model whatever
    # its window; now a known model keeps its own, and the note says so.
    args = main_mod.build_parser().parse_args([str(SUBJ_001), "--public_max_context_tokens", "64000"])
    settings = main_mod.apply_cli_settings(reload_settings(), args, interactive=False)
    assert settings.effective_context_window(settings.models.predictor_model, role="predictor") != 64_000
    assert "--public_max_context_tokens 64,000 has no effect" in capsys.readouterr().err

    args = main_mod.build_parser().parse_args(
        [str(SUBJ_001), "--public_model", "acme/unlisted-model", "--public_max_context_tokens", "64000"]
    )
    settings = main_mod.apply_cli_settings(reload_settings(), args, interactive=False)
    assert settings.effective_context_window(settings.models.predictor_model, role="predictor") == 64_000
    assert "--public_max_context_tokens" not in capsys.readouterr().err
