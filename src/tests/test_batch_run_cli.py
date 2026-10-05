"""
batch_run.py forwards the current main.py command line: Predictor and companion
models, routing, decision-model settings, context windows and the offline
configuration check. Every forwarded command is parsed and validated by
main.py's own parser, so a renamed or removed main.py flag fails here.
"""

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import main as main_mod  # noqa: E402

BATCH_RUN = REPO / "src" / "full_stack" / "backend" / "utils" / "batch_run.py"
PSEUDO_INPUTS = REPO / "src" / "full_stack" / "backend" / "data" / "pseudo_data" / "inputs"
JEV = "typesafe/jev-1.13"
LLM = "deepseek/deepseek-v4-flash-0731"


def _load_batch_run():
    spec = importlib.util.spec_from_file_location("compass_batch_run_cli_under_test", BATCH_RUN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def batch_run():
    return _load_batch_run()


def _command(batch_run, argv, path="/data/participant_ID01", target="CASE"):
    args, extra = batch_run.build_parser().parse_known_args(argv)
    return batch_run.build_main_command(path, target, batch_run.batch_args_from_args(args, extra))


def _parse_with_main(cmd):
    """Parse a forwarded command exactly as main.py would (exit 2 on refusal)."""
    assert Path(cmd[1]).name == "main.py"
    parser = main_mod.build_parser()
    ns = parser.parse_args(cmd[2:])
    main_mod.validate_cli_args(parser, ns)
    return ns


def _dests(parser):
    return {action.dest for action in parser._actions}


# --- coverage of the main.py command line -------------------------------------------


def test_requested_main_flags_are_batch_options(batch_run):
    batch = _dests(batch_run.build_parser())
    main_flags = _dests(main_mod.build_parser())
    decision = {name for name in main_flags if name.startswith("decision_")}
    assert decision, "main.py has no decision_* flags any more"
    requested = {
        "predictor_model",
        "companion_model",
        "orchestration",
        "orchestration_threshold",
        "context_window",
        "role_context_window",
        "check_config",
        "public_model",
        *decision,
    }
    assert requested <= main_flags
    assert not requested - batch, f"batch_run.py does not offer: {sorted(requested - batch)}"


def test_forwarded_option_names_are_main_flags(batch_run):
    main_flags = _dests(main_mod.build_parser())
    forwarded = {
        *batch_run.TASK_VALUE_FLAGS,
        *batch_run.HOSTED_VALUE_FLAGS,
        *batch_run.LOCAL_VALUE_FLAGS,
        *batch_run.LOCAL_SWITCH_FLAGS,
        *batch_run.RUN_VALUE_FLAGS,
        *batch_run.RUN_BOOLEAN_FLAGS,
        *batch_run.RUN_REPEATED_FLAGS,
    }
    assert not forwarded - main_flags, f"not main.py flags: {sorted(forwarded - main_flags)}"


def test_help_has_no_em_dash(batch_run):
    assert "\u2014" not in BATCH_RUN.read_text(encoding="utf-8")
    assert "\u2014" not in batch_run.build_parser().format_help()


# --- forwarded commands ---------------------------------------------------------------


def test_hosted_decision_predictor_with_every_setting(batch_run):
    cmd = _command(
        batch_run,
        [
            "--backend", "openrouter",
            "--public_model", LLM,
            "--predictor_model", JEV,
            "--companion_model", LLM,
            "--orchestration", "auto",
            "--orchestration_threshold", "20000",
            "--decision_provider", "typesafe",
            "--decision_choice_orders", "2",
            "--decision_score_levels", "7",
            "--no-decision_refine",
            "--decision_compiler_model", LLM,
            "--decision_stability_threshold", "0.3",
            "--decision_regression_stability_threshold", "0.8",
            "--decision_sufficiency_threshold", "0.2",
            "--context_window", "200000",
            "--role_context_window", "predictor=128000,tool=32000",
            "--role_context_window", "critic=64000",
            "--reasoning_effort", "off",
            "--max_agent_input", "60000",
        ],
    )
    ns = _parse_with_main(cmd)
    assert ns.backend == "openrouter"
    assert ns.public_model == LLM
    assert ns.predictor_model == JEV
    assert ns.companion_model == LLM
    assert ns.orchestration == "auto"
    assert ns.orchestration_threshold == 20000
    assert ns.decision_provider == "typesafe"
    assert ns.decision_choice_orders == 2
    assert ns.decision_score_levels == 7
    assert ns.decision_refine is False
    assert ns.decision_compiler_model == LLM
    assert ns.decision_stability_threshold == pytest.approx(0.3)
    assert ns.decision_regression_stability_threshold == pytest.approx(0.8)
    assert ns.decision_sufficiency_threshold == pytest.approx(0.2)
    assert ns.context_window == 200000
    assert ns.role_context_window == ["predictor=128000,tool=32000", "critic=64000"]
    assert ns.reasoning_effort == "off"
    assert ns.max_agent_input == 60000
    assert ns.detailed_log and ns.quiet and not ns.check_config
    # Local backend options are not forwarded to a hosted run.
    assert "--local_engine" not in cmd and "--max_tokens" not in cmd and "--model" not in cmd


def test_decision_refine_on_is_forwarded(batch_run):
    cmd = _command(batch_run, ["--backend", "openrouter", "--predictor_model", JEV, "--decision_refine"])
    assert "--decision_refine" in cmd and "--no-decision_refine" not in cmd
    assert _parse_with_main(cmd).decision_refine is True


def test_local_backend_with_decision_predictor(batch_run):
    cmd = _command(
        batch_run,
        [
            "--backend", "local",
            "--model", "Qwen/Qwen3-14B-AWQ",
            "--local_max_model_len", "32768",
            "--local_enforce_eager",
            "--predictor_model", JEV,
            "--orchestration", "never",
            "--context_window", "32768",
        ],
    )
    ns = _parse_with_main(cmd)
    assert ns.backend == "local"
    assert ns.model == "Qwen/Qwen3-14B-AWQ"
    assert ns.local_max_model_len == 32768
    assert ns.local_enforce_eager is True
    assert ns.predictor_model == JEV
    assert ns.orchestration == "never"
    assert ns.context_window == 32768
    assert "--public_model" not in cmd


def test_unset_options_keep_main_defaults(batch_run):
    cmd = _command(batch_run, ["--backend", "openrouter"])
    ns = _parse_with_main(cmd)
    defaults = main_mod.build_parser().parse_args([])
    for name in (
        "public_model",
        "predictor_model",
        "companion_model",
        "orchestration",
        "orchestration_threshold",
        "context_window",
        "role_context_window",
        "decision_provider",
        "decision_choice_orders",
        "decision_score_levels",
        "decision_refine",
        "decision_compiler_model",
        "decision_stability_threshold",
        "decision_regression_stability_threshold",
        "decision_sufficiency_threshold",
        "public_max_context_tokens",
        "reasoning_effort",
        "max_agent_input",
    ):
        assert getattr(ns, name) == getattr(defaults, name), name
    assert not any(arg.startswith("--decision") or arg.startswith("--orchestration") for arg in cmd)


def test_model_on_hosted_backend_is_used_as_public_model(batch_run):
    cmd = _command(batch_run, ["--backend", "openrouter", "--model", "qwen/qwen3-32b"])
    ns = _parse_with_main(cmd)
    assert ns.public_model == "qwen/qwen3-32b"
    explicit = _parse_with_main(
        _command(batch_run, ["--backend", "openrouter", "--model", "x/ignored", "--public_model", LLM])
    )
    assert explicit.public_model == LLM


def test_unknown_flags_are_passed_through_to_main(batch_run):
    cmd = _command(
        batch_run,
        ["--backend", "openrouter", "--iterations", "2", "--global_instruction", "Be brief.", "--predictor_model", JEV],
    )
    assert cmd[-4:] == ["--iterations", "2", "--global_instruction", "Be brief."]
    ns = _parse_with_main(cmd)
    assert ns.iterations == 2 and ns.global_instruction == "Be brief." and ns.predictor_model == JEV


def test_check_config_command_is_offline_json(batch_run):
    args, extra = batch_run.build_parser().parse_known_args(["--backend", "openrouter", "--predictor_model", JEV])
    cmd = batch_run.build_main_command("/data/p01", "CASE", batch_run.batch_args_from_args(args, extra), check_config=True)
    ns = _parse_with_main(cmd)
    assert ns.check_config and ns.check_format == "json"
    assert not ns.detailed_log


def test_task_flags_are_forwarded(batch_run):
    cmd = _command(
        batch_run,
        ["--backend", "openrouter", "--prediction_type", "multiclass", "--class_labels", "A,B,C", "--control_label", "B"],
    )
    ns = _parse_with_main(cmd)
    assert ns.prediction_type == "multiclass" and ns.class_labels == "A,B,C" and ns.control_label == "B"
    reg = _parse_with_main(
        _command(batch_run, ["--prediction_type", "regression_univariate", "--regression_output", "total_iq"])
    )
    assert reg.regression_output == "total_iq"


def test_conflicting_regression_outputs_are_refused(batch_run):
    with pytest.raises(SystemExit):
        batch_run.main(["--regression_output", "a", "--regression_outputs", "b,c"])


# --- running the cohort with a stand-in main.py ----------------------------------------


STUB_MAIN = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path

    log = Path(os.environ["BATCH_STUB_LOG"])
    with log.open("a") as fh:
        fh.write(json.dumps(sys.argv[1:]) + "\\n")
    mode = os.environ.get("BATCH_STUB_MODE", "ok")
    if mode == "usage":
        print("usage: main.py [-h] ...", file=sys.stderr)
        print("main.py: error: --decision_score_levels must be between 2 and 10; got 25.", file=sys.stderr)
        sys.exit(2)
    if "--check_config" in sys.argv:
        pid = Path(sys.argv[1]).name
        problems = ["TYPESAFE_API_KEY is not set (environment or .env)."] if mode == "problem" else []
        report = {
            "backend": "openrouter",
            "endpoint": "https://example.invalid/v1",
            "predictor_kind": "decision",
            "roles": {"predictor": {"model": "typesafe/jev-1.13", "context_window": 26666},
                      "orchestrator": {"model": "deepseek/deepseek-v4-flash-0731", "context_window": 1310720}},
            "orchestration": {"mode": "auto", "threshold_tokens": 0},
            "credentials": {"TYPESAFE_API_KEY": mode != "problem"},
            "problems": problems,
            "notes": [],
            "record": {"participant_id": pid, "input_tokens": 1384, "budget_tokens": 26320,
                       "unit": "cl100k tokens of the decision state", "route": "direct"},
        }
        print(json.dumps(report))
        sys.exit(1 if problems else 0)
    results = os.environ.get("BATCH_STUB_RESULTS")
    if results:
        # Write a report the way main.py does: participant_<id>/report_<id>.json, labels [target, control].
        pid = Path(sys.argv[1]).name
        target = sys.argv[sys.argv.index("--target_label") + 1]
        label = target if pid.endswith("01") else "CONTROL"
        folder = Path(results) / f"participant_{pid}"
        folder.mkdir(parents=True, exist_ok=True)
        report = {"participant_id": pid, "prediction": {
            "prediction_task_spec": {"root": {"node_id": "root", "mode": "binary_classification",
                                               "class_labels": [target, "CONTROL"]}},
            "root_prediction": {"node_id": "root", "classification": {"predicted_label": label}}}}
        (folder / f"report_{pid}.json").write_text(json.dumps(report))
        print(f"[Report] Saved to: {folder / f'report_{pid}.json'}")
    sys.exit(0)
    """
)


@pytest.fixture()
def stub_cohort(batch_run, tmp_path, monkeypatch):
    stub = tmp_path / "main.py"
    stub.write_text(STUB_MAIN)
    data_root = tmp_path / "data"
    for pid in ("01", "02", "03"):
        (data_root / f"participant_ID{pid}").mkdir(parents=True)
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("BATCH_STUB_LOG", str(log))
    monkeypatch.setattr(batch_run, "MAIN_SCRIPT", stub)
    monkeypatch.setattr(batch_run, "DATA_ROOT", data_root)
    monkeypatch.setattr(batch_run, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        batch_run,
        "PARTICIPANTS",
        [{"id": pid, "expected": "CASE", "target_str": "CASE"} for pid in ("01", "02", "03")],
    )

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    return calls


def test_check_config_checks_every_participant_and_runs_nothing(batch_run, stub_cohort, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        batch_run.main(["--backend", "openrouter", "--predictor_model", JEV, "--check_config"])
    assert info.value.code == 0
    calls = stub_cohort()
    assert len(calls) == 3
    assert all("--check_config" in argv and "--detailed_log" not in argv for argv in calls)
    assert all(argv[argv.index("--predictor_model") + 1] == JEV for argv in calls)
    out = capsys.readouterr().out
    assert "3 direct" in out and "0 with problems" in out
    saved = json.loads((tmp_path / "results" / "participant_02" / "config_check_02.json").read_text())
    assert saved["record"]["route"] == "direct"


def test_check_config_reports_problems_with_exit_code_one(batch_run, stub_cohort, monkeypatch, capsys):
    monkeypatch.setenv("BATCH_STUB_MODE", "problem")
    with pytest.raises(SystemExit) as info:
        batch_run.main(["--backend", "openrouter", "--predictor_model", JEV, "--check_config"])
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert "TYPESAFE_API_KEY is not set" in out and "3 with problems" in out


def test_preflight_check_blocks_the_cohort_on_a_problem(batch_run, stub_cohort, monkeypatch):
    monkeypatch.setenv("BATCH_STUB_MODE", "problem")
    with pytest.raises(SystemExit) as info:
        batch_run.main(["--backend", "openrouter", "--predictor_model", JEV, "--preflight_check"])
    assert info.value.code == 1
    assert all("--check_config" in argv for argv in stub_cohort())


def test_preflight_check_then_runs_the_cohort(batch_run, stub_cohort):
    batch_run.main(["--backend", "openrouter", "--predictor_model", JEV, "--preflight_check"])
    calls = stub_cohort()
    assert sum("--check_config" in argv for argv in calls) == 3
    runs = [argv for argv in calls if "--check_config" not in argv]
    assert len(runs) == 3 and all("--detailed_log" in argv for argv in runs)


def test_refused_command_line_stops_the_batch(batch_run, stub_cohort, monkeypatch, capsys):
    monkeypatch.setenv("BATCH_STUB_MODE", "usage")
    batch_run.main(["--backend", "openrouter"])
    assert len(stub_cohort()) == 1
    out = capsys.readouterr().out
    assert "remaining participants are skipped" in out
    assert "must be between 2 and 10" in out and "usage:" not in out
    assert out.count("[SKIPPED]") == 2


def test_check_config_stops_after_a_refused_command_line(batch_run, stub_cohort, monkeypatch, capsys):
    monkeypatch.setenv("BATCH_STUB_MODE", "usage")
    with pytest.raises(SystemExit) as info:
        batch_run.main(["--backend", "openrouter", "--check_config"])
    assert info.value.code == 1
    assert len(stub_cohort()) == 1
    out = capsys.readouterr().out
    assert "must be between 2 and 10" in out and "usage:" not in out


def test_missing_participant_folder_is_skipped(batch_run, stub_cohort, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        batch_run,
        "PARTICIPANTS",
        [{"id": "01", "expected": "CASE", "target_str": "CASE"}, {"id": "99", "expected": "CONTROL", "target_str": "CASE"}],
    )
    batch_run.main(["--backend", "openrouter"])
    assert len(stub_cohort()) == 1
    assert "[MISSING]" in capsys.readouterr().out


def test_classification_summary_reads_the_report_json(batch_run, stub_cohort, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BATCH_STUB_RESULTS", str(tmp_path / "results"))
    monkeypatch.setattr(
        batch_run,
        "PARTICIPANTS",
        [
            {"id": "01", "expected": "CASE", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
            {"id": "02", "expected": "CONTROL", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
            {"id": "03", "expected": "CASE", "target_str": "MAJOR_DEPRESSIVE_DISORDER"},
        ],
    )
    batch_run.main(["--backend", "openrouter"])
    out = capsys.readouterr().out
    assert "Accuracy: 2/3" in out
    assert "TP: 1  FN: 1" in out and "FP: 0  TN: 1" in out


_REAL_REPORT = REPO / "src/full_stack/backend/data/pseudo_data/outputs/participant_SUBJ_001_PSEUDO/report_SUBJ_001_PSEUDO.json"


@pytest.mark.skipif(not _REAL_REPORT.is_file(), reason="needs a local engine run of SUBJ_001_PSEUDO (outputs are not tracked)")
def test_binary_outcome_from_a_real_report():
    report = json.loads(_REAL_REPORT.read_text())
    batch = _load_batch_run()
    assert batch.binary_outcome_from_report(report) == "CASE"
    root = report["prediction"]["root_prediction"]["classification"]
    root["predicted_label"] = report["prediction"]["prediction_task_spec"]["root"]["class_labels"][1]
    assert batch.binary_outcome_from_report(report) == "CONTROL"
    root["predicted_label"] = "something else"
    report["prediction"]["classification"] = None
    assert batch.binary_outcome_from_report(report) == "UNKNOWN"


def test_stale_reports_are_ignored(batch_run, tmp_path, monkeypatch):
    monkeypatch.setattr(batch_run, "RESULTS_DIR", tmp_path)
    folder = tmp_path / "participant_ID07"
    folder.mkdir()
    report = {"prediction": {"root_prediction": {"classification": {"predicted_label": "CASE"}}}}
    (folder / "report_ID07.json").write_text(json.dumps(report))
    assert batch_run.read_binary_outcome("07", "/data/ID07") == "CASE"
    import time

    assert batch_run.read_binary_outcome("07", "/data/ID07", since=time.time() + 60) == "UNKNOWN"


def test_report_path_is_read_from_the_run_log(batch_run, tmp_path, monkeypatch):
    # The record's own id names the output folder, so only the log points at it.
    monkeypatch.setattr(batch_run, "RESULTS_DIR", tmp_path / "results")
    folder = tmp_path / "results" / "participant_SUB-XYZ"
    folder.mkdir(parents=True)
    report = {"prediction": {"prediction_task_spec": {"root": {"class_labels": ["MDD", "HC"]}},
                             "root_prediction": {"classification": {"predicted_label": "HC"}}}}
    (folder / "report_SUB-XYZ.json").write_text(json.dumps(report))
    log = tmp_path / "batch_out_07.txt"
    log.write_text(f"[1/5] Loading ...\n[Report] Saved to: {folder / 'report_SUB-XYZ.json'}\n")
    assert batch_run.read_binary_outcome("07", "/data/participant_ID07") == "UNKNOWN"
    assert batch_run.read_binary_outcome("07", "/data/participant_ID07", log_path=log) == "CONTROL"


# --- the real main.py, offline ----------------------------------------------------------


def test_real_main_check_config_on_pseudo_participant(batch_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("COMPASS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(batch_run, "DATA_ROOT", PSEUDO_INPUTS)
    monkeypatch.setattr(batch_run, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(
        batch_run, "PARTICIPANTS", [{"id": "SUBJ_002_PSEUDO", "expected": "CONTROL", "target_str": "CASE"}]
    )
    with pytest.raises(SystemExit) as info:
        batch_run.main(
            [
                "--backend", "local",
                "--model", "Qwen/Qwen3-14B-AWQ",
                "--local_max_model_len", "32768",
                "--control_label", "CONTROL",
                "--orchestration", "auto",
                "--role_context_window", "tool=16000",
                "--check_config",
            ]
        )
    out = capsys.readouterr().out
    assert info.value.code == 0, out
    report = json.loads((tmp_path / "results" / "participant_SUBJ_002_PSEUDO" / "config_check_SUBJ_002_PSEUDO.json").read_text())
    assert report["backend"] == "local"
    assert report["record"]["route"] in {"direct", "orchestrate", "orchestrated"}
    assert report["roles"]["tool"]["context_window"] == 16000
    assert "SUBJ_002_PSEUDO" in out
