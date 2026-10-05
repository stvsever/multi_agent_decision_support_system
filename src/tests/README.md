# tests

Unit tests of the engine, the dashboard API and the evaluation library. No test calls a model or the
network: `conftest.py` sets a placeholder OpenRouter key when none is configured, and tests that need
locally recorded runs are skipped when those runs are absent.

| Files | Cover |
|---|---|
| `test_predictor_*`, `test_prediction_*`, `test_critic*`, `test_plan_*`, `test_orchestrator_*`, `test_integrator_*`, `test_executor_*` | Agents and their prompts, parsing and guardrails |
| `test_decision_*`, `test_input_routing.py` | Structured decision-model Predictor and evidence routing |
| `test_cli_*`, `test_context_windows.py`, `test_main_*` | Command line, configuration checks, context windows |
| `test_dashboard_*`, `test_cost_engine_parity.py`, `test_ui_*` | Dashboard API, run manager, cost estimate against the engine |
| `test_annotated_validation_*`, `test_validation_guide_notebook.py`, `test_batch_run_cli.py` | Evaluation library and batch runner |
| other `test_*.py` | Data loading, fusion, RAG scoring, clients, paths and reports |

```bash
python -m pytest src/tests -q
```
