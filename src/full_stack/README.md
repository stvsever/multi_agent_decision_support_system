# full_stack

The COMPASS engine and its dashboard. `main.py` at the repository root is the entry point for both.

| Folder | Content |
|---|---|
| `backend/agents/` | Orchestrator, Executor, Integrator, Predictor, Critic and Communicator; `agents/decision/` holds the structured decision-model Predictor |
| `backend/tools/` | Tools the Executor calls (narratives, compression, ranking, hypotheses) |
| `backend/api/` | FastAPI service behind the dashboard: runs, settings, cost estimates, reports |
| `backend/config/` | Settings: models, token budgets, context windows, routing |
| `backend/data/` | Data models (task specification, prediction result) and bundled pseudo participants |
| `backend/runtime/` | Event bus that streams a run to the dashboard |
| `backend/utils/` | Data loading, record rendering and routing, LLM clients, logging, evaluation library |
| `backend/hpc/` | Apptainer, Slurm and vLLM scripts for running on a cluster (own README) |
| `backend/assets/` | Figures used by the documentation |
| `frontend/` | React dashboard (Vite, TypeScript), built into `frontend/dist/` and served by the API |

```bash
python3 main.py <participant_dir> --prediction_type binary --target_label CASE --control_label CONTROL
```

```bash
npm --prefix src/full_stack/frontend install && npm --prefix src/full_stack/frontend run build && python3 main.py --ui
```
