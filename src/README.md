# src

All engine code. `main.py` and `COMPASS_demo.ipynb` at the repository root are the entry points; run them
from the root so that `src.full_stack...` imports resolve.

| Folder | Content |
|---|---|
| [`full_stack/`](full_stack/README.md) | The engine (agents, tools, configuration, data models, utilities, HPC scripts) and the dashboard (FastAPI backend, React frontend) |
| [`tests/`](tests/README.md) | Unit tests of the engine, the dashboard API and the evaluation library; no test calls a model or the network |

```bash
python3 main.py --help
python -m pytest src/tests -q
```
