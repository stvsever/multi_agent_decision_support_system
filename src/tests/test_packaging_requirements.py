"""
The dependency lists cover what main.py imports at startup, and the Docker GPU
image installs the same engine extras as its Apptainer twin.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO / "requirements.txt"
UI_REQUIREMENTS = REPO / "docker" / "requirements.ui.txt"
DOCKERFILE_GPU = REPO / "docker" / "Dockerfile.gpu"
APPTAINER_DEF = REPO / "src" / "full_stack" / "backend" / "hpc" / "apptainer.def"
DECISION_CLIENT = REPO / "src" / "full_stack" / "backend" / "agents" / "decision" / "client.py"

EXTRAS_MARKER = "What the engine itself imports beyond the dashboard list"


def _requirement_names(path: Path) -> set:
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _extras_block(path: Path) -> set:
    """Quoted package specifiers of the engine-extras pip install that follows the marker comment."""
    text = path.read_text()
    start = text.index(EXTRAS_MARKER)
    block = text[start:]
    install = block.index("pip install")
    end = block.find("\n\n", install)
    specs = re.findall(r'"([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?[<>=!~][^"]*)"', block[install:end if end != -1 else None])
    return set(specs)


def test_requests_is_a_dependency_of_main():
    # The decision client imports requests, and main.py imports the client at startup.
    assert re.search(r"^\s*import requests|^\s*from requests", DECISION_CLIENT.read_text(), re.M)
    assert "requests" in _requirement_names(REQUIREMENTS)
    assert "requests" in _requirement_names(UI_REQUIREMENTS)


def test_ui_requirements_cover_figures():
    assert "matplotlib" in _requirement_names(UI_REQUIREMENTS)
    assert "matplotlib" in _requirement_names(REQUIREMENTS)


def test_docker_gpu_installs_the_apptainer_engine_extras():
    apptainer = _extras_block(APPTAINER_DEF)
    docker = _extras_block(DOCKERFILE_GPU)
    assert apptainer, "apptainer.def has no engine-extras block"
    assert {"requests>=2.31", "matplotlib>=3.7.0"} <= apptainer
    assert docker == apptainer


def test_dependency_files_have_no_em_dash():
    for path in (REQUIREMENTS, UI_REQUIREMENTS, DOCKERFILE_GPU):
        assert "\u2014" not in path.read_text(encoding="utf-8"), path
