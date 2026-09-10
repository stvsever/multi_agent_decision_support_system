"""
Turning a local backend configuration into commands somebody can actually run.

Open weights are only useful if the person choosing them can see what the choice
costs and how to start it, so this module answers three questions from one
configuration: which engine will serve the model, roughly how much accelerator
memory it needs, and the exact command for the runtime and scheduler in use.

Every size here is an estimate. Repositories publish a weight count, not a
memory footprint, and the real number moves with batch size, cache dtype, and
how much of the context is actually filled.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from .schemas import DashboardConfig, LocalBackendConfig

#: Assets the container and cluster workstream owns; referenced, never written.
DOCKERFILE = "docker/Dockerfile.gpu"
COMPOSE_FILE = "docker/docker-compose.gpu.yml"
APPTAINER_DEF = "src/full_stack/backend/hpc/apptainer.def"
VLLM_SERVE_SCRIPT = "src/full_stack/backend/hpc/06_serve_vllm.sh"

DEFAULT_DOCKER_IMAGE = "compass-engine:gpu"
DEFAULT_APPTAINER_IMAGE = "src/full_stack/backend/hpc/compass-engine.sif"

#: A real participant folder, so every command below runs as printed.
SAMPLE_PARTICIPANT = "src/full_stack/backend/data/pseudo_data/inputs/SUBJ_001_PSEUDO"

HF_CACHE_HOST = "$HOME/.cache/huggingface"
HF_CACHE_CONTAINER = "/root/.cache/huggingface"

#: Bytes per stored weight, by declared quantization. A four bit checkpoint
#: still carries full precision scales and zero points, hence the margin above
#: a flat 0.5.
_QUANTIZATION_BYTES: Dict[str, float] = {
    "": 0.0,
    "none": 0.0,
    "awq": 0.6,
    "awq_marlin": 0.6,
    "gptq": 0.6,
    "gptq_marlin": 0.6,
    "4bit": 0.6,
    "int4": 0.6,
    "nf4": 0.6,
    "bitsandbytes": 0.6,
    "8bit": 1.1,
    "int8": 1.1,
    "fp8": 1.1,
    "w8a8": 1.1,
}

_DTYPE_BYTES: Dict[str, float] = {
    "auto": 2.0,
    "bfloat16": 2.0,
    "float16": 2.0,
    "half": 2.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "float32": 4.0,
    "fp32": 4.0,
    "float8": 1.0,
    "fp8": 1.0,
}

#: Parameter count written into a repository name, such as `Qwen3-14B-AWQ`.
_SIZE_IN_NAME = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?:[-_.]|$)")

#: Quantization markers as they appear inside a repository name, longest first
#: so `awq_marlin` is not shortened to `awq` with a different size.
_NAME_MARKERS: Tuple[str, ...] = tuple(
    sorted((k for k in _QUANTIZATION_BYTES if k not in ("", "none")), key=len, reverse=True)
)

GIB = 1024 ** 3


def _local_of(config: Any) -> LocalBackendConfig:
    """Accept either the whole dashboard configuration or just its local section."""
    if isinstance(config, DashboardConfig):
        return config.local
    if isinstance(config, LocalBackendConfig):
        return config
    if isinstance(config, dict):
        return LocalBackendConfig.model_validate(config.get("local", config))
    local = getattr(config, "local", None)
    if isinstance(local, LocalBackendConfig):
        return local
    raise TypeError("plan_deployment needs a dashboard configuration or a local backend configuration.")


def parameters_from_name(model_name: str) -> Optional[int]:
    """The size a repository advertises in its own name, when nothing else says."""
    match = _SIZE_IN_NAME.search(str(model_name or ""))
    if not match:
        return None
    try:
        return int(float(match.group(1)) * 1_000_000_000)
    except ValueError:
        return None


def _quantization_in_name(model_name: str) -> str:
    """
    The quantization a repository advertises in its own name.

    The shipped default is `Qwen/Qwen3-14B-AWQ`, and sizing that at the dtype
    table's two bytes a weight overstates it roughly threefold, which is the
    difference between "this fits on one card" and "this does not".
    """
    name = str(model_name or "").strip().lower()
    if not name:
        return ""
    for marker in _NAME_MARKERS:
        if re.search(rf"(?:^|[-_./]){re.escape(marker)}(?:[-_./]|$)", name):
            return marker
    return ""


def _bytes_per_weight(
    local: LocalBackendConfig, model_detail: Optional[Dict[str, Any]] = None
) -> Tuple[float, str]:
    """
    Bytes one stored weight occupies, and where that number came from.

    The configured quantization wins because it is a deliberate choice; the
    repository's own `config.json` comes next because it describes the actual
    checkpoint; the name is a last resort before the dtype table, which
    describes an unquantized checkpoint and is wrong for every other kind.
    """
    configured = str(local.quantization or "").strip().lower()
    quantized = _QUANTIZATION_BYTES.get(configured)
    if quantized:
        return quantized, f"{configured} weights, as configured"

    declared = str((model_detail or {}).get("quantization_method") or "").strip().lower()
    quantized = _QUANTIZATION_BYTES.get(declared)
    if quantized:
        return quantized, f"{declared} weights, declared by the repository"

    from_name = _quantization_in_name(local.model_name)
    quantized = _QUANTIZATION_BYTES.get(from_name)
    if quantized:
        return quantized, f"{from_name} weights, read from the repository name"

    dtype = str(local.dtype or "auto").strip().lower()
    return _DTYPE_BYTES.get(dtype, 2.0), f"{dtype} weights, from the dtype setting"


def estimate_vram(
    local: LocalBackendConfig,
    parameter_count: Optional[int],
    model_detail: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Roughly what this model needs to load and serve, and how it splits per GPU.

    Weights dominate, the working cache scales with them, and a runtime reserve
    covers the CUDA context and activations. The result is deliberately labelled
    an estimate: nothing here reads the machine.
    """
    per_weight, basis = _bytes_per_weight(local, model_detail)
    if not parameter_count:
        return {
            "known": False,
            "parameter_count": None,
            "bytes_per_weight": per_weight,
            "basis": f"{basis}; the repository does not publish a weight count",
            "weights_gb": None,
            "kv_cache_gb": None,
            "overhead_gb": None,
            "total_gb": None,
            "per_gpu_gb": None,
            "is_estimate": True,
        }

    weights_gb = (parameter_count * per_weight) / GIB
    # A serving cache large enough to be useful, never smaller than a couple of
    # gigabytes, because a token cache below that serves one short request.
    kv_cache_gb = max(2.0, 0.15 * weights_gb)
    overhead_gb = 2.0
    total_gb = weights_gb + kv_cache_gb + overhead_gb
    # vLLM claims a fraction of each card, so the card has to be larger than the
    # share it is asked to hold.
    per_gpu_gb = total_gb / max(1, local.gpu_count) / max(0.05, local.gpu_memory_utilization)
    return {
        "known": True,
        "parameter_count": parameter_count,
        "bytes_per_weight": per_weight,
        "basis": f"{parameter_count / 1e9:.1f}B parameters at {per_weight} bytes each ({basis})",
        "weights_gb": round(weights_gb, 1),
        "kv_cache_gb": round(kv_cache_gb, 1),
        "overhead_gb": round(overhead_gb, 1),
        "total_gb": round(total_gb, 1),
        "per_gpu_gb": round(per_gpu_gb, 1),
        "is_estimate": True,
    }


def _resolve_engine(local: LocalBackendConfig, probe: Optional[Dict[str, Any]]) -> Tuple[str, str]:
    if local.engine == "vllm":
        return "vllm", "The configuration asks for vLLM."
    if local.engine == "transformers":
        return "transformers", "The configuration asks for transformers."
    if local.tensor_parallel_size > 1 or local.pipeline_parallel_size > 1:
        return "vllm", "Only vLLM splits one model across several GPUs."
    if probe and probe.get("vllm_installed") is False and probe.get("transformers_installed"):
        return "transformers", "vLLM is not installed on this machine."
    return "vllm", "vLLM is the default when it is available, and it falls back to transformers."


def _compass_command(local: LocalBackendConfig, participant: str = SAMPLE_PARTICIPANT) -> str:
    """The engine loads the weights in process, so this is the run command."""
    parts = [
        "python main.py",
        shlex.quote(participant),
        "--backend local",
        f"--model {shlex.quote(local.model_name)}",
        f"--max_tokens {local.max_tokens}",
        f"--local_engine {local.engine}",
        f"--local_dtype {shlex.quote(local.dtype or 'auto')}",
    ]
    if local.quantization:
        parts.append(f"--local_quant {shlex.quote(local.quantization)}")
    if local.kv_cache_dtype and local.kv_cache_dtype != "auto":
        parts.append(f"--local_kv_cache_dtype {shlex.quote(local.kv_cache_dtype)}")
    parts.append(f"--local_tensor_parallel {local.tensor_parallel_size}")
    parts.append(f"--local_pipeline_parallel {local.pipeline_parallel_size}")
    parts.append(f"--local_gpu_mem_util {local.gpu_memory_utilization}")
    if local.max_model_len:
        parts.append(f"--local_max_model_len {local.max_model_len}")
    if local.enforce_eager:
        parts.append("--local_enforce_eager")
    if local.trust_remote_code:
        parts.append("--local_trust_remote_code")
    if local.attn_implementation and local.attn_implementation != "auto":
        parts.append(f"--local_attn {shlex.quote(local.attn_implementation)}")
    return " ".join(parts)


def _vllm_server_command(local: LocalBackendConfig) -> str:
    """A shared OpenAI-compatible server, for one node serving several runs."""
    parts = [
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {shlex.quote(local.model_name)}",
        f"--served-model-name {shlex.quote(local.model_name)}",
        f"--tensor-parallel-size {local.tensor_parallel_size}",
        f"--pipeline-parallel-size {local.pipeline_parallel_size}",
        f"--gpu-memory-utilization {local.gpu_memory_utilization}",
        f"--max-model-len {local.max_model_len or local.max_tokens}",
        f"--dtype {shlex.quote(local.dtype or 'auto')}",
    ]
    if local.quantization:
        parts.append(f"--quantization {shlex.quote(local.quantization)}")
    if local.kv_cache_dtype and local.kv_cache_dtype != "auto":
        parts.append(f"--kv-cache-dtype {shlex.quote(local.kv_cache_dtype)}")
    if local.enforce_eager:
        parts.append("--enforce-eager")
    if local.trust_remote_code:
        parts.append("--trust-remote-code")
    parts.append("--host 0.0.0.0 --port 8000")
    return " ".join(parts)


def _docker_prefix(local: LocalBackendConfig, image: str) -> str:
    devices = ",".join(str(i) for i in range(local.gpu_count))
    return (
        f"docker run --rm -it --gpus '\"device={devices}\"' --shm-size=16g -p 8000:8000 "
        f'-v "$PWD":/workspace -v "{HF_CACHE_HOST}":{HF_CACHE_CONTAINER} '
        f"-e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN -w /workspace {shlex.quote(image)}"
    )


def _apptainer_prefix(image: str) -> str:
    return (
        "apptainer exec --nv "
        f'--bind "$PWD":/workspace --bind "{HF_CACHE_HOST}":{HF_CACHE_CONTAINER} '
        f'--env HF_TOKEN="$HF_TOKEN" --env HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" '
        f"--pwd /workspace {shlex.quote(image)}"
    )


def _sbatch_script(local: LocalBackendConfig, inner: str) -> str:
    """A submission script parameterised by the cluster fields, ready to sbatch."""
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=compass",
    ]
    if local.slurm_partition:
        lines.append(f"#SBATCH --partition={local.slurm_partition}")
    if local.slurm_account:
        lines.append(f"#SBATCH --account={local.slurm_account}")
    lines.append(f"#SBATCH --time={local.slurm_time or '04:00:00'}")
    lines.append(f"#SBATCH --nodes={local.slurm_nodes}")
    lines.append("#SBATCH --ntasks-per-node=1")
    lines.append(f"#SBATCH --gpus-per-node={local.slurm_gpus_per_node}")
    lines.append(f"#SBATCH --cpus-per-task={local.slurm_cpus_per_task}")
    if local.slurm_mem_gb:
        lines.append(f"#SBATCH --mem={local.slurm_mem_gb}G")
    lines.append("#SBATCH --output=logs/compass-%j.out")
    lines.append("#SBATCH --error=logs/compass-%j.err")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "mkdir -p logs",
            "",
            "# Weights are pulled once into a shared cache; a compute node is",
            "# usually offline, so keep this on a filesystem every node can read.",
            f'export HF_HOME="${{HF_HOME:-{HF_CACHE_HOST}}}"',
            'export HF_TOKEN="${HF_TOKEN:-}"',
            'export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"',
            "export TOKENIZERS_PARALLELISM=false",
            f"export COMPASS_EXECUTOR_MAX_WORKERS={max(1, local.slurm_cpus_per_task // 2)}",
            "",
            f"srun --gpus-per-node={local.slurm_gpus_per_node} bash -lc {shlex.quote(inner)}",
            "",
        ]
    )
    return "\n".join(lines)


def _warnings(
    local: LocalBackendConfig,
    engine: str,
    vram: Dict[str, Any],
    detail: Optional[Dict[str, Any]],
    hf_token: bool,
    probe: Optional[Dict[str, Any]],
) -> List[str]:
    warnings: List[str] = []
    if not str(local.model_name or "").strip():
        # Everything below describes a command line with an empty --model, which
        # is worse than no command at all, so this case is stated first and the
        # recipe itself is withheld.
        warnings.append(
            "No model is chosen, so there is no runnable recipe. Pick a model first "
            "and the launch commands appear here."
        )
    needed = local.tensor_parallel_size * local.pipeline_parallel_size
    if needed > local.gpu_count:
        warnings.append(
            f"The parallelism needs {needed} GPUs but only {local.gpu_count} are declared."
        )
    if local.trust_remote_code:
        warnings.append(
            "Trusting remote code runs whatever Python the repository ships. "
            "Leave it on only for a repository you have reviewed or trust."
        )
    if detail and detail.get("gated") and not hf_token:
        warnings.append(
            "This repository is gated and no HuggingFace token is stored, so the download will be refused."
        )
    if engine == "transformers" and local.tensor_parallel_size > 1:
        warnings.append("The transformers engine does not shard a model across GPUs; tensor parallelism is ignored.")
    if local.scheduler == "slurm" and not local.slurm_partition:
        warnings.append("No Slurm partition is set, so the job goes to the cluster default queue.")
    if local.scheduler == "slurm" and local.slurm_gpus_per_node * local.slurm_nodes < needed:
        warnings.append(
            f"The job requests {local.slurm_gpus_per_node * local.slurm_nodes} GPUs but the model needs {needed}."
        )
    context = (detail or {}).get("context_length")
    requested = local.max_model_len or local.max_tokens
    if context and requested > int(context):
        warnings.append(
            f"The requested context of {requested} tokens is longer than the {context} this model serves."
        )
    if probe:
        if probe.get("gpu_count") and local.gpu_count > int(probe["gpu_count"]):
            warnings.append(
                f"This machine reports {probe['gpu_count']} GPUs but the configuration declares {local.gpu_count}."
            )
        total_vram = probe.get("total_vram_gb")
        if vram.get("known") and total_vram and vram["total_gb"] > float(total_vram):
            warnings.append(
                f"The estimate of {vram['total_gb']} GB exceeds the {total_vram} GB this machine reports."
            )
        if local.runtime == "docker" and probe.get("docker_available") is False:
            warnings.append("Docker was not found on this machine.")
        if local.runtime == "apptainer" and probe.get("apptainer_available") is False:
            warnings.append("Apptainer was not found on this machine.")
        if local.scheduler == "slurm" and probe.get("slurm_available") is False:
            warnings.append("No Slurm client was found on this machine, so the job has to be submitted elsewhere.")
    return warnings


def _endpoint_hint(local: LocalBackendConfig, engine: str) -> str:
    """Where a served model answers, for the person wiring the base URL to it."""
    if engine != "vllm":
        return (
            "The transformers engine loads the weights inside the run process, "
            "so there is no server to point a base URL at."
        )
    host = "http://localhost:8000/v1"
    if local.scheduler == "slurm":
        return (
            f"Started as a server the job serves {host} on its allocated node, so a client "
            "elsewhere needs that node's hostname rather than localhost."
        )
    return (
        f"Started as a server the model answers at {host}. Set that as the base URL to share "
        "one loaded model across several runs."
    )


def _command_list(
    local: LocalBackendConfig,
    engine: str,
    engine_reason: str,
    commands: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    One copy-pasteable recipe per runtime, in the order the runtime picker shows.

    Each entry is self-contained: a container recipe carries its build line above
    the run line, because running the second without the first fails on an image
    that was never built.
    """
    native = commands["native"]
    docker = commands["docker"]
    apptainer = commands["apptainer"]

    rows: List[Dict[str, str]] = [
        {
            "runtime": "native",
            "title": "Run on this machine",
            "command": native["run"],
            "note": f"{engine_reason} The engine loads the weights in process; no server is started.",
        },
        {
            "runtime": "docker",
            "title": "Run in Docker",
            "command": f"{docker['build']}\n{docker['run']}",
            "note": (
                f"Built from {DOCKERFILE}. The HuggingFace cache is mounted from {HF_CACHE_HOST}, "
                f"so the weights are downloaded once. `{docker['compose']}` starts the same stack with a server."
            ),
        },
        {
            "runtime": "apptainer",
            "title": "Run with Apptainer",
            "command": f"{apptainer['build']}\n{apptainer['run']}",
            "note": (
                f"Built from {APPTAINER_DEF}. Apptainer runs as your own user, which is what a "
                "shared cluster usually allows where Docker does not."
            ),
        },
    ]

    batch = commands.get("slurm")
    if batch:
        rows.append(
            {
                "runtime": "slurm",
                "title": f"Submit to Slurm ({batch['script_name']})",
                "command": batch["sbatch_script"],
                "note": (
                    f"Save this as {batch['script_name']}, then `{batch['submit']}`. "
                    f"Follow it with `{batch['watch']}`."
                ),
            }
        )
    return rows


def plan_deployment(
    config: Any,
    *,
    model_detail: Optional[Dict[str, Any]] = None,
    hf_token: bool = False,
    probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    A ready-to-run recipe for the local backend as currently configured.

    ``model_detail`` is a HuggingFace lookup when one is available; without it
    the size estimate falls back to the parameter count in the repository name.
    """
    local = _local_of(config)
    engine, engine_reason = _resolve_engine(local, probe)

    parameter_count = (model_detail or {}).get("parameter_count") or parameters_from_name(local.model_name)
    vram = estimate_vram(local, parameter_count, model_detail)

    run_command = _compass_command(local)
    serve_command = _vllm_server_command(local) if engine == "vllm" else ""

    docker_image = local.image.strip() or DEFAULT_DOCKER_IMAGE
    apptainer_image = local.image.strip() or DEFAULT_APPTAINER_IMAGE
    docker_prefix = _docker_prefix(local, docker_image)
    apptainer_prefix = _apptainer_prefix(apptainer_image)

    commands: Dict[str, Any] = {
        "native": {
            "run": run_command,
            "serve": f"bash {VLLM_SERVE_SCRIPT}" if serve_command else "",
            "serve_direct": serve_command,
        },
        "docker": {
            "build": f"docker build -f {DOCKERFILE} -t {docker_image} .",
            "run": f"{docker_prefix} {run_command}",
            "serve": f"{docker_prefix} {serve_command}" if serve_command else "",
            "compose": f"docker compose -f {COMPOSE_FILE} up",
        },
        "apptainer": {
            "build": f"apptainer build {apptainer_image} {APPTAINER_DEF}",
            "run": f"{apptainer_prefix} {run_command}",
            "serve": f"{apptainer_prefix} {serve_command}" if serve_command else "",
        },
    }

    selected = commands[local.runtime]
    if local.scheduler == "slurm":
        commands["slurm"] = {
            "script_name": "compass.sbatch",
            "sbatch_script": _sbatch_script(local, selected["run"]),
            "submit": "sbatch compass.sbatch",
            "watch": "squeue --me",
        }
    else:
        commands["slurm"] = None

    runnable = bool(str(local.model_name or "").strip())
    flat = _command_list(local, engine, engine_reason, commands) if runnable else []

    notes = [
        "Every memory figure here is an estimate from the published weight count, not a measurement.",
        f"Weights are cached in {HF_CACHE_HOST}; mount it so a container does not download them again.",
    ]
    if local.runtime != "native" and not local.image.strip():
        notes.append(
            f"No image was named, so the recipe uses the shipped default built from {DOCKERFILE}."
            if local.runtime == "docker"
            else f"No image was named, so the recipe uses the default built from {APPTAINER_DEF}."
        )
    if engine == "vllm" and (local.tensor_parallel_size > 1 or local.pipeline_parallel_size > 1):
        notes.append(
            "Multi-GPU serving needs shared memory for NCCL, which is why the container line sets --shm-size."
        )

    return {
        "model_name": local.model_name,
        "engine": engine,
        "engine_reason": engine_reason,
        "runtime": local.runtime,
        "scheduler": local.scheduler,
        "image": local.image.strip() or (DEFAULT_DOCKER_IMAGE if local.runtime == "docker" else (DEFAULT_APPTAINER_IMAGE if local.runtime == "apptainer" else "")),
        "gpu_count": local.gpu_count,
        "tensor_parallel_size": local.tensor_parallel_size,
        "pipeline_parallel_size": local.pipeline_parallel_size,
        "gpus_required": local.tensor_parallel_size * local.pipeline_parallel_size,
        "context_tokens": local.max_model_len or local.max_tokens,
        "vram": vram,
        # The flat trio the client reads; `vram` keeps the full breakdown.
        "estimated_vram_gb": vram.get("total_gb"),
        "estimate_basis": vram.get("basis") or "",
        "endpoint_hint": _endpoint_hint(local, engine) if runnable else "",
        "warnings": _warnings(local, engine, vram, model_detail, hf_token, probe),
        "notes": notes,
        "hf_token_present": bool(hf_token),
        "model_detail": model_detail or None,
        "assets": {
            "dockerfile": DOCKERFILE,
            "compose": COMPOSE_FILE,
            "apptainer_def": APPTAINER_DEF,
            "serve_script": VLLM_SERVE_SCRIPT,
        },
        # One entry per runtime, which is what the settings panel searches by
        # name; the per-runtime breakdown stays available beside it.
        "commands": flat,
        "runtime_commands": commands,
        "selected_runtime_commands": selected,
    }


# --- what this machine actually has ------------------------------------------


def _which(name: str) -> bool:
    try:
        return shutil.which(name) is not None
    except Exception:
        return False


def _installed(module: str) -> bool:
    """Presence without an import: importing vLLM initialises CUDA."""
    try:
        from importlib.util import find_spec

        return find_spec(module) is not None
    except Exception:
        return False


def _nvidia_smi() -> Tuple[List[str], float]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except Exception:
        return [], 0.0
    if completed.returncode != 0:
        return [], 0.0

    names: List[str] = []
    total_mib = 0.0
    for line in (completed.stdout or "").splitlines():
        if "," not in line:
            continue
        name, _, memory = line.partition(",")
        names.append(name.strip())
        try:
            total_mib += float(memory.strip())
        except ValueError:
            continue
    return names, round(total_mib / 1024, 1)


def probe_machine() -> Dict[str, Any]:
    """
    What is on this machine, without importing anything heavy.

    The planner shows a configuration; this shows the hardware, and the gap
    between the two is the only thing worth warning about. It never raises,
    because a probe that fails is a probe that found nothing.
    """
    names, total_vram_gb = _nvidia_smi()
    return {
        "cuda_available": bool(names),
        "gpu_count": len(names),
        "gpu_names": names,
        "total_vram_gb": total_vram_gb,
        "vllm_installed": _installed("vllm"),
        "transformers_installed": _installed("transformers"),
        "docker_available": _which("docker"),
        "apptainer_available": _which("apptainer") or _which("singularity"),
        "slurm_available": _which("sbatch"),
    }
