# Docker Setup

Two paths, and within the hosted path three shapes, so a container carries only
what you actually run:

| Path | You get | Files | GPU needed |
| --- | --- | --- | --- |
| Hosted models | The engine locally, models called at OpenRouter | `Dockerfile`, `docker-compose.yml` | no |
| Self-hosted open weights | The dashboard plus a vLLM server holding a HuggingFace model | `Dockerfile.gpu`, `docker-compose.gpu.yml` | yes |

| Shape | Command | What it builds |
| --- | --- | --- |
| Dashboard and API | `docker compose -f docker/docker-compose.yml up --build` | client bundle plus service |
| API only | `docker compose -f docker/docker-compose.yml --profile api up --build` | service only, no Node stage |
| One run, then exit | `docker compose -f docker/docker-compose.yml --profile cli run --rm compass-cli <main.py args>` | service only, no server |

The API-only and CLI shapes pass `--build-arg WEB_STAGE=web-none`, which skips
the Node stage outright rather than building a bundle and ignoring it. The
service still answers every `/api` route; only the browser client is absent.

Both paths expose the dashboard at `http://localhost:5005`. The self-hosted path
also exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

For several nodes on a cluster, use `src/full_stack/backend/hpc/` instead: the
Apptainer definition and `06_serve_vllm.sh` there cover Slurm, multi-node
pipeline parallelism, and Ray.

## Files in this folder

- `Dockerfile`: CPU image for the dashboard and engine (curated dependencies)
- `Dockerfile.full`: optional CPU image with the full `requirements.txt`
- `Dockerfile.gpu`: CUDA image that serves models with vLLM and runs the dashboard
- `docker-compose.yml`: the hosted-model path
- `docker-compose.gpu.yml`: the self-hosted path (vLLM server plus dashboard)
- `requirements.ui.txt`: curated dashboard and engine dependencies
- `entrypoint.sh`: starts `main.py --ui` and binds to `0.0.0.0:5005`
- `entrypoint.gpu.sh`: picks the role for the GPU image, `serve` or `dashboard`
- `.env.example`: every environment variable the compose files read
- `.dockerignore`: exclusions for the `tar | docker buildx build` flow below
- `Dockerfile.dockerignore`, `Dockerfile.gpu.dockerignore`: the same exclusions
  for builds whose context is the repository root, which is what Compose uses.
  BuildKit reads an ignore file that sits next to the Dockerfile, so these exist
  because the canonical one lives in `docker/` rather than at the root. Keep the
  three files in step.

## Prerequisites

- Docker Engine 23 or newer, or Docker Desktop, with BuildKit enabled (default).
- For the self-hosted path: an NVIDIA GPU, a recent driver, and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check it with `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`.
- An `OPENROUTER_API_KEY` for the hosted path.

Compose reads `docker/.env`. Start from the annotated example beside it, which
documents every knob including the GPU and vLLM settings:

```bash
cp docker/.env.example docker/.env
```

Two mounts are worth setting there. `COMPASS_DATA_DIR` is your folder of
participant directories, mounted read only at `/data`; add `/data` as a data
root in the dashboard, or pass a path under it on the command line. It defaults
to the bundled synthetic samples. `COMPASS_RESULTS_DIR` is where run outputs
land: a path mounts that folder, a bare name uses a Docker volume.

## Path 1: hosted models (no GPU)

```bash
# from the repository root
docker compose -f docker/docker-compose.yml up --build
```

Then open `http://localhost:5005`. The default public model is
`deepseek/deepseek-v4-flash-0731` through OpenRouter. Settings, stored
credentials, and run history live in the `compass-home` named volume, so they
survive a rebuild.

Stop it with `Ctrl+C`, or `docker compose -f docker/docker-compose.yml down`.

### Without Compose

The manual flow still works and produces the same image under a different tag.
It pipes the context through `tar` so that `docker/.dockerignore` applies:

```bash
tar --exclude-from=docker/.dockerignore -cf - . | docker buildx build \
  --platform linux/arm64 \
  -f docker/Dockerfile \
  -t compass-ui:local \
  --load \
  -

docker run --rm \
  -p 5005:5005 \
  -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
  --name compass-ui \
  compass-ui:local
```

Use `--platform linux/amd64` on Intel Mac, Linux, and Windows Docker Desktop. On
Windows, run these commands in WSL or Git Bash so the pipeline works as written.

## Path 2: self-hosted open weights on GPU

The GPU image is one image in two roles. `serve` starts a vLLM server that speaks
the OpenAI protocol; `dashboard` starts COMPASS pointed at that server. Weights
are never baked into the image: they are downloaded once into the `hf-cache`
named volume and reused.

`serve` builds the same `vllm serve` command line as the `%runscript` in
`src/full_stack/backend/hpc/apptainer.def`, flag for flag, so a model that serves
on a cluster serves here too. Keep the two in step when you change either.

### One GPU

```bash
docker compose -f docker/docker-compose.gpu.yml up --build
```

The first start downloads the model, so give it time. The dashboard waits for the
server's health check before it comes up. Then open `http://localhost:5005`.

To serve the weights without the dashboard, for instance on a GPU box that other
machines call over the network, start the one service:

```bash
docker compose -f docker/docker-compose.gpu.yml up vllm
```

It exposes the OpenAI-compatible API on port 8000. Point any COMPASS install at
`http://<host>:8000/v1` with any non-empty key.

### How the dashboard finds the server

The compose file sets `OPENROUTER_BASE_URL=http://vllm:8000/v1` on the dashboard
service. That variable seeds the dashboard's stored configuration on **first
start only**, when the `compass-home` volume does not yet hold a
`dashboard.json` (`~/.compass/dashboard.json`, mounted at
`/home/compass/.compass/dashboard.json` in the container).

From the moment that file exists, the base URL saved in it wins and the
environment is no longer consulted, so a value you chose in Settings is never
overwritten by the container it happens to run in. Note that the file is written
by the first save of any setting, not only of the connection: a stack whose
settings were ever saved keeps the base URL recorded then, which for most users
is OpenRouter.

A fresh stack therefore wires itself to the local server, and an existing
installation does not. Nothing rewires an existing configuration for you: open
Settings, Connection, and set the base URL to `http://vllm:8000/v1` with any
non-empty key.

To check what is actually in effect, read the file the dashboard uses:

```bash
docker compose -f docker/docker-compose.gpu.yml exec compass \
  cat /home/compass/.compass/dashboard.json
```

Starting from scratch instead means discarding that volume, along with the
stored settings and run history it holds:

```bash
docker compose -f docker/docker-compose.gpu.yml down
docker volume rm compass-gpu_compass-home
```

### Several GPUs on one node

Tensor parallelism splits one model across several GPUs. Give the container the
GPUs and tell vLLM how many to shard across. The two numbers must match, and the
tensor-parallel size must divide the model's attention head count:

```bash
# docker/.env
COMPASS_GPU_COUNT=4
COMPASS_VLLM_TP=4
COMPASS_VLLM_MODEL=Qwen/Qwen3-32B-AWQ
COMPASS_VLLM_MAX_MODEL_LEN=32768
```

```bash
docker compose -f docker/docker-compose.gpu.yml up --build
```

### Settings

All of these go in `docker/.env`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `COMPASS_VLLM_MODEL` | `Qwen/Qwen3-14B-AWQ` | HuggingFace repo id to serve |
| `COMPASS_VLLM_TP` | `1` | GPUs to split the model across (tensor parallel) |
| `COMPASS_VLLM_PP` | `1` | pipeline stages, for a model too large for one node |
| `COMPASS_GPU_COUNT` | `all` | GPUs handed to the container |
| `COMPASS_VLLM_MAX_MODEL_LEN` | `32768` | context window served |
| `COMPASS_VLLM_GPU_MEM_UTIL` | `0.90` | fraction of VRAM vLLM may hold |
| `COMPASS_VLLM_QUANT` | unset | quantization hint, for example `awq_marlin` |
| `COMPASS_VLLM_SERVED_NAME` | repo id | name clients must ask for |
| `COMPASS_VLLM_API_KEY` | `local-vllm` | key the server requires and the dashboard sends |
| `COMPASS_VLLM_TRUST_REMOTE_CODE` | `1` | allow code shipped in the model repo; `0` refuses |
| `COMPASS_VLLM_EXTRA_ARGS` | empty | any other `vllm serve` flags |
| `COMPASS_VLLM_PORT` | `8000` | host port for the model API |
| `COMPASS_UI_PORT` | `5005` | host port for the dashboard |
| `VLLM_VERSION` | `0.8.5` | vLLM release baked into the image at build time |
| `HF_TOKEN` | empty | only for gated repositories, read at run time |

`HF_TOKEN` is passed to the running container, never into an image layer. Nothing
in these images carries a credential.

### Build and run without Compose

```bash
# from the repository root
docker build -f docker/Dockerfile.gpu -t compass-engine:gpu .

# serve a model
docker run --rm --gpus all --shm-size=16g \
  -p 8000:8000 \
  -v compass-hf-cache:/home/compass/.cache/huggingface \
  -e COMPASS_VLLM_MODEL=Qwen/Qwen3-14B-AWQ \
  -e COMPASS_VLLM_TP=1 \
  --name compass-vllm \
  compass-engine:gpu serve

# run the dashboard against it
docker run --rm \
  -p 5005:5005 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENROUTER_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENROUTER_API_KEY=local-vllm \
  -v compass-home:/home/compass/.compass \
  --name compass-ui \
  compass-engine:gpu dashboard
```

`--add-host` is what makes `host.docker.internal` resolve on Linux; Docker
Desktop provides it already. `--shm-size` matters too: vLLM moves tensors between
workers through shared memory, and the 64 MB Docker default deadlocks
tensor-parallel startup.

### Pointing an existing COMPASS at the server

A self-hosted server is configured exactly like a hosted provider, because both
speak the OpenAI protocol. In the dashboard, open Settings, Connection, and set
the base URL to `http://localhost:8000/v1` with any non-empty key. From the CLI:

```bash
export OPENROUTER_BASE_URL="http://localhost:8000/v1"
export OPENROUTER_API_KEY="local-vllm"
python3 main.py <participant_dir> --backend openrouter --model "Qwen/Qwen3-14B-AWQ" ...
```

## Prediction task setup in the dashboard

Inside the run wizard:
- Step 1/2 configures the engine and runtime backend.
- Step 2/2 configures the prediction task: binary classification, multi-class
  classification, univariate regression, multivariate regression, or a
  hierarchical JSON task specification. Mode-specific fields appear as needed,
  so comparator labels only show up for binary classification.

## Verification

Container logs should show `Launching COMPASS Dashboard...` and
`Dashboard live at http://0.0.0.0:5005`. Two quick probes:

```bash
curl -fsS http://localhost:5005/api/health
curl -fsS http://localhost:8000/v1/models   # self-hosted path only
docker compose -f docker/docker-compose.gpu.yml ps   # both services healthy
docker inspect --format '{{.State.Health.Status}}' compass-vllm
```

The image's health probe follows the role it was started in: `serve` probes
`/health` on `COMPASS_VLLM_PORT`, `dashboard` probes `/api/health` on
`COMPASS_UI_PORT`. The plain `docker run` commands above therefore report
healthy without being told which URL to use.

The entrypoint writes its role's probe URL to `/tmp/compass-health-url` and the
health check reads that file. It has to work that way: Docker builds the probe
as a sibling process from the image's configured environment, so a variable
exported by the running entrypoint never reaches it, while the filesystem is
shared. Set `COMPASS_HEALTHCHECK_URL` with `-e` to override the file, or
`COMPASS_HEALTH_URL_FILE` to move it.

## Troubleshooting

- `could not select device driver "nvidia"`: the NVIDIA Container Toolkit is not
  installed or the Docker daemon was not restarted after installing it.
- The vLLM container is stuck in `starting`: a large model can take many minutes
  to download and load. Watch it with
  `docker compose -f docker/docker-compose.gpu.yml logs -f vllm`.
- Out of memory at load time: lower `COMPASS_VLLM_MAX_MODEL_LEN`, lower
  `COMPASS_VLLM_GPU_MEM_UTIL`, or serve a quantized checkpoint.
- Tensor-parallel startup hangs: raise `shm_size`, and confirm
  `COMPASS_VLLM_TP` is not larger than `COMPASS_GPU_COUNT`.
- A huge build context: your Docker is too old to read the per-Dockerfile ignore
  file. Upgrade to Docker 23 or newer, or use the `tar | docker buildx build`
  flow shown above.

## Optional variant: full dependency image

This CPU image adds the local inference dependencies (`torch`, `transformers`,
`bitsandbytes`) for users who want them without CUDA:

```bash
tar --exclude-from=docker/.dockerignore -cf - . | docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.full \
  -t compass-full:amd64 \
  --load \
  -
```

## Local install vs containers

Both are valid. A local install
(`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`)
is best for active development and editing code in place. Containers are better
for reproducible execution: they pin the runtime at build time, avoid
host-specific Python and toolchain problems, and give every user the same
entrypoint and the same port.
