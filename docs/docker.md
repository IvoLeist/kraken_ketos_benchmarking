# Run the suite with Docker

The CPU image contains Python 3.12, the exact `uv.lock` environment, Kraken, plotting tools, pytest, Zensical, and the C++ compiler required by TorchInductor. A host Python installation is unnecessary. Docker Engine with Docker Compose is required; Docker Desktop supplies both on macOS and Windows. Linux x86-64 is the primary benchmark platform. Do not compare native runs with emulated containers as equivalent hardware.

## Build and prepare storage

Run these commands from the repository root in a POSIX shell (use WSL on Windows):

```sh
mkdir -p training_data artifacts/docker .docker-cache site
# Linux: match the container's output-file ownership to your user.
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export BENCHMARK_HOSTNAME="$(hostname)"
docker compose build benchmark
```

Supply at least 17 JPG/PAGE XML pairs in `training_data/`. The default runtime user is UID/GID 1000; `.env.example` lists optional overrides. To persist settings, copy it to `.env` and edit the values. Create any custom host directories before starting containers. Missing bind-mount sources fail rather than being silently created as root.

The image build excludes local data, results, credentials, caches and the host virtual environment. Dependencies are installed during the build with `uv sync --locked`; runtime commands do not need to install packages. Rebuild after changing code, dependencies or documentation. Base-image tags can receive updates; retain the built image ID alongside published measurements if exact image reproduction matters.

## Compile, train, and plot

```sh
docker compose run --rm benchmark make datasets
docker compose run --rm benchmark make dry-run

# Pipeline verification, kept separate from benchmark measurements:
docker compose run --rm benchmark make smoke RUNS=artifacts/smoke-runs
docker compose run --rm benchmark make plots \
  RUNS=artifacts/smoke-runs PLOTS=artifacts/smoke-plots

# Full default matrix: 81 serial, three-epoch training runs.
docker compose run --rm benchmark make benchmark
docker compose run --rm benchmark make plots

# Smaller matrix, with eager execution instead of Torch compilation:
docker compose run --rm benchmark make benchmark \
  'SIZES=5 10' 'WORKERS=0 2' 'THREADS=1 2' 'BATCH_SIZES=4 8' \
  REPEATS=3 TORCH_COMPILE=disabled

# Compile, benchmark and plot in one job:
docker compose run --rm benchmark make all
```

All regular Make variables work inside the image. Quote arguments containing lists so each remains one Make assignment. The smoke target uses reduced maximum width, AdamW, and disabled Torch compilation; it is a pipeline check, not a default-matrix measurement. Runtime, memory and CPU observations include the same subprocess work as native execution.

Optional host Make shortcuts:

```sh
make docker-build
make docker-run TASK=datasets
make docker-run TASK=benchmark ARGS='SIZES=5 WORKERS=0 THREADS=1 BATCH_SIZES=4 TORCH_COMPILE=disabled'
```

The first storage-preparation step is still required. Use direct Compose commands for complex list-valued arguments.

## Persistent files and caches

| Host path (default) | Container path | Access |
| --- | --- | --- |
| `training_data/` | `/app/training_data` | Read-only source data |
| `artifacts/docker/` | `/app/artifacts` | Compiled datasets, manifests, runs and plots |
| `.docker-cache/` | `/cache` | Persistent Torch, Matplotlib and pytest caches |
| `site/` | `/app/site` | Built documentation |

Removing a container does not remove these files. Dataset manifests use absolute paths: compile them inside Docker, and keep the `/app` mount layout when reusing them. Native manifests pointing into your home directory cannot be used inside the image. Container outputs are separate from the native `artifacts/datasets` and `artifacts/runs` directories by default. Plot container results through Docker too, since compilation metadata references container paths.

Use the same `BENCHMARK_HOSTNAME` for repeated invocations on one machine; choose different values on different machines. Results use this hostname to separate experiments. Zensical uses its own ephemeral `/app/.cache` directory so clean documentation builds cannot remove Torch caches. The Torch compiler cache persists across runs; a warm cache can change runtime. Use a fresh `CACHE_DIR` for explicitly cold-cache experiments and record that choice.

## Resource limits and stopping

The image is CPU-only. Merely setting `DEVICE=cuda:0` does not add GPU support. The Compose services allocate a 2 GiB shared-memory ceiling for worker communication; increase `SHM_SIZE` if worker processes exhaust it. This is a ceiling, not a reservation of physical RAM.

The supplied `compose.limits.yaml` is sized conservatively for the shared KVM VM with 12 vCPUs and 31 GiB RAM:

| Service | CPU allowance | Hard RAM ceiling | Swap | Process/thread ceiling |
| --- | --- | --- | --- | --- |
| Benchmark | CPUs 0–3, at most 4 CPU equivalents | 8 GiB | Disabled | 512 |
| Documentation preview | CPU 4, at most 1 CPU equivalent | 1 GiB | Disabled | 128 |

With one of each container running, at least 22 GiB of the VM's RAM remains outside these ceilings, and the containers cannot use CPUs 5–11. CPU affinity does not reserve CPUs exclusively; other VM workloads can still compete on CPUs 0–4. The benchmark's 2 GiB shared-memory ceiling is **included** in its 8 GiB RAM limit. Equal `mem_limit` and `memswap_limit` disable container swap, avoiding benchmark-driven swap thrashing. OOM killing remains enabled: an oversized run can fail inside its container. Docker console logs rotate at 10 MiB with three files.

**Activate the override for every invocation** (it is not loaded automatically):

```sh
export COMPOSE_FILE=compose.yaml:compose.limits.yaml
docker compose config --quiet
docker compose run --rm benchmark make datasets
docker compose run --rm benchmark make benchmark
docker compose run --rm benchmark make plots
# The same exported setting also applies to make docker-run and docker-docs.
```

Alternatively, use explicit file arguments:

```sh
docker compose -f compose.yaml -f compose.limits.yaml run --rm benchmark make benchmark
```

Start with `SIZES=5 WORKERS=0 THREADS=1 BATCH_SIZES=4` if checking a new dataset. Worker/thread combinations still share the same four CPUs; large batches can hit the RAM ceiling. Reduce batch size or workers if a run is killed, rather than removing the limit.

These are per-container limits: run only one benchmark container at a time. They constrain runtime containers, not image builds or native Python commands. They cannot guarantee VM stability if unrelated workloads exhaust resources. Check free disk space as checkpoints and files in `artifacts/` and caches are not capped by these limits; Docker log rotation does not rotate benchmark output files. Retain the override with your results and use a separate results directory when changing limits.

CPU utilization is normalized to affinity-visible logical CPUs, so CPU pinning with `cpuset` is reflected in the denominator. Cgroup CPU quotas (`cpus`) and Docker Desktop VM limits may be narrower and are not used in that denominator. Report the container limits and host/VM configuration with results. Process RSS is not the entire container's cgroup memory usage; container OOM termination can prevent final summaries from being written.

For a detached job:

```sh
docker compose run -d --name ketos-benchmark benchmark make benchmark
# Follow the command output; detailed training logs are in artifacts/docker/runs/.
docker logs -f ketos-benchmark
# Orderly interruption (allow child shutdown and summary writing):
docker stop --time 30 ketos-benchmark
# Remove the stopped container when no longer needed:
docker rm ketos-benchmark
```

The image includes Tini to forward signals and reap children. The monitor handles SIGTERM and marks interrupted runs, which plotting excludes. Forced kills, daemon crashes and OOM kills cannot guarantee cleanup or a summary. The Compose stop grace period is 30 seconds. Only run one benchmark job at a time when measuring performance.

## Documentation and tests

```sh
docker compose run --rm benchmark make test
# Tiny synthetic pages; checks compilation, 3 epochs with a worker, and plotting:
docker compose run --rm benchmark make integration-test
docker compose run --rm benchmark make docs
docker compose --profile docs up -d docs
# Open http://localhost:8000
docker compose --profile docs down
```

The documentation preview listens on the container network interface but is published on host loopback only. Set `DOCS_PORT` to change the host port. Rebuild after editing documentation. Use the existing GitHub Pages workflow for public static hosting; the Zensical preview service is for local inspection. `make docs` exports the static site to the host `site/` directory.

A Docker validation workflow builds the image, runs the harness tests, checks a tiny synthetic three-epoch training pipeline, checks the benchmark command, and builds documentation on pushes and pull requests. It does not publish an image or launch the full training matrix.

References: [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/), [Docker Compose services](https://docs.docker.com/reference/compose-file/services/).

The image and benchmark runner disable TorchDynamo by default with `TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1`, matching Galaxy. Training commands enforce and record both flags. Use `TORCH_COMPILE=enabled` only when explicitly benchmarking compilation.
