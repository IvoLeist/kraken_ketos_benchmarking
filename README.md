# Kraken / ketos benchmarking

Measure how training-set size, data-loader workers, CPU threads and batch size affect **Kraken 7.1.1 PP-OCRv6 tiny** training. Each run trains from scratch for **three fixed epochs**, using compiled Arrow data and a separate, fixed validation set.

## Quick start

Requires `uv`, GNU Make and Linux or macOS. `uv` creates `.venv` with Python 3.12; Kraken is installed from PyPI. The locked Linux/Windows PyTorch packages use the official CPU wheel index. The monitor requires POSIX process groups, so use WSL on Windows.

```sh
make setup
# Put at least 17 annotated JPG/PAGE XML pairs in training_data/.
make datasets
make smoke                 # 5 pages, 3 epochs, reduced max width for a quick check
make plots
make docs                  # Static documentation in site/
```

The existing `training_data/` directory remains ignored by Git. PAGE XML must reference an existing JPG through `Page/@imageFilename`, relative to its XML file. Filenames may contain spaces. XML needs transcribed text lines with usable polygons/baselines. Images and annotations are not downloaded or published.

`make datasets` deterministically shuffles pairs with seed 42, reserves two pages for validation, and compiles nested training subsets of **5, 10 and 15 page pairs**. Page count is not line count: Kraken trains on the text lines extracted from each page. A dataset manifest records source paths, SHA-256 hashes, page counts and actual Arrow file sizes. Compilation is measured too. Every call recompiles; this avoids silently retaining stale data after source edits.

## Docker

Run the entire CPU suite without installing Python or uv on the host:

```sh
mkdir -p training_data artifacts/docker .docker-cache site
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export BENCHMARK_HOSTNAME="$(hostname)"
docker compose build benchmark
docker compose run --rm benchmark make datasets
docker compose run --rm benchmark make smoke RUNS=artifacts/smoke-runs
docker compose run --rm benchmark make plots RUNS=artifacts/smoke-runs PLOTS=artifacts/smoke-plots
```

Use `docker compose run --rm benchmark make benchmark` for the full matrix, or `make all` inside the container to compile, train and plot in sequence. All Make variables remain configurable. Outputs persist under `artifacts/docker/`; source data is mounted read-only. The image runs as a non-root user and includes development and documentation tools.

```sh
docker compose run --rm benchmark make test
docker compose run --rm benchmark make docs
docker compose --profile docs up -d docs  # Preview at http://localhost:8000
```

See [Docker deployment](docs/docker.md) for storage, user IDs, resource limits, stopping jobs, custom datasets and repeatable measurements. Recompile datasets inside Docker: native manifests contain host paths. `.env.example` documents optional Compose settings.

## Training matrix

```sh
make dry-run               # Print commands without executing or requiring compiled files
make benchmark             # 3 sizes × 3 workers × 3 threads × 3 batches = 81 runs
make plots

# A smaller matrix, with three repetitions per configuration:
make benchmark WORKERS='0 2' THREADS='1 4' BATCH_SIZES='4 8' REPEATS=3

# Change the datasets and validation size:
make datasets SIZES='5 10 15 20' VALIDATION_PAGES=3
make benchmark SIZES='5 10 15 20'

# Further controls:
make benchmark WORKERS=2 THREADS=4 BATCH_SIZES=8 ACCUMULATE=2 PRECISIONS=32-true
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIZES` | `5 10 15` | Training page counts |
| `WORKERS` | `0 2 4` | Loader worker processes; zero loads in the main process |
| `THREADS` | `1 2 4` | PyTorch intra-op and numerical-library thread limits |
| `BATCH_SIZES` | `4 8 16` | Text-line examples per batch |
| `OPTIMIZER` | `AdamW` | Explicit CPU-friendly optimizer; `AdamW+Muon` selects Kraken’s PP-OCRv6 default |
| `TORCH_COMPILE` | `enabled` | Torch model compilation; `disabled` skips compilation |
| `HEIGHT`, `MAX_WIDTH` | `96`, `2560` | Normalized line height and maximum input width |
| `SMOKE_MAX_WIDTH` | `512` | Reduced maximum width used only by the smoke target |
| `EPOCHS` | `3` | Fixed epochs, with early stopping disabled |
| `REPEATS` | `1` | Repetitions; use at least three for variability estimates |
| `PRECISIONS` | `32-true` | Kraken/Lightning precision values |
| `DEVICE` | `cpu` | Explicit compute device |
| `ACCUMULATE` | `1` | Gradient accumulation; effective batch = batch × accumulation |
| `SEED` | `42` | Dataset selection and training seed |
| `INTERVAL` | `0.2` | Resource sampling interval, seconds |
| `VALIDATION_PAGES` | `2` | Distinct held-out pages, outside the training counts |
| `COMPILE_WORKERS`, `COMPILE_THREADS` | `1`, `1` | Compilation concurrency |
| `DATA_DIR` | `training_data` | Source directory, scanned recursively |
| `DATASETS`, `RUNS`, `PLOTS` | `artifacts/datasets`, `artifacts/runs`, `artifacts/plots` | Output directories |

Runs execute serially in a seeded shuffled order, with unique output directories. Re-running adds measurements. Failures retain logs and summaries, the matrix continues, and the command exits unsuccessfully if any run failed. Use separate `RUNS` folders when comparing different machines or software environments. Do not run multiple benchmark invocations concurrently.

CPU is the supported default. Selecting `DEVICE=cuda:0` also requires replacing the CPU Torch installation with compatible CUDA wheels and maintaining a separate environment/lock. Mixed precision must be supported by your selected hardware. AdamW is selected explicitly; Kraken’s PP-OCRv6 default is AdamW+Muon, selectable with `OPTIMIZER=AdamW+Muon`. No pretrained weights are downloaded; this is a training-resource benchmark, not an OCR-quality comparison.

## Results and interpretation

Each run includes `command.json`, `process.log`, `samples.csv`, `summary.json`, checkpoints and training logs. `make plots` writes:

- `results.csv`: all run summaries, including failed runs and exit codes.
- `dataset-vs-*.png` / `.svg`: Arrow size versus peak RAM, wall time and CPU utilization, one series per configuration.
- `settings-*.png` / `.svg`: workers/threads/batch comparisons for each dataset and fixed experimental conditions.
- `compilation.png` / `.svg`: resource usage when compiling datasets.

Only successful training runs enter plots. Repeated equivalent configurations show their mean and sample standard deviation. The smoke target disables Torch compilation and uses a reduced maximum width (512 pixels); it validates the pipeline and is not comparable to default-width benchmarks. For full-default verification, use `make benchmark SIZES=5 WORKERS=0 THREADS=1 BATCH_SIZES=4`. Keep smoke runs in a separate `RUNS` directory if you do not want them included in subsequent comparisons.

**RAM** is sampled summed resident memory (RSS) of the training process and descendants; shared pages can be counted more than once. **CPU utilization** is sampled process-tree CPU time divided by wall time and the logical CPUs available through affinity. It measures occupancy, not computational efficiency or speedup; a faster run can use fewer CPU resources. Short-lived workers and peaks between samples may be missed. Cgroup CPU quotas are not accounted for. GPU memory is not measured. See [the methodology](docs/methodology.md).

## Documentation and GitHub Pages

```sh
make docs-serve             # Local Zensical preview
make docs                  # Build static site
make test                  # Harness tests; no full training matrix
```

The Zensical site configuration is in `zensical.toml`. The GitHub Actions workflow builds documentation on pull requests and deploys pushes to `main`. After pushing this repository to GitHub, select **Settings → Pages → Build and deployment → Source: GitHub Actions**. No remote is assumed or created. Training data, checkpoints and local benchmark outputs are excluded from deployment. See [publishing instructions](docs/publishing.md) to add reviewed plots to the site.

## Project layout

```text
scripts/datasets.py   Pair validation, seeded subsets and measured compilation
scripts/benchmark.py  Training commands and serial matrix execution
scripts/monitor.py    Process-tree resource sampling
scripts/plot.py       CSV export and PNG/SVG plots
artifacts/            Generated data, runs and plots (ignored)
docs/                 Zensical documentation source
uv.lock               Exact dependency resolution
```

Dependencies are defined in `pyproject.toml` and installed with `uv sync --locked --group docs`. To deliberately upgrade, edit the version constraints, run `uv lock`, then repeat tests and a smoke benchmark. Python 3.14 is not supported by this Kraken release.

References: [Kraken recognition training](https://kraken.re/main/user_guide/training_recognition.html), [Kraken on PyPI](https://pypi.org/project/kraken/7.1.1/), [Zensical publishing](https://zensical.org/docs/publish-your-site/).
