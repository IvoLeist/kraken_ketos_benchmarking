.DEFAULT_GOAL := help
UV ?= uv
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
PYTHON := .venv/bin/python
DOCS_ADDRESS ?= 127.0.0.1:8000
DATA_DIR ?= training_data
DATASETS ?= artifacts/datasets
RUNS ?= artifacts/runs
PLOTS ?= artifacts/plots
SIZES ?= 5 10 15
VALIDATION_PAGES ?= 2
WORKERS ?= 0 2 4
THREADS ?= 1 2 4
BATCH_SIZES ?= 4 8 16
PRECISIONS ?= 32-true
DEVICE ?= cpu
OPTIMIZER ?= AdamW
TORCH_COMPILE ?= disabled
HEIGHT ?= 96
MAX_WIDTH ?= 2560
SMOKE_MAX_WIDTH ?= 512
EPOCHS ?= 3
REPEATS ?= 1
SEED ?= 42
ACCUMULATE ?= 1
COMPILE_WORKERS ?= 1
COMPILE_THREADS ?= 1
INTERVAL ?= 0.2
BENCH_ARGS = --datasets "$(DATASETS)" --output "$(RUNS)" --sizes $(SIZES) --workers $(WORKERS) --threads $(THREADS) --batch-sizes $(BATCH_SIZES) --precisions $(PRECISIONS) --device "$(DEVICE)" --optimizer "$(OPTIMIZER)" --torch-compile $(TORCH_COMPILE) --height $(HEIGHT) --max-width $(MAX_WIDTH) --epochs $(EPOCHS) --repeats $(REPEATS) --seed $(SEED) --accumulate $(ACCUMULATE) --interval $(INTERVAL)

.PHONY: help setup check-env datasets benchmark dry-run smoke plots test docs docs-serve all
help:
	@printf '%s\n' 'make setup       Install locked Python 3.12 environment with uv' 'make datasets    Compile nested 5/10/15-page Arrow files and held-out validation' 'make dry-run     Preview the training matrix (81 runs by default)' 'make smoke       Run one 5-page, 3-epoch training benchmark' 'make benchmark   Run serial training matrix; capture resources and logs' 'make plots       Export CSV, PNG and SVG comparisons' 'make test        Run harness tests' 'make docs        Build Zensical site' 'make docs-serve  Preview documentation' 'make all         Compile, benchmark, plot (setup required)' 'make docker-build Build the Docker image' 'make docker-run TASK=help ARGS=...  Run a Make target inside Docker' 'make docker-docs  Start the container documentation preview' 'make docker-down Stop Compose services'
setup:
	$(UV) sync --locked --group docs
check-env:
	@test -x "$(PYTHON)" || (echo 'Run make setup first'; exit 1)
datasets: check-env
	$(PYTHON) -m scripts.datasets --source "$(DATA_DIR)" --output "$(DATASETS)" --sizes $(SIZES) --validation-pages $(VALIDATION_PAGES) --seed $(SEED) --workers $(COMPILE_WORKERS) --threads $(COMPILE_THREADS)
benchmark: check-env
	$(PYTHON) -m scripts.benchmark $(BENCH_ARGS)
dry-run: check-env
	$(PYTHON) -m scripts.benchmark $(BENCH_ARGS) --dry-run
smoke: check-env
	$(PYTHON) -m scripts.benchmark --datasets "$(DATASETS)" --output "$(RUNS)" --sizes 5 --workers 0 --threads 1 --batch-sizes 4 --epochs 3 --max-width $(SMOKE_MAX_WIDTH) --torch-compile disabled --device "$(DEVICE)"
plots: check-env
	$(PYTHON) -m scripts.plot --input "$(RUNS)" --datasets "$(DATASETS)" --output "$(PLOTS)"
test: check-env
	$(PYTHON) -m pytest -q
docs: check-env
	.venv/bin/zensical build --clean
docs-serve: check-env
	.venv/bin/zensical serve --dev-addr "$(DOCS_ADDRESS)"
all: datasets
	$(MAKE) benchmark
	$(MAKE) plots

# Native Python/uv are not required for these targets.
DOCKER_COMPOSE ?= docker compose
TASK ?= help
ARGS ?=
.PHONY: docker-build docker-run docker-docs docker-down
docker-build:
	$(DOCKER_COMPOSE) build benchmark
docker-run:
	$(DOCKER_COMPOSE) run --rm benchmark make $(TASK) $(ARGS)
docker-docs:
	$(DOCKER_COMPOSE) --profile docs up -d docs
docker-down:
	$(DOCKER_COMPOSE) --profile docs down

.PHONY: integration-test
integration-test: check-env
	$(PYTHON) tests/container_smoke.py
