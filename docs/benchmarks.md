# Running benchmarks

## Environment

Install `uv` and GNU Make, then run `make setup`. This creates a Python 3.12 virtual environment and installs the locked dependencies. Kraken 7.1.1 comes from PyPI; Linux CPU PyTorch wheels come from the official PyTorch index. Run on Linux or macOS (WSL on Windows).

## Data

Place JPG images and corresponding PAGE XML under `training_data/`. Each XML's `Page/@imageFilename` must resolve relative to the XML file. Annotations must include transcribed text and line geometry.

```sh
make datasets SIZES='5 10 15' VALIDATION_PAGES=2 SEED=42
```

Selection checks for missing images, duplicate image references and missing transcriptions. Two distinct pages are held out. Each larger training subset contains the smaller subset. `artifacts/datasets/datasets.json` records provenance, file hashes and byte sizes. Compilation logs and sampled resources are retained under `compile-runs/`. Recompilation invalidates old metadata until all subsets succeed.

## Experiments

```sh
make dry-run
make smoke
make benchmark WORKERS='0 2 4' THREADS='1 2 4' BATCH_SIZES='4 8 16'
```

The default matrix has 81 runs. The smoke target executes one 5-page run with zero workers, one thread and batch size four, for three epochs, with Torch compilation disabled and maximum line width reduced to 512 pixels for speed. Override `SMOKE_MAX_WIDTH=2560` to validate at the full benchmark width. `make benchmark` requires already compiled datasets. `make all` compiles, benchmarks and plots in sequence.

```sh
make benchmark REPEATS=3 RUNS=artifacts/repeated
make plots RUNS=artifacts/repeated PLOTS=artifacts/repeated-plots
```

TorchDynamo compilation is disabled by default, matching Galaxy. The runner explicitly sets `TORCHDYNAMO_DISABLE=1` and `TORCH_COMPILE_DISABLE=1` for every training process, overriding inherited environment values. The second flag is used by the pinned PyTorch version. Both values are recorded with each run. Explicitly setting `TORCH_COMPILE=enabled` (or `--torch-compile enabled`) sets both flags to 0 for a separate compilation experiment. The smoke target always disables compilation. `HEIGHT` and `MAX_WIDTH` default to 96 and 2560 pixels for the matrix. Smaller maximum widths reduce the cost of the auxiliary training head but can change which lines are accepted. `PRECISIONS`, `DEVICE`, `ACCUMULATE`, `EPOCHS`, `SEED` and `INTERVAL` control additional dimensions. `DATA_DIR`, `DATASETS`, `RUNS` and `PLOTS` override paths. Use `COMPILE_WORKERS` and `COMPILE_THREADS` to change compilation concurrency.

The harness selects AdamW explicitly. Use `OPTIMIZER=AdamW+Muon` to select Kraken’s PP-OCRv6 default instead; it can be costly on CPUs. Optimizer choice is recorded and kept separate in plots. Runs use fixed epochs, no augmentation, a constant learning rate schedule and no warmup. Model initialization and training seed stay fixed across repeats. Numerical and loader scheduling can still vary. Matrix order is shuffled reproducibly; runs are serial to avoid competing benchmark processes.

Failures preserve their logs, do not stop the rest of the matrix, and cause a nonzero exit status at the end. Inspect each `process.log`. Successful runs alone enter plots. Interrupted runs are marked and excluded.

## Compatibility

The default environment is CPU-only. GPU experiments require a separate compatible PyTorch installation and lock; changing `DEVICE` alone does not install CUDA support. Precision modes also depend on hardware. These experiments measure resource usage, not model accuracy or quality after convergence.
