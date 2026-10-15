# Measurement methodology

## What is measured

| Metric | Definition | Caveat |
| --- | --- | --- |
| Wall time | BenchExec wall-clock duration | Includes imports, model compilation when enabled, dataset loading, training, validation and checkpoint writes |
| Peak memory | BenchExec peak cgroup memory for the command and descendants | Different accounting from RSS; not GPU memory |
| CPU seconds | BenchExec cgroup CPU time for the command and descendants | Includes worker processes |
| Mean CPU cores | CPU seconds / wall seconds | One fully occupied logical core is approximately 1 |
| CPU utilization | 100 × CPU seconds / (wall seconds × available logical CPUs) | Affinity is respected; cgroup quotas are not |
| Dataset size | Actual compiled Arrow file bytes | Page count and extracted text-line count are different |

BenchExec replaces custom process-tree polling and cleanup. The adapter requires CPU and memory controllers and rejects missing measurements. `benchexec.json` preserves the raw result; `summary.json` adds experiment provenance and derived utilization. The `peak_memory_bytes` field deliberately differs from legacy `peak_rss_bytes`; plots keep the two measurement methods in separate groups. There is no `samples.csv` or `INTERVAL` setting.

The monitor runs outside the measured process tree. CPU utilization is **occupancy**, not algorithmic efficiency: judge it together with runtime. Dividing by `threads × workers` would be misleading because those settings do not reserve CPU cores. Thread limits are supplied to Kraken and the OpenMP, MKL, OpenBLAS and NumExpr environments; inter-op threads and library behavior may still differ.

## Fair comparisons

- Training subsets are nested, with fixed validation pages outside all training subsets.
- Each run starts from scratch with the same architecture and seed.
- Commands, software version, OS, available CPUs, dataset hashes and BenchExec measurements are recorded.
- Runs are sequential; order is shuffled using the seed to reduce order-related bias.
- Repeats share the seed to measure runtime variability. Use three or more for useful error bars.
- Dataset plots keep training settings and validation identity separate. Settings plots keep dataset identity and fixed experimental conditions separate.

Keep the machine otherwise idle. Record power settings, CPU model, storage, available RAM and thermal conditions alongside any published report. Use separate output directories for different hosts or software environments. TorchDynamo compilation is disabled by default in the benchmark matrix, smoke target and Docker image, matching Galaxy (`TORCHDYNAMO_DISABLE=1`, plus `TORCH_COMPILE_DISABLE=1` for the pinned PyTorch). Initial compilation can dominate short runs; Torch compiler caches are not cleared between runs. Compare the two modes separately. Filesystem caches are not flushed, and there is no automatic warmup run; this represents normal repeated invocations. All runs, including the first, are included unless you isolate them in another directory.

## Limits

Different subsets also change line lengths, image content and the inferred character alphabet. These measurements capture the combined effect of adding pages, rather than an isolated byte-size effect. Validation characters absent from a training subset can produce alphabet warnings; do not treat the smoke run as an accuracy evaluation.

Three epochs make a short resource benchmark; startup and checkpoint overhead can dominate. Tiny datasets do not establish production throughput or OCR quality. Large batch sizes can exceed the available line count; Kraken may reject or warn about such configurations. Inspect training logs before interpreting results.

No CUDA memory or energy measurements are collected. The adapter does not currently set per-run CPU or RAM limits; external cgroup limits still apply. Source hashes allow provenance checking, but changed source data requires recompilation. Binary hashes are verified before training; do not modify manifests or datasets during a matrix run.
