# Measurement methodology

## What is measured

| Metric | Definition | Caveat |
| --- | --- | --- |
| Wall time | Monotonic elapsed seconds around the subprocess | Includes imports, model compilation when enabled, dataset loading, training, validation and checkpoint writes |
| Peak RSS | Maximum sampled sum of resident bytes for the root process and descendants | Shared memory may be double-counted; not GPU memory |
| CPU seconds | Sum of each observed process's maximum user + system CPU time | Short-lived processes and final increments can be missed |
| Mean CPU cores | CPU seconds / wall seconds | One fully occupied logical core is approximately 1 |
| CPU utilization | 100 × CPU seconds / (wall seconds × available logical CPUs) | Affinity is respected; cgroup quotas are not |
| Dataset size | Actual compiled Arrow file bytes | Page count and extracted text-line count are different |

Sampling defaults to 0.2 seconds. PID plus process creation time identifies a process, preventing CPU time from being counted twice across samples or PID reuse. CPU time already recorded for exited workers is retained. This is a sampled estimate, not kernel-level accounting. Memory spikes and processes between observations can be missed.

The monitor runs outside the measured process tree. CPU utilization is **occupancy**, not algorithmic efficiency: judge it together with runtime. Dividing by `threads × workers` would be misleading because those settings do not reserve CPU cores. Thread limits are supplied to Kraken and the OpenMP, MKL, OpenBLAS and NumExpr environments; inter-op threads and library behavior may still differ.

## Fair comparisons

- Training subsets are nested, with fixed validation pages outside all training subsets.
- Each run starts from scratch with the same architecture and seed.
- Commands, software version, OS, available CPUs, dataset hashes and resource samples are recorded.
- Runs are sequential; order is shuffled using the seed to reduce order-related bias.
- Repeats share the seed to measure runtime variability. Use three or more for useful error bars.
- Dataset plots keep training settings and validation identity separate. Settings plots keep dataset identity and fixed experimental conditions separate.

Keep the machine otherwise idle. Record power settings, CPU model, storage, available RAM and thermal conditions alongside any published report. Use separate output directories for different hosts or software environments. TorchDynamo compilation is disabled by default in the benchmark matrix, smoke target and Docker image, matching Galaxy (`TORCHDYNAMO_DISABLE=1`, plus `TORCH_COMPILE_DISABLE=1` for the pinned PyTorch). Initial compilation can dominate short runs; Torch compiler caches are not cleared between runs. Compare the two modes separately. Filesystem caches are not flushed, and there is no automatic warmup run; this represents normal repeated invocations. All runs, including the first, are included unless you isolate them in another directory.

## Limits

Different subsets also change line lengths, image content and the inferred character alphabet. These measurements capture the combined effect of adding pages, rather than an isolated byte-size effect. Validation characters absent from a training subset can produce alphabet warnings; do not treat the smoke run as an accuracy evaluation.

Three epochs make a short resource benchmark; startup and checkpoint overhead can dominate. Tiny datasets do not establish production throughput or OCR quality. Large batch sizes can exceed the available line count; Kraken may reject or warn about such configurations. Inspect training logs before interpreting results.

No CUDA memory or energy measurements are collected. Sampling does not enforce CPU or RAM limits. Source hashes allow provenance checking, but changed source data requires recompilation. Binary hashes are verified before training; do not modify manifests or datasets during a matrix run.
