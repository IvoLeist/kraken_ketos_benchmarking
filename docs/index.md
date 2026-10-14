# Kraken / ketos benchmarking

A reproducible harness for **PP-OCRv6 tiny**, trained from scratch for **three epochs** with Kraken 7.1.1.

Compare 5-, 10- and 15-page compiled Arrow datasets while varying data-loader workers, CPU threads and batch size. Capture runtime, peak resident memory and CPU utilization for every run.

```sh
make setup
make datasets
make smoke
make plots
```

Use at least 17 JPG/PAGE XML pairs in `training_data/`. Two pages are reserved for validation; training subsets are nested and selected using seed 42. Data stays local and is ignored by Git.

Read [Running benchmarks](benchmarks.md) for controls, [Measurement methodology](methodology.md) for interpretation, and [Results](results.md) for the output format. [GitHub Pages](publishing.md) explains deployment.

The entire suite can also run in containers: see [Docker deployment](docker.md).
