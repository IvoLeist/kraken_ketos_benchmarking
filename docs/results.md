# Results

Results are generated from your local dataset and machine. No example numbers are presented as benchmark findings.

```sh
make plots
```

The output in `artifacts/plots/` includes a CSV of every run, plots of compiled dataset size against runtime, RAM and CPU utilization, and per-dataset comparisons of workers/threads/batch size. A separate chart covers dataset compilation. PNG and SVG formats are generated.

Repeated equivalent runs are summarized by mean and sample standard deviation. Failed and interrupted runs remain in the CSV but are omitted from charts. Per-run `samples.csv` retains the time series for more detailed analysis. Every run has its own log, command record and summary.

## Publish selected results

Copy reviewed PNG or SVG plots to `docs/assets/` and reference them from this page. Include your hardware, operating system, software versions, dataset provenance, settings, repeat count and date. Keep raw data, model weights and paths containing private information out of the published site.

GitHub Pages builds the documentation only. It does not train models or automatically copy ignored artifacts.
