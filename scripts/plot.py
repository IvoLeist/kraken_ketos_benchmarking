"""Export benchmark summaries and plots without pooling unlike configurations."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METRICS = [('peak_memory_bytes', 'Peak memory (MiB; accounting shown in legend)', 2**20),
           ('wall_seconds', 'Wall time (s)', 1),
           ('cpu_utilization_pct', 'Available CPU utilization (%)', 1)]


def load_results(root):
    rows = []
    for path in sorted(Path(root).rglob('summary.json')):
        row = json.loads(path.read_text())
        row['run_directory'] = str(path.parent)
        rows.append(row)
    return rows


def plot_training(rows, output):
    # Legacy sampled RSS and cgroup memory have different semantics: never pool them.
    rows = [dict(r, peak_memory_bytes=r.get('peak_memory_bytes', r.get('peak_rss_bytes')),
                 memory_measurement=r.get('memory_measurement', 'sampled RSS')) for r in rows]
    groups = defaultdict(list)
    fields = ('workers', 'threads', 'batch_size', 'precision', 'device', 'epochs',
              'accumulate', 'seed', 'validation_sha256', 'platform', 'kraken_version', 'available_cpus', 'hostname', 'lock_sha256', 'optimizer', 'torch_compile', 'height', 'max_width')
    for row in rows:
        groups[tuple(row[x] for x in fields) + (row['memory_measurement'],)].append(row)
    for key, ylabel, divisor in METRICS:
        fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
        for config, records in groups.items():
            by_size = defaultdict(list)
            for r in records:
                by_size[(r['dataset_bytes'], r['dataset_sha256'])].append(r[key] / divisor)
            ordered = sorted(by_size)
            ax.errorbar([x[0] / 2**20 for x in ordered], [mean(by_size[x]) for x in ordered],
                        yerr=[stdev(by_size[x]) if len(by_size[x]) > 1 else 0 for x in ordered],
                        marker='o', capsize=3,
                        label=f'w={config[0]}, t={config[1]}, b={config[2]}, {config[3]}, {config[4]}, {config[-5]}, compile={config[-4]}, shape={config[-3]}×{config[-2]}, {config[-1]}')
        ax.set(xlabel='Compiled training dataset (MiB)', ylabel=ylabel,
               title='PP-OCRv6 tiny • mean ± sample standard deviation')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        for ext in ('png', 'svg'):
            fig.savefig(output / f'dataset-vs-{key}.{ext}', dpi=160)
        plt.close(fig)
    # Separate setting comparisons by dataset and all fixed experimental conditions.
    panels = defaultdict(list)
    for r in rows:
        key = tuple(r[x] for x in ('pages', 'dataset_sha256', 'validation_sha256', 'device',
                                  'precision', 'epochs', 'accumulate', 'seed', 'platform',
                                  'kraken_version', 'available_cpus', 'hostname', 'lock_sha256', 'optimizer', 'torch_compile', 'height', 'max_width', 'memory_measurement'))
        panels[key].append(r)
    for index, (conditions, records) in enumerate(panels.items()):
        configurations = defaultdict(list)
        for r in records:
            configurations[(r['workers'], r['threads'], r['batch_size'])].append(r)
        keys = sorted(configurations)
        fig, axes = plt.subplots(3, 1, figsize=(max(10, len(keys) * .45), 10), layout='constrained')
        for ax, (metric, label, divisor) in zip(axes, METRICS):
            values = [[r[metric] / divisor for r in configurations[k]] for k in keys]
            ax.bar(range(len(keys)), [mean(v) for v in values],
                   yerr=[stdev(v) if len(v) > 1 else 0 for v in values], capsize=3)
            ax.set(ylabel=label)
            ax.set_xticks(range(len(keys)), [f'{w}/{t}/{b}' for w, t, b in keys], rotation=60)
            ax.grid(axis='y', alpha=.2)
        axes[-1].set_xlabel('Workers / threads / batch size')
        fig.suptitle(f'{conditions[0]} training pages • {conditions[3]} • {conditions[4]} • {conditions[5]} epochs • {conditions[-3]}×{conditions[-2]} px • {conditions[-1]}')
        for ext in ('png', 'svg'):
            fig.savefig(output / f'settings-{conditions[0]}pages-{index}.{ext}', dpi=160)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('artifacts/runs'))
    p.add_argument('--datasets', type=Path, default=Path('artifacts/datasets'))
    p.add_argument('--output', type=Path, default=Path('artifacts/plots'))
    args = p.parse_args()
    rows = load_results(args.input)
    if not rows:
        p.error('No benchmark summaries found; run make benchmark first')
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (args.output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    successful = [r for r in rows if r.get('returncode') == 0 and not r.get('interrupted') and not r.get('measurement_error') and not r.get('termination_reason') and r.get('kind') == 'train']
    print(f'{len(successful)} successful training runs; {len(rows)-len(successful)} excluded from plots')
    if not successful:
        p.error('No successful training runs; results.csv contains failures for inspection')
    plot_training(successful, args.output)
    meta_path = args.datasets / 'datasets.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())['datasets']
        data = [(r, json.loads((Path(r['compile_run']) / 'summary.json').read_text()))
                for name, r in meta.items() if name != 'validation']
        data.sort(key=lambda pair: pair[0]['pages'])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for ax, (metric, label, divisor) in zip(axes, METRICS):
            ax.plot([r['bytes'] / 2**20 for r, _ in data], [(s.get('peak_memory_bytes', s.get('peak_rss_bytes')) if metric == 'peak_memory_bytes' else s[metric]) / divisor for _, s in data], 'o-')
            ax.set(xlabel='Compiled dataset (MiB)', ylabel=label)
            ax.grid(alpha=.2)
        fig.suptitle('Dataset compilation resources • ' + ', '.join(sorted({s.get('memory_measurement', 'sampled RSS') for _, s in data})))
        for ext in ('png', 'svg'):
            fig.savefig(args.output / f'compilation.{ext}', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
