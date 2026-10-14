"""Run a serial Cartesian matrix of PP-OCRv6 tiny training experiments."""
import argparse
import itertools
import json
import os
import random
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

from scripts.datasets import sha256
from scripts.monitor import measure


def command_for(ketos, dataset, validation, run, settings, args):
    pages, workers, threads, batch, precision, repeat = settings
    return [str(ketos), '--device', args.device, '--workers', str(workers),
            '--threads', str(threads), '--seed', str(args.seed), '--precision', precision,
            'train', '--arch', 'ppocrv6', '--variant', 'tiny', '--format-type', 'binary',
            '--training-data', str(dataset), '--evaluation-data', str(validation),
            '--quit', 'fixed', '--epochs', str(args.epochs), '--min-epochs', str(args.epochs),
            '--optimizer', args.optimizer, '--height', str(args.height), '--max-width', str(args.max_width),
            '--batch-size', str(batch), '--output', str(run / 'checkpoints'),
            '--log-dir', str(run), '--schedule', 'constant', '--warmup', '0',
            '--no-augment', '--accumulate-grad-batches', str(args.accumulate)]


def matrix(args):
    values = list(itertools.product(args.sizes, args.workers, args.threads, args.batch_sizes,
                                   args.precisions, range(1, args.repeats + 1)))
    random.Random(args.seed).shuffle(values)
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', type=Path, default=Path('artifacts/datasets'))
    p.add_argument('--output', type=Path, default=Path('artifacts/runs'))
    p.add_argument('--sizes', nargs='+', type=int, default=[5, 10, 15])
    p.add_argument('--workers', nargs='+', type=int, default=[0, 2, 4])
    p.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    p.add_argument('--batch-sizes', nargs='+', type=int, default=[4, 8, 16])
    p.add_argument('--precisions', nargs='+', default=['32-true'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--optimizer', choices=['Adam', 'AdamW', 'AdamW+Muon', 'SGD', 'RMSprop'], default='AdamW')
    p.add_argument('--torch-compile', choices=['enabled', 'disabled'], default='enabled')
    p.add_argument('--height', type=int, default=96)
    p.add_argument('--max-width', type=int, default=2560)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--accumulate', type=int, default=1)
    p.add_argument('--interval', type=float, default=0.2)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if (min(args.sizes + args.threads + args.batch_sizes + [args.epochs, args.repeats, args.accumulate, args.height, args.max_width]) < 1
            or min(args.workers) < 0 or args.interval <= 0):
        p.error('Sizes/threads/batches/epochs/repeats/accumulation/interval must be positive; workers >= 0')
    ketos = Path(sys.executable).parent / 'ketos'
    records = None
    if not args.dry_run:
        records = json.loads((args.datasets / 'datasets.json').read_text())['datasets']
        for name in [*map(str, args.sizes), 'validation']:
            record = records[name]
            manifest = args.datasets / f'{name}.txt'
            if manifest.read_text().splitlines() != [record['path']]:
                raise ValueError(f'Manifest changed: {manifest}; recompile before benchmarking')
            if sha256(record['path']) != record['sha256']:
                raise ValueError(f'Dataset changed: {name}; recompile before benchmarking')
        help_text = subprocess.check_output([str(ketos), 'train', '--help'], text=True)
        if '--arch' not in help_text or '--variant' not in help_text:
            raise RuntimeError('Installed Kraken does not support PP-OCRv6; run make setup')
    failures = 0
    for settings in matrix(args):
        pages, workers, threads, batch, precision, repeat = settings
        run = (args.output / f'p{pages}-w{workers}-t{threads}-b{batch}-r{repeat}-{uuid.uuid4().hex[:10]}').resolve()
        command = command_for(ketos, (args.datasets / f'{pages}.txt').resolve(),
                              (args.datasets / 'validation.txt').resolve(), run, settings, args)
        overrides = dict(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                         OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads),
                         TOKENIZERS_PARALLELISM='false',
                         TORCH_COMPILE_DISABLE='1' if args.torch_compile == 'disabled' else '0')
        print(shlex.join([f'{k}={v}' for k, v in overrides.items()]) + ' ' + shlex.join(command), flush=True)
        if args.dry_run:
            continue
        env = dict(os.environ, **overrides)
        result = measure(command, run, dict(kind='train', environment=overrides, pages=pages, workers=workers,
                         threads=threads, batch_size=batch, precision=precision, repeat=repeat,
                         epochs=args.epochs, optimizer=args.optimizer, torch_compile=args.torch_compile, height=args.height, max_width=args.max_width, seed=args.seed, device=args.device, accumulate=args.accumulate,
                         dataset_bytes=records[str(pages)]['bytes'],
                         dataset_lines=records[str(pages)].get('lines'),
                         dataset_sha256=records[str(pages)]['sha256'],
                         validation_sha256=records['validation']['sha256']), args.interval, env)
        failures += result['returncode'] != 0
        print(f'Exit {result["returncode"]}; {result["wall_seconds"]:.1f}s; log: {run / "process.log"}', flush=True)
    if failures:
        raise SystemExit(f'{failures} training runs failed; see their process.log files')


if __name__ == '__main__':
    main()
