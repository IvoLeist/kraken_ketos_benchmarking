import argparse
import json
import sys
from pathlib import Path

import pytest

from scripts.benchmark import command_for, matrix
from scripts.datasets import discover, select
from scripts.monitor import measure


def test_subsets_are_nested_and_disjoint():
    pairs = list(range(20))
    subsets, validation = select(pairs, [5, 10, 15], 2, 42)
    assert subsets['10'][:5] == subsets['5']
    assert subsets['15'][:10] == subsets['10']
    assert not set(subsets['15']) & set(validation)
    assert (subsets, validation) == select(pairs, [5, 10, 15], 2, 42)
    with pytest.raises(ValueError, match='Need 17'):
        select(pairs[:16], [5, 10, 15], 2, 42)


def test_discovery_uses_page_image_reference(tmp_path):
    (tmp_path / 'actual image.jpg').write_bytes(b'image')
    xml = tmp_path / 'different.page.xml'
    xml.write_text('<PcGts><Page imageFilename="actual image.jpg"><TextLine><TextEquiv><Unicode>text</Unicode></TextEquiv></TextLine></Page></PcGts>')
    assert discover(tmp_path) == [(xml, tmp_path / 'actual image.jpg')]
    (tmp_path / 'actual image.jpg').unlink()
    with pytest.raises(ValueError, match='Missing JPG'):
        discover(tmp_path)


def test_matrix_and_fixed_epoch_command(tmp_path):
    args = argparse.Namespace(sizes=[5, 10, 15], workers=[0, 2, 4], threads=[1, 2, 4],
                              batch_sizes=[4, 8, 16], precisions=['32-true'], repeats=1,
                              seed=42, device='cpu', epochs=3, accumulate=1, height=96, max_width=2560, optimizer='AdamW')
    runs = matrix(args)
    assert len(runs) == len(set(runs)) == 81
    assert runs == matrix(args)
    cmd = command_for('ketos', tmp_path / '5.txt', tmp_path / 'validation.txt', tmp_path, runs[0], args)
    assert cmd.index('--workers') < cmd.index('train')
    for option, value in [('--arch', 'ppocrv6'), ('--variant', 'tiny'), ('--quit', 'fixed'),
                          ('--epochs', '3'), ('--min-epochs', '3')]:
        assert cmd[cmd.index(option) + 1] == value
    assert '--evaluation-data' in cmd


def test_monitor_records_success_and_failure(tmp_path):
    for code in (0, 7):
        run = tmp_path / str(code)
        result = measure([sys.executable, '-c', f'import time; x=bytearray(1000000); time.sleep(.15); raise SystemExit({code})'],
                         run, {'kind': 'test'}, interval=.02)
        assert result['returncode'] == code
        assert result['wall_seconds'] >= .15
        assert result['peak_rss_bytes'] > 0
        assert json.loads((run / 'summary.json').read_text())['returncode'] == code
        assert (run / 'samples.csv').read_text().count('\n') > 2


def test_monitor_includes_worker_memory(tmp_path):
    code = ('import subprocess,sys,time; '
            'p=subprocess.Popen([sys.executable,"-c",'
            '"import time; x=bytearray(32000000); time.sleep(.4)"]); '
            'p.wait()')
    result = measure([sys.executable, '-c', code], tmp_path / 'tree', {'kind': 'test'}, interval=.02)
    assert result['returncode'] == 0
    assert result['peak_rss_bytes'] > 32_000_000
    import csv
    with (tmp_path / 'tree' / 'samples.csv').open() as f:
        assert max(int(r['processes']) for r in csv.DictReader(f)) >= 2


def test_plot_export_keeps_failed_runs(tmp_path, monkeypatch):
    from scripts import plot
    run = tmp_path / 'runs' / 'failed'
    run.mkdir(parents=True)
    (run / 'summary.json').write_text(json.dumps({'kind': 'train', 'returncode': 7}))
    output = tmp_path / 'plots'
    monkeypatch.setattr(sys, 'argv', ['plot', '--input', str(tmp_path / 'runs'), '--output', str(output)])
    with pytest.raises(SystemExit):
        plot.main()
    assert 'returncode' in (output / 'results.csv').read_text()
    assert not list(output.glob('*.png'))


def test_plots_for_successful_repeats(tmp_path):
    from scripts.plot import plot_training
    base = dict(workers=0, threads=1, batch_size=4, precision='32-true', device='cpu',
                epochs=3, accumulate=1, seed=42, validation_sha256='validation',
                platform='test', kraken_version='test', available_cpus=2,
                hostname='test', lock_sha256='lock', torch_compile='disabled', optimizer='AdamW', height=96, max_width=2560, pages=5, dataset_bytes=1024,
                dataset_sha256='training', peak_rss_bytes=1024**2,
                wall_seconds=1.0, cpu_utilization_pct=50.0)
    plot_training([base, dict(base, wall_seconds=2.0)], tmp_path)
    assert len(list(tmp_path.glob('*.png'))) == 4
    assert len(list(tmp_path.glob('*.svg'))) == 4


def test_sigterm_preserves_interrupted_summary(tmp_path):
    import subprocess
    import time

    run = tmp_path / 'terminated'
    code = ('from scripts.monitor import measure; import sys; '
            f'measure([sys.executable,"-c","import time; time.sleep(60)"], '
            f'{str(run)!r}, {{"kind":"test"}}, interval=.02)')
    process = subprocess.Popen([sys.executable, '-c', code],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            samples = run / 'samples.csv'
            if samples.exists() and len(samples.read_text().splitlines()) > 1:
                break
            if process.poll() is not None:
                pytest.fail('Monitor exited before receiving SIGTERM')
            time.sleep(.02)
        else:
            pytest.fail('Monitor did not start sampling')
        process.terminate()
        process.wait(timeout=15)
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['interrupted'] is True
        assert summary['returncode'] is not None
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


@pytest.mark.parametrize('mode,expected', [(None, '1'), ('enabled', '0')])
def test_training_compilation_environment(monkeypatch, capsys, mode, expected):
    from scripts import benchmark

    for flag in ('TORCHDYNAMO_DISABLE', 'TORCH_COMPILE_DISABLE'):
        monkeypatch.setenv(flag, '0' if expected == '1' else '1')
    argv = ['benchmark', '--dry-run', '--sizes', '5', '--workers', '0',
            '--threads', '1', '--batch-sizes', '4']
    if mode:
        argv += ['--torch-compile', mode]
    monkeypatch.setattr(sys, 'argv', argv)
    benchmark.main()
    command = capsys.readouterr().out
    assert f'TORCHDYNAMO_DISABLE={expected}' in command
    assert f'TORCH_COMPILE_DISABLE={expected}' in command
