"""Adapter tests need no cgroup privileges; real execution is opt-in."""
import json
import os
import signal
import sys

import pytest
from benchexec.util import ProcessExitCode

from scripts import monitor


@pytest.mark.parametrize('exitcode,expected', [
    (ProcessExitCode.create(value=0), 0),
    (ProcessExitCode.create(value=7), 7),
    (ProcessExitCode.create(signal=9), -9),
])
def test_measure_results(tmp_path, monkeypatch, exitcode, expected):
    class Executor:
        def execute_run(self, **kwargs):
            assert kwargs['environments'] == {'keepEnv': {}, 'newEnv': {'TEST': 'value'}}
            assert kwargs['args'] == ['tool', 'a file']
            return dict(exitcode=exitcode, walltime=2.0, cputime=3.0, memory=1234)

    monkeypatch.setattr(monitor, 'create_executor', Executor)
    monkeypatch.setattr(monitor, 'available_cpus', lambda: 4)
    run = tmp_path / 'run'
    result = monitor.measure(['tool', 'a file'], run, {'kind': 'test'}, env={'TEST': 'value'})
    assert result['returncode'] == expected
    assert result['peak_memory_bytes'] == 1234
    assert result['cpu_utilization_pct'] == 37.5
    assert 'peak_rss_bytes' not in result
    assert json.loads((run / 'summary.json').read_text()) == result
    assert (run / 'benchexec.json').is_file()
    assert not (run / 'samples.csv').exists()


def test_interrupt_stops_executor_and_restores_handlers(tmp_path, monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    class Executor:
        stopped = False
        def stop(self):
            self.stopped = True
        def execute_run(self, **kwargs):
            signal.raise_signal(signal.SIGTERM)
            assert self.stopped
            return dict(exitcode=ProcessExitCode.create(signal=9), walltime=1.,
                        cputime=.2, memory=42, terminationreason='killed')
    monkeypatch.setattr(monitor, 'create_executor', Executor)
    run = tmp_path / 'run'
    with pytest.raises(KeyboardInterrupt):
        monitor.measure(['tool'], run, {})
    assert signal.getsignal(signal.SIGTERM) == previous
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['interrupted'] and summary['termination_reason'] == 'killed'


def test_missing_measurements_are_not_successful(tmp_path, monkeypatch):
    class Executor:
        def execute_run(self, **kwargs):
            return dict(exitcode=ProcessExitCode.create(value=0), walltime=1., cputime=.2)
    monkeypatch.setattr(monitor, 'create_executor', Executor)
    run = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='complete'):
        monitor.measure(['tool'], run, {})
    assert json.loads((run / 'summary.json').read_text())['measurement_error']


def test_cgroups_required(monkeypatch):
    import benchexec.runexecutor
    def unavailable(**kwargs):
        assert kwargs['use_namespaces'] is False
        raise SystemExit('unavailable')
    monkeypatch.setattr(benchexec.runexecutor, 'RunExecutor', unavailable)
    with pytest.raises(RuntimeError, match='writable CPU/memory cgroups'):
        monitor.create_executor()


@pytest.mark.skipif(os.environ.get('BENCHEXEC_INTEGRATION') != '1',
                    reason='Set BENCHEXEC_INTEGRATION=1 inside a delegated cgroup scope')
def test_real_worker_accounting(tmp_path):
    code = ('import subprocess,sys; '
            'subprocess.run([sys.executable,"-c",'
            '"import time; x=bytearray(32000000); time.sleep(.2)"], check=True); '
            'raise SystemExit(7)')
    result = monitor.measure([sys.executable, '-c', code], tmp_path / 'run', {})
    assert result['returncode'] == 7
    assert result['peak_memory_bytes'] > 32_000_000
    assert result['wall_seconds'] >= .2
    assert result['cpu_seconds'] > 0
