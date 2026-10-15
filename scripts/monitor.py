"""Record run provenance; delegate execution and resource accounting to BenchExec."""
import hashlib
import json
import os
import platform
import signal
import sys
from importlib.metadata import version
from pathlib import Path

import psutil


def available_cpus():
    """Logical CPUs available through affinity (cgroup quotas may be narrower)."""
    return len(os.sched_getaffinity(0))


def create_executor():
    if sys.platform != 'linux':
        raise RuntimeError('BenchExec requires Linux with delegated cgroups; see docs/benchmarks.md')
    from benchexec.runexecutor import RunExecutor
    try:
        # Keep the existing filesystem/environment; cgroups still account for children.
        executor = RunExecutor(use_namespaces=False)
        cgroups = executor.cgroups
        if any(s not in cgroups for s in (cgroups.CPU, cgroups.MEMORY)):
            raise SystemExit('CPU or memory controller unavailable')
        return executor
    except SystemExit as exc:
        raise RuntimeError('BenchExec needs writable CPU/memory cgroups. '
                           'See docs/benchmarks.md for delegated-scope setup.') from exc


def measure(command, directory, metadata, env=None):
    executor = create_executor()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cpus = available_cpus()
    info = dict(metadata, command=command, platform=platform.platform(),
                python=platform.python_version(), available_cpus=cpus,
                kraken_version=version('kraken'), torch_version=version('torch'),
                measurement_backend='benchexec', benchexec_version=version('benchexec'),
                memory_measurement='cgroup',
                hostname=platform.node(), ram_total_bytes=psutil.virtual_memory().total,
                lock_sha256=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest() if Path('uv.lock').exists() else None)
    (directory / 'command.json').write_text(json.dumps(info, indent=2) + '\n')
    info.update(returncode=None, interrupted=False)

    def stop(signum, frame):
        info['interrupted'] = True
        executor.stop()

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        result = executor.execute_run(
            args=list(map(str, command)), output_filename=str(directory / 'process.log'),
            environments={} if env is None else {'keepEnv': {}, 'newEnv': env})
        # Preserve all original measurements, including termination reasons and I/O.
        (directory / 'benchexec.json').write_text(json.dumps(result, indent=2, default=str) + '\n')
        exitcode = result.get('exitcode')
        info.update(returncode=(exitcode.value if exitcode.value is not None else -exitcode.signal)
                    if exitcode is not None else None,
                    termination_reason=result.get('terminationreason'),
                    wall_seconds=result.get('walltime'), cpu_seconds=result.get('cputime'),
                    peak_memory_bytes=result.get('memory'))
        if any(info[k] is None for k in ('wall_seconds', 'cpu_seconds', 'peak_memory_bytes')):
            raise RuntimeError('BenchExec did not provide complete CPU/memory measurements; see benchexec.json')
        wall = info['wall_seconds']
        info.update(mean_cpu_cores=info['cpu_seconds'] / wall if wall else 0,
                    cpu_utilization_pct=100 * info['cpu_seconds'] / wall / cpus if wall else 0)
    except BaseException as exc:
        info['measurement_error'] = str(exc)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        (directory / 'summary.json').write_text(json.dumps(info, indent=2) + '\n')
    if info['interrupted']:
        raise KeyboardInterrupt('Benchmark interrupted; partial results saved')
    return info
