"""Measure a subprocess tree, preserving logs and sampled resource usage."""
import csv
from contextlib import contextmanager
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import psutil


def available_cpus():
    """Logical CPUs available through affinity (cgroup quotas may be narrower)."""
    try:
        return len(psutil.Process().cpu_affinity())
    except AttributeError:
        return os.cpu_count() or 1


def _stop_requested(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


@contextmanager
def interrupt_on_termination():
    """Let Docker stop write a summary and shut down the measured process tree."""
    previous = signal.signal(signal.SIGTERM, _stop_requested)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def measure(command, directory, metadata, interval=0.2, env=None):
    with interrupt_on_termination():
        return _measure(command, directory, metadata, interval, env)


def _measure(command, directory, metadata, interval=0.2, env=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    cpus = available_cpus()
    info = dict(metadata, command=command, platform=platform.platform(),
                python=platform.python_version(), available_cpus=cpus,
                kraken_version=version('kraken'), torch_version=version('torch'),
                hostname=platform.node(), ram_total_bytes=psutil.virtual_memory().total,
                sample_interval_s=interval,
                lock_sha256=hashlib.sha256(Path('uv.lock').read_bytes()).hexdigest() if Path('uv.lock').exists() else None)
    (directory / 'command.json').write_text(json.dumps(info, indent=2) + '\n')
    seen_cpu = {}
    peak_rss = 0
    start = time.monotonic()
    interrupted = False
    with (directory / 'process.log').open('w') as log, (directory / 'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['elapsed_s', 'rss_bytes', 'cpu_seconds', 'processes'])
        writer.writeheader()
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        root = psutil.Process(proc.pid)
        try:
            while True:
                rss = 0
                count = 0
                try:
                    processes = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                for child in processes:
                    try:
                        with child.oneshot():
                            key = (child.pid, child.create_time())
                            cpu = child.cpu_times()
                            seen_cpu[key] = max(seen_cpu.get(key, 0), cpu.user + cpu.system)
                            rss += child.memory_info().rss
                            count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                peak_rss = max(peak_rss, rss)
                writer.writerow(dict(elapsed_s=time.monotonic() - start, rss_bytes=rss,
                                     cpu_seconds=sum(seen_cpu.values()), processes=count))
                f.flush()
                if proc.poll() is not None:
                    break
                time.sleep(interval)
        except BaseException:
            interrupted = True
            # A process-group stop can arrive again while the child is exiting.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
        finally:
            elapsed = time.monotonic() - start
            cpu_seconds = sum(seen_cpu.values())
            info.update(returncode=proc.poll(), interrupted=interrupted,
                        wall_seconds=elapsed, peak_rss_bytes=peak_rss,
                        cpu_seconds=cpu_seconds, mean_cpu_cores=cpu_seconds / elapsed,
                        cpu_utilization_pct=100 * cpu_seconds / elapsed / cpus)
            (directory / 'summary.json').write_text(json.dumps(info, indent=2) + '\n')
    return info
