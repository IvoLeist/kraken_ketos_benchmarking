"""Check privilege dropping in the actual Compose benchmark container."""
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get('BENCHEXEC_CGROUP_SETUP') != '1',
                    reason='Requires the Compose benchmark entrypoint')
def test_runtime_is_unprivileged():
    assert os.getuid() == int(os.environ['LOCAL_UID']) != 0
    assert os.getgid() == int(os.environ['LOCAL_GID']) != 0
    assert os.getgroups() == []
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    for field in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'):
        assert int(status[field].strip(), 16) == 0
    assert status['NoNewPrivs'].strip() == '1'
    assert Path('/sys/fs/cgroup/benchexec').stat().st_uid == os.getuid()
