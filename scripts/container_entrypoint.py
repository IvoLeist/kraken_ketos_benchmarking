#!/usr/local/bin/python3
"""Delegate the container's private cgroup hierarchy, then drop privileges."""
import os
from pathlib import Path
import sys


def main():
    command = ['tini', '-g', '--', *sys.argv[1:]]
    if os.environ.get('BENCHEXEC_CGROUP_SETUP') != '1':
        os.execvp(command[0], command)

    uid = int(os.environ.get('LOCAL_UID', '1000'))
    gid = int(os.environ.get('LOCAL_GID', '1000'))
    if uid <= 0 or gid <= 0:
        raise ValueError('LOCAL_UID and LOCAL_GID must be positive non-root IDs')
    if os.geteuid() != 0:
        raise ValueError('Cgroup setup must start as root; use the benchmark Compose service')
    # Never configure an ancestor of the container's own cgroup.
    if Path('/proc/self/cgroup').read_text().strip() != '0::/':
        raise ValueError('A private cgroups v2 namespace is required; use cgroup: private')
    root = Path('/sys/fs/cgroup')
    controllers = (root / 'cgroup.controllers').read_text().split()
    if not {'cpu', 'memory'} <= set(controllers):
        raise ValueError('Docker must delegate the cpu and memory controllers')

    # Keep Tini, Make and other wrapper processes out of BenchExec's subtree.
    init = root / 'init'
    delegated = root / 'benchexec'
    init.mkdir()
    delegated.mkdir()
    (init / 'cgroup.procs').write_text(str(os.getpid()))
    enabled = ' '.join('+' + controller for controller in controllers)
    (root / 'cgroup.subtree_control').write_text(enabled)
    (delegated / 'cgroup.subtree_control').write_text(enabled)
    # Migration needs write permission on the common ancestor's cgroup.procs.
    for path in [root / 'cgroup.procs', init / 'cgroup.procs', delegated,
                 *delegated.iterdir()]:
        os.chown(path, uid, gid)

    os.execvp('setpriv', ['setpriv', f'--reuid={uid}', f'--regid={gid}',
                         '--clear-groups', '--bounding-set=-all', '--inh-caps=-all',
                         '--ambient-caps=-all', '--no-new-privs', *command])


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError) as exc:
        sys.exit(f'Container cgroup setup failed: {exc}. See docs/docker.md.')
