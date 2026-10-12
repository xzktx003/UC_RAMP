"""Linux user-namespace isolation for EvalScope HumanEval checks.

EvalScope 1.4.1 executes HumanEval completions on the host when its optional
Docker sandbox is disabled.  This module supplies an API-compatible checker
that runs each completion in fresh user, mount, PID, and network namespaces.
Only a read-only system Python runtime is exposed inside an ephemeral tmpfs.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import math
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import subprocess
import tempfile
from typing import Any


_MAX_OUTPUT_BYTES = 64 * 1024
_MEMORY_BYTES = 512 * 1024 * 1024
_TMPFS_BYTES = 128 * 1024 * 1024

_NAMESPACE_SCRIPT = r"""
set -eu
root=$1
memory_bytes=$2
cpu_seconds=$3
tmpfs_bytes=$4

mount --make-rprivate /
mount -t tmpfs -o "size=${tmpfs_bytes},nosuid,nodev" tmpfs "$root"
mkdir -p "$root/usr" "$root/lib" "$root/lib64" "$root/tmp"
ln -s usr/bin "$root/bin"
ln -s usr/sbin "$root/sbin"

mount --rbind /usr "$root/usr"
mount -o remount,ro,bind "$root/usr"
mount --rbind /lib "$root/lib"
mount -o remount,ro,bind "$root/lib"
if [ -d /lib64 ]; then
    mount --rbind /lib64 "$root/lib64"
    mount -o remount,ro,bind "$root/lib64"
fi
mount -t tmpfs -o size=32m,nosuid,nodev tmpfs "$root/tmp"

exec /usr/bin/prlimit \
    "--as=${memory_bytes}" \
    "--cpu=${cpu_seconds}" \
    --fsize=1048576 \
    --nofile=64 \
    --nproc=32 \
    -- /usr/sbin/chroot "$root" \
    /usr/bin/env -i PATH=/usr/bin:/bin \
    /bin/sh -c 'cd /tmp && exec /usr/bin/python3 -I -S -B -'
"""


@lru_cache(maxsize=1)
def userns_sandbox_available() -> bool:
    """Return whether the required Linux namespace primitives are available."""

    if platform.system() != "Linux":
        return False
    required = ("unshare", "mount", "prlimit", "chroot")
    if any(shutil.which(command) is None for command in required):
        return False
    try:
        probe = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--mount", "true"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _read_capped(stream: Any, limit: int = _MAX_OUTPUT_BYTES) -> str:
    stream.seek(0)
    return stream.read(limit).decode("utf-8", errors="replace")


def _failure_summary(stderr: str, stdout: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    detail = lines[-1] if lines else "sandbox process exited without a pass marker"
    return f"failed: {detail[:1000]}"


def _run_isolated_python(code: str, timeout: float) -> tuple[bool, str]:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if not userns_sandbox_available():
        raise RuntimeError("Linux user-namespace sandbox is unavailable")

    pass_marker = f"__RAMP_HUMANEVAL_PASS_{secrets.token_hex(16)}__"
    program = f"{code.rstrip()}\nprint({pass_marker!r})\n"
    root = Path(tempfile.mkdtemp(prefix="ramp-humaneval-", dir="/tmp"))
    cpu_seconds = max(1, math.ceil(timeout))
    command = [
        "unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--net",
        "/bin/sh",
        "-c",
        _NAMESPACE_SCRIPT,
        "sh",
        str(root),
        str(_MEMORY_BYTES),
        str(cpu_seconds),
        str(_TMPFS_BYTES),
    ]

    timed_out = False
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                process.communicate(
                    input=program.encode("utf-8"), timeout=timeout + 0.5
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()

            stdout_text = _read_capped(stdout)
            stderr_text = _read_capped(stderr)
            if timed_out:
                return False, "timed out"
            if process.returncode == 0 and stdout_text.rstrip().endswith(pass_marker):
                return True, "passed"
            if process.returncode in (-signal.SIGXCPU, -signal.SIGKILL):
                return False, "timed out"
            return False, _failure_summary(stderr_text, stdout_text)
    finally:
        try:
            root.rmdir()
        except FileNotFoundError:
            pass


def check_correctness_userns(
    problem: Mapping[str, Any],
    completion: str,
    timeout: float,
    completion_id: int | None = None,
) -> dict[str, Any]:
    """EvalScope-compatible HumanEval checker using namespace isolation."""

    required = ("task_id", "prompt", "test", "entry_point")
    missing = [key for key in required if key not in problem]
    if missing:
        raise ValueError(f"HumanEval problem is missing fields: {missing}")
    check_program = (
        str(problem["prompt"])
        + completion
        + "\n"
        + str(problem["test"])
        + "\n"
        + f"check({problem['entry_point']})"
    )
    passed, result = _run_isolated_python(check_program, timeout)
    return {
        "task_id": problem["task_id"],
        "passed": passed,
        "result": result,
        "completion_id": completion_id,
        "execution_backend": "linux_userns_mount_pid_net_tmpfs",
    }


def install_evalscope_humaneval_patch() -> Any:
    """Install the isolated checker and return EvalScope's previous checker."""

    if not userns_sandbox_available():
        raise RuntimeError("Linux user-namespace sandbox is unavailable")
    from evalscope.benchmarks.humaneval import utils

    previous = utils.check_correctness
    utils.check_correctness = check_correctness_userns
    return previous


__all__ = [
    "check_correctness_userns",
    "install_evalscope_humaneval_patch",
    "userns_sandbox_available",
]
