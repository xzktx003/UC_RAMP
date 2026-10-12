from __future__ import annotations

from pathlib import Path

import pytest

from ramp_moe.humaneval_sandbox import (
    check_correctness_userns,
    install_evalscope_humaneval_patch,
    userns_sandbox_available,
)


pytestmark = pytest.mark.skipif(
    not userns_sandbox_available(), reason="Linux user namespaces are unavailable"
)


def _problem(
    test: str = "def check(candidate):\n    assert candidate(2) == 3",
    *,
    prompt: str = "def add_one(value):\n",
) -> dict:
    return {
        "task_id": "HumanEval/test",
        "prompt": prompt,
        "entry_point": "add_one",
        "test": test,
    }


def test_userns_checker_accepts_correct_completion() -> None:
    result = check_correctness_userns(
        _problem(), "    return value + 1\n", timeout=2.0
    )

    assert result["passed"] is True
    assert result["result"] == "passed"


def test_userns_checker_rejects_incorrect_completion() -> None:
    result = check_correctness_userns(
        _problem(), "    return value - 1\n", timeout=2.0
    )

    assert result["passed"] is False
    assert str(result["result"]).startswith("failed:")


def test_userns_checker_times_out_infinite_loop() -> None:
    result = check_correctness_userns(
        _problem(), "    while True:\n        pass\n", timeout=0.25
    )

    assert result["passed"] is False
    assert result["result"] == "timed out"


def test_userns_checker_hides_host_workspace() -> None:
    host_path = Path(__file__).resolve()
    problem = _problem(
        "def check(candidate):\n"
        f"    assert candidate({str(host_path)!r}) is False"
    )
    completion = "    import os\n    return os.path.exists(value)\n"

    result = check_correctness_userns(problem, completion, timeout=2.0)

    assert result["passed"] is True


def test_userns_checker_disables_network() -> None:
    problem = _problem(
        "def check(candidate):\n"
        "    assert candidate() != 0",
        prompt="def add_one():\n",
    )
    completion = (
        "    import socket\n"
        "    sock = socket.socket()\n"
        "    try:\n"
        "        return sock.connect_ex(('1.1.1.1', 80))\n"
        "    finally:\n"
        "        sock.close()\n"
    )

    result = check_correctness_userns(problem, completion, timeout=2.0)

    assert result["passed"] is True


def test_evalscope_patch_replaces_local_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    from evalscope.benchmarks.humaneval import utils

    original = utils.check_correctness
    monkeypatch.setattr(utils, "check_correctness", original)

    replaced = install_evalscope_humaneval_patch()

    assert replaced is original
    assert utils.check_correctness is check_correctness_userns
