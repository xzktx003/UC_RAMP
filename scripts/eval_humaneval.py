#!/usr/bin/env python3
"""Run EvalScope HumanEval with the project user-namespace checker installed."""

from __future__ import annotations

import sys

from ramp_moe.humaneval_sandbox import install_evalscope_humaneval_patch


def main() -> None:
    if "--use-sandbox" in sys.argv:
        raise SystemExit(
            "Do not combine --use-sandbox with this wrapper; it uses the audited "
            "Linux user-namespace backend instead of EvalScope's Docker backend."
        )
    if "humaneval" not in sys.argv:
        raise SystemExit("This wrapper only runs EvalScope HumanEval evaluations.")

    install_evalscope_humaneval_patch()
    print(
        "HumanEval execution backend: linux user+mount+pid+net namespaces, "
        "read-only runtime, ephemeral tmpfs",
        file=sys.stderr,
        flush=True,
    )
    from evalscope.cli.cli import run_cmd

    run_cmd()


if __name__ == "__main__":
    main()
