#!/usr/bin/env python3
"""Reproduce UC-RAMP masks and downstream accuracy with local artifacts only.

The runner intentionally wraps the same primitives used by the paper experiments:

1. CAPE Stage-I: ``review_stage1_envelope`` with four-domain 448x2048 support.
2. Strict RAMP Stage-II: ``review_stage1_ramp`` initialized from the frozen
   Stage-I checkpoint and constrained to preserve every expert width.
3. WikiText-2 PPL, model export/verification, and EvalScope ten-task accuracy.

It does not download models or datasets.  All model/dataset paths are supplied by
the caller, and the default GPU gate is limited to GPU ids 0--3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Iterable
from urllib.error import URLError
from urllib.request import urlopen


SEED = 1234
FORMAL_NSAMPLES = 448
FORMAL_SEQUENCE_LENGTH = 2048
MODEL_NAMES = {
    "qwen3": "local/Qwen3-30B-A3B-Instruct-2507",
    "gemma4": "local/gemma-4-26B-A4B-it",
    "qwen36": "local/Qwen3.6-35B-A3B",
}
MODEL_LABELS = {
    "qwen3": "Qwen3-30B-A3B-Instruct-2507",
    "gemma4": "Gemma4-26B-A4B-it",
    "qwen36": "Qwen3.6-35B-A3B",
}
PPL_MAX = {
    "qwen3": 20.0,
    "gemma4": 300.0,
    "qwen36": 20.0,
}
PPL_MIN_WINDOWS = {
    "qwen3": 145,
    "gemma4": 140,
    "qwen36": 145,
}
TASK_GROUPS = (
    ("knowledge", 32, "knowledge", ("hellaswag", "winogrande", "race", "gpqa_diamond")),
    ("general", 8, "general", ("piqa", "arc", "gsm8k")),
    ("math500", 8, "general", ("math_500",)),
)
DATASET_ARGS = {
    "knowledge": {
        "hellaswag": {"shuffle": False},
        "winogrande": {"shuffle": False},
        "race": {"few_shot_num": 3, "shuffle": False},
        "gpqa_diamond": {"shuffle": False},
    },
    "general": {
        "piqa": {"shuffle": False},
        "arc": {"shuffle": False},
        "gsm8k": {"few_shot_num": 4, "shuffle": False},
    },
    "math500": {"math_500": {"shuffle": False}},
    "humaneval": {"humaneval": {"shuffle": False}},
}


@dataclass(frozen=True)
class CommandStep:
    name: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]
    log: Path | None = None
    exit_marker: Path | None = None


@dataclass(frozen=True)
class Layout:
    label: str
    retained_endpoint: int
    checkpoints: Path
    results: Path
    logs: Path
    exports: Path
    eval_root: Path

    @property
    def cape_checkpoint(self) -> Path:
        return self.checkpoints / f"{self.label}_cape.pt"

    @property
    def cape_result(self) -> Path:
        return self.results / f"{self.label}_cape_maskgen.json"

    @property
    def ucramp_checkpoint(self) -> Path:
        return self.checkpoints / f"{self.label}_ucramp.pt"

    @property
    def ucramp_result(self) -> Path:
        return self.results / f"{self.label}_ucramp_maskgen.json"

    @property
    def ppl_result(self) -> Path:
        return self.results / f"{self.label}_wikitext2.json"

    @property
    def export_dir(self) -> Path:
        return self.exports / self.label

    @property
    def export_result(self) -> Path:
        return self.results / f"{self.label}_export.json"

    @property
    def export_verify_result(self) -> Path:
        return self.results / f"{self.label}_export_verification.json"

    @property
    def run_label(self) -> str:
        return f"{self.label}_evalscope"

    @property
    def eval_result_root(self) -> Path:
        return self.eval_root / self.run_label

    @property
    def eval_exit(self) -> Path:
        return self.logs / f"{self.run_label}.exit"

    @property
    def math_pathology(self) -> Path:
        return self.results / f"{self.label}_math500_generation_pathology.json"

    @property
    def summary(self) -> Path:
        return self.results / f"{self.label}_evalscope_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen UC-RAMP reproduction protocol.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("experiments"))
    parser.add_argument("--arch", choices=("qwen3", "gemma4", "qwen36"), required=True)
    parser.add_argument("--pruning-rate", type=float, choices=(0.25, 0.50), required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--competition-math-source", type=Path)
    parser.add_argument("--competition-math-exclusion-manifest", type=Path)
    parser.add_argument("--general-dataset-dir", type=Path)
    parser.add_argument("--knowledge-dataset-dir", type=Path)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--evalscope-bin", default="evalscope")
    parser.add_argument("--gpu", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--use-reference-mask",
        action="store_true",
        help=(
            "Import the bundled reference mask checkpoint instead of rebuilding "
            "CAPE/RAMP. Requires --reference-manifest and --reference-run."
        ),
    )
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-run")
    parser.add_argument(
        "--phase",
        action="append",
        choices=("all", "mask", "stage1", "stage2", "ppl", "export", "verify", "eval", "math-diagnostics", "summarize"),
        help="Repeat to run a subset. Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.gpu_memory_utilization is not None:
        try:
            gpu_memory = float(args.gpu_memory_utilization)
        except ValueError:
            parser.error("--gpu-memory-utilization must be a finite decimal in (0, 1)")
        if not math.isfinite(gpu_memory) or not 0 < gpu_memory < 1:
            parser.error("--gpu-memory-utilization must be a finite decimal in (0, 1)")
    if args.use_reference_mask and (
        args.reference_manifest is None or args.reference_run is None
    ):
        parser.error("--use-reference-mask requires --reference-manifest and --reference-run")
    return args


def retained_endpoint(pruning_rate: float) -> int:
    return int(round((1.0 - pruning_rate) * 100))


def normalize_paths_and_defaults(args: argparse.Namespace, phases: set[str]) -> None:
    args.repo_root = args.repo_root.resolve()
    if not args.work_root.is_absolute():
        args.work_root = args.repo_root / args.work_root
    if args.reference_manifest is not None and not args.reference_manifest.is_absolute():
        args.reference_manifest = args.repo_root / args.reference_manifest
    if args.manifest is not None and not args.manifest.is_absolute():
        args.manifest = args.repo_root / args.manifest
    if args.gpu_memory_utilization is None:
        args.gpu_memory_utilization = "0.78" if args.arch == "gemma4" else "0.90"
    apply_reference_defaults(args, phases)


def selected_phases(raw: list[str] | None) -> set[str]:
    phases = set(raw or ["all"])
    if "all" in phases:
        return {"stage1", "stage2", "ppl", "export", "verify", "eval", "math-diagnostics", "summarize"}
    if "mask" in phases:
        phases.remove("mask")
        phases.update({"stage1", "stage2"})
    return phases


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
        }
    )
    pythonpath_parts = [str(args.repo_root), str(args.repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def no_proxy_env(env: dict[str, str]) -> dict[str, str]:
    clean = dict(env)
    for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        clean.pop(key, None)
    clean["NO_PROXY"] = "127.0.0.1,localhost"
    return clean


def build_layout(args: argparse.Namespace) -> Layout:
    endpoint = retained_endpoint(args.pruning_rate)
    label = args.label or f"uc_ramp_{args.arch}_exact{endpoint}"
    if not all(ch.isalnum() or ch in "._-" for ch in label):
        raise SystemExit("--label must contain only letters, digits, '.', '_' and '-'")
    root = args.work_root if args.work_root.is_absolute() else args.repo_root / args.work_root
    return Layout(
        label=label,
        retained_endpoint=endpoint,
        checkpoints=root / "checkpoints",
        results=root / "results",
        logs=root / "logs",
        exports=root / "exported_models",
        eval_root=root / "evalscope" / "formal",
    )


def require_paths(args: argparse.Namespace, phases: set[str]) -> None:
    needs_mask_rebuild = bool({"stage1", "stage2"} & phases) and not args.use_reference_mask
    if args.dry_run:
        return
    required: list[tuple[Path, str]] = [
        (args.repo_root / "scripts" / "prune_universal_moe.py", "pruning script"),
        (args.repo_root / "scripts" / "eval_wikitext.py", "PPL script"),
        (args.repo_root / "scripts" / "export_masked_model.py", "export script"),
        (args.repo_root / "scripts" / "verify_masked_model_export.py", "export verifier"),
        (args.repo_root / "scripts" / "eval_humaneval.py", "HumanEval wrapper"),
        (args.repo_root / "scripts" / "analyze_math_generation.py", "MATH diagnostic script"),
        (args.repo_root / "scripts" / "summarize_evalscope_full.py", "EvalScope summary script"),
        (args.model_source, "local model"),
    ]
    if needs_mask_rebuild:
        if args.competition_math_source is None or args.competition_math_exclusion_manifest is None:
            raise SystemExit(
                "--competition-math-source and --competition-math-exclusion-manifest "
                "are required when rebuilding CAPE/RAMP masks"
            )
        required.extend(
            [
                (args.competition_math_source, "competition MATH parquet"),
                (
                    args.competition_math_exclusion_manifest,
                    "competition MATH exclusion manifest",
                ),
            ]
        )
    for path, description in required:
        if not path.exists():
            raise SystemExit(f"missing {description}: {path}")
    if args.use_reference_mask:
        assert args.reference_manifest is not None
        if not args.reference_manifest.exists():
            raise SystemExit(f"missing reference manifest: {args.reference_manifest}")
        if not (args.repo_root / "tools" / "mask_artifact.py").exists():
            raise SystemExit("missing tools/mask_artifact.py required by --use-reference-mask")
    if "eval" in phases:
        for binary, description in (
            (args.vllm_bin, "vLLM executable"),
            (args.evalscope_bin, "EvalScope executable"),
        ):
            if shutil.which(binary) is None and not Path(binary).exists():
                raise SystemExit(f"missing {description}: {binary}")
        if args.general_dataset_dir is None or args.knowledge_dataset_dir is None:
            raise SystemExit("--general-dataset-dir and --knowledge-dataset-dir are required for --phase eval")
        for path, description in (
            (args.general_dataset_dir, "general EvalScope dataset directory"),
            (args.knowledge_dataset_dir, "knowledge EvalScope dataset directory"),
        ):
            if not path.exists():
                raise SystemExit(f"missing {description}: {path}")


def refuse_overwrite(paths: Iterable[Path], *, dry_run: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not dry_run:
        joined = "\n  ".join(str(path) for path in existing)
        raise SystemExit(f"refusing to overwrite existing artifacts:\n  {joined}")


def step(
    name: str,
    argv: Iterable[object],
    *,
    args: argparse.Namespace,
    log: Path | None,
    exit_marker: Path | None,
    extra_env: dict[str, str] | None = None,
) -> CommandStep:
    env = base_env(args)
    if extra_env:
        env.update(extra_env)
    return CommandStep(
        name=name,
        argv=tuple(str(item) for item in argv),
        cwd=args.repo_root,
        env=env,
        log=log,
        exit_marker=exit_marker,
    )


def build_steps(args: argparse.Namespace, layout: Layout, phases: set[str]) -> list[CommandStep]:
    model_name = args.model_name or MODEL_NAMES[args.arch]
    model_revision = args.model_revision or "local-20260826-review-v1"
    if {"stage1", "stage2"} & phases and not args.use_reference_mask:
        if args.competition_math_source is None or args.competition_math_exclusion_manifest is None:
            raise SystemExit(
                "--competition-math-source and --competition-math-exclusion-manifest "
                "are required when rebuilding CAPE/RAMP masks"
            )
    common_prune = (
        args.python_bin,
        "scripts/prune_universal_moe.py",
        "--pruning-rate",
        args.pruning_rate,
        "--model-source",
        args.model_source,
        "--model-name",
        model_name,
        "--model-revision",
        model_revision,
        "--competition-math-source",
        args.competition_math_source,
        "--competition-math-exclusion-manifest",
        args.competition_math_exclusion_manifest,
        "--chunk-size",
        1,
    )
    steps: list[CommandStep] = []
    skip_mask_build = bool(args.use_reference_mask)
    if "stage1" in phases and not skip_mask_build:
        steps.append(
            step(
                "stage1_cape_mask",
                (
                    *common_prune,
                    "--profile",
                    "review_stage1_envelope",
                    "--checkpoint",
                    layout.cape_checkpoint,
                    "--output",
                    layout.cape_result,
                ),
                args=args,
                log=layout.logs / f"{layout.label}_stage1_cape_gpu{args.gpu}.log",
                exit_marker=layout.logs / f"{layout.label}_stage1_cape.exit",
            )
        )
    if "stage2" in phases and not skip_mask_build:
        steps.append(
            step(
                "stage2_strict_ramp",
                (
                    *common_prune,
                    "--profile",
                    "review_stage1_ramp",
                    "--fixed-stage1-checkpoint",
                    layout.cape_checkpoint,
                    "--checkpoint",
                    layout.ucramp_checkpoint,
                    "--output",
                    layout.ucramp_result,
                ),
                args=args,
                log=layout.logs / f"{layout.label}_stage2_ramp_gpu{args.gpu}.log",
                exit_marker=layout.logs / f"{layout.label}_stage2_ramp.exit",
            )
        )
    if "ppl" in phases:
        steps.append(
            step(
                "wikitext2_ppl",
                (
                    args.python_bin,
                    "scripts/eval_wikitext.py",
                    "--model",
                    model_name,
                    "--model-source",
                    args.model_source,
                    "--tokenizer-source",
                    args.model_source,
                    "--revision",
                    model_revision,
                    "--mask-checkpoint",
                    layout.ucramp_checkpoint,
                    "--output",
                    layout.ppl_result,
                    "--local-files-only",
                ),
                args=args,
                log=layout.logs / f"{layout.label}_ppl_gpu{args.gpu}.log",
                exit_marker=layout.logs / f"{layout.label}_ppl.exit",
            )
        )
    if "export" in phases:
        steps.append(
            step(
                "export_masked_model",
                (
                    args.python_bin,
                    "scripts/export_masked_model.py",
                    "--model",
                    model_name,
                    "--revision",
                    model_revision,
                    "--model-source",
                    args.model_source,
                    "--mask-checkpoint",
                    layout.ucramp_checkpoint,
                    "--output-dir",
                    layout.export_dir,
                    "--result-output",
                    layout.export_result,
                ),
                args=args,
                log=layout.logs / f"{layout.label}_export.log",
                exit_marker=layout.logs / f"{layout.label}_export.exit",
            )
        )
    if "verify" in phases:
        steps.append(
            step(
                "verify_export",
                (
                    args.python_bin,
                    "scripts/verify_masked_model_export.py",
                    "--model",
                    model_name,
                    "--revision",
                    model_revision,
                    "--model-source",
                    layout.export_dir,
                    "--mask-checkpoint",
                    layout.ucramp_checkpoint,
                    "--output",
                    layout.export_verify_result,
                ),
                args=args,
                log=layout.logs / f"{layout.label}_export_verify.log",
                exit_marker=layout.logs / f"{layout.label}_export_verify.exit",
            )
        )
    if "math-diagnostics" in phases:
        steps.append(
            step(
                "math500_generation_diagnostics",
                (
                    args.python_bin,
                    "scripts/analyze_math_generation.py",
                    "--prediction-dir",
                    layout.eval_result_root / "math500",
                    "--label",
                    layout.label,
                    "--output",
                    layout.math_pathology,
                    "--expected-samples",
                    500,
                ),
                args=args,
                log=layout.logs / f"{layout.label}_math_diagnostics.log",
                exit_marker=layout.logs / f"{layout.label}_math_diagnostics.exit",
            )
        )
    if "summarize" in phases:
        steps.append(
            step(
                "summarize_evalscope",
                (
                    args.python_bin,
                    "scripts/summarize_evalscope_full.py",
                    "--eval-root",
                    layout.eval_result_root,
                    "--served-name",
                    layout.label,
                    "--method-label",
                    "UC-RAMP",
                    "--model-label",
                    MODEL_LABELS[args.arch],
                    "--mask-result",
                    layout.ucramp_result,
                    "--mask-checkpoint",
                    layout.ucramp_checkpoint,
                    "--ppl-result",
                    layout.ppl_result,
                    "--math-pathology",
                    layout.math_pathology,
                    "--terminal-exit",
                    layout.eval_exit,
                    "--output",
                    layout.summary,
                ),
                args=args,
                log=layout.logs / f"{layout.label}_summary.log",
                exit_marker=layout.logs / f"{layout.label}_summary.exit",
            )
        )
    return steps


def outputs_for_phases(layout: Layout, phases: set[str]) -> list[Path]:
    outputs: list[Path] = []
    if "stage1" in phases:
        outputs += [layout.cape_checkpoint, layout.cape_result, layout.logs / f"{layout.label}_stage1_cape.exit"]
    if "stage2" in phases:
        outputs += [layout.ucramp_checkpoint, layout.ucramp_result, layout.logs / f"{layout.label}_stage2_ramp.exit"]
    if "ppl" in phases:
        outputs += [layout.ppl_result, layout.logs / f"{layout.label}_ppl.exit"]
    if "export" in phases:
        outputs += [layout.export_dir, layout.export_result, layout.logs / f"{layout.label}_export.exit"]
    if "verify" in phases:
        outputs += [layout.export_verify_result, layout.logs / f"{layout.label}_export_verify.exit"]
    if "eval" in phases:
        outputs += [layout.eval_result_root, layout.eval_exit]
    if "math-diagnostics" in phases:
        outputs += [layout.math_pathology, layout.logs / f"{layout.label}_math_diagnostics.exit"]
    if "summarize" in phases:
        outputs += [layout.summary, layout.logs / f"{layout.label}_summary.exit"]
    return outputs


def reference_run_payload(args: argparse.Namespace) -> dict[str, object]:
    assert args.reference_manifest is not None
    assert args.reference_run is not None
    payload = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
    try:
        run = payload["runs"][args.reference_run]
    except KeyError as exc:
        raise SystemExit(f"reference run not found: {args.reference_run}") from exc
    if run.get("architecture") != args.arch:
        raise SystemExit("reference manifest architecture does not match --arch")
    if float(run.get("prune_ratio")) != float(args.pruning_rate):
        raise SystemExit("reference manifest prune_ratio does not match --pruning-rate")
    return run


def apply_reference_defaults(args: argparse.Namespace, phases: set[str]) -> None:
    if not args.use_reference_mask:
        if args.model_revision is None:
            args.model_revision = "local-20260826-review-v1"
        return
    run = reference_run_payload(args)
    if {"stage1", "stage2"} & phases and not bool(run.get("mask_available", False)):
        raise SystemExit(
            f"reference run {args.reference_run} does not include a redistributable mask"
        )
    reference_model = str(run.get("model_name"))
    reference_revision = str(run.get("model_revision"))
    if args.model_name is None:
        args.model_name = reference_model
    elif args.model_name != reference_model:
        raise SystemExit(
            f"--model-name {args.model_name!r} does not match reference {reference_model!r}"
        )
    if args.model_revision is None:
        args.model_revision = reference_revision
    elif args.model_revision != reference_revision:
        raise SystemExit(
            "--model-revision "
            f"{args.model_revision!r} does not match reference {reference_revision!r}"
        )


def task_dataset_args(args: argparse.Namespace, group: str) -> dict[str, object]:
    """Return pinned EvalScope dataset args for one task group.

    When a reference manifest/run is provided, prefer the exact task config
    snapshot bundled with that run.  This keeps replay aligned with the
    successful historical run instead of relying on EvalScope's moving defaults.
    """

    if args.reference_manifest is not None and args.reference_run is not None:
        config = args.reference_manifest.parent / "task_configs" / args.reference_run / f"{group}.json"
        if config.exists():
            payload = json.loads(config.read_text(encoding="utf-8"))
            dataset_args = payload.get("dataset_args")
            if isinstance(dataset_args, dict):
                return dataset_args
    return DATASET_ARGS[group]


def resolve_reference_path(args: argparse.Namespace, manifest_value: object) -> Path | None:
    if manifest_value is None:
        return None
    path = Path(str(manifest_value))
    return path if path.is_absolute() else args.repo_root / path


def import_reference_masks(args: argparse.Namespace, layout: Layout) -> None:
    run = reference_run_payload(args)
    if not bool(run.get("mask_available", False)):
        raise SystemExit(
            f"reference run {args.reference_run} does not include a redistributable mask"
        )
    tool = args.repo_root / "tools" / "mask_artifact.py"
    pairs = (
        ("stage1_mask_artifact", "stage1_maskgen", layout.cape_checkpoint, layout.cape_result),
        ("mask_artifact", "maskgen", layout.ucramp_checkpoint, layout.ucramp_result),
    )
    for artifact_key, maskgen_key, checkpoint, maskgen in pairs:
        artifact = resolve_reference_path(args, run.get(artifact_key))
        metadata = resolve_reference_path(args, run.get(maskgen_key))
        if artifact is None or metadata is None:
            if artifact_key == "stage1_mask_artifact":
                continue
            raise SystemExit(f"reference run missing {artifact_key} or {maskgen_key}")
        for path, description in ((artifact, artifact_key), (metadata, maskgen_key)):
            if not path.exists():
                raise SystemExit(f"missing reference {description}: {path}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        maskgen.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(
            [args.python_bin, str(tool), "unpack", str(artifact), str(checkpoint)],
            cwd=args.repo_root,
            env=base_env(args),
            check=False,
        ).returncode
        if rc != 0:
            write_exit_marker(layout.logs / f"{layout.label}_reference_import.exit", rc)
            raise SystemExit(rc)
        shutil.copyfile(metadata, maskgen)
    write_exit_marker(layout.logs / f"{layout.label}_reference_import.exit", 0)


def write_exit_marker(path: Path | None, rc: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"__EXPERIMENT_EXIT__={rc}\n", encoding="utf-8")


def run_step(step_: CommandStep) -> int:
    if step_.log is None:
        process = subprocess.run(step_.argv, cwd=step_.cwd, env=step_.env, check=False)
        rc = process.returncode
    else:
        step_.log.parent.mkdir(parents=True, exist_ok=True)
        with step_.log.open("wb") as log_handle:
            process = subprocess.run(
                step_.argv,
                cwd=step_.cwd,
                env=step_.env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rc = process.returncode
    write_exit_marker(step_.exit_marker, rc)
    return rc


def pre_eval_steps(steps: Iterable[CommandStep]) -> list[CommandStep]:
    return [
        item
        for item in steps
        if item.name not in {"math500_generation_diagnostics", "summarize_evalscope"}
    ]


def post_eval_steps(steps: Iterable[CommandStep]) -> list[CommandStep]:
    return [
        item
        for item in steps
        if item.name in {"math500_generation_diagnostics", "summarize_evalscope"}
    ]


def command_plan(steps: list[CommandStep], phases: set[str]) -> list[dict[str, object]]:
    plan = [
        {
            "name": item.name,
            "argv": list(item.argv),
            "log": None if item.log is None else str(item.log),
            "exit_marker": None if item.exit_marker is None else str(item.exit_marker),
        }
        for item in pre_eval_steps(steps)
    ]
    if "eval" in phases:
        plan.append(
            {
                "name": "evalscope_full",
                "argv": ["<internal:vllm serve + EvalScope full suite>"],
                "log": None,
                "exit_marker": "<logs>/<label>_evalscope.exit",
            }
        )
    plan.extend(
        {
            "name": item.name,
            "argv": list(item.argv),
            "log": None if item.log is None else str(item.log),
            "exit_marker": None if item.exit_marker is None else str(item.exit_marker),
        }
        for item in post_eval_steps(steps)
    )
    return plan


def check_ppl_gate(path: Path, *, arch: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    ppl = float(metrics["ppl"])
    windows = int(metrics["num_windows"])
    if windows < PPL_MIN_WINDOWS[arch]:
        raise SystemExit(f"WikiText-2 PPL used too few windows: {windows}")
    if ppl > PPL_MAX[arch]:
        raise SystemExit(f"WikiText-2 PPL gate failed for {arch}: {ppl} > {PPL_MAX[arch]}")


def wait_for_vllm(port: int, process: subprocess.Popen[bytes], timeout_seconds: int = 1200) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                if response.status == 200:
                    return True
        except URLError:
            pass
        except TimeoutError:
            pass
        time.sleep(5)
    return False


def run_evalscope(args: argparse.Namespace, layout: Layout) -> int:
    port = args.port or (9300 + args.gpu)
    env = base_env(args)
    env["VLLM_PLUGINS"] = ""
    env["VLLM_USE_FLASHINFER_SAMPLER"] = env.get("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.arch == "gemma4":
        env["UC_RAMP_GEMMA4_VLLM_COMPAT"] = "1"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(args.repo_root / "scripts" / "compat"), env["PYTHONPATH"]]
        )
    vllm_path = shutil.which(args.vllm_bin) or args.vllm_bin
    evalscope_path = shutil.which(args.evalscope_bin) or args.evalscope_bin
    server_args = [
        vllm_path,
        "serve",
        str(layout.export_dir),
        "--served-model-name",
        layout.label,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        args.gpu_memory_utilization,
        "--enforce-eager",
    ]
    if args.arch == "qwen36":
        server_args.extend(["--hf-config-path", str(args.model_source), "--language-model-only"])
    server_log = layout.logs / f"{layout.run_label}_server_gpu{args.gpu}.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    with server_log.open("wb") as handle:
        server = subprocess.Popen(
            server_args,
            cwd=args.repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        if not wait_for_vllm(port, server):
            write_exit_marker(layout.eval_exit, 15)
            return 15
        eval_common = [
            "eval",
            "--model",
            layout.label,
            "--api-url",
            f"http://127.0.0.1:{port}/v1",
            "--api-key",
            "EMPTY",
            "--dataset-hub",
            "modelscope",
            "--generation-config",
            '{"temperature":0,"max_tokens":2048}',
            "--seed",
            str(SEED),
            "--no-timestamp",
        ]
        processes: list[subprocess.Popen[bytes]] = []
        for group, batch_size, dataset_kind, datasets in TASK_GROUPS:
            dataset_dir = args.knowledge_dataset_dir if dataset_kind == "knowledge" else args.general_dataset_dir
            assert dataset_dir is not None
            log = layout.logs / f"{layout.run_label}_{group}_gpu{args.gpu}.log"
            argv = [
                evalscope_path,
                *eval_common,
                "--datasets",
                *datasets,
                "--dataset-args",
                json.dumps(task_dataset_args(args, group), separators=(",", ":")),
                "--eval-batch-size",
                str(batch_size),
                "--dataset-dir",
                str(dataset_dir),
                "--work-dir",
                str(layout.eval_result_root / group),
            ]
            with log.open("wb") as handle:
                processes.append(
                    subprocess.Popen(
                        argv,
                        cwd=args.repo_root,
                        env=no_proxy_env(env),
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                )
        assert args.general_dataset_dir is not None
        human_log = layout.logs / f"{layout.run_label}_humaneval_gpu{args.gpu}.log"
        human_argv = [
            args.python_bin,
            "scripts/eval_humaneval.py",
            "eval",
            "--model",
            layout.label,
            "--api-url",
            f"http://127.0.0.1:{port}/v1",
            "--api-key",
            "EMPTY",
            "--datasets",
            "humaneval",
            "--dataset-args",
            json.dumps(task_dataset_args(args, "humaneval"), separators=(",", ":")),
            "--dataset-hub",
            "modelscope",
            "--dataset-dir",
            str(args.general_dataset_dir),
            "--eval-batch-size",
            "8",
            "--generation-config",
            '{"temperature":0,"max_tokens":2048}',
            "--seed",
            str(SEED),
            "--work-dir",
            str(layout.eval_result_root / "humaneval"),
            "--no-timestamp",
        ]
        human_env = no_proxy_env(env)
        human_env["PYTHONPATH"] = os.pathsep.join([str(args.repo_root / "src"), human_env["PYTHONPATH"]])
        with human_log.open("wb") as handle:
            processes.append(
                subprocess.Popen(
                    human_argv,
                    cwd=args.repo_root,
                    env=human_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        rc = 0
        for process in processes:
            if process.wait() != 0:
                rc = 1
        write_exit_marker(layout.eval_exit, rc)
        return rc
    finally:
        if server.poll() is None:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait()


def manifest(args: argparse.Namespace, layout: Layout, steps: list[CommandStep], phases: set[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": "UC-RAMP strict reproduction",
        "arch": args.arch,
        "model_name": args.model_name or MODEL_NAMES[args.arch],
        "model_source": str(args.model_source),
        "pruning_rate": args.pruning_rate,
        "retained_endpoint": layout.retained_endpoint,
        "seed": SEED,
        "calibration": {
            "domains": ["wikitext", "reasoning", "hard_math", "code"],
            "nsamples": FORMAL_NSAMPLES,
            "sequence_length": FORMAL_SEQUENCE_LENGTH,
            "competition_math_source": str(args.competition_math_source),
            "competition_math_exclusion_manifest": str(args.competition_math_exclusion_manifest),
        },
        "settings": {
            "stage1_profile": "review_stage1_envelope",
            "stage2_profile": "review_stage1_ramp",
            "stage2_fixed_stage1_checkpoint": str(layout.cape_checkpoint),
            "interaction_strength": 0.2,
            "max_swaps_per_expert": 1,
            "camera_alpha": 0.9,
            "anchor_general_weight": 0.9,
            "trust_radius_divisor": 8,
            "evalscope": {
                "temperature": 0,
                "max_tokens": 2048,
                "seed": SEED,
                "mmlu": False,
                "groups": {
                    "knowledge": ["hellaswag", "winogrande", "race", "gpqa_diamond"],
                    "general": ["piqa", "arc", "gsm8k"],
                    "math500": ["math_500"],
                    "humaneval": ["humaneval"],
                },
                "dataset_args": {
                    group: task_dataset_args(args, group)
                    for group in ("knowledge", "general", "math500", "humaneval")
                },
                "qwen36_language_model_only": args.arch == "qwen36",
            },
        },
        "phases": sorted(phases),
        "artifacts": {
            "cape_checkpoint": str(layout.cape_checkpoint),
            "cape_result": str(layout.cape_result),
            "ucramp_checkpoint": str(layout.ucramp_checkpoint),
            "ucramp_result": str(layout.ucramp_result),
            "ppl_result": str(layout.ppl_result),
            "export_dir": str(layout.export_dir),
            "export_verification": str(layout.export_verify_result),
            "eval_root": str(layout.eval_result_root),
            "summary": str(layout.summary),
        },
        "commands": command_plan(steps, phases),
        "reference_import": (
            None
            if not args.use_reference_mask
            else {
                "manifest": str(args.reference_manifest),
                "run": args.reference_run,
                "tool": "tools/mask_artifact.py unpack",
            }
        ),
    }


def main() -> None:
    args = parse_args()
    phases = selected_phases(args.phase)
    normalize_paths_and_defaults(args, phases)
    layout = build_layout(args)
    require_paths(args, phases)
    refuse_overwrite(outputs_for_phases(layout, phases), dry_run=args.dry_run)
    steps = build_steps(args, layout, phases)
    payload = manifest(args, layout, steps, phases)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(payload, indent=2), flush=True)
        return
    for directory in (layout.checkpoints, layout.results, layout.logs, layout.exports, layout.eval_root):
        directory.mkdir(parents=True, exist_ok=True)
    if args.use_reference_mask and ({"stage1", "stage2"} & phases):
        print("importing bundled reference mask", flush=True)
        import_reference_masks(args, layout)
    for item in pre_eval_steps(steps):
        if item.name == "summarize_evalscope" and "ppl" not in phases:
            check_ppl_gate(layout.ppl_result, arch=args.arch)
        print(f"running {item.name}", flush=True)
        rc = run_step(item)
        if rc != 0:
            raise SystemExit(rc)
        if item.name == "wikitext2_ppl":
            check_ppl_gate(layout.ppl_result, arch=args.arch)
    if "eval" in phases:
        print("running evalscope_full", flush=True)
        rc = run_evalscope(args, layout)
        if rc != 0:
            raise SystemExit(rc)
    for item in post_eval_steps(steps):
        if item.name == "summarize_evalscope" and "ppl" not in phases:
            check_ppl_gate(layout.ppl_result, arch=args.arch)
        print(f"running {item.name}", flush=True)
        rc = run_step(item)
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
