# UC-RAMP

UC-RAMP is a post-training channel-pruning method for routed mixture-of-experts models. **CAPE** allocates an exact channel budget across experts using capability-conditioned calibration evidence. **RAMP** then refines channel membership while keeping every expert's allocated width fixed. The method removes coupled gate/up rows and matching down-projection columns without backpropagation or weight recovery.

This repository packages the algorithm, a portable reproduction runner, frozen evaluation settings, reported results, and Boolean pruning masks. Model weights, datasets, private experiment logs, and the development repository's Git history are not included.

## Reproducibility status

Five reported endpoints include masks that were verified bit-for-bit against the original experiment checkpoints. They can be evaluated without rerunning calibration. All six endpoints have result summaries and pruning metadata, including calibration offsets and per-layer budgets.


The unrounded scores, correct counts, and sample counts for all ten tasks are in [reference/results](reference/results). Results include regressions: for example, the Qwen3.6 25% endpoint is not a consistent improvement over the comparison methods. Gemma4 PPL is retained exactly as recorded and is not directly comparable to Qwen PPL.

## Installation

Use Python 3.10 and a CUDA environment compatible with the pinned PyTorch/vLLM builds. The five newer runs used PyTorch `2.13.0+cu129`, Transformers `5.16.1`, vLLM `0.27.1+cu129`, and EvalScope `1.4.1`. CUDA wheel build suffixes matter; install the matching CUDA builds rather than CPU wheels.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
python -m pip install -e . --no-deps
python -m pip install -r requirements-dev.txt
PYTHONPATH=src:. python -m pytest -q tests
```

The pinned requirements record observed versions. A fresh dependency installation has not been validated on every platform. The historical Qwen3 50% experiment used a different environment; install [requirements-historical-qwen3-50.txt](requirements-historical-qwen3-50.txt) in a separate environment if reconstructing that endpoint. Do not install both version sets together.

## Local inputs

The runner uses offline loading. Obtain model and dataset snapshots separately and supply their local paths; it never downloads a model during an experiment.

- A complete model directory, including the original tokenizer, configuration, and all indexed weight shards.
- For pruning reconstruction: cached Hugging Face train datasets for WikiText-2 raw, GSM8K `main`, MathQA, and MBPP `full`, plus the Competition-MATH training parquet described in [the protocol](docs/REPRODUCIBILITY.md).
- For accuracy evaluation: the EvalScope/ModelScope local dataset caches used by the general and knowledge task groups. These differ from the Hugging Face calibration cache. Include the GSM8K and RACE few-shot train splits.
- For PPL: a local Hugging Face WikiText-2 raw test snapshot.

The default Hugging Face cache location is honored. To use a different cache, set `HF_DATASETS_CACHE` and `HF_HOME` before running. Model/config/tokenizer asset fingerprints are provided in [reference/model_assets](reference/model_assets); shard sizes are recorded, but historical full weight hashes are not available.

```bash
python tools/snapshot_assets.py \
  --model-source /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --verify reference/model_assets/qwen3.json
```

## Evaluate a published mask

Run from the repository root. This example evaluates the included Qwen3 25% mask through PPL, a masked-model export, export validation, and the full ten-task EvalScope suite.

```bash
python reproduction/reproduce_uc_ramp.py \
  --arch qwen3 --pruning-rate 0.25 --gpu 0 \
  --model-source /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --general-dataset-dir /path/to/modelscope/dataset-cache \
  --knowledge-dataset-dir /path/to/evalscope/dataset-cache \
  --use-reference-mask \
  --reference-manifest reference/manifest.json \
  --reference-run qwen3_prune25 \
  --work-root outputs/qwen3_prune25 \
  --manifest outputs/qwen3_prune25-plan.json
```

Add `--dry-run` to inspect the plan before execution. Replace the architecture, pruning rate, run ID, and model path for another endpoint. Reference IDs are `qwen3_prune25`, `gemma4_prune25`, `gemma4_prune50`, `qwen36_prune25`, and `qwen36_prune50`. `qwen3_prune50` rejects fixed-mask replay because its original mask is missing.

The runner writes logs and `__EXPERIMENT_EXIT__=<code>` markers, waits for its subprocesses, and refuses to overwrite completed artifacts. GPU IDs 0–3 are accepted by default. A full BF16 export requires substantial host RAM and roughly one extra model's worth of disk space; typical runs use a single 80-GB GPU. No GPU evaluation is started by the test suite.

After completion, check every task against the reported exact counts:

```bash
python tools/check_results.py \
  --actual outputs/qwen3_prune25/results/uc_ramp_qwen3_exact75_evalscope_summary.json \
  --reference reference/results/qwen3_prune25.json
```

The checker defaults to exact task accuracy and identical sample counts. An optional `--accuracy-tolerance` is expressed as an absolute fraction and is applied to each task, not just the macro average. Hardware or framework differences can change greedy outputs; report deviations rather than silently adjusting evaluation settings.

## Reconstruct CAPE and strict RAMP

Omit `--use-reference-mask` to rebuild both stages. The reconstruction uses the fixed four-domain 448×2048 calibration protocol, alpha=0.9, general-anchor weight=0.9, capacity trust divisor=8, interaction strength=0.2, and at most one channel exchange per expert. Stage II is initialized from the generated Stage-I mask and preserves every expert width.

```bash
python reproduction/reproduce_uc_ramp.py \
  --arch qwen3 --pruning-rate 0.25 --gpu 0 \
  --model-source /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --competition-math-source /path/to/competition_math/train.parquet \
  --competition-math-exclusion-manifest reference/calibration/competition_math_train_decontamination.json \
  --reference-manifest reference/manifest.json --reference-run qwen3_prune25 \
  --phase mask --work-root outputs/qwen3_prune25_rebuild \
  --manifest outputs/qwen3_prune25-rebuild-plan.json
```

For a full reconstruction and evaluation, omit `--phase mask` and supply both evaluation cache paths. Individual phases can also be selected repeatedly: `stage1`, `stage2`, `ppl`, `export`, `verify`, `eval`, `math-diagnostics`, and `summarize`. Use the same work root when continuing a phased run.

## Accuracy protocol

Accuracy uses **EvalScope generation-based evaluation** with greedy decoding, maximum 2048 output tokens, seed 1234, and no MMLU. GSM8K is 4-shot; RACE is 3-shot; the remaining tasks are zero-shot. Task templates, splits, grading settings, and shuffle settings are frozen in [reference/task_configs](reference/task_configs) for the five newer endpoints. Ten-task macro accuracy is the unweighted mean of PIQA, ARC-Easy, ARC-Challenge, GSM8K, HellaSwag, WinoGrande, RACE, GPQA-Diamond, MATH500, and HumanEval pass@1.

The accuracy path exports zeroed dense matrices and serves them through vLLM. It reproduces the evaluation representation used for the reported scores. The separate physical compact kernel is an efficiency implementation and is not substituted into the accuracy protocol.

## Optional compact inference

`scripts/serve_compact_vllm.py` registers a Qwen3 adapter that physically packs retained expert channels and invokes the repository's Triton ragged kernels. Attention remains in vLLM. Weights are packed without expert-width alignment; gate/up channel tiles are 16, down-reduction tiles are 32, and down-output tiles are 64. The temporary activation workspace still uses the layer's maximum expert width.

`scripts/benchmark_vllm_http.py` compares this implementation with the original BF16 vLLM model using HTTP SSE token timestamps. The recorded efficiency experiment used Qwen3 **25%** pruning, a single A100-SXM4-80GB, eager execution, max context 2048, and 32 generated tokens. Prefix caching was enabled with repeated synthetic prompts; its TTFT is not an uncached real-text prefill benchmark. See [the protocol](docs/REPRODUCIBILITY.md).

## Layout

```text
src/ramp_moe/       CAPE/RAMP, model adapters, mask application, compact kernels
scripts/           Pruning, PPL, export/verification, EvalScope helpers, serving
reproduction/      Portable experiment orchestration
reference/         Results, bitpacked masks, calibration metadata, task snapshots
tools/             Mask conversion, asset fingerprints, result comparison
tests/             CPU regression and protocol tests
docs/              Reproduction details and release limitations
```

## Release and attribution

This is a preparation repository for anonymous review. Git history starts at this release; commit authorship contains no personal identity. The current GitHub hosting account must be replaced or concealed through an appropriate anonymous review hosting service before sharing a public review link. No project-wide redistribution license has been selected yet.

CAMERA scoring is an independent implementation; this release does not vendor third-party pruning repositories. EvalScope, vLLM, PyTorch, Transformers, datasets, and Triton retain their upstream licenses. Model and benchmark usage remain subject to their respective terms. See [attribution](docs/ATTRIBUTION.md).
