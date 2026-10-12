# Frozen experimental protocol

## Pruning

The pruning unit is one expert intermediate channel, coupled across gate, up, and down projections. Only routed experts are pruned; attention, router, embedding, shared modules, and expert count remain unchanged. Rates 0.25 and 0.50 refer to the total routed-expert intermediate channel count, not total model parameters or active FLOPs.

CAPE uses the `review_stage1_envelope` profile. Strict RAMP uses `review_stage1_ramp` with an explicit fixed Stage-I checkpoint. The width constraint is enforced for every expert. The formal settings are seed 1234, alpha=0.9, general-anchor weight=0.9, trust radius divisor=8, next-RMS output scale, a 512-entry expert statistics reservoir, and one exchange per expert at interaction strength 0.2. Per-layer budgets use the frozen tail-protected normalized schedule; actual integer budgets are included in each pruning reference.

Calibration consists of 448 windows of 2048 tokens: WikiText-2 raw train128, GSM8K train64, MathQA train64, Competition-MATH train128, and MBPP full train64. Competition-MATH selects Levels 4 and 5 across seven categories. The checked exclusion manifest removes 21 training problems overlapping with MATH500 under the registered exact/5-gram audit. Calibration is train-only. The evaluation questions are not calibration samples.

Recorded dataset fingerprints are WikiText `7c4dea6941cc4a0a`, GSM8K `097157d8187d2b97`, MathQA `7211603f1601a16d`, Competition-MATH `721bcef2ef10224e`, and MBPP `5cdc16311fd8e220`. A datasets fingerprint depends on processing and software versions; retain source hashes, row order, tokenizer assets, offsets, and serialized token statistics as well.

Competition-MATH training parquet SHA256:

```text
33ced5214cca5a09fc6a518a698db19ef98b188330705b601a5a1fb0982631c1
```

MATH500 problem source SHA256 recorded by the decontamination audit:

```text
35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132
```

## Accuracy evaluation

The original full pipeline was PPL → masked BF16 export → zero-mask export validation → vLLM/EvalScope → exact-count summary. The masked export preserves original matrix shapes. It is distinct from physical compact serving.

All accuracy runs used EvalScope1.4.1, temperature0, max output tokens2048, seed1234, and generation-based grading. vLLM ran eager BF16, TP1, maximum context8192. Qwen3.6 requires the language-only flag and the original HF config path. Gemma4 used `Gemma4ForConditionalGeneration`, the opt-in config compatibility shim, GPU memory utilization0.78, and did **not** use language-only mode. The tested local Gemma configuration contains `num_kv_shared_layers=0`, local head dimension256, global head dimension512, and global KV heads2; inspect the asset fingerprint if your snapshot differs. The runner does not silently rewrite a user's model config.

Four evaluator groups ran concurrently against one server: knowledge batch32 (HellaSwag, WinoGrande, RACE, GPQA-Diamond), general batch8 (PIQA, ARC, GSM8K), MATH500 batch8, and HumanEval batch8. GSM8K uses four demonstrations and RACE three. Choice shuffling and dataset shuffling are disabled. Full task configurations are extracted from successful historical evaluator YAMLs for the five newer runs.

Expected sample counts, in macro-average order:

| Task | Samples |
|---|---:|
| PIQA | 1838 |
| ARC-Easy | 2376 |
| ARC-Challenge | 1172 |
| GSM8K | 1319 |
| HellaSwag | 10042 |
| WinoGrande | 1267 |
| RACE | 4934 |
| GPQA-Diamond | 198 |
| MATH500 | 500 |
| HumanEval | 164 |

PIQA/HellaSwag/WinoGrande use validation splits, GPQA-Diamond uses its train-named evaluation split, and the other tasks use test splits. GSM8K/RACE require their train splits for demonstrations. HumanEval uses the repository's scoring wrapper, including isolated subprocess execution. Run generated benchmark code in an isolated evaluation environment.

## Environment split

| Run family | PyTorch | Transformers | vLLM | Observed hardware |
|---|---|---|---|---|
| Qwen3 50%, historical strict beta0.2 | 2.10.0+cu128 | 5.13.0 | 0.19.1 | RTX PRO 6000 Blackwell Server Edition |
| Five newer strict endpoints | 2.13.0+cu129 | 5.16.1 | 0.27.1+cu129 | A100-class server environment |

The local model revision labels in the mask metadata are experiment identifiers, not immutable upstream revision hashes. `reference/model_assets` fingerprints the current model configuration/tokenizer files and records shard sizes. These are useful compatibility checks, but they do not establish historical weight identity. Full model weight hashes and portable snapshots of every dataset are not included.

## Fixed-mask artifacts

`reference/masks/*.json` contains only bitpacked Boolean channel selections plus method metadata. Every layer carries a SHA256 checksum of its packed mask bits. `tools/mask_artifact.py unpack` restores the checkpoint format accepted by pruning/evaluation scripts. Re-serialized checkpoint file hashes may differ from the original `.pt` serialization; mask-bit equality is the relevant structural invariant. The original checkpoint SHA256 remains in `reference/manifest.json` and result records for provenance.

Five strict masks and their five CAPE references are included and verified against the original local checkpoints. Original Qwen3-50 masks and evaluator configuration YAMLs are unavailable locally. Its historical summary is preserved, and the runner offers a reconstruction path; a new reconstruction must be checked against the original metrics and must not be labeled byte-identical without evidence.

This release was tested through CPU regression tests, all-six dry-run command plans, reference-mask import/round-trip checks, and result-count consistency checks. A fresh six-endpoint full GPU rerun has not been performed during release preparation. Publishing this bundle does not turn those checks into a claim of independent end-to-end accuracy reproduction.

## Efficiency measurement

The separate compact measurement uses Qwen3 25% pruning, GPU0 on an A100-SXM4-80GB, vLLM0.27.1 BF16, eager mode, context2048, and GPU memory utilization0.90. Attention uses FlashAttention2. The original model uses vLLM's unquantized Triton MoE backend with its default configuration. Compact experts use the repository's two-stage Triton kernels with packed variable widths.

No storage padding is applied to individual expert widths. Gate/up channel tiles are16, down-channel tiles32, and down-output tiles64; masked tails handle arbitrary widths. Gate/up hidden reduction uses the next power-of-two hidden dimension. The activation workspace has shape `[routed_rows, maximum_layer_expert_width]`, and the first kernel's launch grid depends on that maximum. It therefore retains some width-imbalance overhead.

Workloads are input/batch256/1,512/1,1024/1,1024/2,1024/4, each with output length32 and temperature0. One warmup and three requests were measured per group per round. Two rounds reversed full/compact startup order; reported values pool per-choice medians. HTTP SSE token IDs verify the generated length. Prefix caching was enabled and prompts repeat one synthetic token. Consequently these TTFT values describe the warmed, cache-eligible workload, not uncached real-text prefill. CUDA Graph performance and other model/rate endpoints remain unmeasured by that experiment.
