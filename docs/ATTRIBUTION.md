# Attribution

The `ramp_moe` implementation and experiment orchestration originate from the UC-RAMP research codebase. The clean repository omits private development history and personal author metadata for review preparation.

The CAMERA importance implementation is independent code based on the published method. No third-party pruning repositories or model weights are vendored in this release. Experimental prototype options remain in the shared source modules to preserve the exact dependencies of the formal profiles; they are not additional validated release methods.

Runtime dependencies include PyTorch, Transformers, datasets, vLLM, EvalScope, NumPy, and Triton. Preserve the upstream license and attribution requirements when redistributing their code. Benchmark task templates in `reference/task_configs` are snapshots of EvalScope1.4.1 configuration used by the experiments; they retain their upstream attribution.

The model families and datasets are separately distributed by their owners. This repository grants no additional rights to those weights or dataset contents. A project-wide code redistribution license and final citation will be added after the authors select them for the public release.
