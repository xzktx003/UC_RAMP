# Release validation

Validation performed on 2026-10-12:

- CPU regression suite: **354 passed, 7 skipped**. Skips include optional backend tests; GPU kernel equivalence is not asserted by those skipped tests.
- Actual fixed-mask import through the public runner completed successfully for all five available strict endpoints and their CAPE reference masks.
- Packed mask bits round-trip exactly to the local historical checkpoints, match the exact total pruning budget, and preserve every expert's CAPE width in strict RAMP.
- All six full reconstruction command plans passed CLI dry-run checks.
- Result references pass exact correct-count/sample-count consistency and ten-task macro-average checks.
- Reference evaluator snapshots pin GSM8K four-shot, RACE three-shot, and disabled shuffle settings.
- The repository staging set excludes model weights, generated exports, datasets, runtime logs, and local smoke outputs. Personal source paths and original author Git history are omitted.

These checks validate the release packaging and orchestration. They do not replace fresh full-scale GPU accuracy reruns, dependency installation tests in a clean environment, or historical model/dataset snapshot attestation. The original Qwen3-50 fixed masks and evaluator YAMLs remain unavailable. Its published scores are historical references and its reconstruction path is not yet independently verified.
