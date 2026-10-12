"""Restore Gemma 4 legacy attention fields for vLLM's config reader.

Transformers 5.16 converts ``global_head_dim`` and
``num_global_key_value_heads`` into per-layer overrides, while vLLM 0.27.1
still reads the legacy fields when building full-attention layers.  This
opt-in shim exposes the same values without changing any model weights.
"""

import os


if os.environ.get("UC_RAMP_GEMMA4_VLLM_COMPAT") == "1":
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    _original_post_init = Gemma4TextConfig.__post_init__

    def _post_init_with_legacy_attention_fields(self, **kwargs):
        _original_post_init(self, **kwargs)
        # vLLM 0.27.1 still reads these heterogeneous attributes from the
        # global text config while constructing its per-layer attention map.
        self.allow_global_per_layer_attribute_access = True
        full_indices = [
            index
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        ]
        if full_indices:
            full_config = self.per_layer_config[full_indices[0]]
            self.global_head_dim = full_config.head_dim
            self.num_global_key_value_heads = full_config.num_key_value_heads

    Gemma4TextConfig.__post_init__ = _post_init_with_legacy_attention_fields
