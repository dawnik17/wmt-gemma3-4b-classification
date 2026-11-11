# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin, loaded through vllm.general_plugins: Gemma3ForSequenceClassification, 
and the CUSTOM attention backend that gives it Gemma 3's image attention mask on FlashAttention."""


def register():
    from vllm import ModelRegistry
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    ModelRegistry.register_model(
        "Gemma3ForSequenceClassification",
        "gemma3_cls.model:Gemma3ForSequenceClassification",
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "gemma3_cls.prefix_attention.PrefixFlashAttentionBackend",
    )
