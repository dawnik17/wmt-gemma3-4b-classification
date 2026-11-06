# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""vLLM plugin that lets vLLM serve Gemma3ForSequenceClassification checkpoints, loaded through vllm.general_plugins."""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Gemma3ForSequenceClassification",
        "gemma3_cls.model:Gemma3ForSequenceClassification",
    )
