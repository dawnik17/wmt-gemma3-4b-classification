# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Gemma3ForSequenceClassification for vLLM: Gemma 3's vision-language model with a classification head.
"""
from vllm.model_executor.models.adapters import as_seq_cls_model
from vllm.model_executor.models.gemma3_mm import Gemma3ForConditionalGeneration

Gemma3ForSequenceClassification = as_seq_cls_model(Gemma3ForConditionalGeneration)
