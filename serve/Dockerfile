# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0

# The official vLLM 0.32.0 image plus the gemma3_cls plugin here.
# docker build -t ghcr.io/dawnik17/gemma3-cls-vllm:0.32.0-fa serve
FROM vllm/vllm-openai:v0.32.0
LABEL org.opencontainers.image.source=https://github.com/dawnik17/wmt-gemma3-4b-classification
COPY pyproject.toml /opt/gemma3-cls/
COPY gemma3_cls /opt/gemma3-cls/gemma3_cls
RUN pip install --no-cache-dir --no-deps --no-build-isolation /opt/gemma3-cls
ENV VLLM_PLUGINS=gemma3_cls
