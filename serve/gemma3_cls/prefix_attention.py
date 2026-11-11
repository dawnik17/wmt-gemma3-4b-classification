# Copyright 2026 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
"""Gemma 3's image mask on vLLM's FlashAttention backend, in place of the Triton backend vLLM uses for it.

Gemma 3's attention is causal except within an image, whose tokens see each other both ways. FlashAttention is causal
only, so this runs two passes: a causal pass over the batch, then a non-causal pass over each image's rows, with keys
ending where the image ends, written over their causal rows. The result is the exact mask.

    --attention-backend CUSTOM          (gemma3_cls.register registers it)
"""

from dataclasses import dataclass

import numpy as np
import torch
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)


@dataclass
class ImageRows:
    """A step's image query rows, and the FlashAttention batch that recomputes them: one sequence per image."""

    rows: torch.Tensor  # [rows] int64, indices into the step's query tokens
    cu_seqlens_q: torch.Tensor  # [images + 1] int32
    seqused_k: (
        torch.Tensor
    )  # [images] int32: keys from the request's first token through the image's last
    block_table: (
        torch.Tensor
    )  # [images, blocks]: the block table row of the image's request
    max_seqlen_q: int
    max_seqlen_k: int


def image_rows(common, device):
    """ImageRows for the images among a step's query tokens, None if no image row can see a later token."""
    ranges = common.mm_req_doc_ranges
    if not ranges or common.max_query_len <= 1:
        return None
    
    starts = common.query_start_loc_cpu.numpy()
    seq_lens = (
        common.seq_lens_cpu_upper_bound.numpy()
    )  # exact for prefill rows, the only ones with images
    
    rows, lengths, keys, reqs = [], [], [], []
    for req, spans in ranges.items():
        begin, query_len = int(starts[req]), int(starts[req + 1] - starts[req])
        context = int(seq_lens[req]) - query_len
        for start, end in spans:  # absolute prompt positions, end inclusive
            first, last = max(start, context), min(end, context + query_len - 1)
            if (
                first >= last
            ):  # one row (or none) of the image in this step: nothing later in the image to see
                continue
            rows.append(np.arange(begin + first - context, begin + last - context + 1))
            lengths.append(last - first + 1)
            keys.append(last + 1)
            reqs.append(req)

    if not rows:
        return None

    meta = torch.from_numpy(
        np.array([0, *np.cumsum(lengths), *keys, *reqs], dtype=np.int32)
    ).pin_memory()
    meta = meta.to(device, non_blocking=True)
    n = len(lengths)
    return ImageRows(
        rows=torch.from_numpy(np.concatenate(rows))
        .pin_memory()
        .to(device, non_blocking=True),
        cu_seqlens_q=meta[: n + 1],
        seqused_k=meta[n + 1 : 2 * n + 1],
        block_table=common.block_table_tensor.index_select(0, meta[2 * n + 1 :]),
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(keys),
    )


class PrefixFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.image_rows = image_rows(common_attn_metadata, self.device)
        return metadata


class PrefixFlashAttentionImpl(FlashAttentionImpl):
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        super().forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )
        images = getattr(attn_metadata, "image_rows", None)
        if images is None or self.vllm_flash_attn_version == 4:
            return output
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # Local layers keep their window on the left; on the right, every row may see the end of its image.
        left = self.sliding_window[0]
        window = [left, images.max_seqlen_q - 1] if left >= 0 else [-1, -1]
        descale = (images.seqused_k.shape[0], self.num_kv_heads)
        out = torch.empty(
            (images.rows.shape[0], self.num_heads, self.head_size),
            dtype=output.dtype,
            device=output.device,
        )
        flash_attn_varlen_func(
            q=query.index_select(0, images.rows),
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=images.cu_seqlens_q,
            max_seqlen_q=images.max_seqlen_q,
            seqused_k=images.seqused_k,
            max_seqlen_k=images.max_seqlen_k,
            softmax_scale=self.scale,
            causal=False,
            window_size=window,
            block_table=images.block_table,
            softcap=self.logits_soft_cap,
            fa_version=self.vllm_flash_attn_version,
            k_descale=layer._k_scale.expand(descale),
            v_descale=layer._v_scale.expand(descale),
        )
        output.index_copy_(0, images.rows, out)
        return output


class PrefixFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PrefixFlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        return PrefixFlashAttentionMetadataBuilder

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        if (
            use_mm_prefix
            and kv_cache_dtype is not None
            and is_quantized_kv_cache(kv_cache_dtype)
        ):
            return "the image rows' second pass reads a bf16/fp16 KV cache"
        # FlashAttention's own checks, less "the image mask needs FlashAttention 4": the second pass is the mask.
        return super().supports_combination(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            False,
            device_capability,
        )
