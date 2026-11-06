<!--
Copyright 2026 NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

```
1. Build the Payload (payload.py)
2. Build the reference/gold logits in float32 and bfloat16 using Huggingface (reference.py)
3. Compare any server's logits with these logits (parity.py)
```