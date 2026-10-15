"""Optional SDPA attention path for SmolVLA training on memory-limited GPUs.

The machine's LeRobot development checkout hardcodes the eager attention
implementation in ``smolvlm_with_expert``, which materializes the full attention
matrix; SmolVLA then exhausts a 16 GB card at batch 32 (the September checkout
used a fused path and trained batch 64 in ~8 GB). Installing this shim swaps the
interface to ``torch.nn.functional.scaled_dot_product_attention`` while keeping
the eager math: float32 scores, additive mask semantics, GQA head broadcast,
output cast back to the value dtype.

Enabled only through ``A3_SMOLVLA_SDPA=1`` so that environments running the
pinned public LeRobot 0.5.1 keep their upstream attention implementation. This
module doubles as a training entry point:

    python -m a3_dual_arm_sim.learning.smolvla_attention <lerobot_train args...>

installs the shim (when enabled) and then runs ``lerobot_train`` with the
remaining arguments.
"""

from __future__ import annotations

import os
import types


def sdpa_attention_forward(
    self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
):
    import torch
    import torch.nn.functional as F

    num_heads = self.num_attention_heads
    num_key_value_heads = self.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads

    query = query_states.transpose(1, 2).to(torch.float32)  # (B, H, T, D)
    key = key_states.transpose(1, 2).to(torch.float32)  # (B, KV, S, D)
    value = value_states.transpose(1, 2).to(torch.float32)
    if num_key_value_groups != 1:
        key = key[:, :, None].expand(-1, -1, num_key_value_groups, -1, -1).reshape(
            batch_size, num_heads, -1, head_dim
        )
        value = value[:, :, None].expand(-1, -1, num_key_value_groups, -1, -1).reshape(
            batch_size, num_heads, -1, head_dim
        )

    mask = attention_mask
    if mask is not None:
        # The caller passes (B, T, S); SDPA wants it broadcastable to (B, H, T, S).
        if mask.dim() == 3:
            mask = mask[:, None, :, :]
        if mask.dtype is not torch.bool:
            mask = mask > 0
    if os.environ.get("A3_SDPA_DEBUG") == "1":
        import sys

        print(
            f"[sdpa-debug] heads={num_heads} kv_heads={num_key_value_heads} "
            f"q={tuple(query.shape)} k={tuple(key.shape)} v={tuple(value.shape)} "
            f"mask={None if mask is None else (tuple(mask.shape), mask.dtype)}",
            file=sys.stderr,
            flush=True,
        )

    att_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        scale=head_dim**-0.5,
        is_causal=False,
    )
    att_output = att_output.to(value_states.dtype)
    att_output = att_output.transpose(1, 2)
    # -1 because the sequence length can change between experts.
    return att_output.reshape(batch_size, -1, num_heads * head_dim)


def install_sdpa_attention() -> bool:
    """Replace the SmolVLA attention interface; returns False when unavailable."""
    try:
        from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
    except Exception:  # noqa: BLE001 - never break an unrelated training entrypoint
        return False

    def get_attention_interface(self):  # noqa: ANN001 - mirrors the upstream signature
        return types.MethodType(sdpa_attention_forward, self)

    SmolVLMWithExpertModel.get_attention_interface = get_attention_interface
    return True


def main() -> int:
    import runpy

    if os.environ.get("A3_SMOLVLA_SDPA") == "1":
        install_sdpa_attention()
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())