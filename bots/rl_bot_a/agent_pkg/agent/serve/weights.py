"""Writing `weights.safetensors` for the NumPy serving path.

Lives beside `numpy_net.py` because the two define one format between them:
every name written here is a name that file looks up. Keep them edited together.

This module imports jax/equinox and so is **not** part of the submission — only
`agent/{spec,serve/{protocol,numpy_net,runner}}.py` are copied into the zip
(`scripts/build_submission.py` enforces that with an AST check).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def flatten_params(net) -> dict[str, np.ndarray]:
    """Name every array the way `agent/serve/numpy_net.py` looks it up.

    `eqx.nn.Linear` stores `weight` as (out, in) and computes `W @ x + b`, so the
    NumPy side does `x @ W.T + b`. The arrays are written untransposed.
    """
    out: dict[str, np.ndarray] = {}

    def linear(prefix, mod):
        out[f"{prefix}.weight"] = np.asarray(mod.weight, dtype=np.float32)
        out[f"{prefix}.bias"] = np.asarray(mod.bias, dtype=np.float32)

    def norm(prefix, mod):
        out[f"{prefix}.weight"] = np.asarray(mod.weight, dtype=np.float32)
        out[f"{prefix}.bias"] = np.asarray(mod.bias, dtype=np.float32)

    linear("embedder", net.embedder)
    linear("scalar_proj", net.scalar_proj)
    out["value_token"] = np.asarray(net.value_token, dtype=np.float32)
    out["pos_encoding"] = np.asarray(net.pos_encoding, dtype=np.float32)

    for i, block in enumerate(net.blocks):
        p = f"blocks.{i}"
        norm(f"{p}.norm1", block.norm1)
        for name in ("q", "k", "v", "out"):
            linear(f"{p}.attn.{name}", getattr(block.attn, name))
        norm(f"{p}.norm2", block.norm2)
        linear(f"{p}.ff1", block.ff1)
        linear(f"{p}.ff2", block.ff2)

    norm("norm_out", net.norm_out)
    linear("policy_head", net.policy_head)
    linear("value_head", net.value_head)
    linear("pass_head", net.pass_head)

    if net.num_bins:
        out["bin_centers"] = np.asarray(net.bin_centers, dtype=np.float32)
    return out


def save_weights(net, model_cfg, out_path: str | Path) -> dict[str, np.ndarray]:
    """Write the safetensors file the serving path loads.

    `n_head` cannot be recovered from the tensor shapes, so it rides in the
    metadata header along with the rest of the geometry.
    """
    from safetensors.numpy import save_file

    out_path = Path(out_path)
    tensors = flatten_params(net)
    metadata = {
        "n_head": str(model_cfg.n_head),
        "embed_dim": str(model_cfg.embed_dim),
        "depth": str(model_cfg.depth),
        "ff_factor": str(model_cfg.ff_factor),
        "value_loss": model_cfg.value_loss,
        "num_bins": str(net.num_bins),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=metadata)
    return tensors
