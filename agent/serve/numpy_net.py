"""The forward pass in pure NumPy. AGENT_SPEC.md §9.2.

No JAX, no torch. Both would trigger JIT/warmup inside the first-move budget, and
in-memory caches do not survive from `build.sh` to `run.sh`. This must stay a
line-for-line mirror of `agent/train/net.py`; `tests/test_agent_parity.py` holds
the two to 1e-4.

Layout note: `eqx.nn.Linear` stores `weight` as (out, in) and computes
`W @ x + b`, so every projection here is `x @ W.T + b`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from agent.spec.codec import decode_action, mask_penalty
from agent.spec.constants import (
    FIRST_PATCH_TOKEN,
    GRID_PATCHES,
    N_CHANNELS,
    N_PLANES,
    PAD,
    PATCH,
    SCALAR_TOKEN,
    VALUE_TOKEN,
)

LAYER_NORM_EPS = 1e-5


def _linear(x, w, b):
    return x @ w.T + b


def _layernorm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LAYER_NORM_EPS) * w + b


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _silu(x):
    return x / (1.0 + np.exp(-x))


class NumpyNet:
    """Loads a `weights.safetensors` written by `agent/serve/weights.py`."""

    def __init__(self, weights: dict[str, np.ndarray], n_head: int):
        self.w = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in weights.items()}
        self.depth = 1 + max(
            int(k.split(".")[1]) for k in self.w if k.startswith("blocks.")
        )
        self.n_head = n_head
        self.embed_dim = self.w["pos_encoding"].shape[1]
        self.head_dim = self.embed_dim // n_head
        self.num_bins = self.w["value_head.weight"].shape[0]
        self.num_bins = self.num_bins if self.num_bins > 1 else 0
        self.bin_centers = self.w.get("bin_centers")

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> NumpyNet:
        """`n_head` cannot be inferred from the shapes, so it rides in the
        safetensors metadata header written by `agent/serve/weights.py`."""
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            meta = f.metadata() or {}
            weights = {k: f.get_tensor(k) for k in f.keys()}
        return cls(weights, n_head=int(meta["n_head"]))

    # ------------------------------------------------------------------
    def _attention(self, x, prefix):
        t = x.shape[0]
        w = self.w

        def heads(name):
            p = _linear(x, w[f"{prefix}.{name}.weight"], w[f"{prefix}.{name}.bias"])
            return p.reshape(t, self.n_head, self.head_dim).transpose(1, 0, 2)

        q, k, v = heads("q"), heads("k"), heads("v")
        scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(np.float32(self.head_dim))
        weights = _softmax(scores.astype(np.float32))
        merged = (weights @ v).transpose(1, 0, 2).reshape(t, self.embed_dim)
        return _linear(merged, w[f"{prefix}.out.weight"], w[f"{prefix}.out.bias"])

    def _trunk(self, obs, scalars):
        w = self.w
        c = obs.shape[0]
        patches = (
            obs.reshape(c, GRID_PATCHES, PATCH, GRID_PATCHES, PATCH)
            .transpose(1, 3, 0, 2, 4)
            .reshape(GRID_PATCHES * GRID_PATCHES, c * PATCH * PATCH)
        )

        x = np.concatenate([
            w["value_token"],
            _linear(scalars[None], w["scalar_proj.weight"], w["scalar_proj.bias"]),
            _linear(patches, w["embedder.weight"], w["embedder.bias"]),
        ]).astype(np.float32) + w["pos_encoding"]

        for i in range(self.depth):
            p = f"blocks.{i}"
            h = _layernorm(x, w[f"{p}.norm1.weight"], w[f"{p}.norm1.bias"])
            x = x + self._attention(h, f"{p}.attn")

            h = _layernorm(x, w[f"{p}.norm2.weight"], w[f"{p}.norm2.bias"])
            h = _silu(_linear(h, w[f"{p}.ff1.weight"], w[f"{p}.ff1.bias"]))
            x = x + _linear(h, w[f"{p}.ff2.weight"], w[f"{p}.ff2.bias"])

        return _layernorm(x, w["norm_out.weight"], w["norm_out.bias"])

    def logits_and_value(self, obs, move_m, build_m, scalars, pad_h=0, pad_w=0):
        w = self.w
        x = self._trunk(obs.astype(np.float32), scalars.astype(np.float32))

        value_raw = _linear(x[VALUE_TOKEN], w["value_head.weight"], w["value_head.bias"])
        pass_logit = _linear(x[SCALAR_TOKEN], w["pass_head.weight"], w["pass_head.bias"])
        patch_logits = _linear(
            x[FIRST_PATCH_TOKEN:], w["policy_head.weight"], w["policy_head.bias"]
        )

        spatial = (
            patch_logits.reshape(GRID_PATCHES, GRID_PATCHES, N_PLANES, PATCH, PATCH)
            .transpose(2, 0, 3, 1, 4)
            .reshape(N_PLANES, PAD, PAD)
        )
        logits = np.concatenate([spatial.reshape(-1), pass_logit]).astype(np.float32)
        # pad_h/pad_w block the strip outside the real board, which exists only
        # here — in training the engine pads with real mountains (delta D4).
        logits = logits + mask_penalty(np, move_m, build_m, pad_h, pad_w)

        if self.num_bins:
            value = float(np.sum(_softmax(value_raw) * self.bin_centers))
        else:
            value = float(value_raw[0])
        return logits, value

    def greedy(self, obs, move_m, build_m, scalars, pad_h=0, pad_w=0):
        """Argmax over masked logits. Serving never samples (§9.3)."""
        logits, value = self.logits_and_value(obs, move_m, build_m, scalars, pad_h, pad_w)
        return decode_action(np, int(np.argmax(logits))), value


assert N_CHANNELS * PATCH * PATCH == 378
