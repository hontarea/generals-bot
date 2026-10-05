"""The policy/value transformer. AGENT_SPEC.md §3.

Single-sample Equinox modules; the rollout and the loss both `vmap` over the batch.
Tokens are `[value, scalars, 49 patches]` = 51. Attention is the only operation
that moves information between tokens, which is what makes reading a head off a
fixed token index meaningful.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from agent.spec.codec import decode_action, encode_action, mask_penalty
from agent.spec.constants import (
    ACTION_DIM,
    FIRST_PATCH_TOKEN,
    GRID_PATCHES,
    N_CHANNELS,
    N_PLANES,
    N_SCALARS,
    N_TOKENS,
    PAD,
    PATCH,
    SCALAR_TOKEN,
    VALUE_TOKEN,
)
from agent.train.config import ModelConfig

PATCH_FEATURES = N_CHANNELS * PATCH * PATCH        # 378
PATCH_OUTPUTS = N_PLANES * PATCH * PATCH           # 81


class MultiHeadSelfAttention(eqx.Module):
    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    out: eqx.nn.Linear
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim: int, n_head: int, *, key):
        kq, kk, kv, ko = jrandom.split(key, 4)
        self.q = eqx.nn.Linear(dim, dim, key=kq)
        self.k = eqx.nn.Linear(dim, dim, key=kk)
        self.v = eqx.nn.Linear(dim, dim, key=kv)
        self.out = eqx.nn.Linear(dim, dim, key=ko)
        self.n_head = n_head
        self.head_dim = dim // n_head

    def __call__(self, x):
        t = x.shape[0]

        def heads(proj):
            return jax.vmap(proj)(x).reshape(t, self.n_head, self.head_dim).transpose(1, 0, 2)

        q, k, v = heads(self.q), heads(self.k), heads(self.v)
        # float32 softmax regardless of activation dtype (§3.7): bf16 logits here
        # cost real accuracy and the attention map is cheap to keep wide.
        scores = (q @ k.transpose(0, 2, 1)).astype(jnp.float32) / jnp.sqrt(
            jnp.float32(self.head_dim)
        )
        weights = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        merged = (weights @ v).transpose(1, 0, 2).reshape(t, -1)
        return jax.vmap(self.out)(merged)


class Block(eqx.Module):
    """Pre-norm attention + SiLU feed-forward, both residual."""

    norm1: eqx.nn.LayerNorm
    attn: MultiHeadSelfAttention
    norm2: eqx.nn.LayerNorm
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    def __init__(self, dim: int, n_head: int, ff_factor: int, *, key):
        ka, k1, k2 = jrandom.split(key, 3)
        hidden = ff_factor * dim
        self.norm1 = eqx.nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_head, key=ka)
        self.norm2 = eqx.nn.LayerNorm(dim)
        self.ff1 = eqx.nn.Linear(dim, hidden, key=k1)
        self.ff2 = eqx.nn.Linear(hidden, dim, key=k2)

    def __call__(self, x):
        x = x + self.attn(jax.vmap(self.norm1)(x))
        h = jax.vmap(self.norm2)(x)
        h = jax.nn.silu(jax.vmap(self.ff1)(h))
        return x + jax.vmap(self.ff2)(h)


def _small_init(linear, scale=0.01):
    """Shrink a head's weights and zero its bias so it starts near-constant.

    Equinox's default `Linear` init puts ~0.58 of logit spread on the LayerNormed
    trunk output — uniform(±1/sqrt(d)) weights make that spread independent of
    `embed_dim` — which costs ~0.16 nats at initialization and gives the policy an
    arbitrary starting preference. §8's go/no-go check wants entropy at
    log(3970) = 8.29, and the entropy schedule is calibrated against that.
    """
    return eqx.tree_at(
        lambda m: (m.weight, m.bias), linear,
        (linear.weight * scale, jnp.zeros_like(linear.bias)),
    )


def _to_bf16(tree):
    return jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_inexact_array(x) else x, tree
    )


class PolicyValueNet(eqx.Module):
    embedder: eqx.nn.Linear
    scalar_proj: eqx.nn.Linear
    value_token: jax.Array
    pos_encoding: jax.Array
    blocks: list
    norm_out: eqx.nn.LayerNorm
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear
    pass_head: eqx.nn.Linear

    num_bins: int = eqx.field(static=True)
    bin_centers: jax.Array
    use_bf16: bool = eqx.field(static=True)

    def __init__(self, cfg: ModelConfig, *, key):
        d = cfg.embed_dim
        ke, ks, kv, kp, kpol, kval, kpass, *kb = jrandom.split(key, 7 + cfg.depth)

        self.embedder = eqx.nn.Linear(PATCH_FEATURES, d, key=ke)
        self.scalar_proj = eqx.nn.Linear(N_SCALARS, d, key=ks)

        # A learned constant, identical for every board: it carries zero board
        # information at layer 0, so everything in it arrives through attention.
        self.value_token = jrandom.normal(kv, (1, d)) * 0.02
        self.pos_encoding = jrandom.truncated_normal(kp, -2.0, 2.0, (N_TOKENS, d)) * 0.1

        self.blocks = [
            Block(d, cfg.n_head, cfg.ff_factor, key=k) for k in kb[: cfg.depth]
        ]
        self.norm_out = eqx.nn.LayerNorm(d)
        self.policy_head = _small_init(eqx.nn.Linear(d, PATCH_OUTPUTS, key=kpol))

        # §3.6: config-only switch. 0 -> scalar head + MSE; >0 -> HL-Gauss.
        self.num_bins = cfg.num_bins if cfg.value_loss == "ce" else 0
        self.value_head = eqx.nn.Linear(d, max(self.num_bins, 1), key=kval)
        self.bin_centers = (
            jnp.linspace(cfg.v_min, cfg.v_max, cfg.num_bins)
            if self.num_bins
            else jnp.zeros(0)
        )

        # Hangs off the SCALAR token, not the value token. Giving pass its own
        # 441-index plane (as the reference does) puts P(pass) at ~10% at init
        # and points the entropy bonus straight at passing; under deathtouch a
        # pass habit is lethal. Collapsed to one index, P(pass) ~ 1/3970.
        self.pass_head = _small_init(eqx.nn.Linear(d, 1, key=kpass))
        self.use_bf16 = cfg.use_bf16

    # ------------------------------------------------------------------
    def _trunk(self, obs, scalars):
        # (C, 21, 21) -> (49, C*9). The transpose reorders to
        # (patch_row, patch_col, channel, dy, dx) so flattening groups one tile's
        # 378 values contiguously. Getting it wrong trains fine on scrambled input.
        c = obs.shape[0]
        patches = (
            obs.reshape(c, GRID_PATCHES, PATCH, GRID_PATCHES, PATCH)
            .transpose(1, 3, 0, 2, 4)
            .reshape(GRID_PATCHES * GRID_PATCHES, c * PATCH * PATCH)
        )

        x = jnp.concatenate([
            self.value_token,
            jax.vmap(self.scalar_proj)(scalars[None]),
            jax.vmap(self.embedder)(patches),
        ]) + self.pos_encoding

        for block in self.blocks:
            x = block(x)
        return jax.vmap(self.norm_out)(x)

    def _logits_and_value(self, obs, move_m, build_m, scalars):
        net = _to_bf16(self) if self.use_bf16 else self
        dtype = jnp.bfloat16 if self.use_bf16 else jnp.float32

        x = net._trunk(obs.astype(dtype), scalars.astype(dtype))

        value_raw = net.value_head(x[VALUE_TOKEN]).astype(jnp.float32)
        pass_logit = net.pass_head(x[SCALAR_TOKEN]).astype(jnp.float32)
        patch_logits = jax.vmap(net.policy_head)(x[FIRST_PATCH_TOKEN:]).astype(jnp.float32)

        # Inverse of the patchify transpose, so plane d, cell (r, c) lands at
        # d * 441 + r * 21 + c — the layout codec.decode_action assumes.
        spatial = (
            patch_logits.reshape(GRID_PATCHES, GRID_PATCHES, N_PLANES, PATCH, PATCH)
            .transpose(2, 0, 3, 1, 4)
            .reshape(N_PLANES, PAD, PAD)
        )

        # float32 *before* the penalty is added: -1e9 overflows bf16's useful range.
        logits = jnp.concatenate([spatial.reshape(-1), pass_logit])
        logits = logits + mask_penalty(jnp, move_m, build_m)

        if self.num_bins:
            probs = jax.nn.softmax(value_raw)
            # stop_gradient: `bin_centers` is an inexact array field, so
            # `eqx.partition(net, is_inexact_array)` files it under *params* and the
            # optimizer would happily move it. The support of a histogram is a
            # coordinate system, not a parameter — letting it drift would move the
            # bins toward the returns instead of the predictions toward the target
            # (D20). `loss.py` builds its bin edges from the config for the same
            # reason, so no gradient path reaches this array from either side.
            return logits, jnp.sum(probs * jax.lax.stop_gradient(self.bin_centers)), value_raw
        return logits, value_raw[0], value_raw[0]

    def __call__(self, obs, move_m, build_m, scalars, key, action=None):
        """Sample (`action=None`) or evaluate a supplied action.

        Both paths go through the *same* logits and the *same* `log_softmax`. If
        they could diverge, every PPO ratio would be silently wrong and the loss
        would still look reasonable — this is the load-bearing line of the file.
        """
        logits, value, value_aux = self._logits_and_value(obs, move_m, build_m, scalars)

        if action is None:
            idx = jrandom.categorical(key, logits)
            action = decode_action(jnp, idx)
        else:
            idx = encode_action(jnp, action)

        lp = jax.nn.log_softmax(logits)
        logprob = lp[idx]
        p_dist = jnp.exp(lp)
        entropy = -jnp.sum(p_dist * lp)

        return action, value, logprob, entropy, value_aux, p_dist

    def greedy(self, obs, move_m, build_m, scalars):
        """Argmax over masked logits. Serving never samples (§9.3)."""
        logits, value, _ = self._logits_and_value(obs, move_m, build_m, scalars)
        return decode_action(jnp, jnp.argmax(logits)), value


def param_count(net) -> int:
    return sum(x.size for x in jax.tree.leaves(eqx.filter(net, eqx.is_inexact_array)))


def make_bench_fns(model_cfg: ModelConfig, cfg):
    """Closures for `scripts/bench_gpu.py`: batched forward, and forward+backward."""
    net = PolicyValueNet(model_cfg, key=jrandom.PRNGKey(0))
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def sample_batch(n):
        k = jrandom.PRNGKey(1)
        return (
            jrandom.normal(k, (n, N_CHANNELS, PAD, PAD), dtype=jnp.float32),
            jnp.ones((n, PAD, PAD, 4), dtype=bool),
            jnp.ones((n, PAD, PAD), dtype=bool),
            jrandom.normal(k, (n, N_SCALARS), dtype=jnp.float32),
            jrandom.split(k, n),
        )

    @jax.jit
    def _fwd(params, batch):
        model = eqx.combine(params, static)
        return jax.vmap(model)(*batch)

    @jax.jit
    def _fwd_bwd(params, batch, actions):
        def loss(p):
            model = eqx.combine(p, static)
            _, value, lp, _, _, _ = jax.vmap(model)(*batch, actions)
            return jnp.mean(lp) + jnp.mean(value**2)

        return jax.grad(loss)(params)

    def fwd(n):
        return _fwd(params, sample_batch(n))

    def fwd_bwd(n):
        batch = sample_batch(n)
        actions = jax.vmap(lambda i: decode_action(jnp, i))(jnp.zeros(n, jnp.int32))
        return _fwd_bwd(params, batch, actions)

    return fwd, fwd_bwd


assert PATCH_FEATURES == 378
assert ACTION_DIM == N_PLANES * PAD * PAD + 1
