"""Gate for build step 3 (AGENT_SPEC.md §10).

The evaluation round-trip is the important one: if sampling and evaluation could
ever produce different log-probs for the same action, every PPO ratio is silently
wrong and the loss still looks reasonable.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from agent.spec import codec, phi
from agent.spec.constants import (
    ACTION_DIM,
    BUILD_PLANE,
    CELLS,
    KIND_BUILD,
    KIND_MOVE,
    KIND_PASS,
    N_CHANNELS,
    N_SCALARS,
    N_TOKENS,
    PAD,
    PASS_IDX,
)
from agent.train.config import ModelConfig
from agent.train.net import PolicyValueNet, param_count

MODEL_CFG = ModelConfig(use_bf16=False)
PARAM_TARGET = 3_428_691        # d=256, L=5, 42 input channels; assert within 1%


@pytest.fixture(scope="module")
def net():
    return PolicyValueNet(MODEL_CFG, key=jrandom.PRNGKey(0))


@pytest.fixture(scope="module")
def sample():
    """A realistic board: real masks, real phi output, not random noise."""
    rng = np.random.default_rng(0)
    o = np.zeros((14, PAD, PAD), dtype=np.int32)
    mountains = rng.random((PAD, PAD)) < 0.25
    owned = (rng.random((PAD, PAD)) < 0.25) & ~mountains
    opponent = (rng.random((PAD, PAD)) < 0.15) & ~mountains & ~owned
    o[0] = rng.integers(0, 60, (PAD, PAD)) * (owned | opponent)
    o[3] = mountains
    o[5] = owned
    o[6] = opponent
    o[1, 4, 4] = 1
    o[5, 4, 4] = 1
    o[9:13] = 20
    o[13] = 137

    obs, scalars, move_m, build_m, _ = phi.augment(jnp, jnp.asarray(o), phi.init_phi_state(jnp))
    return obs, move_m, build_m, scalars


# --------------------------------------------------------------------------


def test_param_count_matches_spec(net):
    n = param_count(net)
    assert abs(n - PARAM_TARGET) / PARAM_TARGET < 0.01, f"{n:,} params, expected ~13.41M"


def test_token_count(net):
    assert net.pos_encoding.shape == (N_TOKENS, MODEL_CFG.embed_dim)
    assert net.value_token.shape == (1, MODEL_CFG.embed_dim)


def test_evaluation_round_trip(net, sample):
    """Sample an action, feed it back in, get exactly the same log-prob.

    Same logits, same log_softmax, both paths. This is the whole point of the
    `action=None` switch.
    """
    for seed in range(50):
        key = jrandom.PRNGKey(seed)
        action, value, lp, ent, aux, p = net(*sample, key)
        action2, value2, lp2, ent2, aux2, p2 = net(*sample, key, action)

        assert jnp.array_equal(action, action2)
        assert lp == lp2, f"log-prob diverged at seed {seed}: {lp} vs {lp2}"
        assert value == value2 and ent == ent2
        assert jnp.array_equal(aux, aux2) and jnp.array_equal(p, p2)


def test_sampled_actions_are_always_legal(net, sample):
    """10k samples: every one legal, or a pass."""
    obs, move_m, build_m, scalars = sample
    mm, bm = np.asarray(move_m), np.asarray(build_m)

    keys = jrandom.split(jrandom.PRNGKey(0), 10_000)
    sampled = jax.jit(jax.vmap(lambda k: net(obs, move_m, build_m, scalars, k)[0]))(keys)

    kinds = np.asarray(sampled[:, 0])
    n_pass = n_build = n_move = 0
    for kind, r, c, d, _s in np.asarray(sampled):
        if kind == KIND_PASS:
            n_pass += 1
        elif kind == KIND_BUILD:
            assert bm[r, c], f"illegal build at ({r}, {c})"
            n_build += 1
        else:
            assert kind == KIND_MOVE and mm[r, c, d], f"illegal move at ({r}, {c}) dir {d}"
            n_move += 1

    assert n_move > 0
    assert set(np.unique(kinds)) <= {KIND_MOVE, KIND_PASS, KIND_BUILD}


def test_pass_probability_is_not_inflated_at_init(net, sample):
    """The collapsed pass index must sit near 1/3970, not near 10%.

    A per-cell pass plane (the reference's layout) would give the entropy bonus a
    structural pull toward passing, which is lethal under deathtouch.
    """
    _, _, _, _, _, p = net(*sample, jrandom.PRNGKey(0))
    p_pass = float(p[PASS_IDX])
    assert p_pass < 0.01, f"P(pass) = {p_pass:.4f} at init"


def test_entropy_at_init_is_near_uniform(net):
    """Near log(n_legal); with everything legal that is log(3970) ~ 8.29."""
    obs = jnp.zeros((N_CHANNELS, PAD, PAD), dtype=jnp.float32)
    scalars = jnp.zeros(N_SCALARS, dtype=jnp.float32)
    move_m = jnp.ones((PAD, PAD, 4), dtype=bool)
    build_m = jnp.ones((PAD, PAD), dtype=bool)

    _, _, _, ent, _, _ = net(obs, move_m, build_m, scalars, jrandom.PRNGKey(0))
    assert abs(float(ent) - float(jnp.log(ACTION_DIM))) < 0.05


def test_masked_logits_carry_no_probability(net, sample):
    _, _, _, _, _, p = net(*sample, jrandom.PRNGKey(0))
    pen = np.asarray(codec.mask_penalty(jnp, sample[1], sample[2]))
    assert np.all(np.asarray(p)[pen < 0] == 0.0), "an illegal action has mass"
    assert np.isclose(float(jnp.sum(p)), 1.0, atol=1e-5)


def test_no_nan_when_only_pass_is_legal(net):
    """-1e9 rather than -inf exists precisely for this case."""
    obs = jnp.zeros((N_CHANNELS, PAD, PAD), dtype=jnp.float32)
    scalars = jnp.zeros(N_SCALARS, dtype=jnp.float32)
    move_m = jnp.zeros((PAD, PAD, 4), dtype=bool)
    build_m = jnp.zeros((PAD, PAD), dtype=bool)

    action, value, lp, ent, _, p = net(obs, move_m, build_m, scalars, jrandom.PRNGKey(0))
    assert int(action[0]) == KIND_PASS
    for x in (value, lp, ent):
        assert jnp.isfinite(x)
    assert np.isclose(float(p[PASS_IDX]), 1.0, atol=1e-5)


def test_policy_head_unpatchify_is_the_codec_layout():
    """A logit written at plane d, cell (r, c) must be readable at d*441 + r*21 + c.

    Getting this transpose wrong trains perfectly well on scrambled input, so it
    is checked against the codec directly rather than through the network.
    """
    from agent.spec.constants import GRID_PATCHES, N_PLANES, PATCH

    # Give every (plane, cell) a unique value, run it through the exact reshape
    # the head uses, and check the flat index agrees with decode_action.
    target = np.arange(N_PLANES * PAD * PAD, dtype=np.float32).reshape(N_PLANES, PAD, PAD)
    patch_logits = (
        target.reshape(N_PLANES, GRID_PATCHES, PATCH, GRID_PATCHES, PATCH)
        .transpose(1, 3, 0, 2, 4)
        .reshape(GRID_PATCHES * GRID_PATCHES, N_PLANES * PATCH * PATCH)
    )
    spatial = (
        patch_logits.reshape(GRID_PATCHES, GRID_PATCHES, N_PLANES, PATCH, PATCH)
        .transpose(2, 0, 3, 1, 4)
        .reshape(N_PLANES, PAD, PAD)
    )
    assert np.array_equal(spatial, target)

    flat = spatial.reshape(-1)
    for idx in (0, 1, 440, 441, 3 * CELLS + 7 * PAD + 11, BUILD_PLANE * CELLS + 20 * PAD + 20):
        kind, r, c, d, s = np.asarray(codec.decode_action(np, idx))
        plane = BUILD_PLANE if kind == KIND_BUILD else s * 4 + d
        assert flat[idx] == target[plane, r, c]


def test_value_head_fork_shapes():
    """§3.6: `value` feeds GAE, `value_aux` feeds the loss, switch is config-only."""
    obs = jnp.zeros((N_CHANNELS, PAD, PAD), dtype=jnp.float32)
    scalars = jnp.zeros(N_SCALARS, dtype=jnp.float32)
    mm = jnp.ones((PAD, PAD, 4), dtype=bool)
    bm = jnp.ones((PAD, PAD), dtype=bool)

    mse = PolicyValueNet(ModelConfig(use_bf16=False), key=jrandom.PRNGKey(0))
    _, v, _, _, aux, _ = mse(obs, mm, bm, scalars, jrandom.PRNGKey(0))
    assert v.shape == () and aux.shape == ()

    ce = PolicyValueNet(
        ModelConfig(use_bf16=False, value_loss="ce", num_bins=128), key=jrandom.PRNGKey(0)
    )
    _, v, _, _, aux, _ = ce(obs, mm, bm, scalars, jrandom.PRNGKey(0))
    assert v.shape == () and aux.shape == (128,)
    assert -1.0 <= float(v) <= 1.0, "HL-Gauss value must stay inside [v_min, v_max]"


def test_batched_jit_vmap(net, sample):
    obs, move_m, build_m, scalars = sample
    batch = 16
    stack = lambda x: jnp.broadcast_to(x, (batch, *x.shape))  # noqa: E731
    keys = jrandom.split(jrandom.PRNGKey(0), batch)

    fn = jax.jit(jax.vmap(net))
    action, value, lp, ent, aux, p = fn(
        stack(obs), stack(move_m), stack(build_m), stack(scalars), keys
    )
    assert action.shape == (batch, 5)
    assert value.shape == (batch,) and lp.shape == (batch,)
    assert p.shape == (batch, ACTION_DIM)


def test_bf16_matches_fp32_closely(sample):
    """bf16 is a throughput choice, not a behavior change; the softmax and the
    logits stay float32 (§3.7)."""
    obs, mm, bm, scalars = sample
    key = jrandom.PRNGKey(0)
    fp32 = PolicyValueNet(ModelConfig(use_bf16=False), key=jrandom.PRNGKey(0))
    bf16 = PolicyValueNet(ModelConfig(use_bf16=True), key=jrandom.PRNGKey(0))

    _, v32, _, e32, _, p32 = fp32(obs, mm, bm, scalars, key)
    _, v16, _, e16, _, p16 = bf16(obs, mm, bm, scalars, key)

    assert abs(float(v32) - float(v16)) < 0.15
    assert abs(float(e32) - float(e16)) < 0.05
    assert jnp.max(jnp.abs(p32 - p16)) < 0.01


def test_greedy_is_deterministic_and_legal(net, sample):
    obs, move_m, build_m, scalars = sample
    a1, v1 = net.greedy(obs, move_m, build_m, scalars)
    a2, _ = net.greedy(obs, move_m, build_m, scalars)
    assert jnp.array_equal(a1, a2)

    kind, r, c, d, _ = np.asarray(a1)
    if kind == KIND_MOVE:
        assert np.asarray(move_m)[r, c, d]
    elif kind == KIND_BUILD:
        assert np.asarray(build_m)[r, c]
    assert jnp.isfinite(v1)


def test_gradients_flow_to_every_head(net, sample):
    obs, mm, bm, scalars = sample
    params, static = eqx.partition(net, eqx.is_inexact_array)

    def loss(p):
        model = eqx.combine(p, static)
        _, value, lp, ent, _, _ = model(obs, mm, bm, scalars, jrandom.PRNGKey(0))
        return lp + value + ent

    grads = jax.grad(loss)(params)
    for name in ("policy_head", "value_head", "pass_head", "embedder", "scalar_proj"):
        g = getattr(grads, name).weight
        assert jnp.any(g != 0), f"{name} received no gradient"
    assert jnp.any(grads.value_token != 0)
    assert jnp.any(grads.pos_encoding != 0)
