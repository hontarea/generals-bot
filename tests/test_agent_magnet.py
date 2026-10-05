"""Tests for the expander magnet (AGENT_SPEC.md §6.1).

The magnet is off by default (`cfg.use_magnet`); these pin its behaviour so that
turning it on later is a config change rather than a debugging session.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agent.spec import codec, phi
from agent.spec.constants import (
    ACTION_DIM,
    BUILD_PLANE,
    CELLS,
    N_CHANNELS,
    PAD,
    PASS_IDX,
)
from agent.train.magnet import (
    HALF_MOVE_PENALTY,
    SCORE_BUILD,
    SCORE_ENEMY,
    SCORE_NEUTRAL,
    expander_magnet,
)


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    o = np.zeros((14, PAD, PAD), dtype=np.int32)
    mountains = rng.random((PAD, PAD)) < 0.2
    owned = (rng.random((PAD, PAD)) < 0.3) & ~mountains
    opponent = (rng.random((PAD, PAD)) < 0.15) & ~mountains & ~owned
    o[0] = rng.integers(0, 60, (PAD, PAD)) * (owned | opponent)
    o[3], o[5], o[6] = mountains, owned, opponent
    o[4] = ~(owned | opponent | mountains)
    o[1, 3, 3] = 1
    o[5, 3, 3] = 1
    o[9:13] = 20
    o[13] = 100
    obs, sc, mm, bm, _ = phi.augment(jnp, jnp.asarray(o), phi.init_phi_state(jnp))
    return obs, sc, mm, bm


def test_is_a_distribution(sample):
    obs, _, mm, bm = sample
    m = expander_magnet(obs, mm, bm)
    assert m.shape == (ACTION_DIM,)
    assert float(jnp.sum(m)) == pytest.approx(1.0, abs=1e-5)
    assert bool(jnp.all(m >= 0.0)) and bool(jnp.all(jnp.isfinite(m)))


def test_puts_no_mass_on_illegal_actions(sample):
    obs, _, mm, bm = sample
    m = np.asarray(expander_magnet(obs, mm, bm))
    illegal = np.asarray(codec.mask_penalty(jnp, mm, bm)) < 0
    assert m[illegal].sum() == pytest.approx(0.0, abs=1e-9)


def test_half_moves_score_below_full_moves(sample):
    """The reference reuses one array for both planes, making its magnet
    indifferent to splitting — which no real expander heuristic is."""
    obs, _, mm, bm = sample
    m = np.asarray(expander_magnet(obs, mm, bm))
    full = m[: 4 * CELLS].sum()
    half = m[4 * CELLS: 8 * CELLS].sum()
    assert full > half, f"full {full} !> half {half}"
    # Ratio should be exp(HALF_MOVE_PENALTY) per legal move.
    assert full / half == pytest.approx(np.exp(HALF_MOVE_PENALTY), rel=0.05)


def test_pass_mass_is_negligible(sample):
    """The collapsed pass index already fixes the reference's inflated pass mass;
    under deathtouch a pass habit is lethal."""
    obs, _, mm, bm = sample
    m = expander_magnet(obs, mm, bm)
    assert float(m[PASS_IDX]) < 1e-3


def test_priority_order_is_build_enemy_neutral_default():
    """Scores are logits, so it is the differences that matter."""
    assert SCORE_BUILD > SCORE_ENEMY > SCORE_NEUTRAL > 1.0


def test_prefers_building_when_affordable(sample):
    obs, _, mm, bm = sample
    m = np.asarray(expander_magnet(obs, mm, bm))
    build_mass = m[BUILD_PLANE * CELLS: PASS_IDX].sum()
    n_builds = int(np.asarray(bm).sum())
    if n_builds:
        assert build_mass > 0.1, "affordable builds carry almost no mass"


def test_does_not_wrap_around_the_border():
    """`jnp.roll` would reward moving off the edge; `shift` pads instead."""
    o = np.zeros((14, PAD, PAD), dtype=np.int32)
    o[5, 0, :] = 1                    # own the top row only
    o[0, 0, :] = 30
    o[4] = 1
    o[4, 0, :] = 0
    o[1, 0, 0] = 1
    o[9:13] = 10
    obs, _, mm, bm, _ = phi.augment(jnp, jnp.asarray(o), phi.init_phi_state(jnp))

    m = np.asarray(expander_magnet(obs, mm, bm))
    # Plane 0 is UP; the whole top row moving UP is off-board and illegal.
    up_from_top_row = m[0 * CELLS: 0 * CELLS + PAD]
    assert up_from_top_row.sum() == pytest.approx(0.0, abs=1e-9)


def test_is_jittable_and_vmappable(sample):
    obs, _, mm, bm = sample
    batch = 4
    stack = lambda x: jnp.broadcast_to(x, (batch, *x.shape))  # noqa: E731
    out = jax.jit(jax.vmap(expander_magnet))(stack(obs), stack(mm), stack(bm))
    assert out.shape == (batch, ACTION_DIM)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_survives_a_position_with_no_legal_move():
    """Softmax over all -inf would be NaN; pass keeps the distribution defined."""
    obs = jnp.zeros((N_CHANNELS, PAD, PAD), dtype=jnp.float32)
    mm = jnp.zeros((PAD, PAD, 4), dtype=bool)
    bm = jnp.zeros((PAD, PAD), dtype=bool)
    m = expander_magnet(obs, mm, bm)
    assert bool(jnp.all(jnp.isfinite(m)))
    assert float(m[PASS_IDX]) == pytest.approx(1.0)


def test_magnet_regularizer_is_not_the_entropy_bonus(sample):
    """Sanity that swapping the magnet in actually changes the objective.

    With a uniform magnet `reg` reduces to `-entropy + log(3970)`; with this one
    it must not.
    """
    from agent.train.loss import uniform_magnet

    obs, _, mm, bm = sample
    p = jnp.full((ACTION_DIM,), 1.0 / ACTION_DIM)

    uni = jnp.sum(p * jnp.log(uniform_magnet(obs, mm, bm) + 1e-10))
    exp = jnp.sum(p * jnp.log(expander_magnet(obs, mm, bm) + 1e-10))
    assert not jnp.allclose(uni, exp)
