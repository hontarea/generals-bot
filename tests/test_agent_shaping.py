"""Gate for potential-based shaping (AGENT_SPEC_DELTAS.md D19).

Shaping is only safe because `F = Phi(s') - Phi(s)` cannot change which policy is
optimal, and that argument rests on three things being literally true rather than
approximately true: the potential is antisymmetric between seats, it is bounded,
and it is zeroed at a terminal so the sum over an episode telescopes to a constant.
Each has its own test here, because each fails silently — a run with a subtly wrong
`Phi` looks exactly like a run that is learning slowly.

The behavioural claim of the whole feature is one test:
`test_building_a_castle_raises_the_potential`. If that ever goes negative, shaping
is teaching the agent *not* to build and the weights are wrong.
"""
from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agent.train.config import SMOKE, get_config
from agent.train.gae import prepare_advantages, shaped_rewards
from agent.train.shaping import ShapingWeights, unit_potential, unit_potentials
from generals.core import game
from generals.modifiers import build_castles as bc

W = ShapingWeights()
COEF = 0.2

PASS = jnp.array([1, 0, 0, 0, 0], dtype=jnp.int32)


# --------------------------------------------------------------------------
# state builders — same shape as tests/test_build_castles.py
# --------------------------------------------------------------------------


def make_grid(size=12):
    grid = jnp.zeros((size, size), dtype=jnp.int32)
    grid = grid.at[0, 0].set(1)
    grid = grid.at[size - 1, size - 1].set(2)
    return grid


def give_cell(state, player, ij, army):
    i, j = ij
    return state._replace(
        armies=state.armies.at[i, j].set(army),
        ownership=state.ownership.at[player, i, j].set(True),
        ownership_neutral=state.ownership_neutral.at[i, j].set(False),
    )


def give_castle(state, player, ij):
    state = give_cell(state, player, ij, 1)
    return state._replace(castles=state.castles.at[ij].set(True))


def fresh(size=12):
    return game.create_initial_state(make_grid(size))


# --------------------------------------------------------------------------
# the potential itself
# --------------------------------------------------------------------------


def test_symmetric_opening_has_zero_potential():
    """Both generals, one tile each, no castles — the gap is zero and so is Phi."""
    assert float(unit_potential(fresh(), 0, W)) == pytest.approx(0.0, abs=1e-7)


def test_potential_is_antisymmetric():
    """Phi(s, 1) == -Phi(s, 0) exactly, which is what keeps shaping zero-sum."""
    state = give_castle(give_cell(fresh(), 0, (2, 3), 40), 0, (5, 5))
    state = give_cell(state, 1, (9, 9), 7)

    p0 = float(unit_potential(state, 0, W))
    p1 = float(unit_potential(state, 1, W))
    assert p0 != 0.0, "the fixture should be lopsided enough to be interesting"
    assert p1 == pytest.approx(-p0, abs=1e-7)


def test_potential_is_bounded_under_absurd_material():
    """tanh, not a clip: an unreachable army lead still lands inside [-1, 1]."""
    state = fresh()
    state = state._replace(
        armies=jnp.full_like(state.armies, 10_000),
        ownership=state.ownership.at[0].set(True),
    )
    p = float(unit_potential(state, 0, W))
    assert 0.0 < p <= 1.0
    assert abs(COEF * p) <= COEF


def test_symmetric_growth_produces_no_shaping():
    """The difference form's whole point: both economies growing pays nothing.

    A tick of structure growth hands +1 to each general. If `Phi` read absolute
    material instead of the gap, this would leak a small positive reward to both
    seats on every even turn — and over 277 turns that is not small.
    """
    before = fresh()
    after = before._replace(armies=before.armies + before.generals.astype(before.armies.dtype))

    assert float(unit_potential(after, 0, W)) == pytest.approx(
        float(unit_potential(before, 0, W)), abs=1e-7
    )


def test_batched_potentials_follow_the_row_layout():
    """(N, ...) -> (2N,), seat 0 first, seat 1 the negation."""
    a = give_cell(fresh(), 0, (2, 2), 30)
    b = give_cell(fresh(), 1, (7, 7), 30)
    states = jax.tree.map(lambda *xs: jnp.stack(xs), a, b)

    out = np.asarray(unit_potentials(states, W))
    assert out.shape == (4,)
    assert out[2] == pytest.approx(-out[0], abs=1e-7)
    assert out[3] == pytest.approx(-out[1], abs=1e-7)
    # a favours seat 0, b favours seat 1.
    assert out[0] > 0.0 > out[1]


# --------------------------------------------------------------------------
# the behavioural claim
# --------------------------------------------------------------------------


def test_building_a_castle_raises_the_potential():
    """**The point of the feature.** A build must read as a gain immediately.

    It is not obvious that it does: the castle is +2.5/10 of `z` but the 35 army it
    consumes is -0.2*35/100, so the sign depends entirely on `w_castle : w_army`.
    This is the test that fails first if those weights get retuned carelessly.
    """
    state = give_cell(fresh(), 0, (4, 4), 40)          # far from the general: cost 35
    before = float(unit_potential(state, 0, W))

    after_state, actions = bc.apply_build_actions(
        state, jnp.stack([jnp.array([bc.BUILD, 4, 4, 0, 0], dtype=jnp.int32), PASS])
    )
    assert bool(after_state.castles[4, 4]), "fixture did not actually build"
    assert int(actions[0][0]) == 1, "the build should come back rewritten as a pass"

    after = float(unit_potential(after_state, 0, W))
    delta = COEF * (after - before)
    assert delta > 0.0, f"building read as a loss ({delta:+.4f}) — check w_castle:w_army"
    assert delta == pytest.approx(0.036, abs=0.005), (
        f"the documented +0.036 per build has moved to {delta:+.4f}; "
        "update ShapingWeights' docstring and D19 together"
    )


def test_an_unaffordable_build_is_worth_nothing():
    """Invalid builds are consumed as passes, so they must not move `Phi` at all."""
    state = give_cell(fresh(), 0, (4, 4), 34)          # one short of the 35 cost
    before = float(unit_potential(state, 0, W))

    after_state, _ = bc.apply_build_actions(
        state, jnp.stack([jnp.array([bc.BUILD, 4, 4, 0, 0], dtype=jnp.int32), PASS])
    )
    assert not bool(after_state.castles[4, 4])
    assert float(unit_potential(after_state, 0, W)) == pytest.approx(before, abs=1e-7)


def test_losing_a_castle_costs_what_gaining_one_paid():
    """Symmetry of the potential under a capture: what the loser drops, the
    winner picks up. Nothing is created by the transfer itself."""
    state = give_castle(fresh(), 0, (6, 6))
    captured = give_castle(state, 1, (6, 6))._replace(
        ownership=state.ownership.at[0, 6, 6].set(False).at[1, 6, 6].set(True)
    )
    assert float(unit_potential(captured, 0, W)) < float(unit_potential(state, 0, W))


# --------------------------------------------------------------------------
# the shaped reward
# --------------------------------------------------------------------------


def _fake_rollout(rews, phis, phis_next, terminated, truncated=None, vals=None):
    """(T,) sequences for one row. Only the fields `shaped_rewards` reads."""
    col = lambda x, dt=jnp.float32: jnp.asarray(np.array(x), dtype=dt)[:, None]  # noqa: E731
    t = len(rews)
    return SimpleNamespace(
        rews=col(rews),
        phis=col(phis),
        phis_next=col(phis_next),
        terminated=col(terminated, jnp.bool_),
        truncated=col(truncated if truncated is not None else [0] * t, jnp.bool_),
        vals=col(vals if vals is not None else [0.0] * t),
        next_vals=col([0.0] * t),
    )


def test_zero_coefficient_is_bit_identical():
    """The inertness guarantee: `shaping_coef = 0.0` must not perturb a single
    float, or every pre-D19 comparison quietly stops being a comparison."""
    rng = np.random.default_rng(0)
    r = _fake_rollout(
        rews=rng.normal(size=16),
        phis=rng.normal(size=16),
        phis_next=rng.normal(size=16),
        terminated=rng.random(16) < 0.2,
    )
    out, shaping = shaped_rewards(r, 0.0)
    assert np.array_equal(np.asarray(out), np.asarray(r.rews))
    assert not np.any(np.asarray(shaping))


def test_terminal_next_potential_is_ignored():
    """Guard twin: on a terminal step the stored `phis_next` belongs to the
    auto-reset next game, so it must contribute nothing however large it is."""
    base = _fake_rollout([0, 0, 1], [0.1, 0.2, 0.3], [0.2, 0.3, 0.0], [0, 0, 1])
    poisoned = _fake_rollout([0, 0, 1], [0.1, 0.2, 0.3], [0.2, 0.3, 999.0], [0, 0, 1])
    assert np.allclose(
        np.asarray(shaped_rewards(base, COEF)[0]),
        np.asarray(shaped_rewards(poisoned, COEF)[0]),
    )


def test_shaping_telescopes_to_minus_phi_zero():
    """Sum over a completed episode == -coef * Phi(s_0).

    This is the policy-invariance argument in one line: whatever shaping pays out
    along the way, it takes back by the end, and what remains depends only on the
    *starting* position — which no policy can influence.
    """
    phis = [0.1, 0.25, -0.05, 0.4]
    phis_next = phis[1:] + [0.9]                       # the last one is discarded
    r = _fake_rollout([0, 0, 0, 1], phis, phis_next, [0, 0, 0, 1])

    _, shaping = shaped_rewards(r, COEF)
    assert float(jnp.sum(shaping)) == pytest.approx(-COEF * phis[0], abs=1e-6)


def test_shaped_return_stays_inside_the_hl_gauss_support():
    """|G| <= 1 + coef, which is what `v_min`/`v_max` are set from (D19)."""
    phis = [1.0, -1.0, 1.0, -1.0]                      # worst case for the bound
    r = _fake_rollout([0, 0, 0, -1], phis, phis[1:] + [0.0], [0, 0, 0, 1])
    rews, _ = shaped_rewards(r, COEF)
    assert float(jnp.max(jnp.abs(jnp.cumsum(np.asarray(rews)[::-1])))) <= 1.0 + COEF + 1e-6


def test_prepare_advantages_reports_shaping_diagnostics():
    rng = np.random.default_rng(1)
    t, rows = 12, 4
    r = SimpleNamespace(
        rews=jnp.zeros((t, rows)),
        vals=jnp.asarray(rng.normal(size=(t, rows)), dtype=jnp.float32),
        next_vals=jnp.asarray(rng.normal(size=(t, rows)), dtype=jnp.float32),
        terminated=jnp.zeros((t, rows), dtype=bool),
        truncated=jnp.zeros((t, rows), dtype=bool),
        phis=jnp.asarray(rng.normal(size=(t, rows)) * 0.3, dtype=jnp.float32),
        phis_next=jnp.asarray(rng.normal(size=(t, rows)) * 0.3, dtype=jnp.float32),
    )
    _, diag = prepare_advantages(r, 1.0, 0.9, COEF)
    assert diag.shaping_abs_mean > 0.0
    _, off = prepare_advantages(r, 1.0, 0.9, 0.0)
    assert off.shaping_abs_mean == 0.0


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------


def test_rollout_carries_antisymmetric_potentials():
    """End to end: `Phi` survives the rollout with the seat layout intact.

    The per-row antisymmetry is exact; the *mean* over the buffer only cancels to
    float32 summation order, so it is checked against `shaping_abs_mean` rather
    than against zero. On a real buffer the gap is ~9 orders of magnitude.
    """
    import equinox as eqx
    import jax.random as jrandom

    from agent.train.net import PolicyValueNet
    from agent.train.rollout import collect_rollout, init_rollout_state, make_step_fn
    from generals.core.env import GeneralsEnv

    n_envs, t = 3, 4
    env = GeneralsEnv(mode="competition", pool_size=32)
    net = PolicyValueNet(SMOKE.model.__class__(use_bf16=False), key=jrandom.PRNGKey(0))
    params, static = eqx.partition(net, eqx.is_inexact_array)
    step_fn = make_step_fn(env, static, W)

    pool, states, phi_state, key = init_rollout_state(env, jrandom.PRNGKey(3), n_envs, 32)
    out, _, _, _ = collect_rollout(
        env, step_fn, params, static, states, phi_state, key, pool, t
    )

    assert out.phis.shape == (t, 2 * n_envs)
    assert out.phis_next.shape == (t, 2 * n_envs)

    phis = np.asarray(out.phis)
    assert np.allclose(phis[:, n_envs:], -phis[:, :n_envs], atol=1e-6)

    _, diag = prepare_advantages(out, 1.0, 0.9, COEF)
    assert diag.shaping_abs_mean > 0.0, "shaping produced nothing to check"
    assert abs(diag.shaping_mean) < diag.shaping_abs_mean * 1e-3


def test_marathon_preset_support_covers_the_shaped_range():
    """The two constants that have to agree across three modules."""
    cfg = get_config("marathon")
    assert cfg.model.v_max == pytest.approx(1.0 + cfg.shaping_coef)
    assert cfg.model.v_min == pytest.approx(-(1.0 + cfg.shaping_coef))
    # sigma at ~0.75 of a bin width, per D20
    width = (cfg.model.v_max - cfg.model.v_min) / (cfg.model.num_bins - 1)
    assert cfg.model.hl_sigma == pytest.approx(0.75 * width, rel=0.05)


def test_default_presets_leave_shaping_off():
    """`full` must stay exactly what rl_bot_b ran, or the baseline moves."""
    for name in ("tiny", "smoke", "full"):
        assert get_config(name).shaping_coef == 0.0
        assert get_config(name).ent_target == 0.0
