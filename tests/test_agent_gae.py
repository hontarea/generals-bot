"""Gate for build step 5 (AGENT_SPEC.md §10).

Hand-computed episode, the lambda=1 identity, truncation isolation, and the
zero-signal regime that makes the curriculum a precondition rather than a
convenience.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from agent.train.gae import (
    Diagnostics,
    compute_gae,
    compute_mc_returns,
    prepare_advantages,
)


def _col(*values, dtype=jnp.float32):
    """A single-column (T, 1) array — one game, one seat."""
    return jnp.asarray(np.array(values, dtype=np.asarray(values).dtype)[:, None]).astype(dtype)


def _episode(rewards, values, next_values, terminated, truncated):
    return dict(
        rewards=_col(*rewards),
        values=_col(*values),
        next_values=_col(*next_values),
        terminated=_col(*terminated, dtype=bool),
        truncated=_col(*truncated, dtype=bool),
    )


# --------------------------------------------------------------------------
# hand-computed
# --------------------------------------------------------------------------


def test_hand_computed_three_step_episode():
    """A win on step 2, gamma=1, lambda=0.9, worked out by hand.

        v = [0.1, 0.2, 0.3],  next_v = [0.2, 0.3, 0.9(stale)],  r = [0, 0, 1]

        t=2 terminal: bootstrap 0, d2 = 1 - 0.3           = 0.7
                      a2 = d2                             = 0.7
        t=1:          d1 = 0 + 0.3 - 0.2                  = 0.1
                      a1 = 0.1 + 0.9 * a2                 = 0.73
        t=0:          d0 = 0 + 0.2 - 0.1                  = 0.1
                      a0 = 0.1 + 0.9 * a1                 = 0.757
    """
    ep = _episode([0, 0, 1], [0.1, 0.2, 0.3], [0.2, 0.3, 0.9], [0, 0, 1], [0, 0, 0])
    advs = np.asarray(compute_gae(**ep, gamma=1.0, gae_lambda=0.9))[:, 0]
    assert np.allclose(advs, [0.757, 0.73, 0.7], atol=1e-5)


def test_stale_next_value_is_ignored_on_termination():
    """Guard 1: the bootstrap on a terminal step belongs to the auto-reset next
    game and must contribute nothing, however large it is."""
    base = _episode([0, 0, 1], [0.1, 0.2, 0.3], [0.2, 0.3, 0.0], [0, 0, 1], [0, 0, 0])
    poisoned = dict(base, next_values=_col(0.2, 0.3, 999.0))
    assert np.allclose(
        np.asarray(compute_gae(**base)), np.asarray(compute_gae(**poisoned))
    )


# --------------------------------------------------------------------------
# lambda = 1 identity
# --------------------------------------------------------------------------


def test_lambda_one_identity_recovers_monte_carlo_returns():
    """With lambda=1 and gamma=1, `gae + values` is exactly the MC return for
    every transition whose episode finished inside the window."""
    rng = np.random.default_rng(0)
    t, rows = 24, 5
    terminated = np.zeros((t, rows), dtype=bool)
    terminated[7, :] = True
    terminated[18, :] = True
    truncated = np.zeros((t, rows), dtype=bool)
    rewards = np.zeros((t, rows), dtype=np.float32)
    rewards[terminated] = rng.choice([-1.0, 1.0], size=terminated.sum())
    values = rng.normal(0, 0.5, (t, rows)).astype(np.float32)
    next_values = np.concatenate([values[1:], rng.normal(0, 0.5, (1, rows))]).astype(np.float32)

    args = dict(
        rewards=jnp.asarray(rewards), values=jnp.asarray(values),
        next_values=jnp.asarray(next_values),
        terminated=jnp.asarray(terminated), truncated=jnp.asarray(truncated),
    )
    advs = compute_gae(**args, gamma=1.0, gae_lambda=1.0)
    rets = np.asarray(advs + jnp.asarray(values))
    mc, valid = compute_mc_returns(
        jnp.asarray(rewards), jnp.asarray(terminated), jnp.asarray(truncated), gamma=1.0
    )
    mc, valid = np.asarray(mc), np.asarray(valid)

    assert valid[:19].all(), "everything before the last terminal should resolve"
    assert np.allclose(rets[valid], mc[valid], atol=1e-4)


def test_mc_returns_are_terminal_rewards():
    """Undiscounted, terminal-only rewards: every resolved return is exactly the
    episode's terminal reward. This is what pins the value range to [-1, 1]."""
    rewards = np.zeros((10, 1), dtype=np.float32)
    terminated = np.zeros((10, 1), dtype=bool)
    rewards[4, 0], terminated[4, 0] = 1.0, True
    rewards[9, 0], terminated[9, 0] = -1.0, True

    mc, valid = compute_mc_returns(
        jnp.asarray(rewards), jnp.asarray(terminated), jnp.zeros((10, 1), dtype=bool)
    )
    mc, valid = np.asarray(mc)[:, 0], np.asarray(valid)[:, 0]
    assert valid.all()
    assert np.allclose(mc[:5], 1.0)
    assert np.allclose(mc[5:], -1.0)


# --------------------------------------------------------------------------
# truncation isolation
# --------------------------------------------------------------------------


def test_truncation_stops_backward_propagation():
    """Guard 2: a truncated step's advantage must not leak into earlier steps.

    Steps before the truncation see only their own bootstrapped delta.
    """
    ep = _episode(
        [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 5.0], [0, 0, 0, 0], [0, 0, 1, 0]
    )
    advs = np.asarray(compute_gae(**ep, gamma=1.0, gae_lambda=0.9))[:, 0]

    # t=2 truncates; its own advantage still exists (delta with a live bootstrap)
    # but the carry into t=1 is zeroed.
    assert np.isclose(advs[1], 0.3 - 0.2), "advantage leaked backwards through a truncation"
    assert np.isclose(advs[0], (0.2 - 0.1) + 0.9 * advs[1])


def test_truncated_transitions_are_dropped_by_train_mask():
    """Guard 3: whatever the advantage is, the transition is not learned from."""
    rollout = _fake_rollout(truncate_at=3)
    adv, _ = prepare_advantages(rollout)
    assert float(adv.train_mask[3, 0]) == 0.0
    assert float(adv.train_mask[2, 0]) == 1.0


def test_mc_returns_do_not_resolve_across_a_truncation():
    rewards = np.zeros((8, 1), dtype=np.float32)
    terminated = np.zeros((8, 1), dtype=bool)
    truncated = np.zeros((8, 1), dtype=bool)
    rewards[6, 0], terminated[6, 0] = 1.0, True
    truncated[3, 0] = True

    _, valid = compute_mc_returns(
        jnp.asarray(rewards), jnp.asarray(terminated), jnp.asarray(truncated)
    )
    valid = np.asarray(valid)[:, 0]
    assert valid[4:7].all(), "the completed episode after the truncation should resolve"
    assert not valid[:4].any(), "returns resolved through a truncation boundary"


# --------------------------------------------------------------------------
# ordering and the zero-signal regime
# --------------------------------------------------------------------------


class _R:
    """Minimal stand-in for `Rollout`.

    The shaping step reads `phis`/`phis_next` before anything else runs, so they
    are zero here: the shaping term is then identically zero and these tests stay
    about GAE alone (`tests/test_agent_shaping.py` owns the potentials).
    """

    def __init__(self, rews, vals, next_vals, terminated, truncated,
                 phis=None, phis_next=None):
        self.rews, self.vals, self.next_vals = rews, vals, next_vals
        self.terminated, self.truncated = terminated, truncated
        self.phis = jnp.zeros_like(rews) if phis is None else phis
        self.phis_next = jnp.zeros_like(rews) if phis_next is None else phis_next


def _fake_rollout(terminate_at=None, truncate_at=None, t=12, rows=4, seed=0):
    rng = np.random.default_rng(seed)
    rews = np.zeros((t, rows), dtype=np.float32)
    terminated = np.zeros((t, rows), dtype=bool)
    truncated = np.zeros((t, rows), dtype=bool)
    if terminate_at is not None:
        terminated[terminate_at, :] = True
        rews[terminate_at, :] = 1.0
    if truncate_at is not None:
        truncated[truncate_at, :] = True
    vals = rng.normal(0, 0.3, (t, rows)).astype(np.float32)
    next_vals = np.concatenate([vals[1:], vals[-1:]])
    return _R(*(jnp.asarray(x) for x in (rews, vals, next_vals, terminated, truncated)))


def test_returns_are_computed_before_normalization():
    """`rets = advs + vals` must precede normalization.

    Reversing the two lines rescales `rets` out of [-1, 1] and silently breaks
    HL-Gauss two modules away, with nothing local to show for it. Here: the
    returns must match an unnormalized recomputation, while advs must not.
    """
    rollout = _fake_rollout(terminate_at=6)
    adv, _ = prepare_advantages(rollout)

    raw = compute_gae(
        rollout.rews, rollout.vals, rollout.next_vals,
        rollout.terminated, rollout.truncated, 1.0, 0.9,
    )
    assert np.allclose(np.asarray(adv.rets), np.asarray(raw + rollout.vals), atol=1e-5)
    assert abs(float(jnp.mean(adv.advs))) < 1e-4
    assert abs(float(jnp.std(adv.advs)) - 1.0) < 1e-3
    assert not np.allclose(np.asarray(adv.advs), np.asarray(raw))


def test_returns_stay_in_unit_range_for_completed_episodes():
    rollout = _fake_rollout(terminate_at=6)
    adv, _ = prepare_advantages(rollout)
    rets = np.asarray(adv.rets)[:7]
    assert np.abs(rets).max() <= 1.0 + 1e-4, f"returns escaped [-1, 1]: max {np.abs(rets).max()}"


def test_zero_signal_batch_is_reported_not_hidden():
    """No game terminates: advantages are critic noise, and normalization
    inflates them to unit variance regardless. `terminal_frac` is the alarm."""
    rollout = _fake_rollout(terminate_at=None)
    adv, diag = prepare_advantages(rollout)

    assert diag.terminal_frac == 0.0, "no episode resolved, yet terminal_frac is non-zero"
    assert diag.adv_std_raw < 1.0, "sanity: raw advantages should be small here"
    assert abs(float(jnp.std(adv.advs)) - 1.0) < 1e-3, \
        "normalization is scale-invariant — this is exactly the hazard"
    assert np.isnan(diag.mc_explained_variance), \
        "EV must not be reported off an empty valid set"


def test_diagnostics_are_finite_when_signal_exists():
    rollout = _fake_rollout(terminate_at=5, t=40, rows=8)
    _, diag = prepare_advantages(rollout)
    assert isinstance(diag, Diagnostics)
    assert diag.terminal_frac > 0.0
    assert np.isfinite(diag.mc_explained_variance)
    assert np.isfinite(diag.mc_value_bias)
    assert np.isfinite(diag.explained_variance)


@pytest.mark.parametrize("gae_lambda", [0.0, 0.5, 0.9, 1.0])
def test_gae_shapes_and_finiteness(gae_lambda):
    rollout = _fake_rollout(terminate_at=6, truncate_at=9)
    advs = compute_gae(
        rollout.rews, rollout.vals, rollout.next_vals,
        rollout.terminated, rollout.truncated, 1.0, gae_lambda,
    )
    assert advs.shape == rollout.rews.shape
    assert bool(jnp.all(jnp.isfinite(advs)))
