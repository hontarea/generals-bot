"""Gate for build step 4 (AGENT_SPEC.md §10).

The reference loop is the highest-value test in the build: one env, one game,
plain Python, no jit, no vmap — and the same obs, actions and rewards as the
production path for a fixed key. Almost every ordering, seat-mapping or
memory-reset bug shows up here and essentially nowhere else.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from agent.spec import phi
from agent.spec.constants import N_CHANNELS, N_SCALARS, PAD
from agent.train.config import SMOKE
from agent.train.net import PolicyValueNet
from agent.train.rollout import (
    collect_rollout,
    init_rollout_state,
    make_step_fn,
)
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

N_ENVS = 4
T = 6


@pytest.fixture(scope="module")
def setup():
    env = GeneralsEnv(mode="competition", pool_size=32)
    net = PolicyValueNet(SMOKE.model.__class__(use_bf16=False), key=jrandom.PRNGKey(0))
    params, static = eqx.partition(net, eqx.is_inexact_array)
    step_fn = make_step_fn(env, static)
    return env, net, params, static, step_fn


@pytest.fixture(scope="module")
def rollout(setup):
    env, _, params, static, step_fn = setup
    pool, states, phi_state, key = init_rollout_state(
        env, jrandom.PRNGKey(7), N_ENVS, 32
    )
    out, states, phi_state, key = collect_rollout(
        env, step_fn, params, static, states, phi_state, key, pool, T
    )
    return out, states, phi_state


# --------------------------------------------------------------------------
# shapes and layout
# --------------------------------------------------------------------------


def test_shapes(rollout):
    out, _, _ = rollout
    rows = 2 * N_ENVS
    assert out.obs.shape == (T, rows, N_CHANNELS, PAD, PAD)
    assert out.obs.dtype == jnp.bfloat16
    assert out.move_mask.shape == (T, rows, PAD, PAD, 4)
    assert out.build_mask.shape == (T, rows, PAD, PAD)
    assert out.scalars.shape == (T, rows, N_SCALARS)
    assert out.actions.shape == (T, rows, 5)
    for name in ("lps", "vals", "next_vals", "rews", "terminated", "truncated", "winners"):
        assert getattr(out, name).shape == (T, rows), name


def test_next_vals_is_shifted_vals(rollout):
    """next_vals[t] == vals[t+1] for every t but the last."""
    out, _, _ = rollout
    assert jnp.array_equal(out.next_vals[:-1], out.vals[1:])


def test_stored_obs_reproduce_stored_logprobs(setup, rollout):
    """Re-evaluating the buffer must give back exactly `lps`.

    `obs` is stored as bf16 but `lps` come from the acting forward pass. If the
    network acted on float32 while the buffer kept bf16, every PPO ratio at the
    first minibatch would be off by the quantization error — a systematic bias
    in the importance weights that nothing downstream would flag.
    """
    _, net, _, _, _ = setup
    out, _, _ = rollout

    for t in (0, T // 2, T - 1):
        obs = out.obs[t].astype(jnp.float32)
        keys = jrandom.split(jrandom.PRNGKey(0), obs.shape[0])
        _, _, lp, _, _, _ = jax.vmap(net)(
            obs, out.move_mask[t], out.build_mask[t], out.scalars[t], keys, out.actions[t]
        )
        assert jnp.allclose(lp, out.lps[t], atol=1e-6), f"log-probs drifted at t={t}"


def test_actions_are_legal_against_their_own_masks(rollout):
    """The mask stored at step t must be the mask the action at step t was drawn
    under — the ordering in §4.3 exists to guarantee this."""
    out, _, _ = rollout
    from agent.spec.constants import KIND_BUILD, KIND_MOVE, KIND_PASS

    for t in range(T):
        for r in range(2 * N_ENVS):
            kind, row, col, d, _ = np.asarray(out.actions[t, r])
            if kind == KIND_PASS:
                continue
            if kind == KIND_BUILD:
                assert np.asarray(out.build_mask[t, r])[row, col]
            else:
                assert kind == KIND_MOVE
                assert np.asarray(out.move_mask[t, r])[row, col, d]


# --------------------------------------------------------------------------
# rewards and seats
# --------------------------------------------------------------------------


def test_rewards_are_zero_sum_across_seats(rollout):
    """rews[:, :N] == -rews[:, N:] — taken straight from the engine (§4.2)."""
    out, _, _ = rollout
    assert jnp.array_equal(out.rews[:, :N_ENVS], -out.rews[:, N_ENVS:])


def test_rewards_are_zero_except_on_termination(rollout):
    out, _, _ = rollout
    assert jnp.all(out.rews[~out.terminated] == 0.0)


def test_winners_are_mirrored(rollout):
    out, _, _ = rollout
    w0, w1 = out.winners[:, :N_ENVS], out.winners[:, N_ENVS:]
    ongoing = w0 < 0
    assert jnp.all(w1[ongoing] < 0), "a draw/ongoing marker did not mirror"
    assert jnp.array_equal(w1[~ongoing], 1 - w0[~ongoing])


def test_terminated_and_truncated_are_shared_by_both_seats(rollout):
    out, _, _ = rollout
    assert jnp.array_equal(out.terminated[:, :N_ENVS], out.terminated[:, N_ENVS:])
    assert jnp.array_equal(out.truncated[:, :N_ENVS], out.truncated[:, N_ENVS:])


# --------------------------------------------------------------------------
# the reference loop
# --------------------------------------------------------------------------


def test_matches_a_plain_python_reference_loop(setup):
    """One env, no jit, no vmap, written straight from §4.3 — same numbers.

    The reference deliberately re-derives the whole step rather than calling any
    production helper, so a mistake in the production path cannot hide in a
    shared subroutine.
    """
    env, net, params, static, step_fn = setup
    pool, states, phi_state, key = init_rollout_state(env, jrandom.PRNGKey(3), 1, 32)

    prod, _, _, _ = collect_rollout(
        env, step_fn, params, static, states, phi_state, key, pool, 5
    )

    # --- the reference ---------------------------------------------------
    ref_states = states
    ref_phi = [
        jax.tree.map(lambda x, i=i: x[i], phi_state) for i in range(2)
    ]
    ref_key = key
    ref = {"obs": [], "actions": [], "rews": [], "lps": [], "vals": []}

    for _ in range(5):
        state = jax.tree.map(lambda x: x[0], ref_states)

        obs_arrs = [get_observation(state, seat).as_tensor() for seat in (0, 1)]

        step_obs, step_act, step_lp, step_val = [], [], [], []
        new_phi = []
        for seat in (0, 1):
            o, sc, mm, bm, st = phi.augment(jnp, obs_arrs[seat], ref_phi[seat])
            new_phi.append(st)
            step_obs.append(o)

        ref_key, act_key = jrandom.split(ref_key)
        keys = jrandom.split(act_key, 2)
        for seat in (0, 1):
            o, sc, mm, bm, _ = phi.augment(jnp, obs_arrs[seat], ref_phi[seat])
            # Quantize to the storage dtype before acting, as §4.3 does, so that
            # the stored observation and the stored log-prob describe the same
            # input.
            o = o.astype(jnp.bfloat16).astype(jnp.float32)
            action, value, lp, _, _, _ = net(o, mm, bm, sc, keys[seat])
            step_act.append(action)
            step_lp.append(lp)
            step_val.append(value)

        actions = jnp.stack(step_act)
        timestep, next_state = env.step(state, actions, pool)

        ref["obs"].append(jnp.stack(step_obs).astype(jnp.bfloat16))
        ref["actions"].append(actions)
        ref["lps"].append(jnp.stack(step_lp))
        ref["vals"].append(jnp.stack(step_val))
        ref["rews"].append(timestep.reward)

        done = timestep.terminated | timestep.truncated
        ref_phi = [
            jax.tree.map(lambda x: jnp.where(done, jnp.zeros_like(x), x), st)
            for st in new_phi
        ]
        ref_states = jax.tree.map(lambda x: x[None], next_state)

    # --- compare ---------------------------------------------------------
    for t in range(5):
        assert jnp.array_equal(prod.obs[t], ref["obs"][t]), f"obs differ at t={t}"
        assert jnp.array_equal(prod.actions[t], ref["actions"][t]), f"actions differ at t={t}"
        assert jnp.array_equal(prod.rews[t], ref["rews"][t]), f"rewards differ at t={t}"
        assert jnp.allclose(prod.lps[t], ref["lps"][t], atol=1e-5), f"logprobs differ at t={t}"
        assert jnp.allclose(prod.vals[t], ref["vals"][t], atol=1e-5), f"values differ at t={t}"


# --------------------------------------------------------------------------
# memory isolation
# --------------------------------------------------------------------------


def test_memory_is_reset_after_a_done_but_not_before(setup):
    """A finished game's last observation keeps its memory; the next step starts
    clean. Getting the order wrong silently blanks the most informative frame."""
    env, net, params, static, step_fn = setup
    pool, states, phi_state, key = init_rollout_state(env, jrandom.PRNGKey(11), 2, 32)

    # Force a terminal state: hand game 0 the win by putting its general count
    # to zero for the opponent is not reachable here, so instead drive `time`
    # to truncation, which sets `truncated` on the next step.
    states = states._replace(
        time=jnp.array([env.truncation - 1, 0], dtype=states.time.dtype)
    )
    # Give the memory something non-zero to lose.
    phi_state = phi_state._replace(seen=jnp.ones_like(phi_state.seen))

    _, new_phi, _, out, _, pre_reset_phi = step_fn(states, phi_state, key, params, pool)

    assert bool(out["truncated"][0]), "expected game 0 to truncate"
    assert not bool(out["truncated"][1])

    # Game 0 (rows 0 and 2 of 4) is cleared; game 1 (rows 1 and 3) is untouched.
    seen = np.asarray(new_phi.seen)
    assert not seen[0].any() and not seen[2].any(), "finished game kept memory"
    assert seen[1].all() and seen[3].all(), "live game lost memory"

    # ...but the stored observation still carries the pre-reset memory.
    assert np.asarray(pre_reset_phi.seen)[0].all()


def test_games_carry_across_calls(setup):
    """Rollouts do not start fresh games (§4.6): time advances continuously."""
    env, _, params, static, step_fn = setup
    pool, states, phi_state, key = init_rollout_state(env, jrandom.PRNGKey(5), 2, 32)

    t0 = np.asarray(states.time).copy()
    _, states, phi_state, key = collect_rollout(
        env, step_fn, params, static, states, phi_state, key, pool, 4
    )
    t1 = np.asarray(states.time).copy()
    _, states, _, _ = collect_rollout(
        env, step_fn, params, static, states, phi_state, key, pool, 4
    )
    t2 = np.asarray(states.time)

    assert np.all(t1 > t0) and np.all(t2 > t1)
    assert np.all(t2 - t1 == 4), "a rollout boundary restarted games"
