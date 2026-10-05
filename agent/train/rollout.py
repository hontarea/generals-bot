"""Self-play rollout collection. AGENT_SPEC.md §4.

One network plays both seats. Both seats are flattened into a single batch of
`2N` rows so that phi, the masks and the forward pass all run once per step:

    row r  in [0, N)   -> game r, seat 0
    row r  in [N, 2N)  -> game r - N, seat 1

Games span iterations (§4.6): `final_states` and the phi state feed straight
back in, so every batch mixes openings, midgames and endgames.
"""
from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

from agent.spec import phi
from agent.train import shaping
from generals.core.game import get_observation


class Rollout(NamedTuple):
    """Every field is (T, 2N, ...). `obs[t]`, `actions[t]`, `lps[t]`, `vals[t]`
    and `rews[t]` all describe the same decision."""

    obs: jax.Array           # (T, 2N, 39, 21, 21) bf16
    move_mask: jax.Array     # (T, 2N, 21, 21, 4) bool
    build_mask: jax.Array    # (T, 2N, 21, 21) bool
    scalars: jax.Array       # (T, 2N, 16) f32
    actions: jax.Array       # (T, 2N, 5) i32
    lps: jax.Array           # (T, 2N) f32
    vals: jax.Array          # (T, 2N) f32
    next_vals: jax.Array     # (T, 2N) f32
    rews: jax.Array          # (T, 2N) f32
    terminated: jax.Array    # (T, 2N) bool
    truncated: jax.Array     # (T, 2N) bool
    winners: jax.Array       # (T, 2N) i32, mirrored per seat, logging only
    phis: jax.Array          # (T, 2N) f32, unit potential of the acted-in state
    phis_next: jax.Array     # (T, 2N) f32, same for the PRE-RESET next state


def _both_seats(states):
    """(N, ...) states -> (2N, 14, 21, 21) observation tensors, seat 0 then seat 1."""
    seat0 = jax.vmap(lambda s: get_observation(s, 0).as_tensor())(states)
    seat1 = jax.vmap(lambda s: get_observation(s, 1).as_tensor())(states)
    return jnp.concatenate([seat0, seat1], axis=0)


def _forward(model, obs, move_m, build_m, scalars, keys, actions=None):
    if actions is None:
        return jax.vmap(model)(obs, move_m, build_m, scalars, keys)
    return jax.vmap(model)(obs, move_m, build_m, scalars, keys, actions)


def _value_only(model, states, phi_state, key):
    """One forward pass used solely for the bootstrap value (§4.5)."""
    obs_arr = _both_seats(states)
    obs, scalars, move_m, build_m, _ = jax.vmap(
        lambda o, s: phi.augment(jnp, o, s)
    )(obs_arr, phi_state)
    # Same bf16 round-trip as the acting path, so the bootstrap value is on the
    # same footing as every other value in the buffer.
    obs = obs.astype(jnp.bfloat16).astype(jnp.float32)
    keys = jrandom.split(key, obs.shape[0])
    _, value, _, _, _, _ = _forward(model, obs, move_m, build_m, scalars, keys)
    return value


def make_step_fn(env, static, shaping_w=shaping.ShapingWeights()):
    """Build the single jitted step. Everything inside is one jitted function
    (§4.4) — a Python loop over it costs ~10% dispatch and buys debuggability.

    `shaping_w` holds Python floats, so it closes over into the trace; changing it
    needs a fresh `make_step_fn`, exactly like `env`.
    """

    def one_step(states, phi_state, key, params, pool):
        model = eqx.combine(params, static)
        n = states.time.shape[0]

        # 1. Observe from the PRE-step states: this is the position the action
        #    is taken in.
        obs_arr = _both_seats(states)

        # 2-3. Fold into memory, then read masks and scalars from the NEW phi
        #      state so the newest ring-buffer entry is this turn's.
        obs, scalars, move_m, build_m, new_phi = jax.vmap(
            lambda o, s: phi.augment(jnp, o, s)
        )(obs_arr, phi_state)

        # Quantize to the storage dtype BEFORE the forward pass, then act on the
        # quantized values. The rollout buffer keeps bf16 (§4.1) while `lps` are
        # computed here; if the network saw float32 and the buffer kept bf16, the
        # update would re-evaluate slightly different inputs and the first
        # minibatch's PPO ratio would not be 1.0 — a systematic bias in every
        # importance weight. bf16 -> float32 is exact, so this makes them agree.
        obs = obs.astype(jnp.bfloat16)

        # 4. Act.
        key, act_key = jrandom.split(key)
        keys = jrandom.split(act_key, 2 * n)
        action, value, logprob, _, _, _ = _forward(
            model, obs.astype(jnp.float32), move_m, build_m, scalars, keys
        )

        # 5. Step. Actions go back as (N, 2, 5); the env may terminate and
        #    silently auto-reset.
        env_actions = jnp.stack([action[:n], action[n:]], axis=1)
        timestep, next_states = jax.vmap(env.step, in_axes=(0, 0, None))(
            states, env_actions, pool
        )

        # The engine's reward is already exactly zero-sum and already mirrored
        # per seat (§4.2). Reimplementing the seat-1 flip is a whole class of
        # sign bug, so take it as given.
        rews = jnp.concatenate([timestep.reward[:, 0], timestep.reward[:, 1]])
        terminated = jnp.tile(timestep.terminated, 2)
        truncated = jnp.tile(timestep.truncated, 2)
        winner = timestep.info.winner
        winners = jnp.concatenate([winner, jnp.where(winner >= 0, 1 - winner, winner)])

        dones = timestep.terminated | timestep.truncated

        # The shaping potential of the position acted in, and of the position it
        # led to. `timestep.last_state` is the only place the pre-reset next state
        # survives — reading `states` on the following step instead would, on a
        # done step, hand back the potential of a brand-new game. That is the same
        # trap the three `next_vals` guards exist for, and it is why this is a
        # per-step output rather than a shifted array. `gae.py` supplies the
        # coefficient and the terminal zeroing (D19).
        phis = shaping.unit_potentials(states, shaping_w)
        phis_next = shaping.unit_potentials(timestep.last_state, shaping_w)

        out = {
            "obs": obs,
            "move_mask": move_m,
            "build_mask": build_m,
            "scalars": scalars,
            "actions": action,
            "lps": logprob,
            "vals": value,
            "rews": rews,
            "terminated": terminated,
            "truncated": truncated,
            "winners": winners,
            "phis": phis,
            "phis_next": phis_next,
        }

        # 6. Reset memory for finished games — AFTER the observation was stored,
        #    so a finished game's last observation keeps its full memory and only
        #    the next step starts clean.
        reset_phi = phi.reset_phi_state(jnp, new_phi, jnp.tile(dones, 2))

        # `timestep.last_state` is the only place the terminal position survives
        # the auto-reset; it is what the bootstrap pass must see.
        return next_states, reset_phi, key, out, timestep.last_state, new_phi

    return jax.jit(one_step)


def collect_rollout(env, step_fn, params, static, states, phi_state, key, pool, num_steps):
    """Run `num_steps` steps, returning the buffer and the carried state."""
    buf = defaultdict(list)
    last_pre_reset = last_phi = None

    for _ in range(num_steps):
        states, phi_state, key, out, last_pre_reset, last_phi = step_fn(
            states, phi_state, key, params, pool
        )
        for k, v in out.items():
            buf[k].append(v)

    stacked = {k: jnp.stack(v) for k, v in buf.items()}

    # next_vals[t] = vals[t+1], with the last entry from one extra forward pass
    # on the pre-reset terminal position.
    #
    # On a done step this is SEMANTICALLY WRONG — vals[t+1] belongs to the next
    # game, because the env auto-resets. It is safe only because of three guards
    # in gae.py, all of which must stay:
    #   1. `bootstrap = where(terminated, 0, next_value)`  zeroes it on termination
    #   2. `carry = where(truncated, 0, adv)`              stops backward propagation
    #   3. `train_mask = 1 - truncated`                    drops the transition
    # Anyone who wants to learn from truncated transitions must fix this first;
    # nothing else in the code says so.
    #
    # `phis_next` needs the same care and does NOT get it from here: it is written
    # per step from `timestep.last_state`, so it never has to be shifted. Guard 1
    # has a twin in `gae.py` that zeroes it on termination (D19).
    key, boot_key = jrandom.split(key)
    model = eqx.combine(params, static)
    final_val = _value_only(model, last_pre_reset, last_phi, boot_key)
    stacked["next_vals"] = jnp.concatenate([stacked["vals"][1:], final_val[None]])

    return Rollout(**stacked), states, phi_state, key


def init_rollout_state(env, key, num_envs, pool_size):
    """Fresh games and empty memory. A curriculum stage change must call this
    (§4.6) — in-flight games from the old stage would otherwise leak across."""
    key, reset_key, pool_key = jrandom.split(key, 3)
    pool, _ = env.reset(reset_key)

    idx = jrandom.randint(pool_key, (num_envs,), 0, env.pool_size)
    states = jax.tree.map(lambda x: x[idx], pool)
    phi_state = phi.init_phi_state(jnp, batch=2 * num_envs)
    return pool, states, phi_state, key
