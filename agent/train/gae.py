"""Generalized advantage estimation and the critic diagnostics. AGENT_SPEC.md §5.

`gamma = 1.0` — undiscounted. With terminal-only rewards the return of any
completed episode is exactly +1, -1 or 0.

`gae_lambda = 0.9` — the credit horizon is ~10 steps. The terminal reward
propagates back ten steps on its own; everything beyond that flows through the
critic. **The value function is the credit assignment mechanism.**

**Shaping moves the value range (D19).** With `shaping_coef > 0` the reward gains
`F = Phi(s') - Phi(s)`, so an intermediate return is `+-1 - Phi(s_t)` and the range
widens from [-1, 1] to `[-(1 + coef), 1 + coef]`. HL-Gauss's `v_min`/`v_max` must be
set from that, not from the old bound. A completed episode still returns very close
to +-1: the shaping telescopes to `-Phi(s_0)`, and `Phi` of a symmetric opening is 0.
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Advantages(NamedTuple):
    advs: jax.Array          # (T, rows) normalized
    rets: jax.Array          # (T, rows) TD(lambda) returns, in [-1, 1]
    train_mask: jax.Array    # (T, rows) 1.0 where the transition may be learned from


class Diagnostics(NamedTuple):
    terminal_frac: float
    adv_std_raw: float
    explained_variance: float
    mc_explained_variance: float
    mc_value_bias: float
    mc_valid_frac: float
    # Shaping (D19). `mc_ev_unshaped` scores the critic against the *engine's*
    # return so the number stays comparable to runs made before shaping existed;
    # `mc_explained_variance` above scores it against what it is actually trained
    # on.
    #
    # `shaping_mean` cancels to zero whenever the potential is antisymmetric,
    # because rows [N, 2N) are the negation of rows [0, N). It is therefore not a
    # tuning signal but a free end-to-end check that the seat mirroring survived
    # the rollout. It is zero only to **float32 summation order**, not exactly:
    # measured ~3e-12 on a 24-step buffer against a `shaping_abs_mean` of 7e-4.
    # Read it as "negligible beside `shaping_abs_mean`"; a drift to the same order
    # of magnitude means the seat layout is wrong. The number that says whether
    # shaping is doing anything at all is `shaping_abs_mean`.
    mc_ev_unshaped: float
    shaping_mean: float
    shaping_abs_mean: float


@partial(jax.jit, static_argnames=("gamma", "gae_lambda"))
def compute_gae(rewards, values, next_values, terminated, truncated, gamma=1.0, gae_lambda=0.9):
    """Reverse scan over the time axis. Shapes are (T, rows)."""

    def scan_fn(last_adv, inp):
        reward, value, next_value, term, trunc = inp
        done = (term | trunc).astype(jnp.float32)

        # Guard 1 of the three named at rollout.py's `next_vals` line: on a
        # termination the stored next_value belongs to the auto-reset next game,
        # so it must contribute nothing.
        bootstrap = jnp.where(term, 0.0, next_value)
        delta = reward + gamma * bootstrap - value
        adv = delta + gamma * gae_lambda * (1.0 - done) * last_adv

        # Guard 2: a truncated step's advantage is meaningless, so it must not
        # propagate backwards into earlier steps either.
        carry = jnp.where(trunc, 0.0, adv)
        return carry, adv

    rev = lambda x: x[::-1]  # noqa: E731
    _, advs = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(rewards[0]),
        (rev(rewards), rev(values), rev(next_values), rev(terminated), rev(truncated)),
    )
    return rev(advs)


@partial(jax.jit, static_argnames=("gamma",))
def compute_mc_returns(rewards, terminated, truncated, gamma=1.0):
    """True Monte-Carlo returns, no bootstrap — plus a validity mask.

    Only transitions whose episode actually finished inside the window get a
    real return; everything after the last terminal is unresolved. These are the
    honest critic diagnostics, unlike `explained_variance`, which is partly
    self-referential (its target is built from the critic's own predictions).
    """

    def scan_fn(carry, inp):
        ret, valid = carry
        reward, term, trunc = inp
        # A termination restarts the accumulator and makes everything from here
        # backwards resolvable; a truncation makes it unresolvable.
        ret = jnp.where(term, reward, reward + gamma * ret)
        valid = jnp.where(term, True, jnp.where(trunc, False, valid))
        ret = jnp.where(trunc, 0.0, ret)
        return (ret, valid), (ret, valid)

    rev = lambda x: x[::-1]  # noqa: E731
    init = (jnp.zeros_like(rewards[0]), jnp.zeros_like(rewards[0], dtype=bool))
    _, (rets, valid) = jax.lax.scan(
        scan_fn, init, (rev(rewards), rev(terminated), rev(truncated))
    )
    return rev(rets), rev(valid)


MIN_VALID_FOR_EV = 32


def _explained_variance(pred, target, mask=None):
    """1 - var(target - pred) / var(target), optionally over a subset."""
    if mask is None:
        return 1.0 - jnp.var(target - pred) / (jnp.var(target) + 1e-8)

    n = jnp.sum(mask)

    def masked_var(x):
        mean = jnp.sum(jnp.where(mask, x, 0.0)) / jnp.maximum(n, 1.0)
        return jnp.sum(jnp.where(mask, (x - mean) ** 2, 0.0)) / jnp.maximum(n, 1.0)

    ev = 1.0 - masked_var(target - pred) / (masked_var(target) + 1e-8)
    # An EV computed off a handful of resolved episodes is noise that reads like
    # a signal; report nothing rather than something misleading.
    return jnp.where(n < MIN_VALID_FOR_EV, jnp.nan, ev)


def shaped_rewards(rollout, shaping_coef=0.0):
    """`r + coef * (Phi(s') - Phi(s))`, with `Phi` zeroed at a terminal (D19).

    The zeroing is what makes the shaping policy-invariant: Ng, Harada & Russell
    require `Phi(absorbing) = 0`, and without it the shaped return of an episode
    would carry `Phi(s_T)` — which correlates with winning material — and the agent
    could prefer to finish rich rather than to finish. It is the exact twin of
    `compute_gae`'s `bootstrap = where(terminated, 0, next_value)`, and for the same
    reason: past a termination there is no next state of this game to read.

    Truncation is deliberately left alone. The episode has not ended, only the
    window has, and that transition is dropped by `train_mask` regardless.

    At `shaping_coef = 0.0` this returns `rews + 0.0`, which is bit-identical to
    `rews` — the inertness `test_shaping_off_is_bit_identical` pins.
    """
    delta = jnp.where(rollout.terminated, 0.0, rollout.phis_next) - rollout.phis
    return rollout.rews + shaping_coef * delta, shaping_coef * delta


def prepare_advantages(rollout, gamma=1.0, gae_lambda=0.9, shaping_coef=0.0):
    """GAE, returns, normalization and diagnostics — **in this order**.

    `rets = advs + vals` must come before normalization. Reversing those two
    lines rescales `rets` out of its bound and silently breaks HL-Gauss two modules
    away, with nothing local to show for it.
    """
    rews, shaping = shaped_rewards(rollout, shaping_coef)

    advs = compute_gae(
        rews, rollout.vals, rollout.next_vals,
        rollout.terminated, rollout.truncated, gamma, gae_lambda,
    )
    rets = advs + rollout.vals

    adv_std_raw = float(jnp.std(advs))
    normed = (advs - jnp.mean(advs)) / (jnp.std(advs) + 1e-8)
    train_mask = 1.0 - rollout.truncated.astype(jnp.float32)

    # The critic is trained on the shaped return, so the honest critic diagnostic
    # has to be built from the shaped reward too.
    mc_rets, mc_valid = compute_mc_returns(
        rews, rollout.terminated, rollout.truncated, gamma
    )
    # The validity mask depends only on terminated/truncated, so the unshaped pass
    # reuses `mc_valid` and differs only in what it accumulates.
    mc_rets_raw, _ = compute_mc_returns(
        rollout.rews, rollout.terminated, rollout.truncated, gamma
    )

    diag = Diagnostics(
        # The zero-signal alarm. Normalization is scale-invariant: if no game
        # terminates in a batch, advantages are critic noise, normalization
        # inflates them to unit variance, and the top-|A| filter then selects the
        # most extreme noise and trains on it at full magnitude.
        terminal_frac=float(jnp.mean(mc_valid)),
        adv_std_raw=adv_std_raw,
        explained_variance=float(_explained_variance(rollout.vals, rets)),
        mc_explained_variance=float(_explained_variance(rollout.vals, mc_rets, mc_valid)),
        mc_value_bias=float(
            jnp.sum(jnp.where(mc_valid, rollout.vals - mc_rets, 0.0))
            / jnp.maximum(jnp.sum(mc_valid), 1.0)
        ),
        mc_valid_frac=float(jnp.mean(mc_valid)),
        mc_ev_unshaped=float(_explained_variance(rollout.vals, mc_rets_raw, mc_valid)),
        shaping_mean=float(jnp.mean(shaping)),
        shaping_abs_mean=float(jnp.mean(jnp.abs(shaping))),
    )
    return Advantages(advs=normed, rets=rets, train_mask=train_mask), diag
