"""PPO loss, advantage filtering and the optimizer step. AGENT_SPEC.md §6.

    total = policy_loss + vf_coef * value_loss + ent_coef * reg

`num_epochs = 1`, and `target_kl` is deliberately absent: with one epoch it can
only break a loop that is already ending, against a KL averaged over updates that
have already been applied.
"""
from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax

from agent.spec.constants import ACTION_DIM


class Batch(NamedTuple):
    """Flattened over (T, rows); every field's leading axis is `total`."""

    obs: jax.Array
    move_mask: jax.Array
    build_mask: jax.Array
    scalars: jax.Array
    actions: jax.Array
    old_lps: jax.Array
    advs: jax.Array
    rets: jax.Array
    train_mask: jax.Array


def flatten_batch(rollout, adv) -> Batch:
    """(T, rows, ...) -> (T*rows, ...). Reshape only — these stay views."""
    flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
    return Batch(
        obs=flat(rollout.obs),
        move_mask=flat(rollout.move_mask),
        build_mask=flat(rollout.build_mask),
        scalars=flat(rollout.scalars),
        actions=flat(rollout.actions),
        old_lps=flat(rollout.lps),
        advs=flat(adv.advs),
        rets=flat(adv.rets),
        train_mask=flat(adv.train_mask),
    )


@partial(jax.jit, static_argnames=("n_keep",))
def select_indices(advs, n_keep):
    """Top-|advantage| index set.

    Ranked by **|advantage|**, not signed. PPO's gradient is signed — negative
    advantages are what push probability off bad actions — so a positive-only
    filter is self-imitation with no unlearning mechanism.

    The result is an index set, so the filter applies to all three loss terms,
    not just the policy.
    """
    _, idx = jax.lax.top_k(jnp.abs(advs), n_keep)
    return idx


def uniform_magnet(_obs, _move_m, _build_m):
    """The default prior: uniform over all actions.

    `sum(p * log(m))` is then the constant `log(1/3970)`, its gradient vanishes,
    and the regularizer reduces **exactly** to an entropy bonus (asserted in the
    tests). Swapping in `magnet.py` later is a one-line config change.
    """
    return jnp.full((ACTION_DIM,), 1.0 / ACTION_DIM, dtype=jnp.float32)


def hl_gauss_target(rets, v_min, v_max, num_bins, sigma):
    """Two-hot-with-Gaussian-smoothing targets. `(B,)` returns -> `(B, num_bins)`.

    HL-Gauss (Farebrother et al., *Stop Regressing*): instead of regressing a
    scalar, spread a Gaussian centred on the return over the bins and train a
    classifier against it. Each bin's target is the Gaussian's mass between that
    bin's edges, renormalized by the mass inside the support so a return near a
    boundary still yields a distribution summing to 1.

    **The edges are derived from the config, never from `net.bin_centers`.** Reading
    the model's array here would put a gradient path into the histogram's own
    support and let the loss move the bins instead of the predictions (D20).

    `net.py` defines centres as `linspace(v_min, v_max, num_bins)` — endpoints
    inclusive — and that definition is the contract, because those exact values are
    written into `weights.safetensors` and used by the NumPy serving path. So the
    edges here are the centres offset by half a spacing, not `linspace(v_min, v_max,
    num_bins + 1)`; the latter would put training targets and the served expectation
    on two different grids, which nothing downstream would notice.
    """
    width = (v_max - v_min) / (num_bins - 1)
    edges = jnp.linspace(v_min - width / 2, v_max + width / 2, num_bins + 1)

    z = (edges[None, :] - rets[:, None]) / sigma
    cdf = 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))
    mass = cdf[:, 1:] - cdf[:, :-1]
    inside = cdf[:, -1:] - cdf[:, :1]
    return mass / jnp.maximum(inside, 1e-6)


def update_ent_coef(ent_coef, entropy, cfg):
    """Host-side multiplicative controller holding entropy at `cfg.ent_target`.

    `rl_bot_b` ran a cosine from 0.03 to 0.005 and arrived at 0.41 nats — roughly
    1.5 effective actions out of 3,970 — with `approx_kl` at 0.0030, below §6's own
    0.005-0.03 health band. Exploration ended at ~40k of a 100k run because the
    schedule said so, not because the agent had finished exploring.

    A coefficient is not an entropy, so the fix is to control the thing we actually
    care about: nudge `ent_coef` in log space by the entropy error and clamp it.
    Multiplicative because the useful range spans orders of magnitude; bounded
    because a controller that can reach zero, or run away, is worse than a schedule.

    `cfg.ent_target = 0.0` keeps the old cosine — see `loop.py`.
    """
    err = cfg.ent_target - entropy
    scaled = math.exp(math.log(ent_coef) + cfg.ent_kp * err)
    return float(min(max(scaled, cfg.ent_coef_min), cfg.ent_coef_max))


def make_optimizer(cfg):
    """`lr` is injected as a hyperparameter so the loop can set it per iteration
    from the host.

    Not an optax schedule keyed on optimizer-step count: the reference computes
    steps-per-iteration from the *unfiltered* batch size, ignoring
    `adv_top_frac`, which puts its schedule off by exactly 1/f and confounds
    every cross-filter-fraction comparison.
    """
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=cfg.lr),
    )


def set_learning_rate(opt_state, lr):
    """Host-side per-iteration LR update; keyed on the iteration counter."""
    clip_state, adam_state = opt_state
    hyperparams = dict(adam_state.hyperparams)
    hyperparams["learning_rate"] = jnp.asarray(lr, dtype=jnp.float32)
    return (clip_state, adam_state._replace(hyperparams=hyperparams))


def get_learning_rate(opt_state) -> float:
    return float(opt_state[1].hyperparams["learning_rate"])


def _masked_mean(values, mask):
    """Divide by the surviving count, not the minibatch size."""
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def make_update_fn(static, cfg, optimizer, magnet_fn=uniform_magnet):
    """One PPO epoch over the filtered index set."""
    n_batches = cfg.n_keep // cfg.minibatch_size

    def minibatch_loss(params, mb, key):
        model = eqx.combine(params, static)
        keys = jrandom.split(key, mb.actions.shape[0])

        _, _, lp, ent, value_aux, p_dist = jax.vmap(model)(
            mb.obs.astype(jnp.float32), mb.move_mask, mb.build_mask,
            mb.scalars, keys, mb.actions,
        )

        ratio = jnp.exp(lp - mb.old_lps)
        # max of the negated forms == min of the objectives == pessimistic bound.
        policy_loss = jnp.maximum(
            -mb.advs * ratio,
            -mb.advs * jnp.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps),
        )

        # §3.6's value fork, resolved at trace time on a Python int. Before D20
        # this branch did not exist and MSE was applied to `value_aux`
        # unconditionally — which, with `num_bins > 0`, meant regressing a
        # 128-vector of logits onto a scalar return. The switch looked like a
        # config flag and was actually a trap.
        if cfg.model.num_bins:
            target = hl_gauss_target(
                mb.rets, cfg.model.v_min, cfg.model.v_max,
                cfg.model.num_bins, cfg.model.hl_sigma,
            )
            value_loss = -jnp.sum(target * jax.nn.log_softmax(value_aux, axis=-1), axis=-1)
        else:
            value_loss = 0.5 * (value_aux - mb.rets) ** 2

        # General form from day one: with a uniform magnet the second term is a
        # constant and this is exactly an entropy bonus.
        magnet = jax.vmap(magnet_fn)(mb.obs.astype(jnp.float32), mb.move_mask, mb.build_mask)
        reg = -ent - jnp.sum(p_dist * jnp.log(magnet + 1e-10), axis=-1)

        return policy_loss, value_loss, reg, ratio, ent

    # Deliberately NOT donating params/opt_state: the loop's EMA update reads
    # `params` after this returns, and donation would leave it a deleted buffer.
    # The saving would be ~40 MB against a 4.8 GB rollout buffer.
    @jax.jit
    def update(params, opt_state, batch: Batch, sample_idx, key, ent_coef):
        perm = jrandom.permutation(key, sample_idx.shape[0])
        idx_mb = sample_idx[perm].reshape(n_batches, cfg.minibatch_size)

        def loss_fn(p, mb, k):
            policy_loss, value_loss, reg, ratio, ent = minibatch_loss(p, mb, k)
            total = policy_loss + cfg.vf_coef * value_loss + ent_coef * reg
            loss = _masked_mean(total, mb.train_mask)

            # Schulman k3: non-negative per sample and low variance, unlike
            # the naive `old_lp - lp`.
            approx_kl = ratio - 1.0 - jnp.log(ratio)
            metrics = {
                "loss": loss,
                "policy_loss": _masked_mean(policy_loss, mb.train_mask),
                "value_loss": _masked_mean(value_loss, mb.train_mask),
                "reg": _masked_mean(reg, mb.train_mask),
                "entropy": _masked_mean(ent, mb.train_mask),
                "ratio": _masked_mean(ratio, mb.train_mask),
                "approx_kl": _masked_mean(approx_kl, mb.train_mask),
                "clip_fraction": _masked_mean(
                    (jnp.abs(ratio - 1.0) > cfg.clip_eps).astype(jnp.float32), mb.train_mask
                ),
                "filtered_adv_mean": _masked_mean(mb.advs, mb.train_mask),
                "train_frac": jnp.mean(mb.train_mask),
            }
            return loss, metrics

        def step(carry, inputs):
            params, opt_state = carry
            mb_idx, mb_key = inputs
            mb = jax.tree.map(lambda x: x[mb_idx], batch)

            (_, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                params, mb, mb_key
            )
            metrics["actor_grad_norm"] = optax.tree.norm(grads.policy_head)
            metrics["critic_grad_norm"] = optax.tree.norm(grads.value_head)
            metrics["grad_norm"] = optax.tree.norm(grads)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return (params, opt_state), metrics

        keys = jrandom.split(key, n_batches)
        (params, opt_state), metrics = jax.lax.scan(step, (params, opt_state), (idx_mb, keys))

        # `first_*` keeps the pre-update value visible: at the first minibatch of
        # an epoch every ratio must be exactly 1.0, which is the cheapest possible
        # check that sampling and evaluation agree.
        summary = {k: jnp.mean(v) for k, v in metrics.items()}
        summary["first_ratio"] = metrics["ratio"][0]
        summary["first_approx_kl"] = metrics["approx_kl"][0]
        summary["max_approx_kl"] = jnp.max(metrics["approx_kl"])
        return params, opt_state, summary

    return update
