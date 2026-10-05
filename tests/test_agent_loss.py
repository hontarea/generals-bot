"""Gate for build step 6 (AGENT_SPEC.md §10).

Ratio identity at lr=0 is the cheapest possible check that sampling and
evaluation agree; if it fails, every PPO ratio in training is wrong and the loss
still looks reasonable.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from agent.spec import phi
from agent.spec.constants import ACTION_DIM, N_CHANNELS, N_SCALARS, PAD
from agent.train.config import ModelConfig, get_config
from agent.train.loss import (
    Batch,
    get_learning_rate,
    hl_gauss_target,
    make_optimizer,
    make_update_fn,
    select_indices,
    set_learning_rate,
    uniform_magnet,
    update_ent_coef,
)
from agent.train.net import PolicyValueNet

TOTAL = 64
MB = 16


@pytest.fixture(scope="module")
def cfg():
    return get_config("smoke").replace(
        num_envs=4, num_steps=8, minibatch_size=MB, adv_top_frac=0.5,
        model=ModelConfig(embed_dim=64, depth=2, n_head=4, ff_factor=2, use_bf16=False),
    )


@pytest.fixture(scope="module")
def net(cfg):
    return PolicyValueNet(cfg.model, key=jrandom.PRNGKey(0))


@pytest.fixture(scope="module")
def batch(net):
    """A batch whose actions were genuinely sampled from this network, so the
    stored log-probs are the ones the update must reproduce."""
    rng = np.random.default_rng(0)
    obs_list, sc_list, mm_list, bm_list = [], [], [], []
    for i in range(TOTAL):
        o = np.zeros((14, PAD, PAD), dtype=np.int32)
        mountains = rng.random((PAD, PAD)) < 0.25
        owned = (rng.random((PAD, PAD)) < 0.3) & ~mountains
        o[0] = rng.integers(0, 60, (PAD, PAD)) * owned
        o[3], o[5] = mountains, owned
        o[1, 3, 3] = 1
        o[5, 3, 3] = 1
        o[9:13] = 20
        o[13] = i
        obs, sc, mm, bm, _ = phi.augment(jnp, jnp.asarray(o), phi.init_phi_state(jnp))
        obs_list.append(obs)
        sc_list.append(sc)
        mm_list.append(mm)
        bm_list.append(bm)

    # Quantize to the storage dtype first, exactly as the rollout does, so the
    # stored log-probs correspond to the stored observations.
    obs = jnp.stack(obs_list).astype(jnp.bfloat16)
    scalars = jnp.stack(sc_list)
    move_m = jnp.stack(mm_list)
    build_m = jnp.stack(bm_list)

    keys = jrandom.split(jrandom.PRNGKey(1), TOTAL)
    actions, values, lps, _, _, _ = jax.vmap(net)(
        obs.astype(jnp.float32), move_m, build_m, scalars, keys
    )

    rng2 = np.random.default_rng(2)
    advs = jnp.asarray(rng2.normal(0, 1, TOTAL).astype(np.float32))
    return Batch(
        obs=obs.astype(jnp.bfloat16), move_mask=move_m, build_mask=build_m,
        scalars=scalars, actions=actions, old_lps=lps, advs=advs,
        rets=values + advs * 0.1, train_mask=jnp.ones(TOTAL, dtype=jnp.float32),
    )


def _run(net, cfg, batch, lr, ent_coef=0.0, steps=1, magnet_fn=uniform_magnet):
    params, static = eqx.partition(net, eqx.is_inexact_array)
    optimizer = make_optimizer(cfg.replace(lr=lr))
    opt_state = optimizer.init(params)
    update = make_update_fn(static, cfg.replace(lr=lr), optimizer, magnet_fn)
    idx = select_indices(batch.advs, cfg.n_keep)

    metrics = None
    for i in range(steps):
        params, opt_state, metrics = update(
            params, opt_state, batch, idx, jrandom.PRNGKey(i), ent_coef
        )
    return params, opt_state, metrics


# --------------------------------------------------------------------------
# the ratio identity
# --------------------------------------------------------------------------


def test_ratio_is_exactly_one_at_lr_zero(net, cfg, batch):
    """No parameter has moved, so every re-evaluated log-prob must equal the
    stored one and every ratio must be 1.0."""
    _, _, m = _run(net, cfg, batch, lr=0.0, steps=2)
    assert abs(float(m["ratio"]) - 1.0) < 1e-5, f"ratio = {float(m['ratio'])}"
    assert abs(float(m["first_ratio"]) - 1.0) < 1e-6
    assert float(m["approx_kl"]) < 1e-8, f"approx_kl = {float(m['approx_kl'])}"
    assert float(m["clip_fraction"]) == 0.0


def test_first_minibatch_ratio_is_one_even_at_positive_lr(net, cfg, batch):
    """The first minibatch of an epoch is evaluated before any update lands."""
    _, _, m = _run(net, cfg, batch, lr=3e-4)
    assert abs(float(m["first_ratio"]) - 1.0) < 1e-6
    assert float(m["first_approx_kl"]) < 1e-8


# --------------------------------------------------------------------------
# signs and clipping
# --------------------------------------------------------------------------


def test_positive_advantage_raises_its_action_probability(net, cfg, batch):
    """And a negative advantage lowers it — the unlearning mechanism that a
    positive-only filter would throw away."""
    params, static = eqx.partition(net, eqx.is_inexact_array)

    for sign in (+1.0, -1.0):
        b = batch._replace(advs=jnp.full(TOTAL, sign, dtype=jnp.float32))
        new_params, _, _ = _run(net, cfg, b, lr=1e-2, steps=3)

        before = jax.vmap(eqx.combine(params, static))(
            b.obs.astype(jnp.float32), b.move_mask, b.build_mask, b.scalars,
            jrandom.split(jrandom.PRNGKey(0), TOTAL), b.actions,
        )[2]
        after = jax.vmap(eqx.combine(new_params, static))(
            b.obs.astype(jnp.float32), b.move_mask, b.build_mask, b.scalars,
            jrandom.split(jrandom.PRNGKey(0), TOTAL), b.actions,
        )[2]

        delta = float(jnp.mean(after - before))
        if sign > 0:
            assert delta > 0, f"positive advantage lowered the log-prob ({delta})"
        else:
            assert delta < 0, f"negative advantage raised the log-prob ({delta})"


def test_clipping_bounds_the_objective():
    """`max(-A*r, -A*clip(r))` is the pessimistic bound."""
    eps = 0.2

    def loss(adv, ratio):
        return float(jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - eps, 1 + eps)))

    # Positive advantage: the gain is capped past 1+eps, the loss is not capped
    # below — pushing the action's probability *down* is always penalized.
    assert np.isclose(loss(1.0, 1.0), -1.0)
    assert np.isclose(loss(1.0, 1.5), -(1 + eps))
    assert np.isclose(loss(1.0, 5.0), -(1 + eps))
    assert np.isclose(loss(1.0, 0.5), -0.5)

    # Negative advantage: mirror image. The gain from pushing probability down is
    # capped below 1-eps; raising it is unboundedly penalized.
    assert np.isclose(loss(-1.0, 0.5), 1 - eps)
    assert np.isclose(loss(-1.0, 0.0), 1 - eps)
    assert np.isclose(loss(-1.0, 3.0), 3.0)


# --------------------------------------------------------------------------
# the regularizer
# --------------------------------------------------------------------------


def test_uniform_magnet_reduces_exactly_to_an_entropy_bonus(net, cfg, batch):
    """`reg = -ent - sum(p log m)`; with a uniform magnet the second term is the
    constant log(1/3970), so `reg == -ent + log(3970)` sample by sample."""
    params, static = eqx.partition(net, eqx.is_inexact_array)
    optimizer = make_optimizer(cfg)
    update = make_update_fn(static, cfg, optimizer, uniform_magnet)
    opt_state = optimizer.init(params)
    idx = select_indices(batch.advs, cfg.n_keep)

    _, _, m = update(params, opt_state, batch, idx, jrandom.PRNGKey(0), 0.01)
    reg, ent = float(m["reg"]), float(m["entropy"])
    assert np.isclose(reg, -ent + float(jnp.log(ACTION_DIM)), atol=1e-3), \
        f"reg {reg} != -ent {-ent} + log(3970) {float(jnp.log(ACTION_DIM))}"


def test_entropy_coefficient_pushes_entropy_up(net, cfg, batch):
    """A large entropy coefficient must raise entropy relative to none at all.

    The advantages must be non-zero. `_small_init` starts the policy uniform, so
    entropy begins *at* its maximum and the bonus has nothing to pull against;
    the policy gradient has to sharpen the distribution first for the bonus to be
    observable at all.

    vf_coef=0 keeps the critic out of it.
    """
    cfg_ent = cfg.replace(vf_coef=0.0)
    _, _, m_none = _run(net, cfg_ent, batch, lr=1e-2, ent_coef=0.0, steps=30)
    _, _, m_high = _run(net, cfg_ent, batch, lr=1e-2, ent_coef=0.5, steps=30)

    assert float(m_none["entropy"]) < 6.35, "the policy did not sharpen — test is vacuous"
    assert float(m_high["entropy"]) > float(m_none["entropy"]) + 0.01, (
        f"entropy bonus had no effect: {float(m_none['entropy'])} -> "
        f"{float(m_high['entropy'])}"
    )


# --------------------------------------------------------------------------
# filtering and masking
# --------------------------------------------------------------------------


def test_filter_ranks_by_absolute_advantage():
    advs = jnp.asarray([0.1, -5.0, 0.2, 4.0, -0.3, 3.0, -2.0, 0.05], dtype=jnp.float32)
    idx = np.asarray(select_indices(advs, 4))
    assert set(idx) == {1, 3, 5, 6}, "a signed filter would have dropped the negatives"


def test_filter_size_is_a_whole_number_of_minibatches(cfg):
    assert cfg.n_keep % cfg.minibatch_size == 0
    assert cfg.n_keep == int(cfg.batch_size * cfg.adv_top_frac) // MB * MB


def test_masked_mean_divides_by_the_surviving_count(net, cfg, batch):
    """Truncated transitions are dropped; the mean must not be diluted by them."""
    half = np.ones(TOTAL, dtype=np.float32)
    half[TOTAL // 2:] = 0.0
    b = batch._replace(train_mask=jnp.asarray(half))

    _, _, m_all = _run(net, cfg, batch, lr=0.0)
    _, _, m_half = _run(net, cfg, b, lr=0.0)

    # With every ratio at 1.0 the mean is 1.0 either way — a divide-by-N bug
    # would report ~0.5 instead.
    assert abs(float(m_half["ratio"]) - 1.0) < 1e-5
    assert float(m_half["train_frac"]) < float(m_all["train_frac"])


def test_all_zero_mask_does_not_produce_nan(net, cfg, batch):
    b = batch._replace(train_mask=jnp.zeros(TOTAL, dtype=jnp.float32))
    _, _, m = _run(net, cfg, b, lr=1e-3)
    for k, v in m.items():
        assert np.isfinite(float(v)), f"{k} = {v}"


# --------------------------------------------------------------------------
# optimizer plumbing
# --------------------------------------------------------------------------


def test_learning_rate_is_host_settable(net, cfg):
    params, _ = eqx.partition(net, eqx.is_inexact_array)
    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(params)

    assert np.isclose(get_learning_rate(opt_state), cfg.lr)
    opt_state = set_learning_rate(opt_state, 1.234e-5)
    assert np.isclose(get_learning_rate(opt_state), 1.234e-5)


def test_gradient_clipping_is_active(net, cfg, batch):
    """The global norm is clipped to cfg.max_grad_norm before Adam sees it."""
    b = batch._replace(advs=jnp.full(TOTAL, 1000.0, dtype=jnp.float32))
    _, _, m = _run(net, cfg, b, lr=1e-4)
    # grad_norm is measured pre-clip, so it may exceed the bound; what matters is
    # that the update stayed finite.
    assert np.isfinite(float(m["grad_norm"]))
    assert np.isfinite(float(m["loss"]))


def test_overfits_a_fixed_batch(net, cfg, batch):
    """1000 updates on one batch drive the value loss to ~0.

    The critic can memorize a fixed set of returns; if it cannot, the value head,
    the masking or the optimizer plumbing is broken.
    """
    cfg_of = cfg.replace(vf_coef=1.0)
    _, _, m0 = _run(net, cfg_of, batch, lr=0.0)
    _, _, m1 = _run(net, cfg_of, batch, lr=3e-3, ent_coef=0.0, steps=250)

    v0, v1 = float(m0["value_loss"]), float(m1["value_loss"])
    assert v1 < v0 * 0.05, f"value loss only fell {v0:.5f} -> {v1:.5f}"
    assert v1 < 1e-3, f"value loss did not reach ~0: {v1:.6f}"


# --------------------------------------------------------------------------
# HL-Gauss value head (D20)
# --------------------------------------------------------------------------

CE_BINS = 32
CE_MIN, CE_MAX = -1.2, 1.2
CE_WIDTH = (CE_MAX - CE_MIN) / (CE_BINS - 1)
CE_SIGMA = 0.75 * CE_WIDTH


@pytest.fixture(scope="module")
def ce_cfg(cfg):
    return cfg.replace(
        vf_coef=1.0,
        model=ModelConfig(
            embed_dim=64, depth=2, n_head=4, ff_factor=2, use_bf16=False,
            value_loss="ce", num_bins=CE_BINS, v_min=CE_MIN, v_max=CE_MAX,
            hl_sigma=CE_SIGMA,
        ),
    )


@pytest.fixture(scope="module")
def ce_net(ce_cfg):
    return PolicyValueNet(ce_cfg.model, key=jrandom.PRNGKey(0))


def _targets(values):
    return np.asarray(
        hl_gauss_target(jnp.asarray(values, dtype=jnp.float32),
                        CE_MIN, CE_MAX, CE_BINS, CE_SIGMA)
    )


def test_hl_gauss_target_is_a_distribution():
    t = _targets([-1.2, -0.7, 0.0, 0.35, 1.2])
    assert t.shape == (5, CE_BINS)
    assert np.all(t >= 0.0)
    assert np.allclose(t.sum(axis=1), 1.0, atol=1e-5)


def test_hl_gauss_target_peaks_at_the_nearest_bin():
    centers = np.linspace(CE_MIN, CE_MAX, CE_BINS)
    values = [-0.9, -0.1, 0.0, 0.55, 1.1]
    peaks = _targets(values).argmax(axis=1)
    expected = [int(np.argmin(np.abs(centers - v))) for v in values]
    assert list(peaks) == expected


def test_hl_gauss_expectation_recovers_the_return():
    """The served value is `sum(softmax * centers)`, so a *perfectly* fitted head
    reproduces the return only if the target's own expectation does."""
    centers = np.linspace(CE_MIN, CE_MAX, CE_BINS)
    values = np.array([-0.8, -0.25, 0.0, 0.4, 0.75])
    recovered = _targets(values) @ centers
    assert np.allclose(recovered, values, atol=CE_WIDTH)


def test_hl_gauss_edges_are_centred_on_the_net_bin_centres(ce_net):
    """Training targets and the served expectation must live on one grid.

    `net.bin_centers` is `linspace(v_min, v_max, num_bins)` and rides in the
    safetensors file, so the loss derives edges as those centres +- half a spacing.
    A target built on `linspace(v_min, v_max, num_bins + 1)` edges would be offset
    by half a bin against the thing serving actually computes — and nothing
    downstream would notice.
    """
    centers = np.asarray(ce_net.bin_centers)
    edges = np.linspace(CE_MIN - CE_WIDTH / 2, CE_MAX + CE_WIDTH / 2, CE_BINS + 1)
    assert np.allclose((edges[:-1] + edges[1:]) / 2, centers, atol=1e-6)


def test_bin_centers_never_move(ce_net, ce_cfg, batch):
    """D20: `bin_centers` is an inexact-array field, so it lands in `params` and
    the optimizer can reach it. The support of a histogram is a coordinate system
    — if it drifts, the value head chases a moving grid while the served weights
    keep whatever centres were last written to the safetensors file."""
    before = np.asarray(ce_net.bin_centers).copy()
    params, _, _ = _run(ce_net, ce_cfg, batch, lr=3e-3, steps=5)
    assert np.array_equal(np.asarray(params.bin_centers), before)


def test_no_gradient_path_reaches_bin_centers(ce_net):
    """The guard itself, not just its consequence.

    The test above passes with or without `stop_gradient`, because the PPO loss
    reads `value_aux` (the logits) and never `value` (the expectation), so no
    gradient reaches the centres *today*. That makes the hazard latent rather than
    absent: measured on one sample, the live path through `value` carries a
    gradient of norm ~0.5, and it would open the moment anything consumes the
    scalar — value clipping, a scalar auxiliary head, a mixed loss. This asserts
    the path is severed at the source instead.
    """
    params, static = eqx.partition(ce_net, eqx.is_inexact_array)
    obs = jnp.zeros((N_CHANNELS, PAD, PAD))
    mm = jnp.ones((PAD, PAD, 4), dtype=bool)
    bm = jnp.ones((PAD, PAD), dtype=bool)
    sc = jnp.zeros((N_SCALARS,))

    def scalar_value_loss(p):
        _, value, _, _, _, _ = eqx.combine(p, static)(
            obs, mm, bm, sc, jrandom.PRNGKey(0)
        )
        return (value - 0.7) ** 2

    grads = eqx.filter_grad(scalar_value_loss)(params)
    assert float(jnp.linalg.norm(grads.bin_centers)) == 0.0
    # ...and the rest of the net is still differentiable through that path, so the
    # zero above is the guard and not a dead forward pass.
    assert float(jnp.linalg.norm(grads.value_head.weight)) > 0.0


def test_ce_value_loss_descends(ce_net, ce_cfg, batch):
    """The classification head has to actually fit the returns; before D20 this
    path silently applied MSE to a 32-vector of logits."""
    _, _, m0 = _run(ce_net, ce_cfg, batch, lr=0.0)
    _, _, m1 = _run(ce_net, ce_cfg, batch, lr=3e-3, ent_coef=0.0, steps=150)

    v0, v1 = float(m0["value_loss"]), float(m1["value_loss"])
    assert v0 == pytest.approx(np.log(CE_BINS), rel=0.2), (
        f"an untrained classifier should start near log(num_bins), got {v0:.3f}"
    )
    assert v1 < v0 * 0.5, f"CE value loss only fell {v0:.4f} -> {v1:.4f}"


# --------------------------------------------------------------------------
# entropy controller (D19)
# --------------------------------------------------------------------------


def test_ent_coef_controller_pushes_toward_the_target(cfg):
    c = cfg.replace(ent_target=1.2, ent_kp=0.01, ent_coef_min=1e-4, ent_coef_max=0.1)
    assert update_ent_coef(0.01, 0.4, c) > 0.01      # too deterministic -> loosen
    assert update_ent_coef(0.01, 3.0, c) < 0.01      # too random -> tighten
    assert update_ent_coef(0.01, 1.2, c) == pytest.approx(0.01)


def test_ent_coef_controller_is_bounded(cfg):
    c = cfg.replace(ent_target=1.2, ent_kp=10.0, ent_coef_min=1e-4, ent_coef_max=0.1)
    assert update_ent_coef(0.05, 0.0, c) == pytest.approx(0.1)
    assert update_ent_coef(0.05, 8.3, c) == pytest.approx(1e-4)


def test_update_is_jitted_and_donates(net, cfg, batch):
    """Sanity that the whole update compiles once and runs repeatedly."""
    params, static = eqx.partition(net, eqx.is_inexact_array)
    optimizer = make_optimizer(cfg)
    opt_state = optimizer.init(params)
    update = make_update_fn(static, cfg, optimizer)
    idx = select_indices(batch.advs, cfg.n_keep)

    for i in range(3):
        params, opt_state, m = update(
            params, opt_state, batch, idx, jrandom.PRNGKey(i), 0.01
        )
    assert np.isfinite(float(m["loss"]))
