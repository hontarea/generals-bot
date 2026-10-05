"""Gate for build step 9 (AGENT_SPEC.md §10).

Resume equivalence is the one that matters: MetaCentrum walltime caps will cut
long runs, and a resume that silently drops the optimizer state or the PRNG key
looks fine in the log and quietly wastes the restart.
"""
from __future__ import annotations

import csv

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from agent.train.checkpoint import TrainState, load, save
from agent.train.config import ModelConfig, get_config
from agent.train.loop import CsvLogger, cosine, ema_update, train
from agent.train.loss import get_learning_rate, make_optimizer
from agent.train.net import PolicyValueNet

TINY_MODEL = ModelConfig(embed_dim=32, depth=1, n_head=4, ff_factor=2, use_bf16=False)


@pytest.fixture(scope="module")
def cfg():
    """Smallest thing that still exercises every code path once per iteration."""
    return get_config("smoke").replace(
        model=TINY_MODEL, num_envs=2, num_steps=3, minibatch_size=2,
        adv_top_frac=0.5, pool_size=32, ckpt_every=2, gate_every=1_000_000,
        num_iters=6,
    )


# --------------------------------------------------------------------------
# schedules and EMA
# --------------------------------------------------------------------------


def test_cosine_is_keyed_on_iterations():
    assert cosine(3e-4, 3e-5, 0, 100) == pytest.approx(3e-4)
    assert cosine(3e-4, 3e-5, 100, 100) == pytest.approx(3e-5)
    assert cosine(3e-4, 3e-5, 50, 100) == pytest.approx((3e-4 + 3e-5) / 2)
    # Past the end it must clamp rather than turn back up.
    assert cosine(3e-4, 3e-5, 500, 100) == pytest.approx(3e-5)


def test_ema_tracks_slowly_and_is_not_the_params():
    net = PolicyValueNet(TINY_MODEL, key=jrandom.PRNGKey(0))
    params, _ = eqx.partition(net, eqx.is_inexact_array)
    ema = jax.tree.map(jnp.copy, params)
    moved = jax.tree.map(lambda p: p + 1.0, params)

    ema = ema_update(ema, moved, 0.999)
    delta = float(jnp.mean(ema.policy_head.weight - params.policy_head.weight))
    assert delta == pytest.approx(0.001, rel=1e-3), "EMA decay is not 0.999"

    for _ in range(999):
        ema = ema_update(ema, moved, 0.999)
    later = float(jnp.mean(ema.policy_head.weight - params.policy_head.weight))
    assert 0.5 < later < 1.0, "EMA never converges toward the params"


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------


def _state(seed=0):
    net = PolicyValueNet(TINY_MODEL, key=jrandom.PRNGKey(seed))
    params, _ = eqx.partition(net, eqx.is_inexact_array)
    optimizer = make_optimizer(get_config("smoke").replace(model=TINY_MODEL))
    return TrainState(
        params=params, ema_params=jax.tree.map(jnp.copy, params),
        opt_state=optimizer.init(params), iteration=17, stage_idx=2,
        iters_in_stage=41, last_gate_score=0.63, low_signal_iters=3,
        key=jrandom.PRNGKey(seed),
    )


def test_checkpoint_round_trip_preserves_everything(tmp_path):
    """Including `opt_state` — Adam moments are half the state of a long run."""
    original = _state(0)
    path = tmp_path / "ckpt.eqx"
    save(path, original)

    restored = load(path, _state(1))          # different structure source

    for field in ("iteration", "stage_idx", "iters_in_stage", "low_signal_iters"):
        assert getattr(restored, field) == getattr(original, field), field
    assert restored.last_gate_score == pytest.approx(original.last_gate_score)
    assert jnp.array_equal(restored.key, original.key), "PRNG key was not restored"

    for a, b in zip(jax.tree.leaves(original.params),
                    jax.tree.leaves(restored.params), strict=True):
        assert jnp.array_equal(a, b)
    for a, b in zip(jax.tree.leaves(original.opt_state),
                    jax.tree.leaves(restored.opt_state), strict=True):
        assert jnp.array_equal(a, b), "optimizer state was not restored"

    assert get_learning_rate(restored.opt_state) == pytest.approx(
        get_learning_rate(original.opt_state)
    )


def test_save_is_atomic(tmp_path, monkeypatch):
    """A write killed midway must leave the previous checkpoint intact.

    This is the walltime-kill scenario: PBS sends SIGTERM, the process dies while
    serialising, and the last good checkpoint must survive.
    """
    path = tmp_path / "ckpt.eqx"
    save(path, _state(0))
    good = path.read_bytes()

    def explode(*_args, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr("agent.train.checkpoint.eqx.tree_serialise_leaves", explode)
    with pytest.raises(OSError, match="disk full"):
        save(path, _state(1))

    assert path.read_bytes() == good, "a failed save clobbered the good checkpoint"
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"


# --------------------------------------------------------------------------
# resume equivalence
# --------------------------------------------------------------------------


def test_training_is_deterministic(cfg, tmp_path):
    """Two runs of the same config must land on identical parameters.

    Any unseeded randomness anywhere in rollout, filtering or the update shows up
    here, and would make every other comparison in this file meaningless.
    """
    a = train(cfg, tmp_path / "a", max_iters=4)
    b = train(cfg, tmp_path / "b", max_iters=4)

    for name in ("params", "ema_params", "opt_state"):
        for x, y in zip(jax.tree.leaves(getattr(a, name)),
                        jax.tree.leaves(getattr(b, name)), strict=True):
            assert jnp.array_equal(x, y), f"{name} differs between identical runs"
    assert jnp.array_equal(a.key, b.key)


def test_resume_picks_up_exactly_where_it_stopped(cfg, tmp_path):
    """A resume must start from the checkpointed state, not a fresh one.

    Note what is *not* claimed: §8 deliberately does not persist env states or
    phi, so a resumed run restarts its in-flight games and its subsequent
    parameters will NOT match an uninterrupted run. That costs one stale rollout
    and is the documented trade for not serialising a 20k-board pool.

    What must hold is that nothing is silently reinitialized — which is checked
    by resuming with no further iterations to run and comparing state directly.
    """
    stopped = train(cfg, tmp_path / "run", max_iters=4)
    ckpt = tmp_path / "run" / "checkpoint.eqx"
    assert ckpt.exists()

    resumed = train(cfg, tmp_path / "run", resume=ckpt, max_iters=4)

    assert resumed.iteration == stopped.iteration == 4
    assert resumed.stage_idx == stopped.stage_idx
    assert resumed.iters_in_stage == stopped.iters_in_stage
    assert jnp.array_equal(resumed.key, stopped.key), "PRNG key was reinitialized"

    for name in ("params", "ema_params", "opt_state"):
        for a, b in zip(jax.tree.leaves(getattr(stopped, name)),
                        jax.tree.leaves(getattr(resumed, name)), strict=True):
            assert jnp.array_equal(a, b), f"{name} was not restored on resume"


def test_resume_continues_training_from_the_checkpoint(cfg, tmp_path):
    """And then actually makes progress, rather than restarting from iteration 0."""
    train(cfg, tmp_path / "run", max_iters=4)
    ckpt = tmp_path / "run" / "checkpoint.eqx"
    before = load(ckpt, _state(0))

    after = train(cfg, tmp_path / "run", resume=ckpt, max_iters=7)
    assert after.iteration == 7

    moved = any(
        not jnp.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before.params),
                        jax.tree.leaves(after.params), strict=True)
    )
    assert moved, "resuming ran iterations but the parameters never changed"
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(after.params))


def test_metrics_csv_has_the_required_columns(cfg, tmp_path):
    train(cfg, tmp_path / "run", max_iters=2)
    rows = list(csv.DictReader((tmp_path / "run" / "metrics.csv").open()))

    assert len(rows) == 2
    required = {
        # health
        "first_ratio", "approx_kl", "clip_fraction", "grad_norm",
        "actor_grad_norm", "critic_grad_norm",
        # learning
        "mc_explained_variance", "mc_value_bias",
        # regime
        "terminal_frac", "adv_std_raw", "post_deathtouch_frac", "mean_turn",
        "draw_rate",
        # behaviour
        "build_rate", "first_build_turn", "pass_rate",
        # bookkeeping
        "iteration", "stage", "lr", "ent_coef", "iter_seconds",
    }
    missing = required - set(rows[0])
    assert not missing, f"metrics.csv is missing {sorted(missing)}"


def test_first_ratio_is_one_in_a_real_iteration(cfg, tmp_path):
    """End-to-end version of the ratio identity: through a genuine rollout, with
    bf16 storage in the buffer, the first minibatch must still price at 1.0."""
    train(cfg, tmp_path / "run", max_iters=2)
    rows = list(csv.DictReader((tmp_path / "run" / "metrics.csv").open()))
    assert float(rows[0]["first_ratio"]) == pytest.approx(1.0, abs=1e-4)
    assert float(rows[0]["approx_kl"]) >= 0.0, "Schulman k3 must be non-negative"


def test_csv_logger_appends_without_duplicating_the_header(tmp_path):
    path = tmp_path / "m.csv"
    for i in range(3):
        logger = CsvLogger(path)
        logger.log({"a": i, "b": i * 2})
        logger.close()
    lines = path.read_text().strip().splitlines()
    assert lines[0] == "a,b"
    assert len(lines) == 4, lines


def test_csv_logger_keeps_columns_aligned_across_a_resume(tmp_path):
    """The existing header wins.

    A resumed run builds its row dict in whatever order the code happens to
    produce; keying an append off that order against an older header shifts every
    column, and the corruption is invisible until someone plots it.
    """
    path = tmp_path / "m.csv"
    logger = CsvLogger(path)
    logger.log({"iteration": 0, "loss": 1.5, "entropy": 8.2})
    logger.close()

    # A different key order, as a later code path might produce.
    logger = CsvLogger(path)
    logger.log({"entropy": 7.1, "iteration": 1, "loss": 0.9})
    logger.close()

    rows = list(csv.DictReader(path.open()))
    assert rows[0] == {"iteration": "0", "loss": "1.5", "entropy": "8.2"}
    assert rows[1] == {"iteration": "1", "loss": "0.9", "entropy": "7.1"}


def test_csv_logger_warns_rather_than_silently_dropping(tmp_path, capsys):
    path = tmp_path / "m.csv"
    logger = CsvLogger(path)
    logger.log({"a": 1})
    logger.close()

    logger = CsvLogger(path)
    logger.log({"a": 2, "surprise": 3})
    logger.close()

    assert "surprise" in capsys.readouterr().out
