"""Job chaining: a run split across walltime-limited links must land where an
uninterrupted run of the same length would, minus the restarted rollouts.

This is the mechanism a multi-day MetaCentrum run depends on, and its failure
mode is silent — a broken handoff still produces logs and checkpoints, it just
never gets anywhere.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from agent.train.checkpoint import load
from agent.train.config import ModelConfig, get_config
from agent.train.loop import train

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_MODEL = ModelConfig(embed_dim=32, depth=1, n_head=4, ff_factor=2, use_bf16=False)


@pytest.fixture(scope="module")
def cfg():
    return get_config("smoke").replace(
        model=TINY_MODEL, num_envs=2, num_steps=3, minibatch_size=2,
        adv_top_frac=0.5, pool_size=32, ckpt_every=1, gate_every=1_000_000,
        num_iters=4,
    )


def _state_like(cfg):
    import equinox as eqx

    from agent.train.loss import make_optimizer
    from agent.train.net import PolicyValueNet
    import jax.random as jrandom

    from agent.train.checkpoint import TrainState

    net = PolicyValueNet(cfg.model, key=jrandom.PRNGKey(0))
    params, _ = eqx.partition(net, eqx.is_inexact_array)
    return TrainState(
        params=params, ema_params=jax.tree.map(jnp.copy, params),
        opt_state=make_optimizer(cfg).init(params), iteration=0, stage_idx=0,
        iters_in_stage=0, last_gate_score=0.0, low_signal_iters=0,
        key=jrandom.PRNGKey(0),
    )


def test_budget_stops_early_and_leaves_a_resumable_checkpoint(cfg, tmp_path):
    """A tiny budget must stop the loop before `num_iters` — with state on disk."""
    out = tmp_path / "run"
    state = train(cfg, out, max_seconds=0.001, max_iters=4)

    assert 0 < state.iteration < 4, f"stopped at {state.iteration}, expected 1..3"
    assert (out / "checkpoint.eqx").exists()
    assert not (out / "DONE").exists(), "marked complete despite stopping early"

    on_disk = load(out / "checkpoint.eqx", _state_like(cfg))
    assert on_disk.iteration == state.iteration


def test_a_chain_of_links_reaches_the_same_iteration_count(cfg, tmp_path):
    """Three budget-limited links must finish the run one straight run finishes."""
    out = tmp_path / "chain"
    ckpt = out / "checkpoint.eqx"

    train(cfg, out, max_seconds=0.001, max_iters=4)
    for _ in range(5):
        state = train(cfg, out, resume=ckpt, max_seconds=0.001, max_iters=4)
        if state.iteration >= 4:
            break

    assert state.iteration == 4, f"chain stalled at {state.iteration}"
    assert (out / "DONE").exists(), "completion marker was not written"
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(state.params))


def test_a_finished_link_exits_without_building_an_env(cfg, tmp_path):
    """The tail of a pre-submitted chain runs after training is done; those links
    must cost nothing, not ~50 s of pool tracing each."""
    out = tmp_path / "run"
    train(cfg, out, max_iters=4)
    assert (out / "DONE").exists()

    import time

    t0 = time.perf_counter()
    state = train(cfg, out, resume=out / "checkpoint.eqx", max_iters=4)
    elapsed = time.perf_counter() - t0

    assert state.iteration == 4
    assert elapsed < 5.0, f"a no-op link took {elapsed:.1f}s — it built an env"


def test_each_link_archives_a_submittable_bot(cfg, tmp_path):
    """`checkpoint.eqx` is a rolling file the next link overwrites.

    Without a per-link archive there is no way to go back to "the bot as of
    link 2" — and §7.4 makes choosing between checkpoints the entire submission
    decision.
    """
    out = tmp_path / "run"

    train(cfg, out, max_iters=2)
    after_link1 = sorted(p.name for p in (out / "weights").iterdir())
    assert after_link1 == ["iter_0000002.safetensors"]

    train(cfg, out, resume=out / "checkpoint.eqx", max_iters=4)
    after_link2 = sorted(p.name for p in (out / "weights").iterdir())
    assert after_link2 == ["iter_0000002.safetensors", "iter_0000004.safetensors"], \
        "link 2 overwrote link 1's archived bot"

    # And each archive is a real, loadable serving artifact.
    from agent.serve.numpy_net import NumpyNet

    for name in after_link2:
        net = NumpyNet.load(out / "weights" / name)
        assert net.embed_dim == cfg.model.embed_dim
        assert net.depth == cfg.model.depth


def test_checkpoint_carries_its_own_model_geometry(cfg, tmp_path):
    """Exporting must not depend on remembering which config produced a run."""
    from agent.train.checkpoint import load_network, read_meta

    out = tmp_path / "run"
    train(cfg, out, max_iters=2)

    meta = read_meta(out / "checkpoint.eqx")
    assert meta["iteration"] == 2
    assert meta["model"]["embed_dim"] == cfg.model.embed_dim
    assert meta["model"]["depth"] == cfg.model.depth

    net, model_cfg, state = load_network(out / "checkpoint.eqx")
    assert state.iteration == 2
    assert model_cfg == cfg.model
    assert net.embedder.weight.shape[0] == cfg.model.embed_dim


def test_export_prefers_ema_and_can_take_raw(cfg, tmp_path):
    """The EMA is what gets submitted (§8); raw params are a different tensor."""
    from agent.train.checkpoint import load_network

    out = tmp_path / "run"
    train(cfg, out, max_iters=3)

    ema, _, _ = load_network(out / "checkpoint.eqx", which="ema")
    raw, _, _ = load_network(out / "checkpoint.eqx", which="raw")

    assert not jnp.allclose(ema.policy_head.weight, raw.policy_head.weight), \
        "EMA and raw params are identical — the EMA is not being tracked"

    with pytest.raises(ValueError, match="ema"):
        load_network(out / "checkpoint.eqx", which="nonsense")


def test_cli_exit_code_signals_whether_the_chain_should_continue(cfg, tmp_path):
    """0 = done, 64 = stopped early. The job script branches on this."""
    out = tmp_path / "cli"
    env_extra = {"PYTHONPATH": str(REPO_ROOT)}

    def run(*args):
        import os

        return subprocess.run(
            [sys.executable, "-m", "agent.train.loop", "--preset", "tiny",
             "--out", str(out), "--iters", "2", *args],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
            env={**os.environ, **env_extra},
        )

    early = run("--max-seconds", "0.001")
    assert early.returncode == 64, f"expected 64, got {early.returncode}\n{early.stdout[-2000:]}"

    finished = run("--resume", str(out / "checkpoint.eqx"))
    assert finished.returncode == 0, (
        f"expected 0, got {finished.returncode}\n{finished.stdout[-2000:]}"
    )
    assert (out / "DONE").exists()
