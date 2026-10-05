"""Gate for build step 7 (AGENT_SPEC.md §9.4, §10).

The NumPy network we submit and the JAX network we trained must agree to 1e-4 on
both logits and value. A silent divergence between them is the most expensive bug
available in this project and it is trivially preventable.

Also covers the protocol round trip: a frame off the wire must rebuild the same
14-channel tensor the engine would have produced.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from agent.serve import protocol, runner
from agent.serve.numpy_net import NumpyNet
from agent.spec import phi
from agent.spec.constants import KIND_BUILD, KIND_MOVE, KIND_PASS, PAD
from agent.train.config import ModelConfig
from agent.train.net import PolicyValueNet
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

REPO_ROOT = Path(__file__).resolve().parent.parent
PARITY_TOL = 1e-4


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """The same weights, loaded into both implementations."""
    from agent.serve.weights import save_weights

    cfg = ModelConfig(use_bf16=False)
    jax_net = PolicyValueNet(cfg, key=jrandom.PRNGKey(0))
    path = tmp_path_factory.mktemp("parity") / "weights.safetensors"
    save_weights(jax_net, cfg, path)
    return jax_net, NumpyNet.load(path), path


@pytest.fixture(scope="module")
def observations():
    """Real observations from a real game, not random arrays."""
    env = GeneralsEnv(mode="competition", pool_size=32)
    pool, state = env.reset(jrandom.PRNGKey(0))

    key = jrandom.PRNGKey(1)
    out = []
    for _ in range(12):
        out.append(np.asarray(get_observation(state, 0).as_tensor()))
        key, k = jrandom.split(key)
        acts = jrandom.randint(k, (2, 5), 0, 4).at[:, 0].set(0)
        acts = acts.at[:, 1].set(acts[:, 1] % PAD).at[:, 2].set(acts[:, 2] % PAD)
        _, state = env.step(state, acts.astype(jnp.int32), pool)
    return out


# --------------------------------------------------------------------------
# the parity test
# --------------------------------------------------------------------------


def test_numpy_and_jax_agree_on_logits_and_value(pair, observations):
    jax_net, np_net, _ = pair
    st_jx = phi.init_phi_state(jnp)
    st_np = phi.init_phi_state(np)

    worst_logits = worst_value = 0.0
    for t, obs_arr in enumerate(observations):
        o_jx, s_jx, mm_jx, bm_jx, st_jx = phi.augment(jnp, jnp.asarray(obs_arr), st_jx)
        o_np, s_np, mm_np, bm_np, st_np = phi.augment(np, obs_arr, st_np)

        lg_jx, v_jx, _ = jax_net._logits_and_value(o_jx, mm_jx, bm_jx, s_jx)
        lg_np, v_np = np_net.logits_and_value(o_np, mm_np, bm_np, s_np)

        d_logits = float(np.max(np.abs(np.asarray(lg_jx) - lg_np)))
        d_value = abs(float(v_jx) - v_np)
        worst_logits = max(worst_logits, d_logits)
        worst_value = max(worst_value, d_value)

        assert d_logits < PARITY_TOL, f"logits diverged by {d_logits} at t={t}"
        assert d_value < PARITY_TOL, f"value diverged by {d_value} at t={t}"

    print(f"\nworst logit delta {worst_logits:.2e}, worst value delta {worst_value:.2e}")


def test_greedy_actions_agree(pair, observations):
    """Argmax is what actually gets played; near-ties are where parity bugs bite."""
    jax_net, np_net, _ = pair
    st_jx, st_np = phi.init_phi_state(jnp), phi.init_phi_state(np)

    for t, obs_arr in enumerate(observations):
        o_jx, s_jx, mm_jx, bm_jx, st_jx = phi.augment(jnp, jnp.asarray(obs_arr), st_jx)
        o_np, s_np, mm_np, bm_np, st_np = phi.augment(np, obs_arr, st_np)

        a_jx, _ = jax_net.greedy(o_jx, mm_jx, bm_jx, s_jx)
        a_np, _ = np_net.greedy(o_np, mm_np, bm_np, s_np)
        assert np.array_equal(np.asarray(a_jx), np.asarray(a_np)), f"actions differ at t={t}"


def test_parity_holds_on_a_padded_board(pair, observations):
    """The padding path runs on every turn of every graded game and never in
    training, so it needs its own parity check."""
    jax_net, np_net, _ = pair
    pad_h, pad_w = 3, 2
    st_jx, st_np = phi.init_phi_state(jnp), phi.init_phi_state(np)

    for obs_arr in observations[:5]:
        o_jx, s_jx, mm_jx, bm_jx, st_jx = phi.augment(
            jnp, jnp.asarray(obs_arr), st_jx, pad_h, pad_w
        )
        o_np, s_np, mm_np, bm_np, st_np = phi.augment(np, obs_arr, st_np, pad_h, pad_w)

        from agent.spec.codec import mask_penalty

        lg_jx, v_jx, _ = jax_net._logits_and_value(o_jx, mm_jx, bm_jx, s_jx)
        lg_jx = lg_jx - mask_penalty(jnp, mm_jx, bm_jx) + mask_penalty(
            jnp, mm_jx, bm_jx, pad_h, pad_w
        )
        lg_np, v_np = np_net.logits_and_value(o_np, mm_np, bm_np, s_np, pad_h, pad_w)

        assert float(np.max(np.abs(np.asarray(lg_jx) - lg_np))) < PARITY_TOL
        assert abs(float(v_jx) - v_np) < PARITY_TOL


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------


def _encode(obs, player_id=0, h=PAD, w=PAD):
    """Render an engine Observation the way `competition/protocol.py` does."""
    sys.path.insert(0, str(REPO_ROOT / "competition"))
    import protocol as engine_protocol

    return engine_protocol.encode_observation(obs)


def test_frame_to_tensor_reproduces_the_engine_tensor():
    """A frame off the wire must rebuild what `Observation.as_tensor()` gave.

    This is the join between training and serving: everything downstream assumes
    the two are the same array.
    """
    env = GeneralsEnv(mode="competition", pool_size=32)
    pool, state = env.reset(jrandom.PRNGKey(2))

    key = jrandom.PRNGKey(3)
    for _ in range(10):
        obs = get_observation(state, 0)
        wire = _encode(obs)
        stream = io.StringIO(wire)

        hs = protocol.Handshake(0, PAD, PAD)
        frame = protocol.read_frame(stream, hs)
        rebuilt = protocol.frame_to_tensor(frame, hs)
        engine = np.asarray(obs.as_tensor())

        from agent.spec.constants import Obs14

        for ch in Obs14:
            assert np.array_equal(rebuilt[ch], engine[ch]), f"channel {ch.name} differs"

        key, k = jrandom.split(key)
        acts = jrandom.randint(k, (2, 5), 0, 4).at[:, 0].set(0)
        acts = acts.at[:, 1].set(acts[:, 1] % PAD).at[:, 2].set(acts[:, 2] % PAD)
        _, state = env.step(state, acts.astype(jnp.int32), pool)


def test_handshake_and_padding_arithmetic():
    hs = protocol.parse_handshake("1 18 19\n")
    assert (hs.player_id, hs.height, hs.width) == (1, 18, 19)
    assert (hs.pad_h, hs.pad_w) == (3, 2)


def test_format_action_round_trips_through_the_engine_decoder():
    sys.path.insert(0, str(REPO_ROOT / "competition"))
    import protocol as engine_protocol

    for action in ([0, 3, 4, 2, 1], [1, 0, 0, 0, 0], [2, 7, 7, 0, 0]):
        line = protocol.format_action(action)
        assert np.array_equal(np.asarray(engine_protocol.decode_action(line)), action)


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------


class _Fail:
    """A network that raises, to exercise the failure path."""

    def greedy(self, *_args, **_kw):
        raise RuntimeError("boom")


def test_runner_emits_a_pass_instead_of_crashing(pair, monkeypatch):
    """An invalid action is a free pass; a crash forfeits the game."""
    _, _, weights = pair

    env = GeneralsEnv(mode="competition", pool_size=32)
    _, state = env.reset(jrandom.PRNGKey(4))
    wire = _encode(get_observation(state, 0))

    monkeypatch.setattr(NumpyNet, "load", staticmethod(lambda _p: _Fail()))
    stdin = io.StringIO("0 21 21\n" + wire)
    stdout, stderr = io.StringIO(), io.StringIO()
    runner.run(weights, stdin, stdout, stderr)

    assert stdout.getvalue().strip() == protocol.PASS_LINE
    assert "boom" in stderr.getvalue()


def test_runner_exits_cleanly_on_eof(pair):
    _, _, weights = pair
    stdout, stderr = io.StringIO(), io.StringIO()
    runner.run(weights, io.StringIO("0 21 21\n"), stdout, stderr)
    assert stdout.getvalue() == ""


def test_runner_plays_legal_actions_over_a_real_game(pair):
    _, _, weights = pair
    env = GeneralsEnv(mode="competition", pool_size=32)
    pool, state = env.reset(jrandom.PRNGKey(5))

    hs = protocol.Handshake(0, PAD, PAD)
    agent = runner.Agent(weights, hs)

    key = jrandom.PRNGKey(6)
    for _ in range(15):
        obs = get_observation(state, 0)
        frame = protocol.read_frame(io.StringIO(_encode(obs)), hs)
        line = agent.act(frame)

        kind, r, c, d, s = (int(x) for x in line.split())
        assert kind in (KIND_MOVE, KIND_PASS, KIND_BUILD)
        assert 0 <= r < PAD and 0 <= c < PAD and 0 <= d < 4 and s in (0, 1)

        key, k = jrandom.split(key)
        acts = jnp.stack([
            jnp.asarray([kind, r, c, d, s], dtype=jnp.int32),
            jrandom.randint(k, (5,), 0, 4).at[0].set(1).astype(jnp.int32),
        ])
        _, state = env.step(state, acts, pool)


# --------------------------------------------------------------------------
# the artifact itself
# --------------------------------------------------------------------------


def test_submission_plays_a_match_against_hunter():
    """End to end through `bash run.sh`, exactly as the harness spawns it."""
    bot = REPO_ROOT / "bots/rl_bot"

    # `run.sh` says `exec python`, so the interpreter's bin dir has to be on
    # PATH — this is what `evaluation/agent_io.py` does when it spawns a bot.
    import os

    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "competition/matchup.py"),
         str(bot / "run.sh"),
         str(REPO_ROOT / "bots/hunter_bot/run.sh"),
         "--mode", "competition", "--seed", "0"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800, env=env,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
