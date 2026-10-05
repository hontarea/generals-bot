"""Tests for the evaluation harness (evaluation/)."""
import json
import os
import textwrap

import numpy as np
import pytest

from evaluation.evaluate import score_bot_a, wilson
from evaluation.run_match import MatchJob, classify_end, run_match, validate_record
from evaluation.seeds import SUITES, parse_seed_expr, resolve_seeds


# ------------------------------------------------------------------- seeds

def test_suites_shapes():
    # smoke is the one frozen suite; freezing it is what makes scores
    # comparable across runs, so pin it here rather than in prose alone.
    assert SUITES["smoke"] == list(range(20))


def test_seed_expr_parsing():
    assert parse_seed_expr("17")[0] == [17]
    assert parse_seed_expr("0-3")[0] == [0, 1, 2, 3]
    assert parse_seed_expr("0-2,42,10-11")[0] == [0, 1, 2, 42, 10, 11]
    assert parse_seed_expr("1,1,1")[0] == [1]  # dedup


def test_seed_expr_random():
    s1, m1 = parse_seed_expr("random:10", entropy=99)
    s2, _ = parse_seed_expr("random:10", entropy=99)
    assert s1 == s2 and len(s1) == 10 == len(set(s1))
    assert m1["random"] and m1["entropy"] == 99 and m1["seeds"] == s1
    assert min(s1) >= 10_000
    s3, _ = parse_seed_expr("random:10", entropy=100)
    assert s3 != s1


@pytest.mark.parametrize("bad", ["", "5-2", "-3", "a-b", "random:0", "1,,2"])
def test_seed_expr_errors(bad):
    with pytest.raises(ValueError):
        parse_seed_expr(bad)


def test_resolve_seeds_exclusive():
    with pytest.raises(ValueError):
        resolve_seeds(None, None)
    with pytest.raises(ValueError):
        resolve_seeds("smoke", "0-5")
    with pytest.raises(ValueError):
        resolve_seeds("nope", None)
    seeds, meta = resolve_seeds("smoke", None)
    assert seeds == SUITES["smoke"] and meta["suite"] == "smoke"


# -------------------------------------------------------------- statistics

def test_wilson_known_values():
    lo, hi = wilson(0.5, 100)
    assert lo == pytest.approx(0.404, abs=0.002)
    assert hi == pytest.approx(0.596, abs=0.002)
    assert wilson(0.5, 0) == (0.0, 1.0)
    lo, hi = wilson(1.0, 6)
    assert hi == pytest.approx(1.0) and 0.5 < lo < 0.7


def _rec(seed, swap, winner):
    return {"seed": seed, "swap": swap, "winner": winner}


def test_score_seat_mapping():
    # bot A wins seat 0 unswapped, wins seat 1 swapped, draws, loses
    records = [_rec(0, 0, 0), _rec(0, 1, 1), _rec(1, 0, None), _rec(1, 1, 0)]
    score, w, d, l = score_bot_a(records)
    assert (w, d, l) == (2, 1, 1)
    assert score == pytest.approx((2 + 0.5) / 4)


# ----------------------------------------------------------- end_reason map

@pytest.mark.parametrize("done,pre_time,winner_val,expect", [
    (True, 100, 0, ("capture", 0)),
    (True, 799, 1, ("capture", 1)),
    (True, 800, 1, ("deathtouch", 1)),
    (True, 1100, -1, ("deathtouch", None)),   # mutual touch draw
    (False, 1200, -1, ("truncation", None)),
])
def test_classify_end(done, pre_time, winner_val, expect):
    assert classify_end(done, pre_time, winner_val, 800) == expect


def test_classify_end_no_deathtouch_rule():
    assert classify_end(True, 900, 0, None) == ("capture", 0)


# ------------------------------------------------------------ record schema

def _good_record():
    return {"seed": 1, "swap": 0, "bot0": "a", "bot1": "b", "winner": 0,
            "end_reason": "capture", "turns": 10, "faults": [0, 0],
            "reply_ms": {"p50": [1, 1], "p99": [1, 1], "max": [1, 1]},
            "final": {"land": [5, 3], "army": [9, 4], "castles": [0, 0]},
            "wall_s": 1.0, "h": 18, "w": 19}


def test_validate_record_accepts_good():
    validate_record(_good_record())


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("winner"),
    lambda r: r.update(end_reason="explosion"),
    lambda r: r.update(winner=2),
    lambda r: r.update(faults=[0]),
    lambda r: r.update(end_reason="crash", winner=None),
])
def test_validate_record_rejects_bad(mutate):
    rec = _good_record()
    mutate(rec)
    with pytest.raises(AssertionError):
        validate_record(rec)


# ------------------------------------------- ruleset modifiers (landmine #1)
# The harness must play by env.step's modifier stack: castle builds must
# land, deathtouch must fire. These test matchup.make_transition directly.

@pytest.fixture(scope="module")
def competition_env():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import matchup
    from generals import GeneralsEnv
    env = GeneralsEnv(mode="competition")
    return env, matchup


def test_build_action_creates_castle(competition_env):
    import jax.numpy as jnp
    env, matchup = competition_env
    state = matchup.make_board(env, 0)
    # give player 0 an owned plain cell with plenty of army, then build there
    r, c = None, None
    gp = np.asarray(state.general_positions)
    for rr in range(state.armies.shape[0]):
        for cc in range(state.armies.shape[1]):
            if bool(state.passable[rr, cc]) and not bool(state.generals[rr, cc]):
                r, c = rr, cc
                break
        if r is not None:
            break
    state = state._replace(
        armies=state.armies.at[r, c].set(500),
        ownership=state.ownership.at[0, r, c].set(True),
        ownership_neutral=state.ownership_neutral.at[r, c].set(False))
    transition = matchup.make_transition(env)
    actions = jnp.array([[2, r, c, 0, 0], [1, 0, 0, 0, 0]], dtype=jnp.int32)
    new_state, _ = transition(state, actions)
    assert bool(new_state.castles[r, c]), \
        "build action did not create a castle — modifier stack not applied"
    assert not bool(state.castles[r, c])
    assert gp is not None  # silence linters


def test_deathtouch_fires_post_threshold(competition_env):
    import jax.numpy as jnp
    env, matchup = competition_env
    state = matchup.make_board(env, 0)
    # single enemy unit adjacent to player 0's general, at time >= 800
    gr, gc = (int(x) for x in np.asarray(state.general_positions)[0])
    H, W = state.armies.shape
    for dr, dc, d_back in ((-1, 0, 1), (1, 0, 0), (0, -1, 3), (0, 1, 2)):
        ar, ac = gr + dr, gc + dc
        if 0 <= ar < H and 0 <= ac < W and bool(state.passable[ar, ac]):
            break
    state = state._replace(
        armies=state.armies.at[ar, ac].set(2),
        ownership=state.ownership.at[1, ar, ac].set(True),
        ownership_neutral=state.ownership_neutral.at[ar, ac].set(False),
        time=jnp.int32(900))
    transition = matchup.make_transition(env)
    # player 1 moves its 2-army stack onto the general (army-irrelevant touch)
    actions = jnp.array([[1, 0, 0, 0, 0], [0, ar, ac, d_back, 0]],
                        dtype=jnp.int32)
    _, info = transition(state, actions)
    assert bool(info.is_done) and int(info.winner) == 1, \
        "deathtouch did not fire — modifier stack not applied"


# ------------------------------------------------- run_match fault handling
# Scripted throwaway bots exercise the crash / malformed / hang paths.

def _write_bot(tmp_path, name, body):
    bot = tmp_path / name
    bot.mkdir()
    (bot / "run.sh").write_text("#!/usr/bin/env bash\nexec python -u main.py\n")
    (bot / "main.py").write_text(textwrap.dedent(body))
    return bot


_BOT_TEMPLATE = """\
    import sys
    hs = sys.stdin.readline()
    pid, H, W = (int(x) for x in hs.split())
    turn = 0
    while True:
        first = sys.stdin.readline()
        if not first:
            sys.exit(0)
        for _ in range(3 * H):
            sys.stdin.readline()
        turn += 1
{action}
"""


def _expander_bot(tmp_path):
    return _write_bot(tmp_path, "okbot", _BOT_TEMPLATE.format(action=(
        "        print('1 0 0 0 0', flush=True)")))


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("bots")


def test_crash_is_a_result_with_stderr(scratch):
    crasher = _write_bot(scratch, "crasher", _BOT_TEMPLATE.format(action="""\
        if turn >= 3:
            print('dying now', file=sys.stderr)
            sys.exit(3)
        print('1 0 0 0 0', flush=True)"""))
    ok = _expander_bot(scratch)
    rec = run_match(MatchJob(seed=0, swap=0, bot_a_dir=crasher, bot_b_dir=ok,
                             replay_path=None))
    validate_record(rec)
    assert rec["end_reason"] == "crash"
    assert rec["winner"] == 1
    assert "dying now" in rec["stderr_tail"]["0"]


def test_malformed_replies_counted_as_faults(scratch):
    garbler = _write_bot(scratch, "garbler", _BOT_TEMPLATE.format(action="""\
        if turn <= 4:
            print('what is a move', flush=True)
        elif turn == 5:
            sys.exit(0)
        else:
            print('1 0 0 0 0', flush=True)"""))
    ok = scratch / "okbot"
    rec = run_match(MatchJob(seed=0, swap=0, bot_a_dir=garbler, bot_b_dir=ok,
                             replay_path=None))
    assert rec["end_reason"] == "crash"      # exits at turn 5
    assert rec["faults"][0] == 4             # 4 malformed replies before that
    assert rec["faults"][1] == 0


def test_hung_bot_forfeits(scratch):
    sleeper = _write_bot(scratch, "sleeper", _BOT_TEMPLATE.format(action="""\
        if turn >= 2:
            import time; time.sleep(60)
        print('1 0 0 0 0', flush=True)"""))
    ok = scratch / "okbot"
    rec = run_match(MatchJob(seed=0, swap=1, bot_a_dir=sleeper, bot_b_dir=ok,
                             replay_path=None, hard_cap_s=0.5))
    assert rec["end_reason"] == "fault_forfeit"
    # sleeper is bot A with swap=1 -> seat 1; opponent seat 0 wins
    assert rec["winner"] == 0


def test_replay_roundtrip_resimulates(scratch, tmp_path):
    from evaluation.run_match import load_replay, states_from_replay
    import jax.numpy as jnp
    import matchup
    from generals import GeneralsEnv

    crasher = scratch / "crasher"
    ok = scratch / "okbot"
    replay_path = tmp_path / "r.npz"
    rec = run_match(MatchJob(seed=0, swap=0, bot_a_dir=ok, bot_b_dir=crasher,
                             replay_path=replay_path))
    rep = load_replay(replay_path)
    assert rep.num_turns == rec["turns"]
    assert rep.meta["end_reason"] == rec["end_reason"]

    states, infos = states_from_replay(rep)
    assert len(states) == rep.num_turns + 1

    env = GeneralsEnv(mode="competition")
    transition = matchup.make_transition(env)
    state = states[0]
    for t in range(rep.num_turns):
        state, _ = transition(state, jnp.asarray(rep.actions_eff[t]))
    assert np.array_equal(np.asarray(state.armies), rep.armies[-1])
    assert np.array_equal(np.asarray(state.ownership), rep.ownership[-1])
