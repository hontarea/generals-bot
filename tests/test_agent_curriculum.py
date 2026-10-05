"""Gate for build step 8 (AGENT_SPEC.md §10, §7.2).

The important test is that a stage transition actually changes the map
distances. The engine caches its pool kernel on `self` identity, so mutating an
attribute is a silent no-op — this is the only thing that catches it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from agent.train.config import get_config
from agent.train.curriculum import (
    COMPETITION_STAGE,
    STAGES,
    CurriculumState,
    assert_stage_distances,
    generals_distance,
    init_curriculum,
    make_env,
    make_stage_env,
    should_advance,
    should_step_back,
    tick,
    transition,
)

POOL = 32


def _pool(stage_idx):
    env = make_stage_env(stage_idx, POOL)
    pool, _ = env.reset(jrandom.PRNGKey(stage_idx))
    return env, pool


# --------------------------------------------------------------------------
# THE test
# --------------------------------------------------------------------------


def test_stage_transition_actually_changes_map_distances():
    """Rebuilding the env must move the sampled BFS distances between stages."""
    _, pool_a = _pool(0)          # distance 2-5
    _, pool_b = _pool(3)          # distance 13-17

    d_a = np.asarray(jax.vmap(generals_distance)(pool_a))
    d_b = np.asarray(jax.vmap(generals_distance)(pool_b))

    assert d_a.max() <= STAGES[0].max_d, f"stage 0 produced distance {d_a.max()}"
    assert d_b.min() >= STAGES[3].min_d, f"stage 3 produced distance {d_b.min()}"
    assert d_a.max() < d_b.min(), "the two stages produced overlapping distances"


def test_mutating_an_env_does_not_change_its_pool():
    """Pins the engine behaviour that `make_stage_env` exists to work around.

    If this ever starts failing, the engine gained cache invalidation and the
    fresh-env requirement can be revisited — until then, mutation is a silent
    no-op and the log would happily print the new stage.
    """
    env = make_env(2, 5, POOL)
    pool_before, _ = env.reset(jrandom.PRNGKey(0))
    d_before = np.asarray(jax.vmap(generals_distance)(pool_before))

    env.min_generals_distance = 17
    env.max_generals_distance = None
    pool_after, _ = env.reset(jrandom.PRNGKey(0))
    d_after = np.asarray(jax.vmap(generals_distance)(pool_after))

    assert np.array_equal(d_before, d_after), (
        "the engine now honours attribute mutation — make_stage_env's "
        "fresh-construction requirement may be revisitable"
    )
    assert d_after.max() <= 5, "distances changed without a rebuild"


def test_assert_stage_distances_accepts_a_correct_pool():
    _, pool = _pool(0)
    lo, hi = assert_stage_distances(pool, 0)
    assert STAGES[0].min_d <= lo <= hi <= STAGES[0].max_d


def test_assert_stage_distances_rejects_a_stale_pool():
    """The failure mode it exists to catch: a stage-3 label on a stage-0 pool."""
    _, pool = _pool(0)
    with pytest.raises(AssertionError, match="rebuilt"):
        assert_stage_distances(pool, 3)


def test_competition_stage_matches_the_engine_preset():
    from generals.core.env import _MODE_PRESETS

    preset = _MODE_PRESETS["competition"]
    stage = STAGES[COMPETITION_STAGE]
    assert stage.min_d == preset["min_generals_distance"]
    assert stage.max_d is None

    env = make_stage_env(COMPETITION_STAGE, POOL)
    for key, value in preset.items():
        if key == "min_generals_distance":
            continue
        assert getattr(env, key) == value, f"{key}: {getattr(env, key)} != {value}"


def test_every_stage_keeps_competition_settings():
    """Only the generals distance may vary across stages."""
    envs = [make_stage_env(i, POOL) for i in range(len(STAGES))]
    pinned = ("min_grid_size", "max_grid_size", "pad_to", "truncation",
              "mountain_density_range", "num_castles_range", "castle_val_range",
              "perfect_info", "build_castles", "deathtouch_turn")
    for field in pinned:
        values = {getattr(e, field) for e in envs}
        assert len(values) == 1, f"{field} varies across stages: {values}"


# --------------------------------------------------------------------------
# stage bookkeeping
# --------------------------------------------------------------------------


def test_advance_requires_both_the_gate_and_the_minimum_iterations():
    cfg = get_config("full")
    cs = CurriculumState(0, cfg.min_iters_per_stage, 0.80, 0)
    assert should_advance(cs, cfg)

    assert not should_advance(cs._replace(iters_in_stage=10), cfg), \
        "advanced before the minimum iterations"
    assert not should_advance(cs._replace(last_gate_score=0.6), cfg), \
        "advanced without clearing the gate"
    assert not should_advance(
        cs._replace(stage_idx=COMPETITION_STAGE), cfg
    ), "advanced past the competition stage"


def test_step_back_fires_only_after_sustained_low_signal():
    cfg = get_config("full")
    cs = CurriculumState(2, 50, 0.4, cfg.stepback_patience)
    assert should_step_back(cs, cfg)
    assert not should_step_back(cs._replace(low_signal_iters=10), cfg)
    assert not should_step_back(cs._replace(stage_idx=0), cfg), "stepped back off stage 0"


def test_tick_resets_the_low_signal_counter_when_signal_returns():
    cfg = get_config("full")
    cs = init_curriculum()
    for _ in range(5):
        cs = tick(cs, terminal_frac=0.0, cfg=cfg)
    assert cs.low_signal_iters == 5 and cs.iters_in_stage == 5

    cs = tick(cs, terminal_frac=0.5, cfg=cfg)
    assert cs.low_signal_iters == 0, "a batch with signal did not clear the alarm"
    assert cs.iters_in_stage == 6


def test_transition_resets_the_per_stage_counters():
    cs = CurriculumState(1, 999, 0.9, 42)
    up = transition(cs, +1)
    assert up.stage_idx == 2
    assert (up.iters_in_stage, up.last_gate_score, up.low_signal_iters) == (0, 0.0, 0)

    assert transition(cs, -1).stage_idx == 0
    assert transition(CurriculumState(0, 0, 0.0, 0), -1).stage_idx == 0
    assert transition(CurriculumState(COMPETITION_STAGE, 0, 0.0, 0), +1).stage_idx \
        == COMPETITION_STAGE


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


class _AlwaysPass:
    def greedy(self, obs, move_m, build_m, scalars):
        from agent.spec.constants import KIND_PASS

        return jnp.asarray([KIND_PASS, 0, 0, 0, 0], dtype=jnp.int32), jnp.float32(0.0)


@pytest.fixture(scope="module")
def passer_gate():
    from agent.train.curriculum import evaluate_vs_expander

    env = make_stage_env(0, POOL)
    pool, _ = env.reset(jrandom.PRNGKey(0))
    return evaluate_vs_expander(
        env, _AlwaysPass(), pool, jrandom.PRNGKey(1), num_games=8, max_turns=1200
    )


def test_a_passing_agent_never_wins(passer_gate):
    """The sharp seat-mapping check.

    A do-nothing policy cannot capture anything, so no game may score 1.0. If the
    seats were crossed, half the games would report the Expander's wins as the
    model's and this would fail immediately.
    """
    per_game = np.asarray(passer_gate.per_game)
    assert not (per_game == 1.0).any(), (
        f"a passing agent 'won' {(per_game == 1.0).sum()} games — seats are crossed"
    )
    assert set(np.unique(per_game)) <= {0.0, 0.5}


def test_a_passing_agent_scores_below_even(passer_gate):
    """Only *below even*, not lopsided.

    Turtling to a draw is a legitimate outcome here: Expander never concentrates
    force, and a general accrues 0.5 army/turn, so a passive agent survives to
    truncation on plenty of boards. The eval harness ships a `turtle` bot for the
    same reason. What the gate must do is rank it worse than even.
    """
    assert passer_gate.score < 0.45, f"a passing agent scored {passer_gate.score}"


def test_gate_plays_both_seats(passer_gate):
    seats = np.asarray(passer_gate.model_seat)
    assert set(np.unique(seats)) == {0, 1}, "the gate did not swap seats"
    assert abs((seats == 0).sum() - (seats == 1).sum()) <= 1, "seats are unbalanced"
