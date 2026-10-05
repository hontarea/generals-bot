"""Distance curriculum and the Expander gate. AGENT_SPEC.md §7.

The curriculum is a **precondition**, not a convenience: at competition distance
17 a fresh policy essentially never terminates a game, and with no terminal
events the advantages are critic noise that normalization inflates to unit
variance (see the zero-signal hazard in gae.py).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jrandom

from generals.agents.expander_agent import ExpanderAgent
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation
from generals.core.grid import bfs_distance_field


class Stage(NamedTuple):
    min_d: int
    max_d: int | None


#: Every other field stays pinned at competition values at every stage.
STAGES: tuple[Stage, ...] = (
    Stage(2, 5),
    Stage(5, 9),
    Stage(9, 13),
    Stage(13, 17),
    Stage(17, None),        # competition
)
COMPETITION_STAGE = len(STAGES) - 1


def make_env(min_d: int, max_d: int | None, pool_size: int) -> GeneralsEnv:
    """The **only** way an environment may be constructed.

    Fields other than the generals distance are the `competition` preset,
    verbatim; `mode=` is not used because it would also pin the distance.
    """
    return GeneralsEnv(
        min_grid_size=18, max_grid_size=21, pad_to=21, truncation=1200,
        mountain_density_range=(0.24, 0.26), num_castles_range=(9, 11),
        castle_val_range=(20, 26), perfect_info=False,
        build_castles=True, deathtouch_turn=800,
        min_generals_distance=min_d, max_generals_distance=max_d,
        pool_size=pool_size,
    )


def make_stage_env(stage_idx: int, pool_size: int) -> GeneralsEnv:
    """A **fresh** env for the stage. Never mutate an existing one.

    `GeneralsEnv._make_pool_batch` is `@partial(jax.jit, static_argnums=(0, 2, 3))`
    with `self` as static argument 0, and `GeneralsEnv` hashes by identity — so
    the compilation cache does not notice attribute mutation and
    `min_generals_distance` stays baked in at trace time:

        env.min_generals_distance = new_d   # DOES NOTHING
        pool, _ = env.reset(key)            # maps at the OLD distance

    Stage 0 works (nothing is cached yet) and every later transition silently
    changes nothing while the log prints the new stage. Costs ~50 s of retracing
    per new object, four times over a run. That is the price of the assertion
    below meaning anything.
    """
    stage = STAGES[stage_idx]
    return make_env(stage.min_d, stage.max_d, pool_size)


# --------------------------------------------------------------------------
# the mandatory assertion
# --------------------------------------------------------------------------


@jax.jit
def generals_distance(state) -> jax.Array:
    """BFS walking distance between the two generals, around mountains.

    Straight-line distance is not what the generator constrains, so measuring it
    would pass while the boards were wrong.
    """
    (r0, c0), (r1, c1) = state.general_positions[0], state.general_positions[1]
    field = bfs_distance_field(state.passable, (r0, c0))
    return field[r1, c1]


def assert_stage_distances(pool, stage_idx: int, sample: int = 64) -> tuple[int, int]:
    """Sample the new pool and check the distances really moved.

    The highest-value assertion in the system: it is the only thing standing
    between a silent no-op transition and hours of training at the wrong stage.
    """
    stage = STAGES[stage_idx]
    n = min(sample, pool.time.shape[0])
    subset = jax.tree.map(lambda x: x[:n], pool)
    distances = jax.vmap(generals_distance)(subset)

    lo, hi = int(jnp.min(distances)), int(jnp.max(distances))
    if lo < stage.min_d:
        raise AssertionError(
            f"stage {stage_idx} wants distance >= {stage.min_d} but the pool "
            f"contains {lo}. The env was mutated instead of rebuilt — see "
            f"make_stage_env."
        )
    if stage.max_d is not None and hi > stage.max_d:
        raise AssertionError(
            f"stage {stage_idx} wants distance <= {stage.max_d} but the pool "
            f"contains {hi}. The env was mutated instead of rebuilt — see "
            f"make_stage_env."
        )
    return lo, hi


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


class GateResult(NamedTuple):
    score: float             # chess convention, averaged over games
    mean_length: float       # mean turn at which a game resolved
    per_game: jax.Array      # (num_games,) individual scores
    model_seat: jax.Array    # (num_games,) which seat the model played
    unfinished: float        # fraction still running at the turn cap


def evaluate_vs_expander(env, model, pool, key, num_games: int, max_turns: int = 1200):
    """Greedy win rate against `ExpanderAgent`, both seats, as a score in [0, 1].

    Expander, not uniform random: random does not get harder with distance, so a
    random gate degenerates into a floor test. Expander never builds, so a
    working agent should clear this comfortably once the economy is discovered —
    if it cannot, the build channels are not earning their place.

    Chess convention: win 1, draw 1/2, loss 0, averaged over games with the seats
    swapped on half of them.
    """
    from agent.spec import phi

    expander = ExpanderAgent()
    half = num_games // 2

    key, pick = jrandom.split(key)
    idx = jrandom.randint(pick, (num_games,), 0, env.pool_size)
    states = jax.tree.map(lambda x: x[idx], pool)

    # Seat assignment: the model plays seat 0 in the first half, seat 1 in the
    # second, so spawn-room luck cancels.
    model_seat = jnp.concatenate([
        jnp.zeros(half, dtype=jnp.int32), jnp.ones(num_games - half, dtype=jnp.int32)
    ])

    phi_state = phi.init_phi_state(jnp, batch=num_games)
    scores = jnp.full(num_games, -1.0)      # -1 = still running
    lengths = jnp.zeros(num_games, dtype=jnp.int32)

    @jax.jit
    def one_turn(states, phi_state, scores, lengths, key):
        def model_action(state, seat, pstate, k):
            obs = get_observation(state, seat).as_tensor()
            o, sc, mm, bm, new_p = phi.augment(jnp, obs, pstate)
            action, _ = model.greedy(o.astype(jnp.bfloat16).astype(jnp.float32), mm, bm, sc)
            return action, new_p

        def expander_action(state, seat, k):
            return expander.act(get_observation(state, seat), k)

        keys = jrandom.split(key, 2 * num_games)
        m_act, new_phi = jax.vmap(model_action)(
            states, model_seat, phi_state, keys[:num_games]
        )
        e_act = jax.vmap(expander_action)(states, 1 - model_seat, keys[num_games:])

        # Place each action at its seat.
        seat0 = jnp.where((model_seat == 0)[:, None], m_act, e_act)
        seat1 = jnp.where((model_seat == 1)[:, None], m_act, e_act)
        actions = jnp.stack([seat0, seat1], axis=1)

        timestep, next_states = jax.vmap(env.step, in_axes=(0, 0, None))(
            states, actions, pool
        )
        winner = timestep.info.winner
        done = timestep.terminated | timestep.truncated

        # win 1, draw 1/2, loss 0; a truncation with no winner is a draw.
        result = jnp.where(winner < 0, 0.5, jnp.where(winner == model_seat, 1.0, 0.0))
        newly_done = done & (scores < 0.0)
        scores = jnp.where(newly_done, result, scores)
        lengths = jnp.where(newly_done, timestep.info.time, lengths)

        return next_states, new_phi, scores, lengths, done

    for _ in range(max_turns):
        key, k = jrandom.split(key)
        states, phi_state, scores, lengths, _ = one_turn(
            states, phi_state, scores, lengths, k
        )
        if bool(jnp.all(scores >= 0.0)):
            break

    # Anything still running at the cap counts as a draw. A turtling agent can
    # genuinely reach truncation against Expander, which never concentrates
    # enough force to crack a general accruing 0.5 army/turn — so this is a real
    # outcome, not a measurement artifact.
    unfinished = float(jnp.mean(scores < 0.0))
    resolved = scores >= 0.0
    scores = jnp.where(resolved, scores, 0.5)
    mean_length = float(
        jnp.sum(jnp.where(resolved, lengths, 0)) / jnp.maximum(jnp.sum(resolved), 1)
    )
    return GateResult(
        score=float(jnp.mean(scores)), mean_length=mean_length,
        per_game=scores, model_seat=model_seat, unfinished=unfinished,
    )


# --------------------------------------------------------------------------
# stage bookkeeping
# --------------------------------------------------------------------------


class CurriculumState(NamedTuple):
    stage_idx: int
    iters_in_stage: int
    last_gate_score: float
    low_signal_iters: int


def init_curriculum() -> CurriculumState:
    return CurriculumState(0, 0, 0.0, 0)


def should_advance(cs: CurriculumState, cfg) -> bool:
    return (
        cs.stage_idx < COMPETITION_STAGE
        and cs.iters_in_stage >= cfg.min_iters_per_stage
        and cs.last_gate_score >= cfg.gate_threshold
    )


def should_step_back(cs: CurriculumState, cfg) -> bool:
    """If the agent stops finishing games after a transition, the stage is too
    hard and the batch has no terminal signal to learn from."""
    return cs.stage_idx > 0 and cs.low_signal_iters >= cfg.stepback_patience


def tick(cs: CurriculumState, terminal_frac: float, cfg) -> CurriculumState:
    low = cs.low_signal_iters + 1 if terminal_frac < cfg.stepback_terminal_frac else 0
    return cs._replace(iters_in_stage=cs.iters_in_stage + 1, low_signal_iters=low)


def transition(cs: CurriculumState, delta: int) -> CurriculumState:
    """Move a stage. In-flight games must be discarded by the caller (§4.6) —
    a half-played game from the old stage would otherwise leak across."""
    new_idx = max(0, min(COMPETITION_STAGE, cs.stage_idx + delta))
    return CurriculumState(new_idx, 0, 0.0, 0)
