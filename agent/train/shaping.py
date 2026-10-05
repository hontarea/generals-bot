"""Potential-based reward shaping for the castle economy. AGENT_SPEC_DELTAS.md D19.

The engine pays ±1 once, at the end of a ~277-turn game. A castle earns 0.5
army/turn against a plain tile's 0.02 and repays its 35 cost in 70 turns, so it is
the dominant economic action in this ruleset — and with `gamma=1`, `lambda=0.9` the
GAE horizon is ten steps, which means the payoff of a build reaches the policy only
through a critic that explains ~a third of the outcome variance. `rl_bot_b` built
0.6 castles per game after 100k iterations.

Shaping fixes the *timing* of the credit, not the objective:

    F(s, s') = gamma * Phi(s') - Phi(s),   gamma = 1

Ng, Harada & Russell: adding `F` to the reward leaves the set of optimal policies
unchanged, provided `Phi(absorbing) = 0`. So this cannot teach the agent to prefer
a worse policy — it can only redistribute an unchanged total over the timeline. The
terminal zeroing is applied in `gae.py`, beside the bootstrap guard it mirrors.

    z(s, seat) = w_land   * (own_land   - opp_land)   / COUNT_SCALE
               + w_army   * (own_army   - opp_army)   / COUNT_SCALE
               + w_castle * (own_castle - opp_castle) / CASTLE_COUNT_SCALE

    Phi_unit = tanh(z)          in [-1, 1]
    Phi      = shaping_coef * Phi_unit

Three properties, each load-bearing and each pinned by `tests/test_agent_shaping.py`:

* **Difference form.** Only the gap moves `Phi`. Both economies growing together —
  which is most of a normal game — produces *zero* shaping reward. The agent is paid
  for out-growing the opponent, never for the clock ticking.
* **Antisymmetry.** `Phi(s, 1) == -Phi(s, 0)` exactly, so shaping stays as zero-sum
  as the engine's own reward and one computation serves both seats. Reimplementing a
  per-seat sign flip is the bug class `rollout.py` already warns about; here it is
  impossible by construction.
* **Bounded.** `tanh` gives `|Phi| <= shaping_coef` without a clip's dead zone. This
  is what pins the shaped return to `[-(1+coef), 1+coef]`, which HL-Gauss's
  `v_min`/`v_max` depend on.

What this module stores is the **unit** potential, not the scaled one: the
coefficient is applied in `gae.py`, so an existing rollout can be re-scored at a
different `shaping_coef` without collecting it again, and `shaping_coef = 0.0`
multiplies to exactly 0.0 and leaves every downstream float bit-identical.

`Phi` reads the true `GameState`, not the seat's observation. The reward function
belongs to the environment, not to the agent — nothing here enters the observation
tensor, `agent/serve/` is untouched, and there is nothing to leak at match time.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from agent.spec.constants import CASTLE_COUNT_SCALE, COUNT_SCALE


class ShapingWeights(NamedTuple):
    """Relative prices inside `z`. Python floats, so they close over into the jit.

    The ratios are what matter, and `w_castle : w_army` is the one that decides
    whether building reads as a gain at all. At the defaults, near `z = 0`:

        castle          +2.5 / 10  = +0.25 z
        its 35 army     -0.2 * 35 / 100 = -0.07 z
        net              +0.18 z    ->  dPhi ~ +0.036 at coef 0.2

    i.e. one castle is worth about 3.6% of a win, paid at the moment of the
    decision rather than 500 turns later. A captured tile is worth ~0.002.
    """

    w_land: float = 1.0
    w_army: float = 0.2
    w_castle: float = 2.5


def unit_potential(state, seat: int, w: ShapingWeights = ShapingWeights()):
    """`tanh(z)` for one game from one seat's point of view. Scalar in [-1, 1].

    `seat` is a Python int so this traces separately per seat and nothing branches
    on a traced value.
    """
    own = state.ownership[seat].astype(jnp.float32)
    opp = state.ownership[1 - seat].astype(jnp.float32)
    armies = state.armies.astype(jnp.float32)
    castles = state.castles.astype(jnp.float32)

    land = jnp.sum(own) - jnp.sum(opp)
    army = jnp.sum(armies * own) - jnp.sum(armies * opp)
    # The general is excluded deliberately: you hold exactly one until you lose,
    # at which point the game is over and Phi is zeroed anyway. Counting it would
    # add a constant to both seats and cancel.
    castle = jnp.sum(castles * own) - jnp.sum(castles * opp)

    z = (
        w.w_land * land / COUNT_SCALE
        + w.w_army * army / COUNT_SCALE
        + w.w_castle * castle / CASTLE_COUNT_SCALE
    )
    return jnp.tanh(z)


def unit_potentials(states, w: ShapingWeights = ShapingWeights()):
    """Batched `(N, ...)` states -> `(2N,)`, in the rollout's row order.

    Row `r` in `[0, N)` is game `r` seat 0, row `r` in `[N, 2N)` is game `r - N`
    seat 1 — the layout `rollout.py` flattens both seats into. Seat 1 is the
    negation, never a second computation.
    """
    phi0 = jax.vmap(lambda s: unit_potential(s, 0, w))(states)
    return jnp.concatenate([phi0, -phi0])
