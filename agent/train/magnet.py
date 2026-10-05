"""The expander magnet: a heuristic prior the regularizer pulls toward.

AGENT_SPEC.md §6.1. **Enable this only after plain-entropy PPO is verified
learning** (`cfg.use_magnet`). It costs a permanent bias toward the heuristic that
only fades as `ent_coef -> 0`.

Why include it at all despite the paper claiming plain entropy suffices: three
mechanisms make terminal events happen early — throughput, curriculum, magnet. We
have roughly a tenth of the reference's throughput, so the other two matter more,
not less.

Two fixes over the reference:

* **Half-moves score below full moves.** The reference reuses one array for both
  planes, making its magnet indifferent to splitting — which no real expander
  heuristic is.
* **Shifts, not rolls.** `jnp.roll` wraps at the border, so the reference's magnet
  rewards moves off the edge of the board.

The collapsed pass index (§3.5) already fixes the reference's inflated pass mass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agent.spec.codec import shift
from agent.spec.constants import (
    ARMY_SCALE,
    DIRECTIONS,
    N_PLANES,
    PAD,
    Ch,
)

SCORE_PASS = 0.2
SCORE_DEFAULT = 1.0
SCORE_NEUTRAL = 2.0
SCORE_ENEMY = 3.0
SCORE_BUILD = 5.0
SCORE_TOPK = 2.0
HALF_MOVE_PENALTY = 0.5
TOP_K_STACKS = 5

#: One army point in normalized units — phi divides army-like channels by 50.
ONE_ARMY = 1.0 / ARMY_SCALE


def expander_magnet(obs, move_m, build_m):
    """(3970,) probability distribution. Treated as a constant: no gradient.

    Scores are **logits**, so the differences are what matter, not the values.
    """
    own_army = obs[Ch.OWN_ARMY]
    enemy_army = obs[Ch.ENEMY_ARMY]
    owned = obs[Ch.OWNED] > 0.5
    opponent = obs[Ch.OPPONENT] > 0.5
    neutral = obs[Ch.NEUTRAL] > 0.5
    affordable = obs[Ch.BUILD_AFFORDABLE] > 0.5

    # Destination properties per direction, padded rather than rolled.
    def dest(grid, fill=0.0):
        return jnp.stack([shift(jnp, grid, dr, dc, fill) for dr, dc in DIRECTIONS])

    dest_enemy_army = dest(enemy_army)
    dest_is_opponent = dest(opponent.astype(jnp.float32)) > 0.5
    dest_is_neutral = dest(neutral.astype(jnp.float32)) > 0.5
    dest_is_owned = dest(owned.astype(jnp.float32)) > 0.5

    # Sequential wheres, so later assignments win: build > enemy > neutral > default.
    scores = jnp.full((4, PAD, PAD), SCORE_DEFAULT, dtype=jnp.float32)

    expansion = ~dest_is_owned
    scores = jnp.where(expansion & dest_is_neutral, SCORE_NEUTRAL, scores)

    beatable = own_army[None] > dest_enemy_army + ONE_ARMY
    scores = jnp.where(expansion & dest_is_opponent & beatable, SCORE_ENEMY, scores)

    # A bonus on the sources worth committing: your biggest stacks.
    flat_own = jnp.where(owned, own_army, -jnp.inf).reshape(-1)
    kth = jax.lax.top_k(flat_own, TOP_K_STACKS)[0][-1]
    is_top_stack = owned & (own_army >= kth) & jnp.isfinite(kth)
    scores = scores + jnp.where(is_top_stack, SCORE_TOPK, 0.0)[None]

    # Illegal moves are masked out by `mask_penalty` downstream, but leaving them
    # scored would still shift the softmax mass, so zero them here too.
    legal = jnp.transpose(move_m, (2, 0, 1))
    full_moves = jnp.where(legal, scores, -jnp.inf)
    half_moves = jnp.where(legal, scores - HALF_MOVE_PENALTY, -jnp.inf)

    builds = jnp.where(build_m & affordable, SCORE_BUILD, -jnp.inf)

    planes = jnp.concatenate([full_moves, half_moves, builds[None]], axis=0)
    logits = jnp.concatenate([planes.reshape(-1), jnp.asarray([SCORE_PASS])])

    # If nothing is legal, softmax over all -inf would be NaN; pass is always
    # finite, so the distribution is well defined.
    return jax.nn.softmax(logits)


assert N_PLANES == 9
