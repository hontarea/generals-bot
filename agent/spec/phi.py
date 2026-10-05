"""Observation augmentation: 14 engine channels + carried memory -> 42 channels.

AGENT_SPEC.md §2.3. Single-sample; training vmaps it, serving calls it once per
turn. `xp`-parameterized like `codec.py` (§1.1): no in-place mutation, no `.at[]`,
no Python branching on array values.

Two deviations from the spec signature, both deliberate:

* `PhiState` is a `NamedTuple`, not a frozen dataclass, so it is a jax pytree
  without registration and `reset_phi_state` can be a plain tree-map.
* `augment` also returns the move and build masks. They are derived from the very
  planes the network is shown, including the serving padding synthesis, so
  returning them here makes it impossible to mask against a different board than
  the one the network saw.

Normalization happens here, before storage (§2.3). `PhiState` holds **raw** army
values and **raw** ages; scaling is applied at emit time only.
"""
from __future__ import annotations

from typing import NamedTuple

from .codec import build_cost_grid, build_mask, move_mask, shift
from .constants import (
    AGE_SCALE,
    ARMY_SCALE,
    CASTLE_COUNT_SCALE,
    COUNT_SCALE,
    DEATHTOUCH_TURN,
    HISTORY_SIZE,
    LAND_GROWTH_PERIOD,
    N_CHANNELS,
    N_SCALARS,
    PAD,
    PAYBACK_SCALE,
    STRUCTURE_GROWTH_PERIOD,
    TEMPORAL_LAG_LONG,
    TEMPORAL_LAG_SHORT,
    TEMPORAL_WINDOW,
    TRUNCATION,
    YIELD_SCALE,
    Ch,
    Obs14,
    Sc,
)


class PhiState(NamedTuple):
    """Everything carried between turns of one game, for one seat."""

    # monotone accumulators — permanent facts, never cleared mid-game
    seen: object
    enemy_seen: object
    enemy_general_seen: object
    mountains_seen: object
    castles_seen: object
    enemy_castles_seen: object

    # value + staleness clock — mutable facts. Raw army, raw step count.
    last_enemy_army_value: object
    last_enemy_age: object

    # bookkeeping for the delta stacks; never emitted
    last_army: object
    last_enemy_army: object

    # shift registers of DELTAS, newest first
    army_stack: object
    enemy_stack: object

    # ring buffers, newest at index -1. Never stored per-step in the rollout.
    opp_army_hist: object
    opp_land_hist: object


def init_phi_state(xp, batch=None):
    """All-zero memory. `batch=None` gives a single game's state."""
    grid = () if batch is None else (batch,)

    def z(*shape, dtype):
        return xp.zeros(grid + shape, dtype=dtype)

    b = z(PAD, PAD, dtype=bool)
    f = z(PAD, PAD, dtype=xp.float32)
    return PhiState(
        seen=b, enemy_seen=b, enemy_general_seen=b,
        mountains_seen=b, castles_seen=b, enemy_castles_seen=b,
        last_enemy_army_value=f, last_enemy_age=f,
        last_army=f, last_enemy_army=f,
        army_stack=z(HISTORY_SIZE, PAD, PAD, dtype=xp.float32),
        enemy_stack=z(HISTORY_SIZE, PAD, PAD, dtype=xp.float32),
        opp_army_hist=z(TEMPORAL_WINDOW, dtype=xp.float32),
        opp_land_hist=z(TEMPORAL_WINDOW, dtype=xp.float32),
    )


def reset_phi_state(xp, state, dones):
    """Zero the memory of finished games. Batched: `dones` is (B,) bool.

    Called *after* the observation is stored (§4.3), so a finished game's last
    observation keeps its full memory and only the next step starts clean.
    """
    def clear(leaf):
        mask = xp.reshape(dones, dones.shape + (1,) * (leaf.ndim - dones.ndim))
        return xp.where(mask, xp.zeros_like(leaf), leaf)

    return PhiState(*[clear(leaf) for leaf in state])


def max_pool_3x3(xp, grid):
    """3x3 dilation, reproducing `generals.core.game.get_visibility`.

    The observation hands you already-masked fields rather than a visibility mask,
    so vision has to be recomputed agent-side.
    """
    out = grid
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out = out | shift(xp, grid, dr, dc, False)
    return out


def _pad_mask(xp, pad_h, pad_w):
    """(21, 21) bool marking cells outside the real board.

    Padding is bottom/right only — `generals/core/grid.py` pads
    `((0, pad_h), (0, pad_w))` with mountains (delta D4).
    """
    rows = xp.arange(PAD)[:, None]
    cols = xp.arange(PAD)[None, :]
    return (rows >= PAD - pad_h) | (cols >= PAD - pad_w)


def _lag(xp, hist, k):
    """Value from `k` steps ago in a ring buffer whose newest entry is at -1."""
    return hist[TEMPORAL_WINDOW - 1 - k]


def augment(xp, obs_arr, state, pad_h=0, pad_w=0):
    """One turn of memory folding.

    Args:
        obs_arr: (14, 21, 21) from `Observation.as_tensor()`. **int32** from the
            engine, so it is cast here.
        state: the `PhiState` carried from the previous turn.
        pad_h, pad_w: static ints. Zero in training (the engine pads the board
            with real mountains inside the game state); non-zero at serving,
            where the handshake gives an exact 18-21 rectangle.

    Returns:
        (obs (42,21,21) f32, scalars (16,) f32, move_m (21,21,4) bool,
         build_m (21,21) bool, new_state)
    """
    o = xp.asarray(obs_arr).astype(xp.float32)

    armies = o[Obs14.ARMIES]
    generals = o[Obs14.GENERALS] > 0
    castles = o[Obs14.CASTLES] > 0
    mountains = o[Obs14.MOUNTAINS] > 0
    neutral = o[Obs14.NEUTRAL] > 0
    owned = o[Obs14.OWNED] > 0
    opponent = o[Obs14.OPPONENT] > 0
    fog = o[Obs14.FOG] > 0
    structures_in_fog = o[Obs14.STRUCTURES_IN_FOG] > 0

    # Broadcast scalar planes: read one real cell. (0, 0) is always inside the
    # board, since padding is bottom/right.
    own_land = o[Obs14.OWNED_LAND, 0, 0]
    own_army = o[Obs14.OWNED_ARMY, 0, 0]
    opp_land = o[Obs14.OPP_LAND, 0, 0]
    opp_army = o[Obs14.OPP_ARMY, 0, 0]
    time = o[Obs14.TIMESTEP, 0, 0]

    visible = max_pool_3x3(xp, owned)

    # --- serving padding synthesis (§2.3) --------------------------------
    # A padded cell you have walked up to is a confirmed mountain; one you have
    # not reached looks like `structures_in_fog`. Not a uniform wall: in training
    # the padding IS real mountains discovered through fog, and a hard wall here
    # would shift the input distribution in the only setting that counts.
    if pad_h or pad_w:
        pad = _pad_mask(xp, pad_h, pad_w)
        pad_mountain_seen = state.mountains_seen | (pad & visible)
        mountains = xp.where(pad & pad_mountain_seen, True, mountains)
        structures_in_fog = xp.where(pad & ~pad_mountain_seen, True, structures_in_fog)
        fog = xp.where(pad, False, fog)
        neutral = xp.where(pad, False, neutral)

    # --- current-frame derived planes ------------------------------------
    own_army_grid = armies * owned
    enemy_army_grid = armies * opponent
    neutral_army_grid = armies * neutral

    own_general = generals & owned
    own_castles = castles & owned
    enemy_castles_vis = castles & opponent

    # --- memory: monotone accumulators (§2.3 pattern 1) -------------------
    new_seen = state.seen | visible
    new_enemy_seen = state.enemy_seen | max_pool_3x3(xp, opponent)
    new_mountains_seen = state.mountains_seen | mountains
    new_castles_seen = state.castles_seen | castles
    new_enemy_general_seen = state.enemy_general_seen | (generals & opponent)
    new_enemy_castles_seen = state.enemy_castles_seen | enemy_castles_vis

    # --- memory: value + staleness clock (pattern 2) ----------------------
    saw_enemy = enemy_army_grid > 0
    new_last_enemy_value = xp.where(saw_enemy, enemy_army_grid, state.last_enemy_army_value)
    new_last_enemy_age = xp.where(saw_enemy, 0.0, state.last_enemy_age + 1.0)

    # --- memory: delta shift registers (pattern 3) ------------------------
    # Deltas, not snapshots: absolute counts are already channels 1 and 2, so a
    # snapshot would be redundant. A delta shows motion directly — a negative
    # blip at the source and a positive one at the destination, same frame.
    army_delta = own_army_grid - state.last_army
    enemy_delta = enemy_army_grid - state.last_enemy_army
    new_army_stack = xp.concatenate([army_delta[None], state.army_stack[:-1]])
    new_enemy_stack = xp.concatenate([enemy_delta[None], state.enemy_stack[:-1]])

    # --- memory: ring buffers (pattern 4) ---------------------------------
    new_opp_army_hist = xp.concatenate([state.opp_army_hist[1:], opp_army[None]])
    new_opp_land_hist = xp.concatenate([state.opp_land_hist[1:], opp_land[None]])

    new_state = PhiState(
        seen=new_seen,
        enemy_seen=new_enemy_seen,
        enemy_general_seen=new_enemy_general_seen,
        mountains_seen=new_mountains_seen,
        castles_seen=new_castles_seen,
        enemy_castles_seen=new_enemy_castles_seen,
        last_enemy_army_value=new_last_enemy_value,
        last_enemy_age=new_last_enemy_age,
        last_army=own_army_grid,
        last_enemy_army=enemy_army_grid,
        army_stack=new_army_stack,
        enemy_stack=new_enemy_stack,
        opp_army_hist=new_opp_army_hist,
        opp_land_hist=new_opp_land_hist,
    )

    # --- build economy ----------------------------------------------------
    own_structures = own_general | own_castles
    cost = build_cost_grid(xp, own_structures)
    build_m = build_mask(xp, armies, owned, generals, castles, cost)
    move_m = move_mask(xp, armies, owned, mountains)

    cost_f = cost.astype(xp.float32)
    buildable_cell = owned & ~generals & ~castles
    surplus = xp.where(
        buildable_cell,
        xp.clip(armies - cost_f, -ARMY_SCALE, ARMY_SCALE) / ARMY_SCALE,
        0.0,
    )

    # --- castle economy, forward-looking (§2.3) ---------------------------
    # Channels 20-22 price a castle. These three say what it is worth, which is
    # the half of the decision the 39-channel tensor only implied — and the
    # 30k run built roughly once per sixty games.
    #
    # What a structure still owes whoever holds it to truncation: +1 per even tick.
    remaining_yield = (
        xp.clip(TRUNCATION - time, 0.0, TRUNCATION) / STRUCTURE_GROWTH_PERIOD
    )

    # Cost to take a cell: arrive with one army more than the occupant. Under
    # fog the occupant is unobservable, so extrapolate what was last seen there
    # at the growth rate that applies — a castle compounds 25x faster than plain
    # land, and nothing else in the tensor says so. Channels 18/19 carry the
    # ingredients, but 19 is log1p(age)/5: recovering age and halving it is a
    # log inversion the network has no reason to find.
    remembered_castle = new_enemy_castles_seen
    remembered_land = new_last_enemy_value > 0
    stale_rate = xp.where(
        remembered_castle, 1.0 / STRUCTURE_GROWTH_PERIOD, 1.0 / LAND_GROWTH_PERIOD
    )
    projected = new_last_enemy_value + new_last_enemy_age * stale_rate

    takeable = ~owned & ~new_mountains_seen
    live_target = visible & takeable
    # Only cells we actually have a memory of: the never-seen ones would read
    # `last_enemy_value = 0` with an age of `t`, i.e. a garrison invented from
    # nothing.
    hidden_target = ~visible & takeable & (remembered_castle | remembered_land)
    occupant = xp.where(live_target, armies, projected)
    capture_cost = xp.where(
        live_target | hidden_target, (occupant + 1.0) / ARMY_SCALE, 0.0
    )

    # Is a castle here worth its price? Deliberately not masked by affordability
    # — channel 22 already carries that, and a cell you cannot pay for yet is
    # exactly where the policy should be marching army.
    payback = xp.where(
        buildable_cell,
        xp.clip(remaining_yield / cost_f, 0.0, PAYBACK_SCALE) / PAYBACK_SCALE,
        0.0,
    )

    # Signed remaining yield of a castle standing here. Live ownership beats
    # memory: `enemy_castles_seen` is monotone, so a castle you captured stays
    # set in it and would otherwise read as the enemy's for the rest of the
    # game. A fogged enemy castle keeps its negative value — it is still earning.
    structure_value = xp.where(
        own_castles,
        remaining_yield / YIELD_SCALE,
        xp.where(
            new_enemy_castles_seen & ~owned, -remaining_yield / YIELD_SCALE, 0.0
        ),
    )

    # --- static coordinates ----------------------------------------------
    rows = xp.broadcast_to(xp.arange(PAD, dtype=xp.float32)[:, None], (PAD, PAD)) / PAD
    cols = xp.broadcast_to(xp.arange(PAD, dtype=xp.float32)[None, :], (PAD, PAD)) / PAD

    def b(flag):
        return xp.asarray(flag).astype(xp.float32)

    channels = [None] * N_CHANNELS
    channels[Ch.ARMIES] = armies / ARMY_SCALE
    channels[Ch.OWN_ARMY] = own_army_grid / ARMY_SCALE
    channels[Ch.ENEMY_ARMY] = enemy_army_grid / ARMY_SCALE
    channels[Ch.NEUTRAL_ARMY] = neutral_army_grid / ARMY_SCALE
    channels[Ch.OWNED] = b(owned)
    channels[Ch.OPPONENT] = b(opponent)
    channels[Ch.NEUTRAL] = b(neutral)
    channels[Ch.FOG] = b(fog)
    channels[Ch.STRUCTURES_IN_FOG] = b(structures_in_fog)
    channels[Ch.OWN_GENERAL] = b(own_general)
    channels[Ch.OWN_CASTLES] = b(own_castles)
    channels[Ch.ENEMY_CASTLES_VISIBLE] = b(enemy_castles_vis)
    channels[Ch.SEEN] = b(new_seen)
    channels[Ch.ENEMY_SEEN] = b(new_enemy_seen)
    channels[Ch.ENEMY_GENERAL_SEEN] = b(new_enemy_general_seen)
    channels[Ch.MOUNTAINS_SEEN] = b(new_mountains_seen)
    channels[Ch.CASTLES_SEEN] = b(new_castles_seen)
    channels[Ch.ENEMY_CASTLES_SEEN] = b(new_enemy_castles_seen)
    channels[Ch.LAST_ENEMY_ARMY] = new_last_enemy_value / ARMY_SCALE
    channels[Ch.LAST_ENEMY_AGE] = xp.log1p(new_last_enemy_age) / AGE_SCALE
    channels[Ch.BUILD_COST] = cost_f / ARMY_SCALE
    channels[Ch.BUILD_SURPLUS] = surplus
    channels[Ch.BUILD_AFFORDABLE] = b(build_m)
    channels[Ch.COORD_X] = cols
    channels[Ch.COORD_Y] = rows
    channels[Ch.CAPTURE_COST] = capture_cost
    channels[Ch.BUILD_PAYBACK] = payback
    channels[Ch.STRUCTURE_VALUE] = structure_value
    for i in range(HISTORY_SIZE):
        channels[int(Ch.ARMY_DELTA_0) + i] = new_army_stack[i] / ARMY_SCALE
        channels[int(Ch.ENEMY_DELTA_0) + i] = new_enemy_stack[i] / ARMY_SCALE

    obs = xp.stack(channels).astype(xp.float32)

    # --- scalars ----------------------------------------------------------
    # Scalars 11-14 replace the reference's ~1M-parameter temporal encoder; they
    # read the ring buffers, which never enter the rollout buffer.
    scalars = [None] * N_SCALARS
    scalars[Sc.TIME] = time / TRUNCATION
    scalars[Sc.LAND_PHASE] = (time % LAND_GROWTH_PERIOD) / LAND_GROWTH_PERIOD
    scalars[Sc.STRUCTURE_PHASE] = time % STRUCTURE_GROWTH_PERIOD
    scalars[Sc.DEATHTOUCH_ACTIVE] = b(time >= DEATHTOUCH_TURN)
    scalars[Sc.DEATHTOUCH_COUNTDOWN] = (
        xp.clip(DEATHTOUCH_TURN - time, 0.0, DEATHTOUCH_TURN) / DEATHTOUCH_TURN
    )
    scalars[Sc.OWN_LAND] = own_land / COUNT_SCALE
    scalars[Sc.OWN_ARMY] = own_army / COUNT_SCALE
    scalars[Sc.OPP_LAND] = opp_land / COUNT_SCALE
    scalars[Sc.OPP_ARMY] = opp_army / COUNT_SCALE
    scalars[Sc.OWN_CASTLES] = xp.sum(b(own_castles)) / CASTLE_COUNT_SCALE
    scalars[Sc.ENEMY_CASTLES_SEEN] = xp.sum(b(new_enemy_castles_seen)) / CASTLE_COUNT_SCALE
    scalars[Sc.OPP_ARMY_D8] = (
        opp_army - _lag(xp, new_opp_army_hist, TEMPORAL_LAG_SHORT)) / ARMY_SCALE
    scalars[Sc.OPP_ARMY_D32] = (
        opp_army - _lag(xp, new_opp_army_hist, TEMPORAL_LAG_LONG)) / ARMY_SCALE
    scalars[Sc.OPP_LAND_D8] = (
        opp_land - _lag(xp, new_opp_land_hist, TEMPORAL_LAG_SHORT)) / ARMY_SCALE
    scalars[Sc.OPP_LAND_D32] = (
        opp_land - _lag(xp, new_opp_land_hist, TEMPORAL_LAG_LONG)) / ARMY_SCALE
    scalars[Sc.BIAS] = xp.asarray(1.0, dtype=xp.float32)

    scalars = xp.stack([xp.asarray(s, dtype=xp.float32) for s in scalars])

    return obs, scalars, move_m, build_m, new_state
