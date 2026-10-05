"""Every magic number shared between training and serving.

AGENT_SPEC.md §2.1. Channel indices are declared **once, here**; re-declaring them
inside `phi.py` is a documented defect of the reference implementation that
corrupts silently. Nothing in this package may import jax (§1.1).
"""
from __future__ import annotations

from enum import IntEnum

# --- board / action layout (§2.2) ---------------------------------------
PAD = 21
CELLS = PAD * PAD                       # 441
N_MOVE_PLANES = 8                       # 4 dirs all-but-one + 4 dirs half
BUILD_PLANE = 8
N_PLANES = 9                            # 8 move + 1 build
ACTION_DIM = N_PLANES * CELLS + 1       # 3970; the last index is PASS
PASS_IDX = ACTION_DIM - 1               # 3969

#: Engine action kinds, the `pass` field of `[kind, row, col, dir, split]`.
KIND_MOVE = 0
KIND_PASS = 1
KIND_BUILD = 2

#: (dr, dc) per direction, matching `generals.core.action.DIRECTIONS`.
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

MASK_PENALTY = -1e9
"""Not -inf: a fully-masked row would give NaN. Also overflows bf16's useful
range, so logits must be cast to float32 *before* this is added (§2.2)."""

# --- observation tensor (§2.3) ------------------------------------------
HISTORY_SIZE = 7
N_CHANNELS = 42
N_SCALARS = 16
TEMPORAL_WINDOW = 64

#: Ring-buffer lags read by scalars 11-14.
TEMPORAL_LAG_SHORT = 8
TEMPORAL_LAG_LONG = 32

# --- patching / tokens (§3.2) -------------------------------------------
PATCH = 3
GRID_PATCHES = PAD // PATCH             # 7
N_PATCH_TOKENS = GRID_PATCHES ** 2      # 49
N_TOKENS = 2 + N_PATCH_TOKENS           # 51: value, scalars, patches
VALUE_TOKEN = 0
SCALAR_TOKEN = 1
FIRST_PATCH_TOKEN = 2

# --- ruleset (§0.1) -----------------------------------------------------
BUILD_BASE_COST = 35
BUILD_PENALTY = 14
BUILD_DECAY = 2
BUILD_RADIUS = (BUILD_PENALTY - 1) // BUILD_DECAY    # 6

DEATHTOUCH_TURN = 800
TRUNCATION = 1200
LAND_GROWTH_PERIOD = 50                 # +1 on every owned cell when time % 50 == 0
STRUCTURE_GROWTH_PERIOD = 2             # +1 on generals/castles on even ticks

# --- normalization (§2.3) -----------------------------------------------
ARMY_SCALE = 50.0
COUNT_SCALE = 100.0
CASTLE_COUNT_SCALE = 10.0
AGE_SCALE = 5.0                         # channel 19 is log1p(age) / 5

#: Army a structure still owes you if you hold it to truncation: 600 at t=0,
#: since it pays +1 on every even tick. Channel 27 divides by this.
YIELD_SCALE = TRUNCATION / STRUCTURE_GROWTH_PERIOD       # 600.0

#: Channel 26 is remaining yield / build cost. On fresh ground at t=0 that is
#: 600/35 ~ 17, so 20 keeps the useful range spread over [0, 1] and clips only
#: the crowded-cheap corner that no longer changes the decision.
PAYBACK_SCALE = 20.0


class Obs14(IntEnum):
    """Channels of `generals.core.observation.Observation.as_tensor()`.

    Fixed by the engine; do not reorder. Note the tensor is **int32** — phi casts.
    """

    ARMIES = 0
    GENERALS = 1
    CASTLES = 2
    MOUNTAINS = 3
    NEUTRAL = 4
    OWNED = 5
    OPPONENT = 6
    FOG = 7
    STRUCTURES_IN_FOG = 8
    OWNED_LAND = 9
    OWNED_ARMY = 10
    OPP_LAND = 11
    OPP_ARMY = 12
    TIMESTEP = 13


N_OBS14 = len(Obs14)


class Ch(IntEnum):
    """The 39 spatial channels phi emits (§2.3)."""

    # current, directly observed
    ARMIES = 0
    OWN_ARMY = 1
    ENEMY_ARMY = 2
    NEUTRAL_ARMY = 3
    OWNED = 4
    OPPONENT = 5
    NEUTRAL = 6
    FOG = 7
    STRUCTURES_IN_FOG = 8
    OWN_GENERAL = 9
    OWN_CASTLES = 10
    ENEMY_CASTLES_VISIBLE = 11

    # memory
    SEEN = 12
    ENEMY_SEEN = 13
    ENEMY_GENERAL_SEEN = 14          # separate from OWN_GENERAL, deliberately
    MOUNTAINS_SEEN = 15
    CASTLES_SEEN = 16
    ENEMY_CASTLES_SEEN = 17
    LAST_ENEMY_ARMY = 18
    LAST_ENEMY_AGE = 19              # log1p(age) / 5

    # build economy
    BUILD_COST = 20
    BUILD_SURPLUS = 21
    BUILD_AFFORDABLE = 22

    # static
    COORD_X = 23
    COORD_Y = 24

    # castle economy, forward-looking. Channels 20-22 price a castle; these
    # three price what it is worth, which is what the build decision turns on.
    CAPTURE_COST = 25                # army to take this cell, fog-extrapolated
    BUILD_PAYBACK = 26               # remaining yield / build cost, own plain cells
    STRUCTURE_VALUE = 27             # signed remaining yield of a castle here

    # deltas, newest first
    ARMY_DELTA_0 = 28                # .. 34
    ENEMY_DELTA_0 = 35               # .. 41


ARMY_DELTA_SLICE = slice(int(Ch.ARMY_DELTA_0), int(Ch.ARMY_DELTA_0) + HISTORY_SIZE)
ENEMY_DELTA_SLICE = slice(int(Ch.ENEMY_DELTA_0), int(Ch.ENEMY_DELTA_0) + HISTORY_SIZE)


class Sc(IntEnum):
    """The 16 scalars phi emits (§2.3)."""

    TIME = 0                         # timestep / 1200
    LAND_PHASE = 1                   # (timestep % 50) / 50
    STRUCTURE_PHASE = 2              # timestep % 2
    DEATHTOUCH_ACTIVE = 3            # timestep >= 800
    DEATHTOUCH_COUNTDOWN = 4         # clip(800 - t, 0, 800) / 800
    OWN_LAND = 5
    OWN_ARMY = 6
    OPP_LAND = 7
    OPP_ARMY = 8
    OWN_CASTLES = 9
    ENEMY_CASTLES_SEEN = 10
    OPP_ARMY_D8 = 11
    OPP_ARMY_D32 = 12
    OPP_LAND_D8 = 13
    OPP_LAND_D32 = 14
    BIAS = 15                        # constant 1.0; keeps K fixed if one is dropped


assert len(Ch) == N_CHANNELS - 2 * (HISTORY_SIZE - 1), "Ch enum names one slot per delta stack"
assert int(Ch.ENEMY_DELTA_0) + HISTORY_SIZE == N_CHANNELS
assert len(Sc) == N_SCALARS
assert PAD % PATCH == 0, "patch size must divide the padded board"
