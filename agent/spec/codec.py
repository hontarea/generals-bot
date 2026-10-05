"""Action encoding/decoding and legality masks, shared by training and serving.

AGENT_SPEC.md §2.2. Every function takes `xp` first and must work under both
`jax.numpy` (traced, vmapped) and `numpy` (eager) — see §1.1 for the rules. No
Python branching on array values, no in-place mutation, no `.at[]`.

Action layout is plane-major so the policy head's reshape works:

    index = d * 441 + r * 21 + c     for d in [0, 9),  index in [0, 3969)
    index = 3969                     pass

    d 0-3  move all-but-one: UP, DOWN, LEFT, RIGHT
    d 4-7  move half:        same four directions
    d 8    build a castle at (r, c)
"""
from __future__ import annotations

from .constants import (
    BUILD_BASE_COST,
    BUILD_DECAY,
    BUILD_PENALTY,
    BUILD_PLANE,
    BUILD_RADIUS,
    CELLS,
    DIRECTIONS,
    KIND_BUILD,
    KIND_MOVE,
    KIND_PASS,
    MASK_PENALTY,
    N_PLANES,
    PAD,
    PASS_IDX,
)

# --------------------------------------------------------------------------
# action codec
# --------------------------------------------------------------------------


def decode_action(xp, idx):
    """Flat action index -> engine action `[kind, row, col, dir, split]` (5,) int32.

    Branchless. `dir` and `split` are forced to 0 for pass and build, without
    which `encode_action` would not invert this.
    """
    idx = xp.asarray(idx, dtype=xp.int32)

    plane = idx // CELLS                       # 9 for the pass index
    pos = idx % CELLS
    row = pos // PAD
    col = pos % PAD

    is_pass = idx == PASS_IDX
    is_build = plane == BUILD_PLANE

    kind = xp.where(is_pass, KIND_PASS, xp.where(is_build, KIND_BUILD, KIND_MOVE))
    is_move = ~(is_pass | is_build)

    row = xp.where(is_pass, 0, row)
    col = xp.where(is_pass, 0, col)
    direction = xp.where(is_move, plane % 4, 0)
    split = xp.where(is_move, plane // 4, 0)

    return xp.stack([kind, row, col, direction, split]).astype(xp.int32)


def encode_action(xp, action):
    """Engine action `[kind, row, col, dir, split]` -> flat index. Inverse of
    `decode_action` for every index in [0, 3970)."""
    action = xp.asarray(action, dtype=xp.int32)
    kind, row, col, direction, split = (action[0], action[1], action[2], action[3], action[4])

    is_pass = kind == KIND_PASS
    is_build = kind == KIND_BUILD

    plane = xp.where(is_build, BUILD_PLANE, split * 4 + direction)
    idx = plane * CELLS + row * PAD + col
    return xp.where(is_pass, PASS_IDX, idx).astype(xp.int32)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------


def shift(xp, grid, dr, dc, fill):
    """`grid` translated so that out[r, c] == grid[r + dr, c + dc], padded with `fill`.

    Written as pad-then-slice because `xp.roll` wraps at the borders, which is
    exactly the bug the reference's magnet has.
    """
    h, w = grid.shape
    # Reading ahead (positive offset) needs room at the far edge; reading back
    # needs it at the near edge.
    padded = xp.pad(
        grid,
        ((max(-dr, 0), max(dr, 0)), (max(-dc, 0), max(dc, 0))),
        mode="constant",
        constant_values=fill,
    )
    r0, c0 = max(dr, 0), max(dc, 0)
    return padded[r0:r0 + h, c0:c0 + w]


def move_mask(xp, armies, owned, mountains):
    """(H, W, 4) bool: is moving from (r, c) in direction d legal?

    Mirrors `generals.core.action.compute_valid_move_mask`: the source must be
    owned with >1 army, and the destination in bounds and passable.
    """
    can_move_from = owned & (armies > 1)
    passable = ~mountains

    # Destination passability, per direction. Out-of-bounds fills with False,
    # which folds the bounds check into the same term.
    dest_ok = xp.stack(
        [shift(xp, passable, dr, dc, False) for dr, dc in DIRECTIONS], axis=-1
    )
    return can_move_from[:, :, None] & dest_ok


def build_cost_grid(xp, own_structures):
    """(H, W) int32 castle price, matching `build_castles.build_cost_grid`.

    `cost(cell) = 35 + sum over your structures s of max(0, 14 - 2*manhattan(cell, s))`,
    where your structures are your general plus every castle you own. Implemented
    as a fixed unrolled loop of shifted adds over the diamond |di| + |dj| <= 6;
    beyond that the penalty is 0.
    """
    struct = xp.asarray(own_structures).astype(xp.int32)
    cost = xp.zeros_like(struct) + BUILD_BASE_COST

    for di in range(-BUILD_RADIUS, BUILD_RADIUS + 1):
        for dj in range(-BUILD_RADIUS, BUILD_RADIUS + 1):
            penalty = BUILD_PENALTY - BUILD_DECAY * (abs(di) + abs(dj))
            if penalty <= 0:
                continue
            # A structure at (r+di, c+dj) surcharges cell (r, c).
            cost = cost + penalty * shift(xp, struct, di, dj, 0)

    return cost.astype(xp.int32)


def build_mask(xp, armies, owned, generals, castles, cost):
    """(H, W) bool: may a castle be built here?

    `build_castles._apply_one`: owned, plain (no general, no castle), and holding
    at least `cost` army.
    """
    return owned & ~generals & ~castles & (armies >= cost)


def mask_penalty(xp, move_m, build_m, pad_h=0, pad_w=0):
    """(3970,) float32: 0 where legal, -1e9 where not.

    Full and half move planes share `move_m` — splitting never changes legality.
    Pass is always legal. `pad_h`/`pad_w` mask the bottom/right strip that exists
    only at serving, where the real board is smaller than 21x21 (delta D4).
    """
    h, w = move_m.shape[0], move_m.shape[1]

    if pad_h or pad_w:
        rows = xp.arange(h)[:, None]
        cols = xp.arange(w)[None, :]
        in_board = (rows < h - pad_h) & (cols < w - pad_w)
        move_m = move_m & in_board[:, :, None]
        build_m = build_m & in_board

    # (4, H, W) -> full moves, half moves, then the build plane.
    moves = xp.transpose(move_m, (2, 0, 1))
    planes = xp.concatenate([moves, moves, build_m[None]], axis=0)

    legal = xp.concatenate([planes.reshape(-1), xp.ones((1,), dtype=bool)])
    return xp.where(legal, xp.float32(0.0), xp.float32(MASK_PENALTY)).astype(xp.float32)


def masks_from_obs(xp, armies, owned, mountains, generals, castles, own_general):
    """Convenience: every mask a turn needs, from observation planes.

    `own_general` is the own-general plane alone; own structures for pricing are
    that plus the castles you own.
    """
    own_structures = own_general | (castles & owned)
    cost = build_cost_grid(xp, own_structures)
    return (
        move_mask(xp, armies, owned, mountains),
        build_mask(xp, armies, owned, generals, castles, cost),
        cost,
    )


assert N_PLANES * CELLS + 1 == PASS_IDX + 1
