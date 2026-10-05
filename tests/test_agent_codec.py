"""Gate for build step 1 (AGENT_SPEC.md §10).

Round-trip every action index, prove behavioral injectivity, and check both masks
cell-for-cell against the engine functions they mirror (delta D5). The engine is
the oracle; `agent/spec/` may not import it.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from agent.spec import codec
from agent.spec.constants import (
    ACTION_DIM,
    BUILD_PLANE,
    CELLS,
    KIND_BUILD,
    KIND_MOVE,
    KIND_PASS,
    MASK_PENALTY,
    PAD,
    PASS_IDX,
)
from generals.core.action import compute_valid_move_mask
from generals.modifiers import build_castles as bc

XPS = pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])


def _decode_all(xp):
    return np.stack([np.asarray(codec.decode_action(xp, i)) for i in range(ACTION_DIM)])


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------


@XPS
def test_round_trip_all_indices(xp):
    """encode(decode(i)) == i for all 3970 indices."""
    for idx in range(ACTION_DIM):
        action = codec.decode_action(xp, idx)
        assert int(codec.encode_action(xp, action)) == idx, f"round trip failed at {idx}"


@XPS
def test_behavioral_injectivity(xp):
    """No two indices decode to the same 5-tuple.

    This is what actually matters: `dir`/`split` must be zeroed for pass and build,
    or several indices collapse onto one engine action and the PPO ratio silently
    prices the wrong thing.
    """
    decoded = _decode_all(xp)
    unique = {tuple(int(v) for v in row) for row in decoded}
    assert len(unique) == ACTION_DIM


@XPS
def test_layout_matches_spec(xp):
    decoded = _decode_all(xp)

    # planes 0-7 are moves: d 0-3 full, d 4-7 half
    for plane in range(8):
        for r, c in ((0, 0), (5, 13), (PAD - 1, PAD - 1)):
            idx = plane * CELLS + r * PAD + c
            kind, row, col, direction, split = decoded[idx]
            assert (kind, row, col) == (KIND_MOVE, r, c)
            assert direction == plane % 4
            assert split == plane // 4

    # plane 8 is build, with dir and split pinned to 0
    for r, c in ((0, 0), (7, 2), (PAD - 1, PAD - 1)):
        idx = BUILD_PLANE * CELLS + r * PAD + c
        assert tuple(decoded[idx]) == (KIND_BUILD, r, c, 0, 0)

    assert tuple(decoded[PASS_IDX]) == (KIND_PASS, 0, 0, 0, 0)


@XPS
def test_encode_ignores_dir_split_for_pass_and_build(xp):
    """The engine hands back arbitrary dir/split on pass/build actions; encoding
    must not depend on them or evaluation-mode log-probs diverge from sampling."""
    for direction in range(4):
        for split in range(2):
            assert int(codec.encode_action(xp, [KIND_PASS, 3, 4, direction, split])) == PASS_IDX
            got = int(codec.encode_action(xp, [KIND_BUILD, 3, 4, direction, split]))
            assert got == BUILD_PLANE * CELLS + 3 * PAD + 4


def test_numpy_and_jax_decode_identically():
    assert np.array_equal(_decode_all(np), _decode_all(jnp))


# --------------------------------------------------------------------------
# masks vs the engine oracles
# --------------------------------------------------------------------------


def _random_board(rng, h=PAD, w=PAD):
    mountains = rng.random((h, w)) < 0.25
    owned = (rng.random((h, w)) < 0.30) & ~mountains
    opponent = (rng.random((h, w)) < 0.20) & ~mountains & ~owned
    armies = rng.integers(0, 40, size=(h, w)).astype(np.int32) * (owned | opponent)
    castles = (rng.random((h, w)) < 0.05) & (owned | opponent)
    generals = np.zeros((h, w), dtype=bool)
    own_gen = np.zeros((h, w), dtype=bool)
    owned_idx = np.argwhere(owned)
    if len(owned_idx):
        r, c = owned_idx[rng.integers(len(owned_idx))]
        generals[r, c] = own_gen[r, c] = True
        castles[r, c] = False
    return armies, owned, opponent, mountains, castles, generals, own_gen


@XPS
def test_move_mask_equals_engine(xp):
    """Cell-for-cell against `generals.core.action.compute_valid_move_mask`."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        armies, owned, _, mountains, *_ = _random_board(rng)
        ours = np.asarray(codec.move_mask(xp, xp.asarray(armies), xp.asarray(owned),
                                          xp.asarray(mountains)))
        theirs = np.asarray(compute_valid_move_mask(jnp.asarray(armies), jnp.asarray(owned),
                                                    jnp.asarray(mountains)))
        assert np.array_equal(ours, theirs)


@XPS
def test_move_mask_no_wraparound(xp):
    """A border cell must not be able to move off the board.

    `jnp.roll` would silently make the top row's UP move legal, wrapping to the
    bottom. `_shift` pads instead; this pins that.
    """
    armies = np.full((PAD, PAD), 10, dtype=np.int32)
    owned = np.ones((PAD, PAD), dtype=bool)
    mountains = np.zeros((PAD, PAD), dtype=bool)
    m = np.asarray(codec.move_mask(xp, xp.asarray(armies), xp.asarray(owned),
                                   xp.asarray(mountains)))
    assert not m[0, :, 0].any(), "top row can move UP"
    assert not m[-1, :, 1].any(), "bottom row can move DOWN"
    assert not m[:, 0, 2].any(), "left column can move LEFT"
    assert not m[:, -1, 3].any(), "right column can move RIGHT"


@XPS
def test_build_cost_grid_equals_engine(xp):
    """Cell-for-cell against `build_castles.build_cost_grid`, which prices from
    live game state rather than from observation planes."""
    from generals.core.game import create_initial_state

    rng = np.random.default_rng(7)
    for trial in range(10):
        grid = np.zeros((PAD, PAD), dtype=np.int32)
        mtn = rng.random((PAD, PAD)) < 0.2
        grid[mtn] = -2
        free = np.argwhere(grid == 0)
        picks = free[rng.choice(len(free), size=2, replace=False)]
        grid[tuple(picks[0])] = 1
        grid[tuple(picks[1])] = 2

        state = create_initial_state(jnp.asarray(grid))
        # Give player 0 a few castles so the surcharge terms actually overlap.
        castles = np.asarray(state.castles).copy()
        ownership = np.asarray(state.ownership).copy()
        plain = np.argwhere((grid == 0))
        for r, c in plain[rng.choice(len(plain), size=3 + trial % 3, replace=False)]:
            castles[r, c] = True
            ownership[0, r, c] = True
            ownership[1, r, c] = False
        state = state._replace(castles=jnp.asarray(castles), ownership=jnp.asarray(ownership))

        theirs = np.asarray(bc.build_cost_grid(state, 0))

        own_structures = (np.asarray(state.generals) & ownership[0]) | (castles & ownership[0])
        ours = np.asarray(codec.build_cost_grid(xp, xp.asarray(own_structures)))
        assert np.array_equal(ours, theirs), f"trial {trial}"


@XPS
def test_build_mask_rules(xp):
    """owned & plain & affordable — and nothing else."""
    armies = np.zeros((PAD, PAD), dtype=np.int32)
    owned = np.zeros((PAD, PAD), dtype=bool)
    generals = np.zeros((PAD, PAD), dtype=bool)
    castles = np.zeros((PAD, PAD), dtype=bool)
    cost = np.full((PAD, PAD), 35, dtype=np.int32)

    owned[1:5, 1:5] = True
    armies[1:5, 1:5] = 40
    armies[1, 1] = 34            # one short
    generals[2, 2] = True        # own general: never buildable
    castles[3, 3] = True         # already a castle

    m = np.asarray(codec.build_mask(xp, *(xp.asarray(a) for a in
                                          (armies, owned, generals, castles, cost))))
    assert not m[1, 1] and not m[2, 2] and not m[3, 3]
    assert m[1, 2] and m[4, 4]
    assert not m[0, 0], "unowned cell is not buildable"
    assert m.sum() == 16 - 3


@XPS
def test_mask_penalty_shape_and_values(xp):
    rng = np.random.default_rng(3)
    armies, owned, _, mountains, castles, generals, own_gen = _random_board(rng)
    mm, bm, _ = codec.masks_from_obs(
        xp, *(xp.asarray(a) for a in (armies, owned, mountains, generals, castles, own_gen))
    )
    pen = np.asarray(codec.mask_penalty(xp, mm, bm))

    assert pen.shape == (ACTION_DIM,)
    assert pen.dtype == np.float32
    assert pen[PASS_IDX] == 0.0, "pass must always be legal"
    assert set(np.unique(pen)) <= {0.0, np.float32(MASK_PENALTY)}

    mm_np, bm_np = np.asarray(mm), np.asarray(bm)
    legal = pen == 0.0
    # Full and half planes share the move mask.
    for plane in range(8):
        block = legal[plane * CELLS:(plane + 1) * CELLS].reshape(PAD, PAD)
        assert np.array_equal(block, mm_np[:, :, plane % 4])
    build_block = legal[BUILD_PLANE * CELLS:(BUILD_PLANE + 1) * CELLS].reshape(PAD, PAD)
    assert np.array_equal(build_block, bm_np)


@XPS
def test_mask_penalty_masks_serving_padding(xp):
    """On an 18x19 board the bottom 3 rows and right 2 columns are not real cells.

    Padding is bottom/right only (delta D4).
    """
    h, w = 18, 19
    pad_h, pad_w = PAD - h, PAD - w
    armies = np.full((PAD, PAD), 10, dtype=np.int32)
    owned = np.ones((PAD, PAD), dtype=bool)
    mountains = np.zeros((PAD, PAD), dtype=bool)
    mm = codec.move_mask(xp, xp.asarray(armies), xp.asarray(owned), xp.asarray(mountains))
    bm = xp.asarray(np.ones((PAD, PAD), dtype=bool))

    pen = np.asarray(codec.mask_penalty(xp, mm, bm, pad_h=pad_h, pad_w=pad_w))
    legal = (pen == 0.0)[:-1].reshape(9, PAD, PAD)

    assert not legal[:, h:, :].any(), "padded rows are selectable"
    assert not legal[:, :, w:].any(), "padded columns are selectable"
    assert legal[8, :h, :w].all(), "real cells were masked out"
