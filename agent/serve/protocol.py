"""stdio protocol: parse the engine's frames, reconstruct the 14-channel tensor.

AGENT_SPEC.md §9.1, mirroring `competition/protocol.py` (which is the engine side
of the same wire format).

    handshake, once:   `player_id H W`
    per turn:          `turn my_land my_army opp_land opp_army`
                       H lines of W ints   -- type
                       H lines of W ints   -- owner
                       H lines of W ints   -- army
    reply:             `kind row col dir split`, one line, flushed
    EOF on stdin:      game over, exit cleanly

Owner codes are perspective-relative, so no remapping is needed. The grids arrive
at the true (H, W) from the handshake and are placed into a 21x21 array with the
padding at the bottom and right, matching how the engine pads in training
(delta D4).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from agent.spec.constants import PAD, Obs14

# Type codes, from `competition/protocol.py`.
TYPE_FOG = 0
TYPE_PLAIN = 1
TYPE_MOUNTAIN = 2
TYPE_CASTLE = 3
TYPE_GENERAL = 4
TYPE_STRUCTURE_IN_FOG = 5

OWNER_NEUTRAL = 0
OWNER_ME = 1
OWNER_OPP = 2


class Handshake(NamedTuple):
    player_id: int
    height: int
    width: int

    @property
    def pad_h(self) -> int:
        return PAD - self.height

    @property
    def pad_w(self) -> int:
        return PAD - self.width


class Frame(NamedTuple):
    turn: int
    my_land: int
    my_army: int
    opp_land: int
    opp_army: int
    type_grid: np.ndarray     # (H, W) int32
    owner_grid: np.ndarray
    army_grid: np.ndarray


def parse_handshake(line: str) -> Handshake:
    player_id, height, width = (int(x) for x in line.split())
    if not (1 <= height <= PAD and 1 <= width <= PAD):
        raise ValueError(f"board {height}x{width} does not fit in {PAD}x{PAD}")
    return Handshake(player_id, height, width)


def read_frame(stream, hs: Handshake) -> Frame | None:
    """Read one turn. Returns None on EOF, which means the game is over."""
    header = stream.readline()
    if not header:
        return None

    turn, my_land, my_army, opp_land, opp_army = (int(x) for x in header.split())

    def grid():
        rows = []
        for _ in range(hs.height):
            line = stream.readline()
            if not line:
                raise EOFError("stream ended mid-frame")
            rows.append([int(x) for x in line.split()])
        return np.array(rows, dtype=np.int32)

    return Frame(turn, my_land, my_army, opp_land, opp_army, grid(), grid(), grid())


def frame_to_tensor(frame: Frame, hs: Handshake) -> np.ndarray:
    """(14, 21, 21) int32, laid out exactly like `Observation.as_tensor()`.

    The engine derives each plane from game state; here they are recovered from
    the three grids. `fog_cells` and `structures_in_fog` are disjoint by
    construction on the engine side (`invisible & ~(mountains | castles)` versus
    `invisible & (mountains | castles)`), and the type codes preserve that.
    """
    h, w = hs.height, hs.width
    out = np.zeros((len(Obs14), PAD, PAD), dtype=np.int32)

    t, owner, army = frame.type_grid, frame.owner_grid, frame.army_grid

    mine = owner == OWNER_ME
    theirs = owner == OWNER_OPP

    visible = t != TYPE_FOG
    visible &= t != TYPE_STRUCTURE_IN_FOG

    # The engine masks every state-derived plane by visibility, so a fogged cell
    # reports zeros rather than stale truth.
    out[Obs14.ARMIES, :h, :w] = army
    out[Obs14.GENERALS, :h, :w] = t == TYPE_GENERAL
    # Generals and castles are disjoint sets in the engine (`build_castles`
    # refuses to build on either), and TYPE_GENERAL only overwrites TYPE_CASTLE
    # in the wire encoding — so a general cell is never a castle.
    out[Obs14.CASTLES, :h, :w] = t == TYPE_CASTLE
    out[Obs14.MOUNTAINS, :h, :w] = t == TYPE_MOUNTAIN
    out[Obs14.NEUTRAL, :h, :w] = visible & (owner == OWNER_NEUTRAL) & (t != TYPE_MOUNTAIN)
    out[Obs14.OWNED, :h, :w] = mine
    out[Obs14.OPPONENT, :h, :w] = theirs
    out[Obs14.FOG, :h, :w] = t == TYPE_FOG
    out[Obs14.STRUCTURES_IN_FOG, :h, :w] = t == TYPE_STRUCTURE_IN_FOG

    out[Obs14.OWNED_LAND] = frame.my_land
    out[Obs14.OWNED_ARMY] = frame.my_army
    out[Obs14.OPP_LAND] = frame.opp_land
    out[Obs14.OPP_ARMY] = frame.opp_army
    out[Obs14.TIMESTEP] = frame.turn

    return out


def format_action(action) -> str:
    """`kind row col dir split`. The caller must flush, or both sides deadlock."""
    kind, row, col, direction, split = (int(x) for x in action)
    return f"{kind} {row} {col} {direction} {split}"


PASS_LINE = "1 0 0 0 0"
