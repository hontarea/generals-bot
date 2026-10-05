"""Random: a uniformly random legal move every turn — the floor any bot must clear.

A move is legal when its source is ours and holds more than one army and its
destination is on the board and not a mountain. Never builds; passes only when
no move is legal. Seeded by player id, so a given seat replays identically.
"""
import random

PASS = (1, 0, 0, 0, 0)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
MOUNTAIN = 2
ME = 1


class Agent:
    def __init__(self, player_id, H, W):
        self.H, self.W = H, W
        self.rng = random.Random(player_id)

    def act(self, obs):
        moves = []
        for r in range(self.H):
            for c in range(self.W):
                if obs.owner_grid[r][c] != ME or obs.army_grid[r][c] <= 1:
                    continue
                for d, (dr, dc) in enumerate(DIRECTIONS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.H and 0 <= nc < self.W and obs.type_grid[nr][nc] != MOUNTAIN:
                        moves.append((r, c, d))
        if not moves:
            return PASS
        r, c, d = self.rng.choice(moves)
        return (0, r, c, d, self.rng.randint(0, 1))
