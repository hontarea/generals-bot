"""Hunter: garrison the general, conveyor the surplus forward, decapitate.

Winning a game means capturing the enemy general — a timeout is a draw. The
Expander spreads into 1-army piles and never defends its general, so Hunter keeps
its own general home as a growing garrison, sends only the surplus out as a
single advancing stack, and takes the enemy general the moment a pile reaches it.

Interface: `Agent(player_id, H, W)` and `act(obs)` returning
`(pass, row, col, direction, split)`; `main.py` drives the stdio loop.
"""
from collections import deque

# A no-op action — used when no valid move exists or as a safe default.
PASS = (1, 0, 0, 0, 0)

# (dr, dc) offsets for direction codes 0=up, 1=down, 2=left, 3=right
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Cell type codes from the wire protocol (obs.type_grid)
FOG, PLAIN, MOUNTAIN, CASTLE, GENERAL, FOG_STRUCTURE = range(6)

GARRISON = 4  # army kept on the general; only its surplus (above 2x) is sent out


class Agent:
    """Agent that garrisons its general and hunts down the enemy general to win."""

    def __init__(self, player_id, H, W):
        self.player_id = player_id
        self.H = H
        self.W = W

    def _bfs(self, sources, passable):
        """Steps from `sources` to every cell over passable terrain (INF = unreachable)."""
        H, W = self.H, self.W
        INF = H * W + 5
        dist = [[INF] * W for _ in range(H)]
        queue = deque()
        for r, c in sources:
            dist[r][c] = 0
            queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and passable[nr][nc] and dist[nr][nc] > dist[r][c] + 1:
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))
        return dist

    def _toward(self, r, c, field, passable):
        """(direction of the lowest-`field` passable neighbour of (r, c), its value)."""
        best_d, best_v = 0, self.H * self.W + 7
        for d, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and passable[nr][nc] and field[nr][nc] < best_v:
                best_d, best_v = d, field[nr][nc]
        return best_d, best_v

    def act(self, obs):
        """Pick one action: capture the general, feed the surplus out, advance, or wait."""
        H, W = obs.H, obs.W
        types, owner, army = obs.type_grid, obs.owner_grid, obs.army_grid
        reach = H * W  # from_gen < reach  <=>  reachable

        # Mountains and fogged structures are impassable; so are castles we don't
        # own (breaking into a neutral castle wastes the stack — walk around them).
        passable = [
            [types[r][c] not in (MOUNTAIN, FOG_STRUCTURE)
             and not (types[r][c] == CASTLE and owner[r][c] != 1)
             for c in range(W)]
            for r in range(H)
        ]

        cells = [(r, c) for r in range(H) for c in range(W)]
        gen = [(r, c) for r, c in cells if types[r][c] == GENERAL and owner[r][c] == 1]
        if not gen:
            return PASS
        gr, gc = gen[0]
        gen_army = army[gr][gc]
        from_gen = self._bfs(gen, passable)

        # Goal: the enemy general, else nearest enemy land, else the farthest cell to scout.
        egen = [(r, c) for r, c in cells if types[r][c] == GENERAL and owner[r][c] == 2]
        enemy = [(r, c) for r, c in cells if owner[r][c] == 2 and types[r][c] != CASTLE]
        fog = [(r, c) for r, c in cells if types[r][c] == FOG and from_gen[r][c] < reach]
        open_ = [(r, c) for r, c in cells
                 if passable[r][c] and owner[r][c] != 1 and from_gen[r][c] < reach]

        def farthest(mask):
            top = max(from_gen[r][c] for r, c in mask)
            return [(r, c) for r, c in mask if from_gen[r][c] == top]

        goal = egen or enemy or (farthest(fog) if fog else None) or (farthest(open_) if open_ else None)
        if not goal:
            return PASS

        to_goal = self._bfs(goal, passable)

        # Priorities: capture general > feed surplus out (keep garrison) > advance stack > wait.
        if egen:
            egen_army = sum(army[r][c] for r, c in egen)
            kill = [(r, c) for r, c in cells
                    if owner[r][c] == 1 and army[r][c] > 1 and to_goal[r][c] == 1
                    and army[r][c] - 1 > egen_army]
            if kill:
                r, c = max(kill, key=lambda rc: army[rc[0]][rc[1]])
                d, _ = self._toward(r, c, to_goal, passable)
                return (0, r, c, d, 0)

        gd, gv = self._toward(gr, gc, to_goal, passable)
        if gen_army >= 2 * GARRISON and gv < to_goal[gr][gc]:
            return (0, gr, gc, gd, 1)  # split: half leaves, half stays as garrison

        fwd = [(r, c) for r, c in cells
               if owner[r][c] == 1 and army[r][c] > 1 and (r, c) != (gr, gc)
               and self._toward(r, c, to_goal, passable)[1] < to_goal[r][c]]
        if fwd:
            r, c = max(fwd, key=lambda rc: army[rc[0]][rc[1]])
            d, _ = self._toward(r, c, to_goal, passable)
            return (0, r, c, d, 0)

        return PASS
