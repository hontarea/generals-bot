# Cross-play results

All games use the harness in `evaluation/`, on real competition boards (18–21 cells per
side, fog of war, castle building, deathtouch from turn 800). Each match is 40 games:
20 map seeds, each played from both seats. Score: win 1, draw 0.5, loss 0, for the first
bot. Interval: 95% Wilson. Measured 2026-10-05 on a laptop, 3 games in parallel.

`rl_bot` is the final submission (Run 3, `rl_bot_c`, 83k iterations).

| Match | Score [95% CI] | W / D / L | Avg length |
| --- | --- | --- | --- |
| `rl_bot_a` vs `hunter_bot` | 0.988 [0.891, 0.999] | 39 / 1 / 0 | 332 turns |
| `rl_bot_b` vs `hunter_bot` | 1.000 [0.912, 1.000] | 40 / 0 / 0 | 422 turns |
| `rl_bot` vs `hunter_bot` | 1.000 [0.912, 1.000] | 40 / 0 / 0 | 226 turns |
| `rl_bot_b` vs `rl_bot_a` | 0.900 [0.769, 0.960] | 36 / 0 / 4 | 393 turns |
| `rl_bot` vs `rl_bot_a` | 0.975 [0.871, 0.996] | 39 / 0 / 1 | 268 turns |
| `rl_bot` vs `rl_bot_b` | 0.875 [0.739, 0.945] | 35 / 0 / 5 | 317 turns |
| `rl_bot` vs `random_bot` | 1.000 [0.912, 1.000] | 40 / 0 / 0 | 212 turns |
| `hunter_bot` vs `random_bot` | 1.000 [0.912, 1.000] | 40 / 0 / 0 | 241 turns |

Reproduce one row:

```bash
python -m evaluation.evaluate bots/rl_bot bots/hunter_bot --suite smoke --workers 3
```

## Full reports

```
rl_bot_a  vs  hunter_bot        suite=smoke  40 games  2026-10-05 16:34
score(rl_bot_a) = 0.988  [0.891, 0.999]   W 39  D 1  L 0
end reasons: capture 38 | deathtouch 1 | truncation 1 | crash 0 | fault_forfeit 0
as p0: 1.000        as p1: 0.975        (seat gap 0.025)
avg length 332 turns   | rl_bot_a castles/game 0.1  | hunter_bot castles/game 0.0
timing rl_bot_a: p50 10.4ms p99* 32.0ms max 78.5ms | faults 0
timing hunter_bot: p50 0.7ms p99* 4.1ms max 25.4ms | faults 0
seeds lost from both seats: (none)
```

```
rl_bot_b  vs  hunter_bot        suite=smoke  40 games  2026-10-05 16:35
score(rl_bot_b) = 1.000  [0.912, 1.000]   W 40  D 0  L 0
end reasons: capture 38 | deathtouch 2 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 1.000        as p1: 1.000        (seat gap 0.000)
avg length 422 turns   | rl_bot_b castles/game 0.2  | hunter_bot castles/game 0.0
timing rl_bot_b: p50 10.1ms p99* 31.7ms max 70.3ms | faults 0
timing hunter_bot: p50 0.7ms p99* 4.1ms max 24.8ms | faults 0
seeds lost from both seats: (none)
```

```
rl_bot  vs  hunter_bot        suite=smoke  40 games  2026-10-05 16:37
score(rl_bot) = 1.000  [0.912, 1.000]   W 40  D 0  L 0
end reasons: capture 40 | deathtouch 0 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 1.000        as p1: 1.000        (seat gap 0.000)
avg length 226 turns   | rl_bot castles/game 0.1  | hunter_bot castles/game 0.0
timing rl_bot: p50 10.1ms p99* 34.3ms max 71.9ms | faults 0
timing hunter_bot: p50 0.7ms p99* 6.6ms max 28.4ms | faults 0
seeds lost from both seats: (none)
```

```
rl_bot_b  vs  rl_bot_a        suite=smoke  40 games  2026-10-05 16:39
score(rl_bot_b) = 0.900  [0.769, 0.960]   W 36  D 0  L 4
end reasons: capture 40 | deathtouch 0 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 0.950        as p1: 0.850        (seat gap 0.100)
avg length 393 turns   | rl_bot_b castles/game 0.4  | rl_bot_a castles/game 0.0
timing rl_bot_b: p50 10.3ms p99* 23.7ms max 84.3ms | faults 0
timing rl_bot_a: p50 10.6ms p99* 26.4ms max 75.5ms | faults 0
seeds lost from both seats: (none)
```

```
rl_bot  vs  rl_bot_a        suite=smoke  40 games  2026-10-05 16:41
score(rl_bot) = 0.975  [0.871, 0.996]   W 39  D 0  L 1
end reasons: capture 40 | deathtouch 0 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 1.000        as p1: 0.950        (seat gap 0.050)
avg length 268 turns   | rl_bot castles/game 0.3  | rl_bot_a castles/game 0.0
timing rl_bot: p50 10.3ms p99* 32.7ms max 85.5ms | faults 0
timing rl_bot_a: p50 10.6ms p99* 36.4ms max 76.2ms | faults 0
seeds lost from both seats: (none)
```

```
rl_bot  vs  rl_bot_b        suite=smoke  40 games  2026-10-05 16:43
score(rl_bot) = 0.875  [0.739, 0.945]   W 35  D 0  L 5
end reasons: capture 40 | deathtouch 0 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 0.900        as p1: 0.850        (seat gap 0.050)
avg length 317 turns   | rl_bot castles/game 0.2  | rl_bot_b castles/game 0.0
timing rl_bot: p50 10.4ms p99* 33.3ms max 78.3ms | faults 0
timing rl_bot_b: p50 10.5ms p99* 35.5ms max 74.2ms | faults 0
seeds lost from both seats: (none)
```

```
rl_bot  vs  random_bot        suite=smoke  40 games  2026-10-05 12:59
score(rl_bot) = 1.000  [0.912, 1.000]   W 40  D 0  L 0
end reasons: capture 40 | deathtouch 0 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 1.000        as p1: 1.000        (seat gap 0.000)
avg length 212 turns   | rl_bot castles/game 0.1  | random_bot castles/game 0.0
timing rl_bot: p50 10.3ms p99* 45.5ms max 73.3ms | faults 0
timing random_bot: p50 0.3ms p99* 4.0ms max 30.9ms | faults 0
seeds lost from both seats: (none)
```

```
hunter_bot  vs  random_bot        suite=smoke  40 games  2026-10-05 11:40
score(hunter_bot) = 1.000  [0.912, 1.000]   W 40  D 0  L 0
end reasons: capture 40 | deathtouch 0 | truncation 0 | crash 0 | fault_forfeit 0
as p0: 1.000        as p1: 1.000        (seat gap 0.000)
avg length 241 turns   | hunter_bot castles/game 0.0  | random_bot castles/game 0.0
timing hunter_bot: p50 0.9ms p99* 8.4ms max 44.2ms | faults 0
timing random_bot: p50 0.3ms p99* 7.2ms max 40.0ms | faults 0
seeds lost from both seats: (none)
```

