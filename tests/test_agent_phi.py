"""Gate for build step 2 (AGENT_SPEC.md §10).

Accumulator monotonicity, reset-to-zero on dones, delta telescoping, the serving
padding path on an 18x19 board, and numpy/jax parity.

Where possible the oracle is a real game driven through the real engine, not a
synthetic array — the padding and fog rules are exactly what a synthetic fixture
would get wrong.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agent.spec import phi
from agent.spec.constants import (
    ARMY_SCALE,
    HISTORY_SIZE,
    N_CHANNELS,
    N_SCALARS,
    PAD,
    PAYBACK_SCALE,
    TEMPORAL_WINDOW,
    TRUNCATION,
    Ch,
    Sc,
)
from generals.core.env import GeneralsEnv
from generals.core.game import get_observation

XPS = pytest.mark.parametrize("xp", [np, jnp], ids=["numpy", "jax"])

MONOTONE = (
    "seen", "enemy_seen", "enemy_general_seen",
    "mountains_seen", "castles_seen", "enemy_castles_seen",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rollout():
    """One real competition game, 100 turns, seat 0's observations and phi states."""
    # pool_size must be >= the 16 (h, w) combos in [18, 21]; below that the
    # per-combo count floors to 0 and reset() returns an empty pool (delta D6).
    env = GeneralsEnv(mode="competition", pool_size=32)
    pool, state = env.reset(jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(1)
    obs_seq, states = [], [phi.init_phi_state(np)]
    st = states[0]
    for _ in range(100):
        obs_arr = np.asarray(get_observation(state, 0).as_tensor())
        obs_seq.append(obs_arr)
        _, _, _, _, st = phi.augment(np, obs_arr, st)
        states.append(st)

        key, k = jax.random.split(key)
        acts = jax.random.randint(k, (2, 5), 0, 4).at[:, 0].set(0)
        acts = acts.at[:, 1].set(acts[:, 1] % PAD).at[:, 2].set(acts[:, 2] % PAD)
        _, state = env.step(state, acts.astype(jnp.int32), pool)

    return obs_seq, states


def _empty_obs(time=0, own_land=1, own_army=1, opp_land=1, opp_army=1):
    o = np.zeros((14, PAD, PAD), dtype=np.int32)
    o[9] = own_land
    o[10] = own_army
    o[11] = opp_land
    o[12] = opp_army
    o[13] = time
    return o


def _empty_obs_owned(time=0):
    """An owned 5x5 block with the general in its corner, so (7, 7) is plain
    land seven manhattan steps away and pays the base build price."""
    o = _empty_obs(time=time)
    o[5, 3:8, 3:8] = 1
    o[0, 3:8, 3:8] = 40
    o[1, 3, 3] = 1
    return o


# --------------------------------------------------------------------------
# shapes and dtypes
# --------------------------------------------------------------------------


@XPS
def test_shapes_and_dtypes(xp):
    st = phi.init_phi_state(xp)
    obs, scalars, move_m, build_m, st2 = phi.augment(xp, _empty_obs(), st)

    assert obs.shape == (N_CHANNELS, PAD, PAD)
    assert scalars.shape == (N_SCALARS,)
    assert move_m.shape == (PAD, PAD, 4)
    assert build_m.shape == (PAD, PAD)
    assert np.asarray(obs).dtype == np.float32
    assert np.asarray(scalars).dtype == np.float32
    for a, b in zip(st, st2, strict=True):
        assert np.asarray(a).shape == np.asarray(b).shape


@XPS
def test_no_nans_on_a_real_game(xp, rollout):
    obs_seq, _ = rollout
    st = phi.init_phi_state(xp)
    for obs_arr in obs_seq[:30]:
        obs, scalars, _, _, st = phi.augment(xp, xp.asarray(obs_arr), st)
        assert np.isfinite(np.asarray(obs)).all()
        assert np.isfinite(np.asarray(scalars)).all()


# --------------------------------------------------------------------------
# memory semantics
# --------------------------------------------------------------------------


def test_accumulators_are_monotone(rollout):
    """A permanent fact, once learned, is never unlearned — over 100 real turns."""
    _, states = rollout
    for prev, cur in zip(states[:-1], states[1:], strict=True):
        for name in MONOTONE:
            a = np.asarray(getattr(prev, name))
            b = np.asarray(getattr(cur, name))
            assert np.all(b >= a), f"{name} lost a bit"


def test_accumulators_actually_accumulate(rollout):
    """Guard against the test above passing because nothing ever gets set."""
    _, states = rollout
    first, last = states[0], states[-1]
    assert np.asarray(last.seen).sum() > np.asarray(first.seen).sum()
    assert np.asarray(last.mountains_seen).sum() > 0


@XPS
def test_reset_zeroes_only_done_games(xp):
    batch = 4
    st = phi.init_phi_state(xp, batch=batch)
    st = st._replace(
        seen=xp.asarray(np.ones((batch, PAD, PAD), dtype=bool)),
        last_army=xp.asarray(np.full((batch, PAD, PAD), 7.0, dtype=np.float32)),
        opp_army_hist=xp.asarray(np.ones((batch, TEMPORAL_WINDOW), dtype=np.float32)),
    )
    dones = xp.asarray(np.array([True, False, True, False]))
    out = phi.reset_phi_state(xp, st, dones)

    for leaf_in, leaf_out in zip(st, out, strict=True):
        a, b = np.asarray(leaf_in), np.asarray(leaf_out)
        assert not b[0].any() and not b[2].any(), "finished game kept memory"
        assert np.array_equal(a[1], b[1]) and np.array_equal(a[3], b[3]), \
            "live game lost memory"


@XPS
def test_delta_stack_telescopes(xp):
    """sum(army_stack) over a window == cur_army - army_at_window_start.

    This is what proves the stack holds deltas rather than snapshots, and that
    the shift register drops exactly one frame per turn.
    """
    rng = np.random.default_rng(0)
    st = phi.init_phi_state(xp)

    armies = np.zeros((PAD, PAD), dtype=np.int32)
    history = []
    for t in range(HISTORY_SIZE):
        armies = np.maximum(armies + rng.integers(-2, 5, (PAD, PAD)), 0).astype(np.int32)
        o = _empty_obs(time=t)
        o[0] = armies
        o[5] = 1                                  # every cell owned
        history.append(armies.copy())
        _, _, _, _, st = phi.augment(xp, o, st)

    stack = np.asarray(st.army_stack)
    assert stack.shape == (HISTORY_SIZE, PAD, PAD)
    # Deltas are stored raw; the window starts from the all-zero initial state.
    assert np.allclose(stack.sum(axis=0), history[-1], atol=1e-4)
    # Newest first: frame 0 is the most recent transition.
    assert np.allclose(stack[0], history[-1] - history[-2], atol=1e-4)


@XPS
def test_staleness_clock(xp):
    """Age resets to 0 on a sighting and ticks up while the cell is unseen."""
    st = phi.init_phi_state(xp)

    seen = _empty_obs(time=0)
    seen[0, 5, 5] = 30
    seen[6, 5, 5] = 1                             # opponent occupies (5, 5)
    _, _, _, _, st = phi.augment(xp, seen, st)
    assert np.asarray(st.last_enemy_army_value)[5, 5] == 30
    assert np.asarray(st.last_enemy_age)[5, 5] == 0

    for step in range(1, 4):
        obs, _, _, _, st = phi.augment(xp, _empty_obs(time=step), st)
        assert np.asarray(st.last_enemy_army_value)[5, 5] == 30, "value forgotten"
        assert np.asarray(st.last_enemy_age)[5, 5] == step
        expected = np.log1p(step) / 5.0
        assert np.isclose(np.asarray(obs)[Ch.LAST_ENEMY_AGE, 5, 5], expected, atol=1e-5)


@XPS
def test_ring_buffer_lags_feed_scalars(xp):
    """Scalars 11-14 are differences against the ring buffer, newest at -1."""
    st = phi.init_phi_state(xp)
    for t in range(40):
        o = _empty_obs(time=t, opp_army=100 + t, opp_land=10 + t)
        _, scalars, _, _, st = phi.augment(xp, o, st)

    s = np.asarray(scalars)
    # Buffer has been fed 40 values, so a lag of 8 is real; a lag of 32 is too.
    assert np.isclose(s[Sc.OPP_ARMY_D8], 8 / ARMY_SCALE, atol=1e-5)
    assert np.isclose(s[Sc.OPP_ARMY_D32], 32 / ARMY_SCALE, atol=1e-5)
    assert np.isclose(s[Sc.OPP_LAND_D8], 8 / ARMY_SCALE, atol=1e-5)
    assert np.isclose(s[Sc.OPP_LAND_D32], 32 / ARMY_SCALE, atol=1e-5)


@XPS
def test_growth_phase_scalars(xp):
    """Scalars 1 and 2 give the network the phase needed to discount growth
    ticks, which otherwise show up as broad uniform blips in the deltas."""
    st = phi.init_phi_state(xp)
    for t in (0, 1, 49, 50, 799, 800, 1199):
        _, scalars, _, _, _ = phi.augment(xp, _empty_obs(time=t), st)
        s = np.asarray(scalars)
        assert np.isclose(s[Sc.LAND_PHASE], (t % 50) / 50)
        assert np.isclose(s[Sc.STRUCTURE_PHASE], t % 2)
        assert np.isclose(s[Sc.DEATHTOUCH_ACTIVE], float(t >= 800))
        assert np.isclose(s[Sc.DEATHTOUCH_COUNTDOWN], max(800 - t, 0) / 800)
        assert np.isclose(s[Sc.TIME], t / 1200)
        assert s[Sc.BIAS] == 1.0


@XPS
def test_own_and_enemy_generals_are_separate_channels(xp):
    """The reference folds both into one accumulator, mixing a trivially-known
    fact with the most valuable hidden one."""
    st = phi.init_phi_state(xp)
    o = _empty_obs()
    o[1, 2, 2] = 1
    o[5, 2, 2] = 1                                # own general
    o[1, 9, 9] = 1
    o[6, 9, 9] = 1                                # enemy general, spotted
    obs, _, _, _, st = phi.augment(xp, o, st)
    obs = np.asarray(obs)

    assert obs[Ch.OWN_GENERAL, 2, 2] == 1 and obs[Ch.OWN_GENERAL, 9, 9] == 0
    assert obs[Ch.ENEMY_GENERAL_SEEN, 9, 9] == 1 and obs[Ch.ENEMY_GENERAL_SEEN, 2, 2] == 0

    # And it is remembered after the enemy general goes back into fog.
    obs2, _, _, _, _ = phi.augment(xp, _empty_obs(time=1), st)
    assert np.asarray(obs2)[Ch.ENEMY_GENERAL_SEEN, 9, 9] == 1


@XPS
def test_build_channels_agree_with_the_cost_grid(xp):
    st = phi.init_phi_state(xp)
    o = _empty_obs()
    o[5, 3:8, 3:8] = 1                            # owned block
    o[0, 3:8, 3:8] = 40
    o[1, 3, 3] = 1                                # own general in the corner of it
    obs, _, _, build_m, _ = phi.augment(xp, o, st)
    obs, build_m = np.asarray(obs), np.asarray(build_m)

    cost = obs[Ch.BUILD_COST] * ARMY_SCALE
    assert np.isclose(cost[3, 3], 35 + 14), "cell under the general pays the full surcharge"
    assert np.isclose(cost[7, 7], 35), "far cell pays base"
    assert not build_m[3, 3], "the general's own cell is never buildable"
    assert build_m[7, 7]
    assert np.isclose(obs[Ch.BUILD_SURPLUS, 7, 7], (40 - 35) / ARMY_SCALE)
    assert obs[Ch.BUILD_AFFORDABLE, 7, 7] == 1


# --------------------------------------------------------------------------
# castle economy, forward-looking (channels 25-27)
# --------------------------------------------------------------------------


@XPS
def test_capture_cost_is_army_plus_one_where_you_can_see(xp):
    """Arrive with one more than the occupant. Unknown cells stay at zero."""
    st = phi.init_phi_state(xp)
    o = _empty_obs()
    o[5, 2, 2] = 1                                # own general at (2, 2)
    o[1, 2, 2] = 1
    o[0, 2, 2] = 5
    o[6, 2, 3] = 1                                # enemy cell, in vision
    o[0, 2, 3] = 7
    o[4, 1, 1] = 1                                # neutral cell, in vision
    obs = np.asarray(phi.augment(xp, o, st)[0])

    assert np.isclose(obs[Ch.CAPTURE_COST, 2, 3], (7 + 1) / ARMY_SCALE)
    assert np.isclose(obs[Ch.CAPTURE_COST, 1, 1], 1 / ARMY_SCALE), "empty cell costs one"
    assert obs[Ch.CAPTURE_COST, 2, 2] == 0, "your own cell is not a target"
    assert obs[Ch.CAPTURE_COST, 15, 15] == 0, "never seen: no garrison may be invented"


@XPS
def test_capture_cost_compounds_a_fogged_castle_at_the_structure_rate(xp):
    """The point of the channel: a castle in fog is still earning +1 per 2 ticks.

    Plain land compounds 25x slower, and channels 18/19 give the network the
    last-seen value behind a log — never the rule that separates the two.
    """
    seen = _empty_obs()
    seen[5, 2, 2] = 1
    seen[1, 2, 2] = 1
    seen[5, 2, 4] = 1                             # a unit standing beside both cells
    seen[0, 2, 4] = 1
    seen[6, 2, 5] = 1                             # enemy castle, 10 army
    seen[2, 2, 5] = 1
    seen[0, 2, 5] = 10
    seen[6, 4, 4] = 1                             # enemy plain land, 4 army
    seen[0, 4, 4] = 4

    fogged = _empty_obs()                         # the unit withdrew; both are dark
    fogged[5, 2, 2] = 1
    fogged[1, 2, 2] = 1
    fogged[8, 2, 5] = 1                           # structure in fog

    st = phi.init_phi_state(xp)
    _, _, _, _, st = phi.augment(xp, seen, st)

    k = 20
    for _ in range(k):
        obs, _, _, _, st = phi.augment(xp, fogged, st)
    obs = np.asarray(obs)

    assert np.isclose(obs[Ch.CAPTURE_COST, 2, 5], (10 + k / 2 + 1) / ARMY_SCALE), \
        "a fogged castle must compound at +1 per 2 ticks"
    assert np.isclose(obs[Ch.CAPTURE_COST, 4, 4], (4 + k / 50 + 1) / ARMY_SCALE), \
        "fogged plain land grows on the 50-tick land clock, not the castle clock"


@XPS
def test_build_payback_is_remaining_yield_over_price(xp):
    st = phi.init_phi_state(xp)
    o = _empty_obs()
    o[5, 3:8, 3:8] = 1
    o[0, 3:8, 3:8] = 40
    o[1, 3, 3] = 1                                # own general
    o[6, 12, 12] = 1                              # enemy cell
    obs = np.asarray(phi.augment(xp, o, st)[0])

    yield_at_0 = TRUNCATION / 2                   # 600 army if held to truncation
    assert np.isclose(obs[Ch.BUILD_PAYBACK, 7, 7], (yield_at_0 / 35) / PAYBACK_SCALE)
    assert np.isclose(obs[Ch.BUILD_PAYBACK, 4, 4], (yield_at_0 / (35 + 14 - 2 * 2)) / PAYBACK_SCALE), \
        "crowding your own general raises the price and lowers the payback"
    assert obs[Ch.BUILD_PAYBACK, 3, 3] == 0, "the general's cell is not buildable"
    assert obs[Ch.BUILD_PAYBACK, 12, 12] == 0, "you cannot build on enemy land"


@XPS
def test_build_payback_decays_to_nothing_by_truncation(xp):
    """Late enough, a castle cannot repay 35 army before the game ends."""
    st = phi.init_phi_state(xp)
    early = np.asarray(phi.augment(xp, _empty_obs_owned(), st)[0])
    late = np.asarray(phi.augment(xp, _empty_obs_owned(time=TRUNCATION - 40), st)[0])
    end = np.asarray(phi.augment(xp, _empty_obs_owned(time=TRUNCATION), st)[0])

    assert early[Ch.BUILD_PAYBACK, 7, 7] > late[Ch.BUILD_PAYBACK, 7, 7] > 0
    assert np.isclose(late[Ch.BUILD_PAYBACK, 7, 7], (20 / 35) / PAYBACK_SCALE)
    assert end[Ch.BUILD_PAYBACK, 7, 7] == 0


@XPS
def test_structure_value_is_signed_and_live_ownership_beats_memory(xp):
    """`enemy_castles_seen` is monotone, so a captured castle is set in both
    masks — it must read as yours from the turn you take it."""
    st = phi.init_phi_state(xp)
    o = _empty_obs()
    o[5, 2, 2] = 1
    o[1, 2, 2] = 1
    o[5, 2, 3] = 1                                # own castle
    o[2, 2, 3] = 1
    o[0, 2, 3] = 8
    o[6, 3, 3] = 1                                # enemy castle
    o[2, 3, 3] = 1
    o[0, 3, 3] = 12
    obs, _, _, _, st = phi.augment(xp, o, st)
    obs = np.asarray(obs)

    assert np.isclose(obs[Ch.STRUCTURE_VALUE, 2, 3], 1.0), "own castle at t=0 is worth a full yield"
    assert np.isclose(obs[Ch.STRUCTURE_VALUE, 3, 3], -1.0), "the enemy's is worth the same, negated"
    assert obs[Ch.STRUCTURE_VALUE, 2, 2] == 0, "the general is not priced as yield"

    captured = o.copy()
    captured[6, 3, 3] = 0
    captured[5, 3, 3] = 1                         # same cell, now ours
    obs2 = np.asarray(phi.augment(xp, captured, st)[0])
    assert obs2[Ch.STRUCTURE_VALUE, 3, 3] > 0, "a captured castle must flip sign"


@XPS
def test_structure_value_decays_with_the_clock(xp):
    st = phi.init_phi_state(xp)
    o = _empty_obs(time=TRUNCATION // 2)
    o[5, 2, 2] = 1
    o[1, 2, 2] = 1
    o[5, 2, 3] = 1
    o[2, 2, 3] = 1
    obs = np.asarray(phi.augment(xp, o, st)[0])
    assert np.isclose(obs[Ch.STRUCTURE_VALUE, 2, 3], 0.5)


# --------------------------------------------------------------------------
# padding (delta D4)
# --------------------------------------------------------------------------


@XPS
def test_padding_18x19_bottom_right(xp):
    """On an 18x19 board rows 18-20 and columns 19-20 are not real cells.

    Unreached padding must look like `structures_in_fog`; padding you have walked
    up to must look like a confirmed mountain. A uniform wall would shift the
    input distribution in the only setting that counts.
    """
    h, w = 18, 19
    pad_h, pad_w = PAD - h, PAD - w
    st = phi.init_phi_state(xp)

    o = _empty_obs()
    o[5, 0:3, 0:3] = 1                            # a small owned blob far from the edge
    o[0, 0:3, 0:3] = 5
    obs, _, _, _, st = phi.augment(xp, o, st, pad_h=pad_h, pad_w=pad_w)
    obs = np.asarray(obs)

    assert obs[Ch.STRUCTURES_IN_FOG, h:, :].all(), "unreached padded rows not in fog"
    assert obs[Ch.STRUCTURES_IN_FOG, :, w:].all(), "unreached padded cols not in fog"
    assert not obs[Ch.MOUNTAINS_SEEN, h:, :].any(), "padding confirmed without seeing it"
    assert not obs[Ch.FOG, h:, :].any(), "padding marked plain-fog instead of structure-fog"

    # Now own a cell adjacent to the padded rows: vision reaches into them.
    o2 = _empty_obs(time=1)
    o2[5, h - 1, 0] = 1
    o2[0, h - 1, 0] = 5
    obs2, _, _, _, _ = phi.augment(xp, o2, st, pad_h=pad_h, pad_w=pad_w)
    obs2 = np.asarray(obs2)

    assert obs2[Ch.MOUNTAINS_SEEN, h, 0] == 1, "adjacent padding not confirmed as mountain"
    assert obs2[Ch.STRUCTURES_IN_FOG, h, 0] == 0
    assert obs2[Ch.STRUCTURES_IN_FOG, h, 10] == 1, "distant padding stopped being fog"


@XPS
def test_padding_is_inert_when_zero(xp):
    """Training passes pad_h = pad_w = 0 and the block must not execute."""
    st = phi.init_phi_state(xp)
    o = _empty_obs()
    o[5, 0:3, 0:3] = 1
    a, _, _, _, _ = phi.augment(xp, o, st, pad_h=0, pad_w=0)
    b, _, _, _, _ = phi.augment(xp, o, st)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.asarray(a)[Ch.STRUCTURES_IN_FOG].any()


# --------------------------------------------------------------------------
# parity and jax integration
# --------------------------------------------------------------------------


def test_numpy_jax_parity_on_a_real_game(rollout):
    """The same code path under both array libraries, on real observations."""
    obs_seq, _ = rollout
    st_np = phi.init_phi_state(np)
    st_jx = phi.init_phi_state(jnp)

    for t, obs_arr in enumerate(obs_seq[:25]):
        o_np, s_np, mm_np, bm_np, st_np = phi.augment(np, obs_arr, st_np)
        o_jx, s_jx, mm_jx, bm_jx, st_jx = phi.augment(jnp, jnp.asarray(obs_arr), st_jx)

        assert np.allclose(o_np, np.asarray(o_jx), atol=1e-6), f"obs diverged at t={t}"
        assert np.allclose(s_np, np.asarray(s_jx), atol=1e-6), f"scalars diverged at t={t}"
        assert np.array_equal(mm_np, np.asarray(mm_jx))
        assert np.array_equal(bm_np, np.asarray(bm_jx))


def test_augment_is_jittable_and_vmappable(rollout):
    """Training vmaps phi over 2N rows inside a single jitted step."""
    obs_seq, _ = rollout
    batch = 4
    obs_b = jnp.asarray(np.stack([obs_seq[i] for i in range(batch)]))
    st_b = phi.init_phi_state(jnp, batch=batch)

    fn = jax.jit(jax.vmap(lambda o, s: phi.augment(jnp, o, s)))
    obs, scalars, move_m, build_m, st2 = fn(obs_b, st_b)

    assert obs.shape == (batch, N_CHANNELS, PAD, PAD)
    assert scalars.shape == (batch, N_SCALARS)
    assert move_m.shape == (batch, PAD, PAD, 4)
    assert build_m.shape == (batch, PAD, PAD)
    assert st2.army_stack.shape == (batch, HISTORY_SIZE, PAD, PAD)
