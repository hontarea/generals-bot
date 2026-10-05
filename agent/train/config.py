"""Training configuration.

Two presets (AGENT_SPEC_DELTAS.md D3): `smoke` fits the 6 GB laptop and is what
every test runs under; `full` is AGENT_SPEC.md §4.1 unchanged and targets a
MetaCentrum L40/A40/A100. Nothing outside this file may hardcode N or T.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """AGENT_SPEC.md §3.2. Serving latency is not the constraint (D2) — shrink
    only to buy training throughput."""

    embed_dim: int = 256
    depth: int = 5
    n_head: int = 8
    ff_factor: int = 3
    use_bf16: bool = True

    # §3.6 value-head fork: 0 -> Linear(d, 1) + MSE; >0 -> HL-Gauss [MARATHON].
    value_loss: str = "mse"
    num_bins: int = 0
    v_min: float = -1.0
    v_max: float = 1.0
    hl_sigma: float = 0.04

    def __post_init__(self) -> None:
        if self.embed_dim % self.n_head:
            raise ValueError(f"embed_dim {self.embed_dim} not divisible by n_head {self.n_head}")
        if self.value_loss not in ("mse", "ce"):
            raise ValueError(f"value_loss must be 'mse' or 'ce', got {self.value_loss!r}")
        if (self.value_loss == "ce") != (self.num_bins > 0):
            raise ValueError("value_loss='ce' requires num_bins > 0, and vice versa")


@dataclass(frozen=True)
class Config:
    run_name: str = "rl_bot"
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)

    # --- rollout (§4.1) -------------------------------------------------
    num_envs: int = 256          # N games; 2N rows once both seats are flattened
    num_steps: int = 256         # T
    num_iters: int = 100_000

    # --- PPO (§5, §6) ---------------------------------------------------
    gamma: float = 1.0           # undiscounted: completed-episode returns are exactly ±1/0
    gae_lambda: float = 0.9
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 1          # >1 would need target_kl, which §6 rules out
    adv_top_frac: float = 0.5
    minibatch_size: int = 2048

    lr: float = 3e-4
    final_lr: float = 3e-5
    ent_coef_start: float = 0.03
    ent_coef_end: float = 0.005
    use_magnet: bool = False     # False -> uniform magnet == plain entropy bonus (D8)

    # --- potential-based shaping (D19) ----------------------------------
    #: Phi_max. 0.0 disables shaping *exactly* — the term multiplies to 0.0 and
    #: every downstream float is bit-identical to a run without it.
    shaping_coef: float = 0.0
    shaping_w_land: float = 1.0
    shaping_w_army: float = 0.2
    shaping_w_castle: float = 2.5

    # --- entropy controller (D19) ---------------------------------------
    #: Target policy entropy in nats. 0.0 keeps the `ent_coef_start -> end` cosine;
    #: anything above it hands `ent_coef` to the controller in `loss.py` instead.
    ent_target: float = 0.0
    ent_kp: float = 0.01
    ent_coef_min: float = 1e-4
    ent_coef_max: float = 0.1

    # --- curriculum (§7) ------------------------------------------------
    pool_size: int = 20_000
    reset_pool_every: int = 25
    gate_every: int = 50
    gate_games: int = 256
    gate_threshold: float = 0.75
    min_iters_per_stage: int = 400
    stepback_patience: int = 200
    stepback_terminal_frac: float = 0.02
    comp_eval_every: int = 250

    # --- loop (§8) ------------------------------------------------------
    ema_decay: float = 0.999
    ckpt_every: int = 500

    @property
    def rows(self) -> int:
        """Rollout rows: both seats of every game, flattened."""
        return 2 * self.num_envs

    @property
    def batch_size(self) -> int:
        return self.num_steps * self.rows

    @property
    def n_keep(self) -> int:
        """Transitions surviving the |advantage| filter, rounded down to whole
        minibatches (§6)."""
        keep = int(self.batch_size * self.adv_top_frac)
        return (keep // self.minibatch_size) * self.minibatch_size

    @property
    def shaping_weights(self):
        """The weights as `rollout.make_step_fn` wants them. Imported locally so
        `config.py` itself stays free of jax."""
        from agent.train.shaping import ShapingWeights

        return ShapingWeights(
            self.shaping_w_land, self.shaping_w_army, self.shaping_w_castle
        )

    def replace(self, **kw) -> Config:
        return dataclasses.replace(self, **kw)


#: Laptop preset: ~0.14 GB of obs instead of 4.5 GB. Every test uses this.
SMOKE = Config(
    run_name="smoke",
    num_envs=32,
    num_steps=64,
    num_iters=100,
    minibatch_size=256,
    pool_size=64,
    reset_pool_every=1_000_000,
    gate_every=1_000_000,
    gate_games=8,
    min_iters_per_stage=1,
    comp_eval_every=1_000_000,
    ckpt_every=10,
)

#: Cluster preset: AGENT_SPEC.md §4.1 verbatim.
FULL = Config(run_name="full")

#: The smallest configuration that still executes every code path once per
#: iteration. For testing the CLI and the job-chaining handoff, not for learning.
TINY = SMOKE.replace(
    run_name="tiny",
    model=ModelConfig(embed_dim=32, depth=1, n_head=4, ff_factor=2, use_bf16=False),
    num_envs=2,
    num_steps=3,
    minibatch_size=2,
    pool_size=32,
    ckpt_every=1,
)

#: The marathon run (`rl_bot_c`): three changes over FULL, chosen because they
#: move three different metrics and so stay separable in one run's logs.
#:
#:   shaping (H2) -> builds_per_1k_steps      the castle economy is unrewarded
#:   HL-Gauss (H1) -> mc_explained_variance   the critic ended rl_bot_b at 0.34
#:   entropy floor (H5) -> entropy, approx_kl both ended below their design band
#:
#: `v_min`/`v_max` are +-(1 + shaping_coef), not +-1: shaping widens the return
#: range and the histogram's support has to cover it (D19). `hl_sigma` is 0.75 of
#: a bin width — 2.4/127 = 0.0189 — rather than the 0.04 that was written for a
#: +-1 range and 128 bins.
MARATHON = FULL.replace(
    run_name="marathon",
    model=ModelConfig(
        value_loss="ce", num_bins=128, v_min=-1.2, v_max=1.2, hl_sigma=0.0142
    ),
    shaping_coef=0.2,
    final_lr=1e-4,
    ent_target=1.2,
)

PRESETS = {"tiny": TINY, "smoke": SMOKE, "full": FULL, "marathon": MARATHON}


def get_config(name: str = "full") -> Config:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; have {sorted(PRESETS)}") from None
