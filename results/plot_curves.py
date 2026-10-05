"""Render the README's training figures from `training_curves.csv`.

The CSV is a 500-iteration-binned extract of the three training runs' logs.
Run from the repo root:  python results/plot_curves.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"

# The palette and type of the generals.bot replay viewer, so figures and GIF match.
BG, INK, INK2, GRID = "#ffffff", "#121214", "#6b6b73", "#e5e5e5"
ACCENT = "#4864db"                  # Run 3, the final bot (the viewer's blue)
RED = "#db4848"                     # the viewer's red
GREEN = "#2f9e44"
RUN_COLOR = {"run1": RED, "run2": ACCENT, "run3": GREEN}   # entropy figure: a red, b blue, c green
matplotlib.rcParams.update({"font.family": "DejaVu Sans Mono", "figure.facecolor": BG,
                            "axes.facecolor": BG, "savefig.facecolor": BG})
LABEL = {"run1": "rl_bot_a", "run2": "rl_bot_b", "run3": "rl_bot_c (final)"}


def load():
    runs = {}
    for r in csv.DictReader(open(HERE / "training_curves.csv")):
        runs.setdefault(r["run"], []).append(r)
    return runs


def col(rows, name):
    return [float(r[name]) if r[name] else float("nan") for r in rows]


def style(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def kfmt(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))


def entropy(runs):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for run in ("run1", "run2", "run3"):
        rows = runs[run]
        x, y = col(rows, "iteration"), col(rows, "entropy")
        final = run == "run3"
        ax.plot(x, y, color=RUN_COLOR[run], linewidth=2.2 if final else 1.6)
        ax.annotate(f"{LABEL[run]}: {sum(y[-10:]) / 10:.2f}", (x[-1], y[-1]), xytext=(6, 10 if final else -4),
                    textcoords="offset points", fontsize=9.5, color=RUN_COLOR[run],
                    fontweight="bold" if final else "normal")
    ax.axhline(1.2, color=INK2, linestyle=(0, (4, 4)), linewidth=0.9)
    ax.text(113_000, 1.12, "target 1.2", ha="right", va="top", fontsize=9, color=INK2)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 115_000)
    kfmt(ax)
    ax.set_xlabel("training iteration", color=INK2, fontsize=9.5)
    ax.set_ylabel("policy entropy (nats)", color=INK2, fontsize=9.5)
    style(ax, "A fixed schedule let exploration collapse; a controller held it",
          "Policy entropy per 500 iterations. a, b: scheduled bonus. c: feedback controller.")
    fig.tight_layout()
    fig.savefig(OUT / "entropy.png")


def castles(runs):
    """Build rate vs self-play game length, competition-distance iterations only (stage 4)."""
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    rs = {}
    for run in ("run1", "run2", "run3"):
        rows = [r for r in runs[run] if r["curriculum_stage"] == "4" and r["mean_game_length"]]
        x, y = col(rows, "mean_game_length"), col(rows, "castle_builds_per_1k_steps")
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        rs[run] = sum((a - mx) * (b - my) for a, b in zip(x, y)) / (
            sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
        ax.scatter(x, y, s=20, color=RUN_COLOR[run], alpha=0.7, edgecolors="white", linewidths=0.5,
                   label=f"{LABEL[run]}  r = {rs[run]:.2f}  (n = {n})")
    ax.legend(loc="upper center", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlabel("mean self-play game length (turns)", color=INK2, fontsize=9.5)
    ax.set_ylabel("castle builds per 1,000 steps", color=INK2, fontsize=9.5)
    ax.set_ylim(bottom=0)
    style(ax, "Castle building vs self-play game length",
          "One point per 500 iterations at full competition distance (curriculum stage 4).")
    fig.tight_layout()
    fig.savefig(OUT / "castles.png")
    return rs


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    runs = load()
    entropy(runs)
    for run, r in castles(runs).items():
        print(f"castles {run}: r = {r:.3f}")
