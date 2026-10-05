"""Render a harness replay (.npz) to an animated GIF in the style of the generals.bot viewer.

    python results/render_replay.py <replay.npz> results/figures/game.gif [--stride 2] [--blue rl_bot]

The whole board is drawn. Cells that neither player has seen yet are dark, the way
the competition's replay viewer shows unexplored ground. Replays are written by
`python -m evaluation.evaluate` by default.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import evaluation  # noqa: F401  (sys.path bootstrap)
from evaluation.run_match import load_replay

C, M, GAP, PANEL = 34, 16, 20, 300            # cell, margin, board-panel gap, panel width
BG, RULE = (255, 255, 255), (228, 228, 228)
INK, INK2 = (18, 18, 20), (107, 107, 115)
PLAIN, MOUNTAIN, FOG, FOG_MOUNTAIN = (229, 229, 229), (186, 186, 186), (75, 80, 87), (117, 122, 130)
BLUE = {"cell": (72, 100, 219), "structure": (59, 81, 175), "general": (50, 67, 142), "card": (233, 236, 248)}
RED = {"cell": (219, 72, 72), "structure": (168, 56, 56), "general": (142, 50, 50), "card": (252, 233, 219)}


def font(size, bold=False):
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def darker(rgb, f=0.78):
    return tuple(int(c * f) for c in rgb)


def dilate(mask):
    """3×3 dilation: the engine's vision rule (you see every cell next to one you own)."""
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.any([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr in (-1, 0, 1) for dc in (-1, 0, 1)], axis=0)


def mountain_icon(d, x0, y0, color):
    pts = [(0.18, 0.74), (0.38, 0.34), (0.5, 0.54), (0.62, 0.30), (0.82, 0.74)]
    d.line([(x0 + C * a, y0 + C * b) for a, b in pts], fill=color, width=2, joint="curve")


def crown_icon(d, x0, y0, color):
    pts = [(0.22, 0.78), (0.22, 0.42), (0.36, 0.58), (0.5, 0.36), (0.64, 0.58), (0.78, 0.42), (0.78, 0.78)]
    d.polygon([(x0 + C * a, y0 + C * b) for a, b in pts], outline=color, width=2)


def castle_icon(d, x0, y0, color):
    pts = [(0.24, 0.80), (0.24, 0.40), (0.34, 0.40), (0.34, 0.50), (0.44, 0.50), (0.44, 0.40),
           (0.56, 0.40), (0.56, 0.50), (0.66, 0.50), (0.66, 0.40), (0.76, 0.40), (0.76, 0.80)]
    d.polygon([(x0 + C * a, y0 + C * b) for a, b in pts], outline=color, width=2)


def frame(rep, t, seen, palettes, names, fonts):
    h, w = rep.mountains.shape
    W = M + w * C + GAP + PANEL + M
    H = M + max(h * C, 200) + M
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    own, armies, castles = rep.ownership[t], rep.armies[t], rep.castles[t]

    for r in range(h):
        for c in range(w):
            x0, y0 = M + c * C, M + r * C
            box = [x0, y0, x0 + C, y0 + C]
            owner = 0 if own[0, r, c] else 1 if own[1, r, c] else -1
            if rep.mountains[r, c]:
                fill = MOUNTAIN if seen[r, c] else FOG_MOUNTAIN
                d.rectangle(box, fill=fill, outline=darker(fill, 0.85))
                mountain_icon(d, x0, y0, (34, 34, 36) if seen[r, c] else (60, 64, 70))
                continue
            if owner < 0:
                fill = PLAIN if seen[r, c] else FOG
                d.rectangle(box, fill=fill, outline=darker(fill, 0.85))
            else:
                pal = palettes[owner]
                kind = "general" if rep.generals[r, c] else "structure" if castles[r, c] else "cell"
                fill = pal[kind]
                d.rectangle(box, fill=fill, outline=darker(pal["cell"], 0.7))
                if kind == "general":
                    crown_icon(d, x0, y0, darker(fill, 0.55))
                elif kind == "structure":
                    castle_icon(d, x0, y0, darker(fill, 0.6))
            a = int(armies[r, c])
            if a > 0 and owner >= 0:
                txt = str(a) if a < 1000 else f"{a / 1000:.1f}k"
                d.text((x0 + C / 2, y0 + C / 2), txt, fill=(255, 255, 255), font=fonts["cell"], anchor="mm")

    # side panel
    px = M + w * C + GAP
    d.text((px, M + 6), "T I C K", fill=INK2, font=fonts["label"])
    d.text((px + PANEL, M + 2), f"{t} / {rep.armies.shape[0] - 1}", fill=INK, font=fonts["tick"], anchor="ra")
    d.line([(px, M + 34), (px + PANEL, M + 34)], fill=RULE, width=1)
    y = M + 46
    for p in range(2):
        pal = palettes[p]
        land = int(own[p].sum())
        army = int((armies * own[p]).sum())
        d.rectangle([px, y, px + PANEL, y + 58], fill=pal["card"])
        d.rectangle([px, y, px + 3, y + 58], fill=pal["cell"])
        d.text((px + 16, y + 9), names[p], fill=INK, font=fonts["name"])
        d.text((px + 16, y + 33), f"{land} land · {army} army", fill=INK2, font=fonts["stat"])
        y += 66
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2, help="draw every Nth turn")
    ap.add_argument("--blue", default="rl_bot", help="bot drawn in blue")
    ap.add_argument("--ms", type=int, default=80, help="milliseconds per frame")
    args = ap.parse_args()

    rep = load_replay(args.replay)
    names = [rep.meta["bot0"], rep.meta["bot1"]]
    palettes = [BLUE, RED] if names[0] == args.blue else [RED, BLUE]
    fonts = {"cell": font(13, True), "label": font(12), "tick": font(18, True),
             "name": font(16, True), "stat": font(13)}

    T = rep.armies.shape[0]
    seen_by_turn, seen = [], np.zeros(rep.mountains.shape, dtype=bool)
    for t in range(T):
        seen |= dilate(rep.ownership[t][0] | rep.ownership[t][1])
        seen_by_turn.append(seen.copy())

    turns = list(range(0, T, args.stride)) + [T - 1]
    frames = [frame(rep, t, seen_by_turn[t], palettes, names, fonts) for t in turns]
    frames += [frames[-1]] * 25  # hold the final position
    frames = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=args.ms, loop=0, optimize=True)
    win = rep.meta["winner"]
    print(f"{args.out}: {len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB, "
          f"winner {names[win] if win is not None else 'draw'} by {rep.meta['end_reason']} at turn {T - 1}")


if __name__ == "__main__":
    main()
