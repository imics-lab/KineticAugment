"""Idea 2: how far and how long each clip's error leaves its normal range. Reads decomp/*.npz.
Exploratory, not project code."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PILOT = {"01986", "01987", "01988", "01991", "01992"}
clips = {p.stem: dict(np.load(p)) for p in sorted(D.glob("*.npz"))}


def series(d):
    """Per-frame component series (frame-mean over joints / landmarks)."""
    g, pf = np.nanmean(d["body_global"], 1), np.nanmean(d["body_perframe"], 1)
    out = {"body total": g, "body shape offset": g - pf, "body non-human arm": pf}
    for side in ("left", "right"):
        hg = np.nanmean(d[f"hand_{side}_global"], 1)
        hp = np.nanmean(d[f"hand_{side}_perframe"], 1)
        out[f"{side} hand total"] = hg
        out[f"{side} hand shape offset"] = hg - hp
        out[f"{side} hand per-frame"] = hp
    return out


S = {c: series(d) for c, d in clips.items()}
KEYS = ["body total", "body shape offset", "body non-human arm", "right hand total",
        "right hand per-frame", "left hand per-frame", "right hand shape offset"]


def robust_band(x):
    x = x[np.isfinite(x)]
    med = np.median(x)
    return med, med + 3 * 1.4826 * np.median(np.abs(x - med))


def excursions(x, hi):
    out = np.isfinite(x) & (x > hi)
    runs, cur = [], 0
    for o in out:
        cur = cur + 1 if o else 0
        if o and cur == 1:
            runs.append(1)
        elif o:
            runs[-1] += 1
    n = np.isfinite(x).sum()
    return (100 * out.sum() / n if n else np.nan, max(runs) if runs else 0, len(runs),
            np.nanmax(x - hi) if out.any() else 0.0)


pop = {k: robust_band(np.concatenate([S[c][k] for c in S])) for k in KEYS}
print("Population band (all 25 clips, median / upper = median + 3 robust SD):")
for k, (m, hi) in pop.items():
    print(f"  {k:<22} median {m:6.2f}  upper {hi:6.2f}")

for k in KEYS:
    print(f"\n{k.upper()}   [% frames out | longest run | episodes | max excess]  "
          f"own-clip band vs population band")
    for c in S:
        x = S[c][k]
        if not np.isfinite(x).sum():
            continue
        own = robust_band(x)
        a = excursions(x, own[1])
        b = excursions(x, pop[k][1])
        tag = "pilot" if c in PILOT else ""
        print(f"  {c} {tag:<5} med {np.nanmedian(x):6.2f} | own up {own[1]:6.2f}: {a[0]:5.1f}% "
              f"run {a[1]:2d} ep {a[2]:2d} exc {a[3]:5.2f} | pop: {b[0]:5.1f}% run {b[1]:2d} "
              f"ep {b[2]:2d} exc {b[3]:5.2f}")

# one panel per clip: body shape offset + non-human arm, right-hand per-frame; with bands
n = len(S)
cols = 5
rows = int(np.ceil(n / cols))
for group, keys, unit in (("body", ["body shape offset", "body non-human arm"], "% sh.w."),
                          ("hands", ["right hand per-frame", "left hand per-frame",
                                     "right hand shape offset"], "mm")):
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows), sharey=True)
    colors = ["#e8762b", "#1f5fbf", "#2ca02c"]
    for ax, c in zip(axes.flat, S):
        for k, col in zip(keys, colors):
            x = S[c][k]
            ax.plot(np.arange(1, len(x) + 1), x, color=col, lw=1.1, label=k)
            ax.axhline(pop[k][1], color=col, lw=0.7, ls="--")
            own = robust_band(x) if np.isfinite(x).sum() else (np.nan, np.nan)
            ax.axhline(own[1], color=col, lw=0.7, ls=":")
        ax.set_title(c + ("  (pilot)" if c in PILOT else ""), fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{group}: per-frame error components ({unit}); dashed = population upper band, "
                 f"dotted = clip's own upper band (median + 3 robust SD)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"deviation_{group}.png", dpi=80)
    print("wrote", OUT / f"deviation_{group}.png")
