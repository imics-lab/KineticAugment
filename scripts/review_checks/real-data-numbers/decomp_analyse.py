import pickle, sys, itertools
from pathlib import Path
import numpy as np
W = Path(__file__).parent
d = pickle.load(open(W / "decomp_sim_rigid.pkl", "rb"))
configs, names, out = d["configs"], d["names"], d["out"]

def pooled(c):
    return np.concatenate([np.concatenate(out[(c, ci)][0]) for ci in range(len(names))])
def perclip(c):
    return [np.median(np.concatenate(out[(c, ci)][0])) * 100 for ci in range(len(names))]
def scale(c):
    return np.median(np.concatenate([np.array(out[(c, ci)][1]) for ci in range(len(names))]))
def rootdeg(c):
    r = np.concatenate([np.array(out[(c, ci)][2]) for ci in range(len(names))])
    return np.degrees(np.linalg.norm(r, axis=1))

print("All configs: pooled median % shoulder width (reduction vs same-mode baseline) | per clip " + " ".join(names) + " | sim scale med | root deg med")
for c in configs:
    base = np.median(pooled((c[0], "none", "rest", "rest")))
    v = pooled(c)
    print(f"  {c[0]:<5} root={c[1]:<4} len={c[2]:<5} sw={c[3]:<5}  {np.median(v)*100:6.3f}%  ({100*(1-np.median(v)/base):5.1f}%)  "
          + " ".join(f"{x:5.2f}" for x in perclip(c)) + f"  | {scale(c):.4f} | {np.median(rootdeg(c)):6.2f}")

print("\nDoc fingerprint: baseline 5.94 | root-only 5.13 (01988: 5.06 -> 5.11) | length-only 3.86 | both 2.64")

def two_by_two(mode, root, length, sw="rest"):
    b = np.median(pooled((mode, "none", "rest", sw)))
    r = np.median(pooled((mode, root, "rest", sw)))
    l = np.median(pooled((mode, "none", length, sw)))
    rl = np.median(pooled((mode, root, length, sw)))
    red = lambda x: 100 * (1 - x / b)
    R_alone, L_alone, both = red(r), red(l), red(rl)
    R_given_L = both - L_alone
    L_given_R = both - R_alone
    inter = both - R_alone - L_alone
    print(f"  [{mode}, root={root}, len={length}, sw={sw}] base {b*100:.2f}  R {r*100:.2f}  L {l*100:.2f}  R+L {rl*100:.2f}")
    print(f"     reductions: root alone {R_alone:.1f}% | root given len {R_given_L:.1f}% | len alone {L_alone:.1f}% | len given root {L_given_R:.1f}% "
          f"| both {both:.1f}% | interaction {inter:+.1f} pts | 'unexplained' {100-both:.1f}%")
    print(f"     Shapley: root {(R_alone+R_given_L)/2:.1f}%  len {(L_alone+L_given_R)/2:.1f}%  (sum {both:.1f}%)")

print("\n2x2 decompositions (percent reduction of the pooled median):")
for mode in ("sim", "rigid"):
    for root in ("opt", "sh"):
        for length in ("clip", "frame"):
            two_by_two(mode, root, length)

# 3-factor Shapley with shoulder width
def shapley3(mode, root, length, sw):
    facs = {"R": ("none", root), "L": ("rest", length), "W": ("rest", sw)}
    def val(S):
        c = (mode, facs["R"][1] if "R" in S else "none", facs["L"][1] if "L" in S else "rest", facs["W"][1] if "W" in S else "rest")
        return np.median(pooled(c))
    b = val(())
    v = {S: 100 * (1 - val(S) / b) for k in range(4) for S in itertools.combinations("RLW", k)}
    import math
    sh = {}
    for f in "RLW":
        others = [x for x in "RLW" if x != f]
        tot = 0
        for k in range(3):
            for S in itertools.combinations(others, k):
                Sf = tuple(sorted(S + (f,), key="RLW".index))
                w = math.factorial(len(S)) * math.factorial(3 - len(S) - 1) / math.factorial(3)
                tot += w * (v[Sf] - v[S])
        sh[f] = tot
    allv = v[("R", "L", "W")]
    print(f"  [{mode} root={root} len={length} sw={sw}] all three {allv:.1f}% -> residual {100-allv:.1f}% | Shapley R {sh['R']:.1f} L {sh['L']:.1f} W {sh['W']:.1f}")
    # all orderings
    for order in itertools.permutations("RLW"):
        S, steps = (), []
        for f in order:
            Sf = tuple(sorted(S + (f,), key="RLW".index))
            steps.append(f"{f}+{v[Sf]-v[S]:.1f}")
            S = Sf
        print("      order " + "".join(order) + ": " + ", ".join(steps))

print("\n3-factor (root, length, shoulder width):")
for mode in ("sim", "rigid"):
    shapley3(mode, "opt", "clip", "clip")
    shapley3(mode, "opt", "frame", "frame")

# bootstrap over clips/frames for the doc-lineage numbers: is 13.7 vs 18.6 within noise? frame-level bootstrap
rng = np.random.default_rng(0)
def frame_lists(c):
    return [e for ci in range(len(names)) for e in out[(c, ci)][0]]
cs = [("sim", "none", "rest", "rest"), ("sim", "opt", "rest", "rest"), ("sim", "none", "clip", "rest"), ("sim", "opt", "clip", "rest")]
F = [frame_lists(c) for c in cs]
nfr = len(F[0])
boots = []
for _ in range(500):
    idx = rng.integers(nfr, size=nfr)
    meds = [np.median(np.concatenate([F[k][i] for i in idx])) for k in range(4)]
    boots.append([100 * (1 - m / meds[0]) for m in meds[1:]])
boots = np.array(boots)
pass
    f"[{np.percentile(boots[:,k],2.5):.1f}, {np.percentile(boots[:,k],97.5):.1f}]" for k in range(3)))
