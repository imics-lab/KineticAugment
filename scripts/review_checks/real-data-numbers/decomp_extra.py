"""Follow-ups on the decomposition: bootstrap CIs, scale spread under oracle shape,
Step-8-like 'shoulder root + translation only' direct number, and per-clip interaction."""
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import decomp as D

W = Path(__file__).parent
d = pickle.load(open(W / "decomp_sim_rigid.pkl", "rb"))
names, out = d["names"], d["out"]
def frames(c):
    return [e for ci in range(len(names)) for e in out[(c, ci)][0]]
def scales(c):
    return np.concatenate([np.array(out[(c, ci)][1]) for ci in range(len(names))])

rng = np.random.default_rng(0)
for mode, root in (("sim", "sh"), ("sim", "opt"), ("rigid", "sh"), ("rigid", "opt")):
    cs = [(mode, "none", "rest", "rest"), (mode, root, "rest", "rest"), (mode, "none", "clip", "rest"), (mode, root, "clip", "rest")]
    F = [frames(c) for c in cs]
    n = len(F[0])
    B = []
    for _ in range(1000):
        idx = rng.integers(n, size=n)
        m = [np.median(np.concatenate([F[k][i] for i in idx])) for k in range(4)]
        red = [100 * (1 - x / m[0]) for x in m[1:]]
        B.append(red + [red[2] - red[0] - red[1]])
    B = np.array(B)
    lab = ["root", "len", "both", "interaction"]
    print(f"frame-bootstrap 95% CI [{mode}, root={root}, len=clip]: " +
          "  ".join(f"{lab[k]} [{np.percentile(B[:,k],2.5):.1f}, {np.percentile(B[:,k],97.5):.1f}]" for k in range(4)))

# clip-level bootstrap (5 clips -> resample clips)
for mode, root in (("sim", "sh"),):
    cs = [(mode, "none", "rest", "rest"), (mode, root, "rest", "rest"), (mode, "none", "clip", "rest"), (mode, root, "clip", "rest")]
    per = [[np.concatenate(out[(c, ci)][0]) for ci in range(len(names))] for c in cs]
    B = []
    for _ in range(2000):
        idx = rng.integers(5, size=5)
        m = [np.median(np.concatenate([per[k][i] for i in idx])) for k in range(4)]
        B.append([100 * (1 - x / m[0]) for x in m[1:]])
    B = np.array(B)
    print(f"clip-bootstrap 95% CI [{mode}, root={root}, len=clip]: root [{np.percentile(B[:,0],2.5):.1f}, {np.percentile(B[:,0],97.5):.1f}]"
          f"  len [{np.percentile(B[:,1],2.5):.1f}, {np.percentile(B[:,1],97.5):.1f}]  both [{np.percentile(B[:,2],2.5):.1f}, {np.percentile(B[:,2],97.5):.1f}]")

print("\nsimilarity scale (median, p5, p95) under each shape oracle (root none):")
for c in [("sim", "none", "rest", "rest"), ("sim", "none", "clip", "rest"), ("sim", "none", "rest", "clip"),
          ("sim", "none", "clip", "clip"), ("sim", "sh", "clip", "clip"), ("sim", "none", "frame", "frame")]:
    s = scales(c)
    print(f"  {c}: {np.median(s):.4f}  [{np.percentile(s,5):.4f}, {np.percentile(s,95):.4f}]")

# residual after doc's 'both' split into shoulder width vs within-clip per-frame length variation
b = np.median(np.concatenate(frames(("sim", "none", "rest", "rest"))))
seq = [("sim", "sh", "clip", "rest"), ("sim", "sh", "clip", "clip"), ("sim", "sh", "frame", "clip"), ("sim", "sh", "frame", "frame")]
print("\nwhat the doc's 'unexplained 44.5%' consists of (sim, sh root):")
prev = None
for c in seq:
    v = np.median(np.concatenate(frames(c)))
    print(f"  {c}: {v*100:.3f}%  remaining {100*v/b:.1f}% of baseline" + (f"  (step {100*(prev-v)/b:.1f} pts)" if prev else ""))
    prev = v

# Step-8-like estimator: shoulder-line root + translation only (no per-frame rotation fit); directions stay exact
clips = D.load_clips()
for lo in ("rest", "clip"):
    errs = []
    for name, pose, oks, width in clips:
        clip_len = {k: float(np.median(np.linalg.norm(pose[oks[:, k[0]] & oks[:, k[1]], k[1]] - pose[oks[:, k[0]] & oks[:, k[1]], k[0]], axis=-1))) for k in D.REST_LEN}
        for frame, ok in zip(pose, oks):
            present = [m for m in D.MP if ok[m]]
            if len(present) < 3:
                continue
            sh = {11: D.rest[16], 12: D.rest[17]}
            L = D.REST_LEN if lo == "rest" else clip_len
            e, _ = D.residual(frame, ok, present, D.shoulder_swing(frame, sh), L, sh, "none", width)
            errs += list(e)
    print(f"\nshoulder-line root + translation only (Step-8-like, directions exact), lengths={lo}: median {np.median(errs)*100:.3f}%  p95 {np.percentile(errs,95)*100:.3f}%")
