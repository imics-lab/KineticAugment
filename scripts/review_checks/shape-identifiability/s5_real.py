"""Q4/Q5 on real clips: length time-series statistics, and what per-clip
regularised / unregularised beta fits deliver (identifiability in practice)."""
import glob, os
import numpy as np
from scipy.optimize import least_squares, minimize
from common import SNAP, OBS, NB, lengths, length_jac, body_stats

np.set_printoptions(precision=2, suppress=True, linewidth=160)
MP = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]   # same order as OBS
names = [n for *_, n in OBS]
VIS = 0.5

def clip_series(path):
    d = np.load(path)
    P = d["landmarks"].astype(np.float64); ok = np.isfinite(P).all(-1) & (d["visibility"] >= VIS)
    S = np.full((len(P), 5), np.nan)
    for i, (a, b) in enumerate(MP):
        m = ok[:, a] & ok[:, b]
        S[m, i] = np.linalg.norm(P[m, a] - P[m, b], axis=-1)
    return S

def acf1(x):
    x = x[np.isfinite(x)]
    if len(x) < 3: return np.nan
    x = x - x.mean()
    return (x[:-1] @ x[1:]) / (x @ x)

def block_boot_median_se(x, block=10, n=2000, rng=np.random.default_rng(0)):
    x = x[np.isfinite(x)]; N = len(x)
    if N < block * 2: return np.nan
    starts = np.arange(N - block + 1); k = int(np.ceil(N / block)); meds = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:N]
        meds.append(np.median(x[idx]))
    return np.std(meds)

files = sorted(glob.glob(f"{SNAP}/data/test_output/*_world.npz"))
L0 = lengths(np.zeros(NB))
print("Template lengths (mm):", np.round(L0 * 1000, 1), names)
med = {}
print("\n=== Per-clip observed length statistics (mm) ===")
for f in files:
    clip = os.path.basename(f)[:5]; S = clip_series(f)
    med[clip] = np.nanmedian(S, 0)
    print(f"clip {clip}: frames={len(S)}")
    for i, n in enumerate(names):
        x = S[:, i]; nv = np.isfinite(x).sum()
        r = acf1(x); neff = nv * (1 - r) / (1 + r) if np.isfinite(r) else np.nan
        print(f"   {n:<15} n={nv:4d} median={np.nanmedian(x)*1000:7.1f} frame-std={np.nanstd(x)*1000:6.2f} "
              f"lag1-acf={r:5.2f} n_eff~{neff:6.1f} iid-SE(median)~{1.2533*np.nanstd(x)/np.sqrt(max(nv,1))*1000:5.2f} "
              f"block-boot-SE={block_boot_median_se(x)*1000:5.2f}")

def map_fit(Lobs, sigma_mm, x0=None):
    w = 1.0 / (sigma_mm / 1000.0)
    fun = lambda b: np.concatenate([(lengths(b) - Lobs) * w, b])
    jac = lambda b: np.vstack([length_jac(b, with_scale=False) * w, np.eye(NB)])
    return least_squares(fun, np.zeros(NB) if x0 is None else x0, jac=jac).x

print("\n=== Per-clip shape fits to the 5 median lengths ===")
results = {}
for clip, Lobs in med.items():
    print(f"\n--- clip {clip}: observed {np.round(Lobs*1000,1)} mm; template residual {np.round((L0-Lobs)*1000,1)} mm")
    for sig in (1.0, 3.0, 10.0):
        b = map_fit(Lobs, sig); st = body_stats(b)
        print(f"  MAP sigma={sig:4.1f}mm |beta|={np.linalg.norm(b):5.2f} resid(mm)={np.round((lengths(b)-Lobs)*1000,1)} "
              f"height={st['height_m']*100:5.1f}cm leg={st['L_leg_m']*100:4.1f}cm vol={st['volume_L']:5.1f}L")
        results[(clip, sig)] = b
    # min-norm EXACT fit (Gauss-Newton on underdetermined system)
    b = np.zeros(NB)
    for _ in range(50):
        Jb = length_jac(b, with_scale=False); r = lengths(b) - Lobs
        b = b - np.linalg.pinv(Jb) @ r
    st = body_stats(b)
    print(f"  min-norm EXACT fit   |beta|={np.linalg.norm(b):6.1f} resid(mm)={np.round((lengths(b)-Lobs)*1000,3)} "
          f"height={st['height_m']*100:5.1f}cm vol={st['volume_L']:6.1f}L  beta={b}")
    # unregularised LS, bounded box |beta_k|<=3, from 20 random inits
    hs, ress, bs = [], [], []
    rng = np.random.default_rng(1)
    for _ in range(20):
        sol = least_squares(lambda b: (lengths(b) - Lobs) * 1000, rng.uniform(-2, 2, NB),
                            jac=lambda b: length_jac(b, with_scale=False) * 1000, bounds=(-3, 3))
        hs.append(body_stats(sol.x)["height_m"] * 100); ress.append(np.abs(sol.fun).max()); bs.append(sol.x)
    hs = np.array(hs); ress = np.array(ress); best = ress < ress.min() + 0.5
    print(f"  unregularised LS, box |b_k|<=3, 20 inits: max|resid| {ress.min():.2f}-{ress.max():.2f} mm; "
          f"among near-best (within 0.5mm) n={best.sum()}: height range {hs[best].min():.1f}-{hs[best].max():.1f} cm, "
          f"|beta| {min(np.linalg.norm(x) for x,k in zip(bs,best) if k):.1f}-{max(np.linalg.norm(x) for x,k in zip(bs,best) if k):.1f}")
np.save("map_fits.npy", {f"{k[0]}_{k[1]}": v for k, v in results.items()}, allow_pickle=True)
