"""Q4 on real clips: are frame-to-frame length variations iid noise (would average
out over frames) or pose-correlated (systematic; frames are not independent)?
Also: observed left/right asymmetry vs what SMPL-X shape space can express."""
import glob, os
import numpy as np
from common import SNAP, NB, lengths, length_jac

MP = {"L upper": (11, 13), "L fore": (13, 15), "R upper": (12, 14), "R fore": (14, 16), "shoulder W": (11, 12)}
rows = []
print(f"{'clip':<6}{'segment':<11}{'n':>4}{'sd mm':>7}{'r(len, |dir_z|)':>17}{'r(len, elbow ang)':>19}{'R2 pose->len':>14}")
for path in sorted(glob.glob(f"{SNAP}/data/test_output/*_world.npz")):
    clip = os.path.basename(path)[:5]
    d = np.load(path); P = d["landmarks"].astype(float)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    for name, (a, b) in MP.items():
        m = ok[:, a] & ok[:, b]
        side = 13 if name.startswith("L") else 14
        sh, wr = (11, 15) if name.startswith("L") else (12, 16)
        if name != "shoulder W":
            m &= ok[:, sh] & ok[:, side] & ok[:, wr]
        if m.sum() < 10: continue
        v = P[m, b] - P[m, a]; L = np.linalg.norm(v, axis=1)
        dz = np.abs(v[:, 2]) / L
        if name != "shoulder W":
            u = P[m, side] - P[m, sh]; f = P[m, wr] - P[m, side]
            ang = np.degrees(np.arccos(np.clip((u * f).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(f, axis=1), -1, 1)))
            X = np.column_stack([np.ones(len(L)), dz, ang, v / L[:, None]])
            r_ang = np.corrcoef(L, ang)[0, 1]
        else:
            X = np.column_stack([np.ones(len(L)), dz, v / L[:, None]]); r_ang = np.nan
        beta, *_ = np.linalg.lstsq(X, L, rcond=None)
        R2 = 1 - ((L - X @ beta) ** 2).sum() / ((L - L.mean()) ** 2).sum()
        r_dz = np.corrcoef(L, dz)[0, 1]
        rows.append((clip, name, R2))
        print(f"{clip:<6}{name:<11}{m.sum():>4}{L.std()*1000:7.1f}{r_dz:17.2f}{r_ang:19.2f}{R2:14.2f}")
R2s = np.array([r[2] for r in rows])
print(f"\nmedian R^2 of length explained by its own bone direction (+elbow angle): {np.median(R2s):.2f} "
      f"(range {R2s.min():.2f}-{R2s.max():.2f}); iid noise would give ~{(5/60):.2f} by chance at n~60, 5 regressors")

# L/R asymmetry vs SMPL-X's expressible asymmetry
Jb = length_jac(np.zeros(NB), with_scale=False); L0 = lengths(np.zeros(NB))
anti_up = np.linalg.norm(Jb[0] - Jb[2]); anti_fo = np.linalg.norm(Jb[1] - Jb[3])
print(f"\nSMPL-X (10 betas) max |d(L-R)| per unit-norm beta: upper {anti_up*1000:.2f} mm, fore {anti_fo*1000:.2f} mm; "
      f"template L-R: upper {(L0[0]-L0[2])*1000:+.1f}, fore {(L0[1]-L0[3])*1000:+.1f} mm")
print(f"{'clip':<6}{'obs upper L-R':>14}{'obs fore L-R':>14}{'|beta| needed (upper)':>23}{'(fore)':>8}")
for path in sorted(glob.glob(f"{SNAP}/data/test_output/*_world.npz")):
    clip = os.path.basename(path)[:5]
    d = np.load(path); P = d["landmarks"].astype(float)
    ok = np.isfinite(P).all(-1) & (d["visibility"] >= 0.5)
    med = {}
    for name, (a, b) in MP.items():
        m = ok[:, a] & ok[:, b]; med[name] = np.median(np.linalg.norm(P[m, b] - P[m, a], axis=1))
    du = med["L upper"] - med["R upper"]; df = med["L fore"] - med["R fore"]
    print(f"{clip:<6}{du*1000:14.1f}{df*1000:14.1f}{abs(du-(L0[0]-L0[2]))/anti_up:23.1f}{abs(df-(L0[1]-L0[3]))/anti_fo:8.1f}")
