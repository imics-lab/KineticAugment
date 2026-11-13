#!/usr/bin/env python3
"""Verifier's independent checks of the noise-floor (A004) cluster.

Own implementations of: Method A/B estimators, One Euro filter (MediaPipe C++ semantics),
Umeyama similarity, round-trip-style metric, ACF. Does NOT import the tester's code.
Run with cwd = snapshot root.
"""
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE / "ext"
TEX = HERE.parent / "noise-floor" / "extract"     # tester's re-extractions (used only for perturbations)
REPO = Path("data/test_output")
CLIPS = ["01986", "01987", "01988", "01991", "01992"]
ARM = [11, 12, 13, 14, 15, 16]
SCRIPT_SET = ARM + [23, 24, 25, 26, 27, 28]
HIPS = [23, 24]


def path(src, clip):
    if src == "repo":
        return REPO / f"{clip}_world.npz"
    if src.startswith("T:"):
        return TEX / src[2:] / f"{clip}_world.npz"
    return EXT / src / f"{clip}_world.npz"


def load(src, clip):
    d = np.load(path(src, clip))
    L = d["landmarks"].astype(np.float64); V = d["visibility"].astype(np.float64)
    ok = np.isfinite(L).all(-1) & (V >= 0.5)
    return L, ok


def sw(L, ok):
    b = ok[:, 11] & ok[:, 12]
    return float(np.median(np.linalg.norm(L[b, 11] - L[b, 12], axis=-1)))


# ---------------------------------------------------------------- estimators (own, vectorised)
def est_A(tr, ok, dims):
    v = ok[:-1] & ok[1:]
    st = np.linalg.norm((tr[1:] - tr[:-1])[:, dims], axis=-1)[v]
    return np.percentile(st, 10) if len(st) >= 5 else np.nan


def est_B(tr, ok, dims):
    v1 = ok[:-1] & ok[1:]
    sp = np.where(v1, np.linalg.norm(tr[1:] - tr[:-1], axis=-1), np.nan)   # speed always 3-D (as script)
    if np.all(np.isnan(sp)):
        return np.nan
    g = np.nanpercentile(sp, 25)
    v3 = ok[:-2] & ok[1:-1] & ok[2:]
    keep = v3 & (np.fmax(sp[:-1], sp[1:]) <= g) & ~np.isnan(sp[:-1]) & ~np.isnan(sp[1:])
    D = (tr[:-2] - 2 * tr[1:-1] + tr[2:])[:, dims][keep]
    if len(D) < 5:
        return np.nan
    return np.linalg.norm(D.std(axis=0)) / math.sqrt(6)


def pooled(src, lms, dims, clips=CLIPS, step=1, offset=0):
    A, B = [], []
    for c in clips:
        L, ok = load(src, c)
        L, ok = L[offset::step], ok[offset::step]
        s = sw(L, ok)
        for j in lms:
            if ok[:, j].mean() < 0.5:
                continue
            a = est_A(L[:, j], ok[:, j], dims) / s; b = est_B(L[:, j], ok[:, j], dims) / s
            if np.isfinite(a): A.append(a)
            if np.isfinite(b): B.append(b)
    return 100 * np.median(A), 100 * np.median(B), len(A), len(B)


# ---------------------------------------------------------------- One Euro (MediaPipe semantics)
class LP:
    def __init__(s): s.init = False; s.raw = 0.0; s.st = 0.0
    def apply(s, v, a):
        r = a * v + (1 - a) * s.st if s.init else v
        s.init = True; s.raw = v; s.st = r; return r


class OE:
    def __init__(s, mc, beta, dc, freq=30.0):
        s.mc, s.beta, s.dc, s.f = mc, beta, dc, freq; s.x = LP(); s.dx = LP(); s.last = 0
    def alpha(s, c):
        te = 1.0 / s.f; tau = 1.0 / (2 * math.pi * c); return 1.0 / (1.0 + tau / te)
    def apply(s, t_us, v):
        if s.last != 0:
            s.f = 1.0 / ((t_us - s.last) * 1e-6)
        s.last = t_us
        dv = (v - s.x.raw) * s.f if s.x.init else 0.0
        e = s.dx.apply(dv, s.alpha(s.dc))
        return s.x.apply(v, s.alpha(s.mc + s.beta * abs(e)))


def oe_filter(raw, mc=0.1, beta=40.0, dc=1.0, t0=0):
    """raw (T,33,3) with NaN rows for missing; filter resets on missing frames (calculator semantics)."""
    T = raw.shape[0]; out = np.full_like(raw, np.nan); fs = None
    for t in range(T):
        ts = t0 + (t + 1) * 33333
        if not np.isfinite(raw[t]).all():
            fs = None; continue
        if fs is None:
            fs = [[OE(mc, beta, dc) for _ in range(3)] for _ in range(33)]
        for j in range(33):
            for k in range(3):
                out[t, j, k] = fs[j][k].apply(ts, raw[t, j, k])
    return out


# ---------------------------------------------------------------- Umeyama
def umeyama(src, tgt):
    ms, mt = src.mean(0), tgt.mean(0); X = src - ms; Y = tgt - mt
    U, S, Vt = np.linalg.svd(Y.T @ X / len(src))
    d = np.sign(np.linalg.det(U @ Vt)); Dm = np.diag([1, 1, d])
    R = U @ Dm @ Vt; s = np.trace(np.diag(S) @ Dm) / (X ** 2).sum(1).mean()
    return s * (R @ X.T).T + mt


def pair(srcA, srcB, align=False, clips=CLIPS, agg=np.median):
    e = []; per_axis = []
    for c in clips:
        a, oka = load(srcA, c); b, okb = load(srcB, c)
        n = min(len(a), len(b)); s = sw(a, oka)
        for t in range(n):
            pres = [m for m in ARM if oka[t, m] and okb[t, m]]
            if len(pres) < 3:
                continue
            P = b[t, pres]; Q = a[t, pres]
            if align:
                P = umeyama(P, Q)
            e.extend(np.linalg.norm(P - Q, axis=1) / s)
            per_axis.extend(np.abs(P - Q) / s)
    e = np.array(e) * 100
    return e


def acf(series, maxlag=2):
    num = np.zeros(maxlag + 1); den = 0
    for x in series:
        x = x - x.mean(); den += (x * x).sum()
        for k in range(maxlag + 1):
            num[k] += (x[:len(x) - k] * x[k:]).sum()
    return num / den


def runs(ok, m):
    out, s = [], None
    for t, v in enumerate(list(ok) + [False]):
        if v and s is None: s = t
        elif not v and s is not None:
            if t - s >= m: out.append((s, t))
            s = None
    return out


def main():
    P = print
    P("=" * 80); P("0. Re-extraction identity checks"); P("=" * 80)
    for c in CLIPS:
        r = np.load(path("repo", c))["landmarks"]; sd = np.load(path("shared_default", c))["landmarks"]
        fd = np.load(path("fresh_default", c))["landmarks"]; td = np.load(path("T:default", c))["landmarks"]
        fr = np.load(path("fresh_raw", c))["landmarks"]; tr_ = np.load(path("T:nosmooth", c))["landmarks"]
        P(f"{c}: repo==my shared_default {np.array_equal(r, sd, equal_nan=True)} | "
          f"my fresh_default==tester default {np.array_equal(fd, td, equal_nan=True)} | "
          f"my fresh_raw==tester nosmooth {np.array_equal(fr, tr_, equal_nan=True)} | "
          f"repo==fresh_default {np.array_equal(r, fd, equal_nan=True)}")

    P("\n" + "=" * 80); P("1. One Euro identification: refilter raw -> compare to smoothed output"); P("=" * 80)
    for mc, beta, dc in [(0.1, 40.0, 1.0), (0.05, 80.0, 1.0), (0.1, 10.0, 1.0), (0.01, 40.0, 1.0)]:
        mx = 0
        for c in CLIPS:
            raw = np.load(path("fresh_raw", c))["landmarks"].astype(np.float64)
            sm = np.load(path("fresh_default", c))["landmarks"].astype(np.float64)
            f = oe_filter(raw, mc, beta, dc)
            m = np.isfinite(f) & np.isfinite(sm)
            mx = max(mx, np.abs(f[m] - sm[m]).max())
        P(f"  OE(min_cutoff={mc}, beta={beta}, dcutoff={dc}) fresh: max |refilt - smoothed| = {mx:.3e} m")
    # raw outputs identical across smoothing on/off? check visibility-independent proxy: image-space not stored;
    # check shared_raw vs shared_default with filter state carried across clips
    mx = 0; t0 = 0; prev = None
    raw_all = np.concatenate([np.load(path("shared_raw", c))["landmarks"].astype(np.float64) for c in CLIPS])
    sm_all = np.concatenate([np.load(path("shared_default", c))["landmarks"].astype(np.float64) for c in CLIPS])
    f = oe_filter(raw_all); m = np.isfinite(f) & np.isfinite(sm_all)
    P(f"  OE(0.1,40,1) shared graph, filter state carried across clips: max diff = {np.abs(f[m]-sm_all[m]).max():.3e} m")

    P("\n" + "=" * 80); P("2. Own Method A/B (pooled median, % sw) -- must match script on repo"); P("=" * 80)
    bands = {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xyz": [0, 1, 2]}
    for src in ["repo", "fresh_default", "fresh_raw", "shared_raw", "T:static"]:
        row = []
        for b, dims in bands.items():
            a, bb, na, nb = pooled(src, SCRIPT_SET, dims)
            row.append(f"{b}:A={a:.3f},B={bb:.3f}")
        P(f"  {src:<14} script set n={nb}  " + "  ".join(row))
    for src in ["repo", "fresh_raw"]:
        a, b, na, nb = pooled(src, ARM, [0, 1, 2]); P(f"  {src:<14} ARMS ONLY xyz A={a:.3f} B={b:.3f} (n={nb})")
    a, b, na, nb = pooled("repo", HIPS, [0, 1, 2]); P(f"  repo HIPS ONLY xyz A={a:.3f} B={b:.3f} (n={nb})")
    for c in CLIPS:
        L, ok = load("repo", c); P(f"    {c} hip observed frac L={ok[:,23].mean():.2f} R={ok[:,24].mean():.2f}")
    # per-axis z ratio
    bx = pooled("repo", SCRIPT_SET, [0])[1]; by = pooled("repo", SCRIPT_SET, [1])[1]; bz = pooled("repo", SCRIPT_SET, [2])[1]
    bxy = pooled("repo", SCRIPT_SET, [0, 1])[1]
    P(f"  z ratio: script B_z/B_xy={bz/bxy:.2f}; per-axis B_z/mean(Bx,By)={bz/((bx+by)/2):.2f}; "
      f"B_z/(B_xy/sqrt2)={bz/(bxy/math.sqrt(2)):.2f}")
    bx = pooled("fresh_raw", SCRIPT_SET, [0])[1]; by = pooled("fresh_raw", SCRIPT_SET, [1])[1]; bz = pooled("fresh_raw", SCRIPT_SET, [2])[1]
    bxy = pooled("fresh_raw", SCRIPT_SET, [0, 1])[1]
    P(f"  raw z ratio: B_z/B_xy={bz/bxy:.2f}; per-axis={bz/((bx+by)/2):.2f}")
    # median shoulder widths
    P("  shoulder widths (repo): " + ", ".join(f"{c}={sw(*load('repo', c)):.4f}" for c in CLIPS))

    P("\n" + "=" * 80); P("3. Frame-rate (decimation) dependence of the script budget max(A_xyz,B_xyz)"); P("=" * 80)
    for src in ["repo", "fresh_raw"]:
        for step in (1, 2, 3):
            vals = []
            for off in range(step):
                a, b, _, _ = pooled(src, SCRIPT_SET, [0, 1, 2], step=step, offset=off)
                vals.append((a, b))
            a = np.mean([v[0] for v in vals]); b = np.mean([v[1] for v in vals])
            a0, b0 = vals[0]
            P(f"  {src:<10} every {step} frame(s): offset0 A={a0:.3f} B={b0:.3f} budget={max(a0,b0):.3f} | "
              f"mean over offsets A={a:.3f} B={b:.3f}")

    P("\n" + "=" * 80); P("4. ACF of second difference D (arm landmarks, runs>=12, all axes)"); P("=" * 80)
    P("  iid theory: lag1=-0.667 lag2=+0.167")
    for src in ["repo", "fresh_default", "fresh_raw", "T:static"]:
        ser = []
        for c in CLIPS:
            L, ok = load(src, c)
            for j in ARM:
                for s, e in runs(ok[:, j], 12):
                    for k in range(3):
                        x = L[s:e, j, k]; ser.append(x[:-2] - 2 * x[1:-1] + x[2:])
        r = acf(ser); P(f"  {src:<14} lag1={r[1]:+.3f} lag2={r[2]:+.3f}")
    ser = []
    for c in CLIPS:
        a, oka = load("fresh_default", c); b, okb = load("fresh_raw", c)
        for j in ARM:
            for s, e in runs(oka[:, j] & okb[:, j], 12):
                for k in range(3):
                    ser.append(a[s:e, j, k] - b[s:e, j, k])
    r = acf(ser); P(f"  filter displacement (smoothed-raw) lag1={r[1]:+.3f} lag2={r[2]:+.3f}")

    P("\n" + "=" * 80); P("5. Same-statistic comparisons: 6 arm landmarks, |a-b|/sw (median, mean, p95) %"); P("=" * 80)
    prs = [("fresh_default", "fresh_raw", "raw vs smoothed (own extraction)"),
           ("fresh_default", "repo", "fresh graph vs shared graph (repo)"),
           ("T:default", "T:resize75", "smoothed, 75% input (tester extracts)"),
           ("T:nosmooth", "T:nosmooth_resize75", "raw, 75% input"),
           ("T:default", "T:padshift", "smoothed, padded"),
           ("T:default", "T:bright", "smoothed, +25 brightness"),
           ("T:default", "T:static", "tracking vs static"),
           ("T:default", "T:complexity2", "full vs heavy model"),
           ("repo", "T:complexity2", "repo vs heavy model")]
    for a, b, lab in prs:
        d = pair(a, b); al = pair(a, b, align=True)
        P(f"  {lab:<40} direct med={np.median(d):6.2f} mean={d.mean():6.2f} p95={np.percentile(d,95):6.2f} | "
          f"aligned med={np.median(al):5.2f} | n={len(d)}")

    P("\n" + "=" * 80); P("6. Shared-graph leak: repo vs own fresh_default per clip (6 arm lms, % sw)"); P("=" * 80)
    for c in CLIPS:
        a, oka = load("fresh_default", c); b, okb = load("repo", c); s = sw(a, oka)
        d = np.linalg.norm(a[:, ARM] - b[:, ARM], axis=-1) / s * 100
        P(f"  {c}: median={np.nanmedian(d):6.2f}  first5={np.nanmedian(d[:5]):6.2f}  rest={np.nanmedian(d[5:]):6.2f}  "
          f"last10={np.nanmedian(d[-10:]):6.2f}  max={np.nanmax(d):6.2f}")
        a, oka = load("fresh_raw", c); b, okb = load("shared_raw", c)
        d = np.linalg.norm(a[:, ARM] - b[:, ARM], axis=-1) / s * 100
        P(f"         raw (no filter) fresh vs shared: median={np.nanmedian(d):6.2f} rest={np.nanmedian(d[5:]):6.2f}")


if __name__ == "__main__":
    main()
