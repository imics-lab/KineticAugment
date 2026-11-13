#!/usr/bin/env python3
"""A004 / noise-floor cluster: empirical checks.

Run with cwd = snapshot root and PYTHONPATH = snapshot/src.
Reads: snapshot data/test_output (repo npz) and work/noise-floor/extract/<variant>
(re-extractions made by reextract.py with the repo's exact extraction loop).
Writes nothing outside stdout.
"""
import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from kinetic_augment.body_model.alignment import apply_similarity, similarity_transform

W = Path(__file__).resolve().parent
EX = W / "extract"
SNAP = Path.cwd()
CLIPS = ["01986", "01987", "01988", "01991", "01992"]
ARM = [11, 12, 13, 14, 15, 16]
HIPS = [23, 24]
VIS = 0.5

spec = importlib.util.spec_from_file_location("mnf", SNAP / "scripts" / "measure_noise_floor.py")
mnf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnf)


def load(variant, clip):
    p = (SNAP / "data/test_output" if variant == "repo" else EX / variant) / f"{clip}_world.npz"
    d = np.load(p)
    pose = d["landmarks"].astype(np.float64)
    ok = np.isfinite(pose).all(axis=-1) & (d["visibility"] >= VIS)
    return pose, ok


def shoulder_w(pose, ok):
    return mnf.shoulder_width(pose, ok)


def runs(ok, min_len):
    """contiguous True runs (start, end) with length >= min_len"""
    out, s = [], None
    for t, v in enumerate(list(ok) + [False]):
        if v and s is None:
            s = t
        elif not v and s is not None:
            if t - s >= min_len:
                out.append((s, t))
            s = None
    return out


# ---------------------------------------------------------------- 1. estimator variants
def pooled(variant, landmarks, dims_list):
    """Method A/B pooled medians (% sw) using the script's own functions."""
    res = {}
    for key, dims in dims_list.items():
        A, B = [], []
        for c in CLIPS:
            pose, ok_all = load(variant, c)
            sw = shoulder_w(pose, ok_all)
            for j in landmarks:
                tr, ok = pose[:, j], ok_all[:, j]
                if ok.mean() < mnf.MIN_OBSERVED_FRACTION:
                    continue
                a = mnf.method_a(tr, ok, dims) / sw
                b = mnf.method_b(tr, ok, dims) / sw
                if np.isfinite(a):
                    A.append(a)
                if np.isfinite(b):
                    B.append(b)
        res[key] = (np.median(A) * 100, np.median(B) * 100, len(B))
    return res


def section1():
    print("=" * 78)
    print("1. Script estimators by variant, landmark set, and per-axis (pooled medians, % sw)")
    print("=" * 78)
    bands = {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xyz": [0, 1, 2]}
    for variant in ["repo", "nosmooth", "static"]:
        for lname, lms in [("script set (arms+hips)", ARM + HIPS + [25, 26, 27, 28]),
                           ("arms only (scored set)", ARM)]:
            r = pooled(variant, lms, bands)
            s = "  ".join(f"{k}: A={v[0]:.3f} B={v[1]:.3f}" for k, v in r.items()) + f"  (n={r['xyz'][2]})"
            print(f"{variant:<9}{lname:<24}{s}")
            if variant == "repo" and lname.startswith("script"):
                print(f"{'':<33}script's 'z is Nx noisier' = B_z/B_xy = {r['z'][1]/r['xy'][1]:.2f}; "
                      f"per-axis B_z/mean(B_x,B_y) = {r['z'][1]/((r['x'][1]+r['y'][1])/2):.2f}")
    # hips alone
    r = pooled("repo", HIPS, {"xyz": [0, 1, 2]})
    print(f"repo hips alone (hip-centred origin!): B_xyz={r['xyz'][1]:.3f}%  (n={r['xyz'][2]})")


# ---------------------------------------------------------------- 2. autocorrelation
def acf_pool(series_list, maxlag=3):
    num = np.zeros(maxlag + 1)
    den = 0.0
    for x in series_list:
        x = x - x.mean()
        den += (x * x).sum()
        for k in range(maxlag + 1):
            num[k] += (x[:len(x) - k] * x[k:]).sum() if k else (x * x).sum()
    return num / den


def theo_acf_highpass(h, maxlag=3):
    c = np.concatenate([np.asarray(h, float), np.zeros(maxlag + 1)])
    r = [np.dot(c[:len(c) - k], c[k:]) for k in range(maxlag + 1)]
    return np.array(r) / r[0]


def collect(variant, kind, landmarks=ARM, gate=False):
    out = []
    for c in CLIPS:
        pose, ok_all = load(variant, c)
        for j in landmarks:
            ok = ok_all[:, j]
            for s, e in runs(ok, 12):
                seg = pose[s:e, j]
                if gate:
                    sp = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                    g = np.percentile(sp, mnf.MOTION_GATE_PERCENTILE)
                for ax in range(3):
                    x = seg[:, ax]
                    if kind == "d1":
                        y = np.diff(x)
                    elif kind == "d2":
                        y = x[:-2] - 2 * x[1:-1] + x[2:]
                    elif kind == "sg":
                        y = x - savgol_filter(x, 7, 2)
                        y = y[3:-3]  # drop edge-mode samples
                    out.append(y)
    return out


def section2():
    print("\n" + "=" * 78)
    print("2. Autocorrelation of high-pass components (6 arm landmarks x 3 axes, pooled)")
    print("=" * 78)
    # SG(7,2) smoothing kernel for the centre point:
    k = savgol_filter(np.eye(7), 7, 2, axis=0, mode="interp")[3]
    hp = -k.copy(); hp[3] += 1.0
    iid = {"d1": theo_acf_highpass([1, -1]), "d2": theo_acf_highpass([1, -2, 1]),
           "sg": theo_acf_highpass(hp)}
    print(f"{'component':<28}{'variant':<12}{'lag1':>8}{'lag2':>8}{'lag3':>8}")
    for kind, label in [("d1", "1st difference"), ("d2", "2nd difference (Method B's D)"),
                        ("sg", "residual from SG(7,2) fit")]:
        print(f"{label:<28}{'iid theory':<12}" + "".join(f"{v:>8.3f}" for v in iid[kind][1:]))
        for variant in ["repo", "default", "nosmooth", "static"]:
            r = acf_pool(collect(variant, kind))
            print(f"{'':<28}{variant:<12}" + "".join(f"{v:>8.3f}" for v in r[1:]))
    # filter error itself
    print("\nSmoothing displacement (default - nosmooth, same raw predictions):")
    ser = []
    for c in CLIPS:
        a, oka = load("default", c); b, okb = load("nosmooth", c)
        for j in ARM:
            ok = oka[:, j] & okb[:, j]
            for s, e in runs(ok, 12):
                for ax in range(3):
                    ser.append(a[s:e, j, ax] - b[s:e, j, ax])
    r = acf_pool(ser)
    print(f"  acf lag1..3 = {r[1]:.3f} {r[2]:.3f} {r[3]:.3f}")


# ---------------------------------------------------------------- 3. One Euro simulation
class LowPass:
    def __init__(self):
        self.init = False; self.raw = None; self.stored = None

    def apply(self, v, a):
        r = v if not self.init else a * v + (1 - a) * self.stored
        self.init = True; self.raw = v; self.stored = r
        return r


class OneEuro:
    def __init__(self, mc=0.1, beta=40.0, dc=1.0, freq=30.0):
        self.mc, self.beta, self.dc, self.freq = mc, beta, dc, freq
        self.x, self.dx = LowPass(), LowPass()

    def alpha(self, cutoff):
        te = 1.0 / self.freq; tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def apply(self, v):
        dv = (v - self.x.raw) * self.freq if self.x.init else 0.0
        edv = self.dx.apply(dv, self.alpha(self.dc))
        return self.x.apply(v, self.alpha(self.mc + self.beta * abs(edv)))


def filt(track):
    out = np.empty_like(track)
    fs = [OneEuro() for _ in range(track.shape[1])]
    for t in range(track.shape[0]):
        for k in range(track.shape[1]):
            out[t, k] = fs[k].apply(track[t, k])
    return out


def method_b_xyz(track):
    ok = np.ones(len(track), bool)
    return mnf.method_b(track, ok, [0, 1, 2])


def method_a_xyz(track):
    ok = np.ones(len(track), bool)
    return mnf.method_a(track, ok, [0, 1, 2])


def section3():
    print("\n" + "=" * 78)
    print("3. Simulation: iid noise -> Method A/B, with and without the identified One Euro filter")
    print("=" * 78)
    rng = np.random.default_rng(0)
    # truth trajectories: the smoothed real arm tracks (smooth, realistic speeds)
    truths = []
    for c in CLIPS:
        pose, ok_all = load("default", c)
        for j in ARM:
            for s, e in runs(ok_all[:, j], 40):
                truths.append(pose[s:e, j])
    stationary = [np.zeros((80, 3)) + 0.0 for _ in range(len(truths))]
    for sigma_mm in (0.5, 0.9, 2.0):
        sig = sigma_mm / 1000.0
        sig3 = sig * math.sqrt(3)
        for name, base in [("stationary", stationary), ("real-motion", truths)]:
            rb_raw, rb_f, ra_raw, ra_f = [], [], [], []
            for _ in range(4):
                for tr in base:
                    noisy = tr + rng.normal(0, sig, tr.shape)
                    f = filt(noisy)
                    rb_raw.append(method_b_xyz(noisy) / sig3)
                    rb_f.append(method_b_xyz(f) / sig3)
                    ra_raw.append(method_a_xyz(noisy) / sig3)
                    ra_f.append(method_a_xyz(f) / sig3)
            print(f"sigma/axis={sigma_mm:.1f}mm {name:<12} B/true: unfiltered={np.nanmedian(rb_raw):.3f} "
                  f"filtered={np.nanmedian(rb_f):.3f} | A/true: unfiltered={np.nanmedian(ra_raw):.3f} "
                  f"filtered={np.nanmedian(ra_f):.3f}")
    # iid theory for A relative to per-axis sigma, and A/B ratios
    from scipy.stats import chi
    print("\niid theory (stationary, per-axis sigma s): A = sqrt2*s*chi_d.ppf(0.10);  B = sqrt(d)*s")
    for band, d in [("z", 1), ("xy", 2), ("xyz", 3)]:
        A = math.sqrt(2) * chi.ppf(0.10, d); B = math.sqrt(d)
        print(f"  {band:<4} A={A:.3f}s  B={B:.3f}s  A/B={A/B:.3f}")
    print(f"  median 3-D Euclidean error of iid noise = {chi.ppf(0.5, 3):.3f}s = "
          f"{chi.ppf(0.5,3)/math.sqrt(3):.3f} x B_xyz")


# ---------------------------------------------------------------- 4. same-statistic floors
def pair_metric(va, vb, align):
    errs = []
    for c in CLIPS:
        a, oka = load(va, c); b, okb = load(vb, c)
        n = min(len(a), len(b))
        sw = shoulder_w(a, oka)
        for t in range(n):
            pres = [m for m in ARM if oka[t, m] and okb[t, m]]
            if len(pres) < 3:
                continue
            src = np.array([b[t, m] for m in pres]); tgt = np.array([a[t, m] for m in pres])
            if align:
                src = apply_similarity(src, *similarity_transform(src, tgt))
            errs.extend(np.linalg.norm(src - tgt, axis=1) / sw)
    e = np.asarray(errs)
    return np.median(e) * 100, np.percentile(e, 95) * 100, len(e)


def self_residual(variant):
    errs = []
    for c in CLIPS:
        pose, ok_all = load(variant, c)
        sw = shoulder_w(pose, ok_all)
        for j in ARM:
            for s, e in runs(ok_all[:, j], 12):
                seg = pose[s:e, j]
                fit = savgol_filter(seg, 7, 2, axis=0)
                errs.extend((np.linalg.norm(seg - fit, axis=1) / sw)[3:-3])
    return np.median(errs) * 100


def section4():
    print("\n" + "=" * 78)
    print("4. Same statistic as the round-trip score: median per-landmark 3-D Euclidean distance")
    print("   / shoulder width over the 6 arm landmarks, between two detections of the SAME frames")
    print("=" * 78)
    pairs = [("default", "default_rerun", "determinism (same settings, rerun)"),
             ("default", "default_sharedgraph", "fresh graph vs repo's shared graph (= repo npz)"),
             ("nosmooth", "default", "raw vs smoothed (filter displacement)"),
             ("nosmooth", "nosmooth_resize75", "raw vs raw, input downscaled to 75%"),
             ("default", "resize75", "smoothed, input downscaled to 75%"),
             ("default", "padshift", "smoothed, image padded/person shifted"),
             ("default", "bright", "smoothed, brightness +25"),
             ("default", "static", "tracking vs static_image_mode"),
             ("default", "complexity2", "model_complexity 1 vs 2 (full vs heavy)")]
    print(f"{'pair':<52}{'direct med':>11}{'p95':>8}{'aligned med':>13}{'p95':>8}")
    for va, vb, lab in pairs:
        dm, dp, n = pair_metric(va, vb, False)
        am, ap, _ = pair_metric(va, vb, True)
        print(f"{lab:<52}{dm:>10.3f}%{dp:>7.2f}%{am:>12.3f}%{ap:>7.2f}%   n={n}")
    print("\nTemporal self-residual in the same statistic (median ||p_t - SG(7,2) fit_t|| / sw):")
    for v in ["repo", "default", "nosmooth", "static"]:
        print(f"  {v:<10}{self_residual(v):.3f}%")


# ---------------------------------------------------------------- 5. shared graph leakage
def section5():
    print("\n" + "=" * 78)
    print("5. Repo extraction reuses one Holistic graph across clips: per-clip effect")
    print("   (repo npz vs fresh-graph re-extraction, 6 arm landmarks, % sw)")
    print("=" * 78)
    for c in CLIPS:
        a, oka = load("default", c); b, okb = load("repo", c)
        sw = shoulder_w(a, oka)
        d = np.linalg.norm(a[:, ARM] - b[:, ARM], axis=-1) / sw * 100
        fin = np.isfinite(d)
        dd = np.where(fin, d, np.nan)
        first = np.nanmedian(dd[:5]); rest = np.nanmedian(dd[5:])
        print(f"  {c}: median={np.nanmedian(dd):.3f}%  max={np.nanmax(dd):.2f}%  "
              f"first5 med={first:.3f}%  rest med={rest:.3f}%  bitwise-equal={np.array_equal(a, b, equal_nan=True)}")


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
    section5()
