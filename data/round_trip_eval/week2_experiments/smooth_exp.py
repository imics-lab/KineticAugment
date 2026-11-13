"""Temporal smoothing experiment: raw vs EMA on outputs vs centred Gaussian on outputs vs
centred Gaussian on fitted parameters (rotations as quaternions). Exploratory, not project code."""
import dataclasses
import sys
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from kinetic_augment.body_model.converter import fit_body, reconstruct_body  # noqa: E402
from kinetic_augment.body_model.hand_converter import (  # noqa: E402
    fit_hand_shape, fit_hands, reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import HandModel  # noqa: E402
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip  # noqa: E402
from kinetic_augment.body_model.shape import fit_shape  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

J = [11, 12, 13, 14, 15, 16]
ARM_BONES = [(11, 13), (13, 15), (12, 14), (14, 16)]


# ----------------------------------------------------------------------------- filters

def runs(mask):
    """Contiguous [start, stop) runs where mask is True."""
    out, start = [], None
    for t, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = t
        elif not m and start is not None:
            out.append((start, t))
            start = None
    return out


def gauss_runs(x, mask, sigma):
    """Centred Gaussian along axis 0 inside each contiguous run of `mask`."""
    y = x.copy()
    for a, b in runs(mask):
        if b - a >= 3:
            y[a:b] = gaussian_filter1d(x[a:b], sigma, axis=0, mode="nearest")
    return y


def ema_runs(x, mask, alpha):
    y = x.copy()
    for a, b in runs(mask):
        for t in range(a + 1, b):
            y[t] = alpha * x[t] + (1 - alpha) * y[t - 1]
    return y


def smooth_rotvecs(rv, mask, sigma):
    """(T, K, 3) axis-angles: sign-continuous quaternions, Gaussian, renormalise."""
    T, K, _ = rv.shape
    q = Rotation.from_rotvec(rv.reshape(-1, 3)).as_quat().reshape(T, K, 4)
    for t in range(1, T):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    q = gauss_runs(q, mask, sigma)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(T, K, 3)


# ----------------------------------------------------------------------------- fits

def body_fits(P, O, sk):
    G = sk.shaped(fit_shape(P, O, sk))
    fits = []
    for t in range(len(P)):
        try:
            fits.append(fit_body(P[t], O[t], G))
        except ValueError:
            fits.append(None)
    return fits


def body_param_smooth(fits, sigma):
    ok = np.array([f is not None for f in fits])
    T = len(fits)
    rv = np.zeros((T, 22, 3))
    tr = np.zeros((T, 3))
    for t, f in enumerate(fits):
        if f is not None:
            rv[t, 0] = f.global_orient
            rv[t, 1:] = f.body_pose.reshape(21, 3)
            tr[t] = f.transl
    rv = smooth_rotvecs(rv, ok, sigma)
    tr = gauss_runs(tr, ok, sigma)
    out = []
    for t, f in enumerate(fits):
        out.append(None if f is None else dataclasses.replace(
            f, global_orient=rv[t, 0], body_pose=rv[t, 1:].reshape(63), transl=tr[t]))
    return out


def body_recon(fits, sk):
    return np.stack([np.full((33, 3), np.nan) if f is None else reconstruct_body(f, sk)
                     for f in fits])[:, J]


def hand_param_smooth(fits, sigma):
    ok = np.array([f.status == "ok" for f in fits])
    T = len(fits)
    rv = np.zeros((T, 16, 3))
    tr = np.zeros((T, 3))
    for t, f in enumerate(fits):
        if ok[t]:
            rv[t, 0] = f.wrist_orient
            rv[t, 1:] = f.hand_pose.reshape(15, 3)
            tr[t] = f.transl
    rv = smooth_rotvecs(rv, ok, sigma)
    tr = gauss_runs(tr, ok, sigma)
    return [dataclasses.replace(f, wrist_orient=rv[t, 0], hand_pose=rv[t, 1:].reshape(45),
                                transl=tr[t]) if ok[t] else f for t, f in enumerate(fits)]


# ----------------------------------------------------------------------------- metrics

def jitter(x):
    """Median |second difference| over time (x: (T, K, 3)), NaN-aware."""
    acc = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1)
    return np.nanmedian(acc)


def glitches(x):
    """Frames whose largest joint jump exceeds median + 5 robust SD of the clip's jumps."""
    jump = np.nanmax(np.linalg.norm(np.diff(x, axis=0), axis=-1), axis=1)
    med = np.nanmedian(jump)
    mad = 1.4826 * np.nanmedian(np.abs(jump - med))
    return int(np.nansum(jump > med + 5 * mad))


def peak_speed(x):
    v = np.linalg.norm(np.diff(x, axis=0), axis=-1)
    return np.nanpercentile(v, 95)


def lag(ref, x):
    """Frame shift (0..4) minimising the gap between x and ref shifted later (x lags by k)."""
    best, out = np.inf, 0
    for k in range(0, 5):
        gap = np.nanmean(np.linalg.norm(x[k:] - ref[:len(ref) - k], axis=-1))
        if gap < best - 1e-12:
            best, out = gap, k
    return out


def err(x, ref):
    return np.nanmean(np.linalg.norm(x - ref, axis=-1))


def bone_cv(x, bones):
    out = []
    for a, b in bones:
        L = np.linalg.norm(x[:, a] - x[:, b], axis=-1)
        out.append(np.nanstd(L) / np.nanmean(L) * 100)
    return np.nanmean(out)


def variants(raw, mask, param_fn):
    return {
        "raw": raw,
        "EMA out a=.5": ema_runs(raw, mask, 0.5),
        "gauss out s=1.5": gauss_runs(raw, mask, 1.5),
        "gauss param s=1": param_fn(1.0),
        "gauss param s=1.5": param_fn(1.5),
        "gauss param s=2.5": param_fn(2.5),
    }


def report(name, inp, vs, bones, unit, scale):
    print(f"  {name}")
    print(f"    {'variant':<18}{'jitter':>8}{'glitch':>7}{'err vs MP':>10}{'peak spd':>9}{'lag':>5}"
          f"{'bone CV%':>9}   ({unit})")
    print(f"    {'MediaPipe input':<18}{jitter(inp) * scale:8.2f}{glitches(inp):7d}{0:10.2f}"
          f"{peak_speed(inp) * scale:9.2f}{0:5d}{bone_cv(inp, bones):9.2f}")
    for k, x in vs.items():
        print(f"    {k:<18}{jitter(x) * scale:8.2f}{glitches(x):7d}{err(x, inp) * scale:10.2f}"
              f"{peak_speed(x) * scale:9.2f}{lag(inp, x):5d}{bone_cv(x, bones):9.2f}")


if __name__ == "__main__":
    sk = SMPLXSkeleton(model_path="models")
    for path in sys.argv[1:]:
        h = load_hand_clip(path)
        clip = path.split("/")[-1].split("_")[0]
        P = h.pose_world.astype(float)
        O = np.isfinite(P).all(-1) & (h.pose_visibility >= 0.5)
        w = np.nanmedian(np.linalg.norm(P[:, 11] - P[:, 12], axis=-1))
        fits = body_fits(P, O, sk)
        mask = np.array([f is not None for f in fits])
        raw = body_recon(fits, sk)
        inp = np.where(O[..., None], P, np.nan)[:, J]
        print(f"\n{clip}  ({len(P)} frames)")
        report("BODY arms", inp, variants(raw, mask, lambda s: body_recon(body_param_smooth(fits, s), sk)),
               [(0, 2), (2, 4), (1, 3), (3, 5)], "% shoulder width", 100 / w)
        for s, side in enumerate(SIDES):
            fr = np.flatnonzero(h.present[:, s])
            if len(fr) < 8:
                continue
            m = HandModel("mano", side)
            ob = np.full((len(P), 21, 3), np.nan)
            ob[fr] = h.world[fr, s]
            v = np.ones((len(fr), 21), bool)
            hf = fits_h = fit_hands(ob[fr], v, m, fit_hand_shape(ob[fr], v, m))
            full = [None] * len(P)
            hmask = np.zeros(len(P), bool)
            hmask[fr] = True
            # wrist-relative points: the hand's own motion, as the hand metric scores it
            def rebuild(fs):
                out = np.full((len(P), 21, 3), np.nan)
                for k, t in enumerate(fr):
                    out[t] = reconstruct_hand(fs[k], m)
                return out - out[:, :1]
            def param_fn(sig):
                sm = hand_param_smooth_runs(fits_h, fr, sig)
                return rebuild(sm)
            def hand_param_smooth_runs(fs, frames, sig):
                seq = [None] * len(P)
                for k, t in enumerate(frames):
                    seq[t] = fs[k]
                dummy = dataclasses.replace(fs[0], status="missing")
                seq = [f if f is not None else dummy for f in seq]
                sm = hand_param_smooth(seq, sig)
                return [sm[t] for t in frames]
            rel_in = ob - ob[:, :1]
            raw_h = rebuild(hf)
            report(f"{side.upper()} HAND (wrist-relative)", rel_in,
                   variants(raw_h, hmask, param_fn), [(0, 5), (5, 6), (6, 7), (7, 8), (0, 17)],
                   "mm", 1000)
