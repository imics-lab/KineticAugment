"""Per-frame error ladder for every clip: body (global / per-frame plausible / floor) and hands
(per-track shape / per-frame shape), plus input and output jitter. Exploratory, not project code."""
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from measure_round_trip import floor_skeleton  # noqa: E402
from kinetic_augment.body_model.converter import fit_body, reconstruct_body  # noqa: E402
from kinetic_augment.body_model.hand_converter import (  # noqa: E402
    fit_hand_shape, fit_hands, fit_hands_per_frame_shape, landmark_errors_mm, reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import HandModel  # noqa: E402
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip  # noqa: E402
from kinetic_augment.body_model.shape import fit_shape  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
J = [11, 12, 13, 14, 15, 16]
sk = SMPLXSkeleton(model_path="models")
hand_models = {s: HandModel("mano", s) for s in SIDES}
CLIPS = [("data/test_output", c) for c in ("01986", "01987", "01988", "01991", "01992")]
CLIPS += [("data/round_trip_eval", p.name.split("_")[0])
          for p in sorted(Path("data/round_trip_eval").glob("*_hands.npz"))]


def recon(p, o, fit_with, rec_with):
    try:
        return reconstruct_body(fit_body(p, o, fit_with), rec_with)
    except ValueError:
        return np.full((33, 3), np.nan)


def joint_err(r, p, o, w):
    e = np.full(len(J), np.nan)
    for k, m in enumerate(J):
        if o[m] and np.isfinite(r[m]).all():
            e[k] = np.linalg.norm(r[m] - p[m]) / w * 100
    return e


for d, c in CLIPS:
    h = load_hand_clip(f"{d}/{c}_hands.npz")
    P = h.pose_world.astype(float)
    O = np.isfinite(P).all(-1) & (h.pose_visibility >= 0.5)
    T = len(P)
    w = np.nanmedian(np.linalg.norm(P[:, 11] - P[:, 12], axis=-1))
    G = sk.shaped(fit_shape(P, O, sk))
    fl = floor_skeleton(sk, P, O)
    res = {k: np.full((T, len(J)), np.nan) for k in ("global", "perframe", "floor")}
    rec_global = np.full((T, 33, 3), np.nan)
    for t in range(T):
        rec_global[t] = recon(P[t], O[t], G, sk)
        res["global"][t] = joint_err(rec_global[t], P[t], O[t], w)
        B = sk.shaped(fit_shape(P[t:t + 1], O[t:t + 1], sk))
        res["perframe"][t] = joint_err(recon(P[t], O[t], B, sk), P[t], O[t], w)
        res["floor"][t] = joint_err(recon(P[t], O[t], fl, fl), P[t], O[t], w)
    out = {f"body_{k}": v for k, v in res.items()}
    out["body_input"] = np.where(O[..., None], P, np.nan)[:, J]
    out["body_recon"] = rec_global[:, J]
    out["shoulder_width"] = np.array(w)

    for s, side in enumerate(SIDES):
        g = np.full((T, 20), np.nan)
        pf = np.full((T, 20), np.nan)
        frames = np.flatnonzero(h.present[:, s])
        if len(frames) >= 2:
            ob = h.world[frames, s].astype(float)
            v = np.ones(ob.shape[:2], bool)
            m = hand_models[side]
            fits = fit_hands(ob, v, m, fit_hand_shape(ob, v, m))
            g[frames] = landmark_errors_mm(np.stack([reconstruct_hand(f, m) for f in fits]), ob)
            pff = fit_hands_per_frame_shape(ob, v, m)
            pf[frames] = landmark_errors_mm(np.stack([reconstruct_hand(f, m) for f in pff]), ob)
        out[f"hand_{side}_global"] = g
        out[f"hand_{side}_perframe"] = pf
    np.savez(OUT / f"{c}.npz", **out)
    gb, pb = np.nanmean(res["global"]), np.nanmean(res["perframe"])
    print(f"{c}: body global {gb:.2f} per-frame {pb:.2f} floor {np.nanmean(res['floor']):.2f}",
          flush=True)
