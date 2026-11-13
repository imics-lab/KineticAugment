"""Side-by-side video: source | raw SMPL-X figure | parameter-smoothed SMPL-X figure, with traces.
Usage: render_smooth.py <hands.npz> <video> <out.mp4> [sigma]. Exploratory, not project code."""
import dataclasses
import types
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
from smooth_exp import body_param_smooth, hand_param_smooth  # noqa: E402
from view_body_hand_round_trip import (  # noqa: E402
    BodyFigure, anchored, draw_mesh, limits, overlay_mediapipe, set_view,
)
from view_round_trip import VideoSink, read_frames  # noqa: E402
from kinetic_augment.body_model.converter import fit_body, reconstruct_body  # noqa: E402
from kinetic_augment.body_model.hand_converter import (  # noqa: E402
    fit_hand_shape, fit_hands, reconstruct_hand,
)
from kinetic_augment.body_model.hand_model import HandModel  # noqa: E402
from kinetic_augment.body_model.hand_observations import SIDES, load_hand_clip  # noqa: E402
from kinetic_augment.body_model.shape import fit_shape  # noqa: E402
from kinetic_augment.body_model.skeleton import SMPLXSkeleton  # noqa: E402

hands_path, video, out_path = sys.argv[1:4]
sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
sk = SMPLXSkeleton(model_path="models")
h = load_hand_clip(hands_path)
P = h.pose_world.astype(float)
O = np.isfinite(P).all(-1) & (h.pose_visibility >= 0.5)
T = len(P)
frames, fps = read_frames("", T, video)

G = sk.shaped(fit_shape(P, O, sk))
fits = []
for t in range(T):
    try:
        fits.append(fit_body(P[t], O[t], G))
    except ValueError:
        fits.append(None)
versions = {"raw": {"body": fits}, "smoothed": {"body": body_param_smooth(fits, sigma)}}

POSE_WRIST = {"left": 15, "right": 16}
for s, side in enumerate(SIDES):
    m = HandModel("mano", side)
    fr = np.flatnonzero(h.present[:, s])
    raw_seq = [None] * T
    if len(fr) >= 2:
        ob = h.world[fr, s].astype(float)
        v = np.ones(ob.shape[:2], bool)
        for k, f in zip(fr, fit_hands(ob, v, m, fit_hand_shape(ob, v, m))):
            raw_seq[k] = f
    template = next((f for f in raw_seq if f is not None), None)
    filled = [f if f is not None else dataclasses.replace(template, status="missing")
              for f in raw_seq] if template is not None else None
    for name in versions:
        if filled is None:
            seq = [None] * T
        else:
            seq = filled if name == "raw" else hand_param_smooth(filled, sigma)
        versions[name][side] = seq
        rec = np.full((T, 21, 3), np.nan)
        for t, f in enumerate(seq):
            if f is not None and f.status == "ok":
                rec[t] = reconstruct_hand(f, m)
        versions[name][side + "_rec"] = rec

figure = BodyFigure(sk)
MISSING = types.SimpleNamespace(status="missing", hand_pose=np.zeros(45))
meshes = {}
traces = {}
for name, v in versions.items():
    meshes[name] = []
    body_rec = np.stack([np.full((33, 3), np.nan) if f is None else reconstruct_body(f, sk)
                         for f in v["body"]])
    traces[name] = {"wrist": body_rec[:, 16, 1],
                    "tip": np.linalg.norm(v["right_rec"][:, 8] - v["right_rec"][:, 0], axis=-1)}
    for t in range(T):
        if v["body"][t] is None:
            meshes[name].append(None)
            continue
        hf = {side: v[side][t] if v[side][t] is not None else MISSING for side in SIDES}
        hr = {side: v[side + "_rec"][t] for side in SIDES}
        verts, _, _ = figure.pose(v["body"][t], hf, hr)
        meshes[name].append(verts)
inp_tip = np.linalg.norm(h.world[:, 1, 8] - h.world[:, 1, 0], axis=-1)
inp_wrist = np.where(O[:, 16], P[:, 16, 1], np.nan)

crop = [m[m[:, 1] < m[:, 1].min() + 0.8] for m in meshes["raw"] if m is not None]  # head to waist
lim = limits([np.concatenate(crop)], "front", pad=1.05)

sink = VideoSink(out_path, fps)
fig = plt.figure(figsize=(16, 7.6), dpi=100)
steps = np.arange(1, T + 1)
for t in range(T):
    fig.clf()
    g = fig.add_gridspec(3, 3, height_ratios=[4.2, 1, 1], hspace=0.35, wspace=0.05,
                         left=0.04, right=0.98, top=0.94, bottom=0.06)
    ax_v = fig.add_subplot(g[0, 0])
    ax_v.imshow(overlay_mediapipe(frames[t], h, t))
    ax_v.set_title(f"frame {t + 1}/{T}  (MediaPipe)", fontsize=10)
    ax_v.axis("off")
    for i, name in enumerate(("raw", "smoothed")):
        ax = fig.add_subplot(g[0, i + 1])
        if meshes[name][t] is not None:
            draw_mesh(ax, meshes[name][t], figure.faces)
        set_view(ax, lim, "raw round trip" if name == "raw"
                 else f"smoothed parameters (centred Gaussian, sigma={sigma:g} frames)")
    for row, (key, inp, label, scale) in enumerate((
            ("wrist", inp_wrist, "R wrist height (cm, up)", -100),
            ("tip", inp_tip, "R index tip to wrist (mm)", 1000))):
        ax = fig.add_subplot(g[row + 1, :])
        ax.plot(steps, inp * scale, color="#1f5fbf", lw=1, alpha=0.6, label="MediaPipe")
        ax.plot(steps, traces["raw"][key] * scale, color="#d62728", lw=1.2, label="raw")
        ax.plot(steps, traces["smoothed"][key] * scale, color="#2ca02c", lw=1.6, label="smoothed")
        ax.axvline(t + 1, color="k", lw=0.8)
        ax.set_xlim(1, T)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7, ncol=3)
    sink.add(fig)
sink.close()
print("wrote", out_path)
