"""Independent helpers: use the real smplx torch forward (not a hand-rolled linear model)."""
import os
import numpy as np
import torch
import smplx

SNAP = "."
MODELS = os.path.join(SNAP, "models")
torch.set_default_dtype(torch.float64)

_cache = {}
def model(nb=10):
    if nb not in _cache:
        m = smplx.create(model_path=MODELS, model_type="smplx", gender="neutral",
                         use_pca=False, flat_hand_mean=True, batch_size=1, num_betas=nb)
        m = m.double()
        _cache[nb] = m
    return _cache[nb]

SEG = [(16, 18, "L upper"), (18, 20, "L fore"), (17, 19, "R upper"), (19, 21, "R fore"), (16, 17, "shoulder W")]

def forward(betas, body_pose=None, global_orient=None, transl=None, nb=None, verts=False):
    betas = torch.as_tensor(np.asarray(betas, float)).reshape(1, -1)
    nb = betas.shape[1] if nb is None else nb
    m = model(nb)
    kw = dict(betas=betas, return_verts=verts)
    if body_pose is not None: kw["body_pose"] = torch.as_tensor(np.asarray(body_pose, float)).reshape(1, 63)
    if global_orient is not None: kw["global_orient"] = torch.as_tensor(np.asarray(global_orient, float)).reshape(1, 3)
    if transl is not None: kw["transl"] = torch.as_tensor(np.asarray(transl, float)).reshape(1, 3)
    out = m(**kw)
    return out

def seg_lengths_t(J):
    return torch.stack([torch.linalg.norm(J[c] - J[p]) for p, c, _ in SEG])

def lengths(betas, body_pose=None):
    with torch.no_grad():
        J = forward(betas, body_pose).joints[0, :55]
    return seg_lengths_t(J).numpy()

def length_jacobian(betas, body_pose=None):
    b = torch.tensor(np.asarray(betas, float), requires_grad=True)
    def f(bb):
        m = model(len(betas))
        kw = dict(betas=bb.reshape(1, -1), return_verts=False)
        if body_pose is not None:
            kw["body_pose"] = torch.as_tensor(np.asarray(body_pose, float)).reshape(1, 63)
        J = m(**kw).joints[0, :55]
        return seg_lengths_t(J)
    return torch.autograd.functional.jacobian(f, b).numpy()

def verts(betas):
    with torch.no_grad():
        return forward(betas, verts=True).vertices[0].numpy()

def height(betas):
    V = verts(betas)
    return V[:, 1].max() - V[:, 1].min()

def load_clips():
    import glob
    out = {}
    for f in sorted(glob.glob(os.path.join(SNAP, "data/test_output/*_world.npz"))):
        d = np.load(f)
        out[os.path.basename(f)[:5]] = (d["landmarks"].astype(np.float64), d["visibility"].astype(np.float64))
    return out

MP_SEG = [(11, 13, "L upper"), (13, 15, "L fore"), (12, 14, "R upper"), (14, 16, "R fore"), (11, 12, "shoulder W")]
