"""Independent check of A001: repo FK vs real SMPL-X forward, all 55 joints,
nonzero betas / root / transl / hands / jaw / eyes, float32 and float64 models.
Model built independently of SMPLXSkeleton.model."""
import warnings, numpy as np, torch, smplx
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
from kinetic_augment.body_model.kinematics import forward_kinematics
warnings.filterwarnings("ignore")

def mk(dtype):
    return smplx.create("models", model_type="smplx", gender="neutral", use_pca=False,
                        flat_hand_mean=True, batch_size=1, dtype=dtype)

m32, m64 = mk(torch.float32), mk(torch.float64)
rng = np.random.default_rng(12345)

def smplx_joints(m, dtype, rots, betas, transl, expr=None):
    T = lambda a: torch.tensor(np.asarray(a), dtype=dtype)[None]
    kw = dict(global_orient=T(rots[0]), body_pose=T(rots[1:22].ravel()),
              jaw_pose=T(rots[22]), leye_pose=T(rots[23]), reye_pose=T(rots[24]),
              left_hand_pose=T(rots[25:40].ravel()), right_hand_pose=T(rots[40:55].ravel()),
              betas=T(betas), transl=T(transl), return_verts=False)
    if expr is not None: kw["expression"] = T(expr)
    with torch.no_grad():
        return m(**kw).joints[0, :55].double().numpy()

res = {"f32": [], "f64": [], "f32_22": [], "large": [], "shape_blind": [], "expr": []}
skel_cache = {}
for trial in range(60):
    betas = rng.normal(0, 1.0 + (trial % 3), 10)        # |b| ~ 3..9
    key = trial // 10
    skel = SMPLXSkeleton(model_path="models", betas=betas)
    rots = rng.normal(0, 0.4, (55, 3)); rots[0] = rng.normal(0, 1.5, 3)
    transl = rng.normal(0, 1.5, 3)
    ours = forward_kinematics(rots, skel) + transl
    j32 = smplx_joints(m32, torch.float32, rots, betas, transl)
    j64 = smplx_joints(m64, torch.float64, rots, betas, transl)
    res["f32"].append(np.abs(ours - j32).max()); res["f64"].append(np.abs(ours - j64).max())
    res["f32_22"].append(np.abs(ours[:22] - j32[:22]).max())
    # shape-blind control: neutral skeleton vs shaped model
    if trial < 5:
        neutral = SMPLXSkeleton(model_path="models")
        res["shape_blind"].append(np.linalg.norm(forward_kinematics(rots, neutral) + transl - j64, axis=1).max())
        expr = rng.normal(0, 2.0, 10)
        jex = smplx_joints(m64, torch.float64, rots, betas, transl, expr)
        res["expr"].append(np.linalg.norm(jex - j64, axis=1).max())
    # large angles
    big = rng.normal(0, 1, (55, 3)); big /= np.linalg.norm(big, axis=1, keepdims=True)
    big *= rng.uniform(2.5, 3.12, (55, 1))
    o2 = forward_kinematics(big, skel) + transl
    res["large"].append(np.abs(o2 - smplx_joints(m64, torch.float64, big, betas, transl)).max())

for k, v in res.items():
    v = np.array(v); print(f"{k:12s} n={len(v):3d} max={v.max():.3e} median={np.median(v):.3e}")

# single pose repro of review's 1.37e-7 style number: betas=0, random pose sigma 0.3
s0 = SMPLXSkeleton(model_path="models")
singles = []
for seed in range(20):
    r = np.random.default_rng(seed); rots = np.zeros((55, 3)); rots[0] = r.normal(0, .3, 3); rots[1:22] = r.normal(0, .3, (21, 3))
    singles.append(np.abs(forward_kinematics(rots, s0) - smplx_joints(m32, torch.float32, rots, np.zeros(10), np.zeros(3))).max())
print("betas=0 sigma0.3 20 poses, all 55 joints f32: max=%.3e min=%.3e" % (max(singles), min(singles)))

# >10 betas
try:
    SMPLXSkeleton(model_path="models", betas=np.zeros(16)); print("16 betas: OK")
except Exception as e:
    print("16 betas:", type(e).__name__, str(e)[:120])
for g in ("male", "female"):
    try:
        SMPLXSkeleton(model_path="models", gender=g); print(g, "OK")
    except Exception as e:
        print(g, type(e).__name__, str(e)[:100])
