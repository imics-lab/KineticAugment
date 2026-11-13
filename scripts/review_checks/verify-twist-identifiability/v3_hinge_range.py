import numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models"); r = sk.rest_joints
for side, (S, E, W) in {"L": (16, 18, 20), "R": (17, 19, 21)}.items():
    u = r[E]-r[S]; u /= np.linalg.norm(u); f = r[W]-r[E]; f /= np.linalg.norm(f)
    print(side, "rest u", np.round(u, 3), "f", np.round(f, 3), "angle %.2f" % np.degrees(np.arccos(u@f)))
    phis = np.radians(np.linspace(-180, 180, 36001))
    for name, ax in [("local y", np.array([0, 1., 0])), ("local z", np.array([0, 0, 1.])),
                     ("normal(u,f)", np.cross(u, f)/np.linalg.norm(np.cross(u, f)))]:
        ang = np.degrees(np.arccos(np.clip([u @ R.from_rotvec(p*ax).apply(f) for p in phis], -1, 1)))
        print(f"   hinge about {name:<12}: bone-angle range {ang.min():6.2f} .. {ang.max():6.2f} deg"
              f"  (u.ax={u@ax:+.3f}, f.ax={f@ax:+.3f})")
