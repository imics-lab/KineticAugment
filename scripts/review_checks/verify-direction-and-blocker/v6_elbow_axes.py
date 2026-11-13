import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial.transform import Rotation as R
from kinetic_augment.body_model.skeleton import SMPLXSkeleton
sk = SMPLXSkeleton(model_path="models"); J = sk.rest_joints
for name, e, w in (("left", 18, 20), ("right", 19, 21)):
    d = (J[w] - J[e]) / np.linalg.norm(J[w] - J[e])
    print(name, "rest forearm dir", d.round(3), "angle to x-axis %.1f deg" % np.degrees(np.arccos(abs(d[0]))))
    for ax in range(3):
        for deg in (60, -60):
            v = np.zeros(3); v[ax] = np.radians(deg)
            nd = R.from_rotvec(v).apply(d)
            print("   rot %+d deg about local %s: forearm moves %.1f deg, new dir %s" % (deg, "xyz"[ax], np.degrees(np.arccos(np.clip(nd.dot(d), -1, 1))), nd.round(2)))
