import numpy as np
from vcommon import *

np.set_printoptions(precision=4, suppress=True, linewidth=160)
rng = np.random.default_rng(123)

# 0) pose invariance of the 5 lengths (collars zero) vs collar motion
b0 = np.zeros(10)
L_rest = lengths(b0)
bp = np.zeros((21, 3))
for j in (16, 17, 18, 19):  # shoulders/elbows in body_pose index j-1
    bp[j - 1] = rng.normal(0, 0.5, 3)
L_pose = lengths(b0, bp.reshape(63))
print("rest lengths (m)", L_rest)
print("max |dL| arm-only pose (mm)", np.abs(L_pose - L_rest).max() * 1e3)
bp2 = bp.copy(); bp2[13 - 1] = [0, 0, 0.2]
print("0.2 rad L collar (z): shoulder W", L_rest[4]*1e3, "->", lengths(b0, bp2.reshape(63))[4]*1e3, "mm")
for ax in range(3):
    bp3 = np.zeros((21, 3)); bp3[13 - 1, ax] = 0.2
    print(f"  collar axis {ax}: dSW = {(lengths(b0, bp3.reshape(63))[4]-L_rest[4])*1e3:.2f} mm")
bp4 = np.zeros((21,3)); bp4[2] = [0.2, 0.1, 0.0]; bp4[5] = [0.1, -0.2, 0.1]; bp4[8]=[0.1,0.1,0.1]
print("spine-only bend: max |dL| (mm)", np.abs(lengths(b0, bp4.reshape(63)) - L_rest).max()*1e3)

# 1) Jacobian via autograd on real smplx, mm per unit beta
def report(J, tag):
    s = np.linalg.svd(J, compute_uv=False)
    print(f"{tag}: shape {J.shape} sv(mm/beta)={s*1e3} rank(tol 1e-9 m)={np.sum(s>1e-9)} cond={s[0]/s[-1]:.0f}")
    return s
J0 = length_jacobian(b0)
s0 = report(J0, "betas=0, 10 betas")
print("J (mm per beta):\n", J0 * 1e3)
# finite difference check
eps = 1e-4
fd = np.stack([(lengths(b0 + eps*np.eye(10)[k]) - lengths(b0 - eps*np.eye(10)[k]))/(2*eps) for k in range(10)], 1)
print("autograd vs FD max diff (mm/beta):", np.abs(fd - J0).max()*1e3)
# Jacobian pose invariance
Jp = length_jacobian(b0, bp.reshape(63))
print("Jacobian change under arm pose (mm/beta max):", np.abs(Jp - J0).max()*1e3)

# with scale column (per unit scale = L itself), and per 1% scale
Js = np.hstack([J0, L_rest[:, None] * 0.01])
report(Js, "betas=0, 10 betas + scale(per 1%)")
# random betas
for i in range(6):
    sd = 1.0 if i < 3 else 2.0
    b = rng.normal(0, sd, 10)
    report(length_jacobian(b), f"betas~N(0,{sd}^2) #{i}")

# 4 arm lengths only
report(J0[:4], "4 arm lengths, 10 betas")

# L/R duplication
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print("cos(L up,R up)=%.5f cos(L fo,R fo)=%.5f" % (cos(J0[0], J0[2]), cos(J0[1], J0[3])))
print("norm of L-R rows (mm per unit-norm beta): upper %.3f fore %.3f ; means: upper %.3f fore %.3f" % (
    np.linalg.norm(J0[0]-J0[2])*1e3, np.linalg.norm(J0[1]-J0[3])*1e3,
    np.linalg.norm((J0[0]+J0[2])/2)*1e3, np.linalg.norm((J0[1]+J0[3])/2)*1e3))

# effective dof with prior N(0,I), isotropic length noise sigma
for sig in (1, 3, 5, 10, 20):
    s = s0 * 1e3
    print(f"sigma_L={sig} mm: eff dof = {np.sum(s**2/(s**2+sig**2)):.2f}")

# scale "in span": trivially true since rank 5 = #rows
x, *_ = np.linalg.lstsq(J0, L_rest*0.01, rcond=None)
print("min-norm beta for +1%% scale: |beta|=%.3f  beta=%s resid=%.2e mm" % (np.linalg.norm(x), x, np.linalg.norm(J0@x - L_rest*0.01)*1e3))
print("cos(J[:,0], L_rest) =", cos(J0[:, 0], L_rest))
