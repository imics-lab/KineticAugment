"""Doc Finding 5 says hips 'Observed in frame' 32%/32%. Reproduce from the image-space caches."""
import glob, numpy as np
L, Rr, Lw, Rw, n = [], [], [], [], 0
for f in sorted(glob.glob("data/test_output/*_landmarks.npy")):
    a = np.load(f)[:, 23:25, :]
    zero = (a == 0).all(-1)
    inside = ((a[..., :2] >= 0) & (a[..., :2] <= 1)).all(-1) & ~zero
    w = np.load(f.replace("_landmarks.npy", "_world.npz"))["visibility"][:, 23:25] >= 0.5
    print(f[-18:-13], "frames", len(a), "zero-sentinel", zero.sum(0), "in-frame L/R %.3f/%.3f" % tuple(inside.mean(0)),
          "| world vis>=0.5 L/R %.3f/%.3f" % tuple(w.mean(0)))
    L.append(inside[:, 0].mean()); Rr.append(inside[:, 1].mean()); Lw.append(w[:, 0].mean()); Rw.append(w[:, 1].mean())
print("mean over clips: in-frame L %.3f R %.3f | visibility L %.3f R %.3f" % (np.mean(L), np.mean(Rr), np.mean(Lw), np.mean(Rw)))
