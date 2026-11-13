"""Check presentation_one.md section 4 numbers on the cached image-space data."""
import glob
import numpy as np

DIMS = {"01986": (320, 240), "01987": (1280, 720), "01988": (640, 480), "01991": (288, 192), "01992": (288, 192)}
BONES = {"thigh": (23, 25), "shoulders": (11, 12), "upper arm": (11, 13), "forearm": (13, 15)}
img = np.load("data/test_output/01986_landmarks.npy")[:, :33].astype(np.float64)
print("01986 image-space bone-length variation (normalised coords as stored):")
for name, (a, b) in BONES.items():
    for dims, lab in [((0, 1), "xy"), ((0, 1, 2), "xyz")]:
        L = np.linalg.norm(img[:, b, dims] - img[:, a, dims], axis=1)
        print(f"  {name:10s} {lab:3s}: CV {L.std()/L.mean()*100:5.1f}%  (max-min)/mean {(L.max()-L.min())/L.mean()*100:5.1f}%  (p95-p5)/median {(np.percentile(L,95)-np.percentile(L,5))/np.median(L)*100:5.1f}%")
# right side variants
for name, (a, b) in {"R thigh": (24, 26), "R upper": (12, 14), "R fore": (14, 16)}.items():
    for dims, lab in [((0, 1), "xy"), ((0, 1, 2), "xyz")]:
        L = np.linalg.norm(img[:, b, dims] - img[:, a, dims], axis=1)
        print(f"  {name:10s} {lab:3s}: CV {L.std()/L.mean()*100:5.1f}%")

w = np.load("data/test_output/01986_world.npz")
print("01986 legs: image y of L hip/knee min", img[:, 23, 1].min().round(3), img[:, 25, 1].min().round(3),
      "| frac frames with hip y>1:", (img[:, 23, 1] > 1).mean(), "knee y>1:", (img[:, 25, 1] > 1).mean())
print("   world visibility mean: L hip", w["visibility"][:, 23].mean().round(3), "L knee", w["visibility"][:, 25].mean().round(3))

print("frame-0 pose-block wrist z vs hand-block wrist z, per clip (pose L/R wrist 15/16; hands at 501/522):")
for clip, (W, H) in DIMS.items():
    a = np.load(f"data/test_output/{clip}_landmarks.npy").astype(np.float64)
    print(f"  {clip}: pose z15 {a[0,15,2]:.3f} z16 {a[0,16,2]:.3f} | blocks 501 z {a[0,501,2]:.3f} 522 z {a[0,522,2]:.3f} | 1.196*W = {1.196*W:.0f}px")

print("fore/upper ratio aggregates on image data (xy / xyz):")
allr = {"xy": [], "xyz": []}
for clip in DIMS:
    a = np.load(f"data/test_output/{clip}_landmarks.npy")[:, :33].astype(np.float64)
    a = a[np.any(a != 0, axis=(1, 2))]
    for dims, lab in [((0, 1), "xy"), ((0, 1, 2), "xyz")]:
        up = np.r_[np.linalg.norm(a[:, 13, dims] - a[:, 11, dims], axis=1), np.linalg.norm(a[:, 14, dims] - a[:, 12, dims], axis=1)]
        fo = np.r_[np.linalg.norm(a[:, 15, dims] - a[:, 13, dims], axis=1), np.linalg.norm(a[:, 16, dims] - a[:, 14, dims], axis=1)]
        allr[lab].append((up, fo))
        print(f"  {clip} {lab}: median(fo)/median(up) {np.median(fo)/np.median(up):.3f}, mean ratio {np.mean(fo/up):.3f}, median ratio {np.median(fo/up):.3f}")
for lab in allr:
    up = np.concatenate([u for u, _ in allr[lab]]); fo = np.concatenate([f for _, f in allr[lab]])
    print(f"  pooled {lab}: median/median {np.median(fo)/np.median(up):.3f}, mean/mean {fo.mean()/up.mean():.3f}, mean ratio {np.mean(fo/up):.3f}, median ratio {np.median(fo/up):.3f}")
