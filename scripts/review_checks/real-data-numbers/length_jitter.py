"""How much do MediaPipe world bone lengths vary frame-to-frame within a clip?
A constant-shape body (any betas) cannot follow this, so it bounds what a per-clip shape fit can explain."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from common import *  # noqa
clips = load_clips()
segs = [(11, 13, "L upper"), (13, 15, "L fore"), (12, 14, "R upper"), (14, 16, "R fore"), (11, 12, "shoulder W")]
print("per-clip |frame length - clip median| (median over frames, % of clip shoulder width) and robust CV (MAD*1.4826/median)")
pooled = {n: [] for *_, n in segs}
for name, pose, oks, width in clips:
    row = []
    for a, b, n in segs:
        v = oks[:, a] & oks[:, b]
        L = np.linalg.norm(pose[v, b] - pose[v, a], axis=-1)
        med = np.median(L)
        dev = np.abs(L - med) / width
        pooled[n] += list(dev)
        row.append(f"{n}: {np.median(dev)*100:5.2f}% (CV {1.4826*np.median(np.abs(L-med))/med*100:4.1f}%, range {L.min():.3f}-{L.max():.3f})")
    print(f"  {name}: " + " | ".join(row))
print("pooled median |dev| % width:", {n: round(float(np.median(v))*100, 2) for n, v in pooled.items()})
