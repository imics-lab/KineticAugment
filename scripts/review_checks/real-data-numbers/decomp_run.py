"""Parallel driver for decomp.py: every (mode, root, len, sw) config x clip as a task."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from multiprocessing import Pool
import decomp as D
from scipy.spatial.transform import Rotation

CLIPS = None

def init():
    global CLIPS
    CLIPS = D.load_clips()

def task(args):
    c, ci = args
    mode, ro, lo, wo = c
    name, pose, oks, width = CLIPS[ci]
    rest = D.rest
    clip_len = {}
    for (a, b) in D.REST_LEN:
        v = oks[:, a] & oks[:, b]
        clip_len[(a, b)] = float(np.median(np.linalg.norm(pose[v, b] - pose[v, a], axis=-1)))
    errs, scales, roots, fidx = [], [], [], []
    for fi, (frame, ok) in enumerate(zip(pose, oks)):
        present = [m for m in D.MP if ok[m]]
        if len(present) < 3:
            continue
        frame_len = {k: (np.linalg.norm(frame[k[1]] - frame[k[0]]) if ok[k[0]] and ok[k[1]] else D.REST_LEN[k]) for k in D.REST_LEN}
        lengths = {"rest": D.REST_LEN, "clip": clip_len, "frame": frame_len}[lo]
        sw = {"rest": D.REST_SW, "clip": width, "frame": float(np.linalg.norm(frame[11] - frame[12]))}[wo]
        mid = (rest[16] + rest[17]) / 2
        half = (rest[16] - rest[17]) / 2
        half = half / np.linalg.norm(half) * sw / 2
        shoulders = {11: mid + half, 12: mid - half}
        if ro == "none":
            Rr = np.eye(3)
        elif ro == "sh":
            Rr = D.shoulder_swing(frame, shoulders)
        else:
            Rr, _ = D.opt_root(frame, ok, present, lengths, shoulders, mode, width)
        e, s = D.residual(frame, ok, present, Rr, lengths, shoulders, mode, width)
        errs.append(e); scales.append(s); roots.append(Rotation.from_matrix(Rr).as_rotvec()); fidx.append((fi, present))
    return c, ci, errs, scales, roots, fidx

if __name__ == "__main__":
    modes = sys.argv[1].split(",") if len(sys.argv) > 1 else ["sim", "rigid"]
    configs = [(m, r, l, w) for m in modes for r in ("none", "sh", "opt") for l in ("rest", "clip", "frame") for w in ("rest", "clip", "frame")]
    init()
    tasks = [(c, ci) for c in configs for ci in range(len(CLIPS))]
    tasks.sort(key=lambda t: (t[0][1] != "opt", -len(CLIPS[t[1]][1])))  # long tasks first
    t0 = time.time()
    out = {}
    with Pool(10, initializer=init) as pool:
        for k, (c, ci, errs, scales, roots, fidx) in enumerate(pool.imap_unordered(task, tasks)):
            out[(c, ci)] = (errs, scales, roots, fidx)
            if k % 20 == 0:
                print(f"  {k+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    import pickle
    tag = "_".join(modes)
    with open(Path(__file__).parent / f"decomp_{tag}.pkl", "wb") as fh:
        pickle.dump({"configs": configs, "names": [c[0] for c in CLIPS], "out": out}, fh)
    print("done", time.time() - t0)
