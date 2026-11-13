"""Re-extract each clip with a FRESH Holistic instance into work/missed-bugs/freshroot (never touches snap/data)."""
import importlib.util, sys
from pathlib import Path
import numpy as np
import mediapipe as mp
spec = importlib.util.spec_from_file_location("ewl", "scripts/extract_world_landmarks.py")
ewl = importlib.util.module_from_spec(spec); spec.loader.exec_module(ewl)
out_dir = Path(sys.argv[1])
for v in sorted(Path("data/sign_videos").glob("*.mp4")):
    h = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
    try:
        L, V = ewl.extract(v, h)
    finally:
        h.close()
    np.savez_compressed(out_dir / f"{v.stem}_world.npz", landmarks=L, visibility=V)
    S = np.load(f"data/test_output/{v.stem}_world.npz")
    d = np.nanmax(np.linalg.norm(L[:, 11:17] - S["landmarks"][:, 11:17], axis=-1), axis=1) * 1000
    print(f"  {v.stem}: per-frame max arm diff vs stored: median {np.nanmedian(d):6.2f} mm, last-10-frames median "
          f"{np.nanmedian(d[-10:]):6.2f} mm; visibility>=0.5 agreement on wrists "
          f"{100*np.mean((V[:, 15:17] >= .5) == (S['visibility'][:, 15:17] >= .5)):.0f}%")
