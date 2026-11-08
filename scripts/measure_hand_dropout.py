#!/usr/bin/env python3
"""
How often do real MediaPipe hand detections drop out in the middle of a clip?

Over the cached `data/round_trip_eval/*_hands.npz`, per hand, an interior run is a stretch of
undetected frames between that hand's first and last detection. Prints the interior run count,
the median / 90th percentile / max run length, and run starts per detected interior hand-frame
(the numbers `temporal.hand_dropout` defaults are set from).

Usage:
    python scripts/measure_hand_dropout.py
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kinetic_augment.body_model.hand_observations import load_hand_clip  # noqa: E402

DATA = REPO / "data" / "round_trip_eval"


def interior_runs(present):
    """Lengths of the undetected runs between the first and last detection of one hand."""
    index = np.flatnonzero(present)
    if len(index) < 2:
        return [], 0
    inside = present[index[0] : index[-1] + 1]
    lengths, run = [], 0
    for seen in inside:
        if seen:
            if run:
                lengths.append(run)
            run = 0
        else:
            run += 1
    return lengths, int(inside.sum())


def main():
    paths = sorted(DATA.glob("*_hands.npz"))
    if not paths:
        print(f"no *_hands.npz under {DATA}")
        return 1
    lengths, detected = [], 0
    for path in paths:
        present = load_hand_clip(path).present
        for hand in range(present.shape[1]):
            runs, count = interior_runs(present[:, hand])
            lengths += runs
            detected += count
    print(f"clips: {len(paths)}")
    print(f"interior runs: {len(lengths)}")
    if lengths:
        print(f"run length median: {np.median(lengths):g}")
        print(f"run length p90: {np.percentile(lengths, 90):g}")
        print(f"run length max: {max(lengths)}")
    print(f"detected interior hand-frames: {detected}")
    print(f"run starts per detected interior hand-frame: {len(lengths) / max(detected, 1):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
