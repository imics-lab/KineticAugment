# Hand pipeline variants: one shape per track, with and without a per-frame scale

**Date:** 2026-10-29. **Why:** the round trip can absorb MediaPipe's drift of hand size into a
per-frame scale ([V1.2 findings §10](V1_2_Findings.md)), and that brings it inside the provisional
targets. Whether a downstream sign classifier is better served by the hand that keeps the drift
or by the hand that does not is not known. Both pipelines are saved so a classifier can decide.

## 1. The two pipelines

Both fit genuine MANO parameters to MediaPipe's 21 hand world landmarks: one shape (10 betas) per
hand track, and per frame a wrist rotation and 15 finger rotations (48 values). Both rebuild the
hand from those values alone and put the model's wrist on MediaPipe's wrist. A *track* is one
side of one clip.

| | `track_shape` | `track_shape_scale` |
| --- | --- | --- |
| What it is | The V1.2 pipeline as it ran: the shape is fitted once over every frame of the track, then each frame's pose at that shape | The same, plus one scale per frame about the wrist |
| Values per frame | 48 rotations | 48 rotations + 1 scale |
| Values per track | 10 betas | 10 betas |
| Hand size over a clip | Fixed. MediaPipe's drift of size is left as error | Follows MediaPipe through the scale |
| How the shape is fitted | Jointly with every frame's pose, inside the plausibility bound | The same, with the scales held to a geometric mean of 1, so the shape carries the hand's typical size and the scales only its drift |
| How the poses are fitted | At the fixed shape | At the fixed shape, each frame with its own free scale |
| Is it a MANO / SMPL-X hand? | Yes | The shape and pose are. The scale is not a model parameter but a residual stored beside them |

From `track_shape_scale` two outputs are saved:

- **`track_shape_scale`** replays each frame's scale: the hand closest to what MediaPipe reported.
- **`track_shape_scale_clean`** drops the scale (scale 1): one size per track, with the shape
  from the scale fit and the poses fitted alongside the scale. This is not the same as
  `track_shape`. Both have one size per track, but in `track_shape` the poses were fitted without
  a scale, so they bend the fingers to make up some of the size drift. In the clean output the
  poses never had to, because the scale took the drift.

So the three outputs differ only in how hand size behaves:

| Output | Size within a clip | What absorbed MediaPipe's drift |
| --- | --- | --- |
| `track_shape` | constant | the error, and partly the finger poses |
| `track_shape_scale` | follows MediaPipe | the scale, replayed |
| `track_shape_scale_clean` | constant | the scale, then discarded |

## 2. How well each reproduces MediaPipe

Pooled mean wrist-relative error against MediaPipe, MANO, in mm (target ≤ 5). From
`data/hand_variants/<set>/manifest.json`.

| Output | Random 20 (26 tracks) | Pilot 5 (9 tracks) |
| --- | ---: | ---: |
| `track_shape` | 5.10 | 7.38 |
| `track_shape_scale` | 3.84 | 4.16 |
| `track_shape_scale_clean` | 7.89 | 14.15 |

`track_shape` reproduces the earlier numbers exactly (ledger P22: 5.10 mm; findings §2: 7.38 mm),
so the export is the pipeline as measured before. Per clip, `track_shape_scale` is the closest to
MediaPipe on all 25 clips.

The clean output sits furthest from MediaPipe, by design: it keeps one size while MediaPipe's
changes, and its poses do not bend the fingers to make up for it. On the pilot, where the drift
is largest (up to 55% between the halves of a clip), it is 14.15 mm. A larger error against
MediaPipe here does not mean a worse hand; it may mean a steadier one. That is the question for
the classifier.

## 3. Files

`scripts/export_hand_variants.py` writes one file per clip, `data/hand_variants/<set>/<clip>_hand_variants.npz`,
and a `manifest.json` with the input hashes and per-clip errors. Saved sets:

- `data/hand_variants/round_trip_eval/`: the random 20 WLASL clips (seed 20261007; ledger P21–P22).
- `data/hand_variants/test_output/`: the five pilot clips.

Every landmark array is laid out like `HandClip.world`: `(T, 2, 21, 3)` metres, side 0 = the
signer's left and 1 = their right, NaN where that side has no hand. `T` is every decoded frame of
the clip, so the arrays line up with the clip's `<clip>_world.npz` body landmarks.

| Key | Shape | Meaning |
| --- | --- | --- |
| `world` | (T, 2, 21, 3) | MediaPipe's own hand world landmarks: the no-round-trip baseline |
| `image`, `present`, `width`, `height`, `fps`, `timestamp_ms` | | Copied from the clip's `HandClip` |
| `mano__track_shape__world` | (T, 2, 21, 3) | Rebuilt by `track_shape` |
| `mano__track_shape_scale__world` | (T, 2, 21, 3) | Rebuilt by `track_shape_scale`, scale replayed |
| `mano__track_shape_scale_clean__world` | (T, 2, 21, 3) | Rebuilt by `track_shape_scale` at scale 1 |
| `mano__<variant>__wrist_orient`, `__hand_pose`, `__transl` | (T, 2, 3), (T, 2, 45), (T, 2, 3) | Fitted parameters |
| `mano__<variant>__betas` | (T, 2, 10) | The track's shape, repeated per frame |
| `mano__<variant>__scale` | (T, 2) | 1 for `track_shape`; the fitted scale for `track_shape_scale` |
| `mano__<variant>__rms_mm`, `__status` | (T, 2) | Per-frame fit residual; `ok`, a failure reason, or `absent` |

```python
import numpy as np

d = np.load("data/hand_variants/round_trip_eval/08917_hand_variants.npz")
hands = {
    "mediapipe": d["world"],
    "track_shape": d["mano__track_shape__world"],
    "track_shape_scale": d["mano__track_shape_scale__world"],
    "track_shape_scale_clean": d["mano__track_shape_scale_clean__world"],
}
```

To export another set (any folder of `<clip>_hands.npz`), or with SMPL-X hands as well:

```bash
python scripts/export_hand_variants.py --data-dir data/round_trip_eval --backends mano smplx
```

## 4. Comparing them on a classifier

Some things to keep fixed so that only the hand differs:

- **Same everything else.** The same clips, split, body landmarks, features, model, seeds and
  epochs for every arm. The body round trip is the same in both pipelines.
- **Four arms.** Raw MediaPipe (`world`) as the reference, then the three outputs above.
- **Two questions, two protocols.**
  - *Is the round-tripped hand a faithful substitute?* Train and test on the same output. Compare
    each arm's accuracy with the MediaPipe arm.
  - *Will it work for augmentation?* Train on the round-tripped (later augmented) hands and test
    on raw MediaPipe, because at deployment the classifier sees MediaPipe. This is the one that
    decides between the pipelines.
- **Small data is noisy.** Run several seeds per arm and compare per-seed differences, or report
  a confidence interval; one run per arm will not separate a 1–2 point difference.
- **Feature frame.** These are world landmarks in metres, wrist on MediaPipe's wrist. If the
  classifier normalises hands (for example wrist-centred and divided by palm length), much of the
  scale difference is normalised away before the classifier sees it. That is worth knowing: in
  that case the pipelines differ mainly in their finger poses.
