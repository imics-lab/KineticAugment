# Round-trip viewer

An interactive web page for checking, clip by clip, the full MediaPipe → SMPL-X → MediaPipe round
trip of sign-language video: pose, both hands and the face mesh, with optional temporal smoothing
of the landmarks before the fit.

For each frame it shows:

1. **Sign video**: the source frame.
2. **MediaPipe extraction (2-D)**: pose, left and right hand and face mesh drawn on the frame. Red
   lines show where the three MediaPipe streams disagree before any fitting: pose wrist ↔ hand
   wrist, and seven pose face points ↔ the matching face-mesh points.
3. **Extraction vs round trip (3-D)**: the extracted landmarks (blue) and the same landmarks read
   back off the one fitted SMPL-X body. The blue hands are rescaled about the wrist to the track's
   median palm size, the fit's input (see "How the numbers are measured"). The read-back points are
   coloured green → red by error, and lines join each pair. Drag to orbit.
4. **SMPL-X body** with the read-back points on it. It shares its camera with panel 3.
5. **Strips over time**, with clip medians beside them:
   - arm error (% of shoulder width)
   - hand error (mm, wrist-relative)
   - finger curl error (degrees per joint)
   - face error (mm RMS)
   - round-trip jitter
   - stream gaps in the image
   - how far smoothing moved the landmarks
   - per-frame flags: fit, face found, hand missing, hand near face

The **smoothing** menu switches between settings exported for the clip: off, Gaussian σ = 1 or 2
frames, and a causal One Euro filter. When smoothing is on, the unsmoothed landmarks and error
curves are drawn underneath in grey for comparison.

Keys: space plays and pauses, ← and → step one frame. Click or drag on the strips to seek. The URL
keeps the clip, smoothing and frame (`#clip=20983&s=gauss2&f=45`), so a link points at an exact
frame.

### Augmentations page

`augment.html` (the **augmentations** link at the top) shows the clip's Tier 1 augmentations
([Augmentation_Grounding.md](../../docs/Augmentation_Grounding.md) §3), each drawn once per clip
from the evidence-based ranges and applied to the unsmoothed round trip: view (yaw ±20°, pitch
±5°), mirror with the sides swapped, duration (0.8–1.25×), arm joints (shoulder, elbow, wrist,
each p = 0.3, ±4°), scale (0.85–1.15), uncentre (up to ±0.1 of the frame, hands kept in it), and
all of them combined. Pick one from the **augmentation** menu. A box at the top says, in plain
words, what it does, what it changes and what it keeps, with this clip's drawn value and the
measured movement. Three 3-D views share one camera:

1. **Original**: the round trip with no augmentation, the fitted SMPL-X and its read-back points.
2. **One over the other**: the original (blue) and the augmented (orange) bodies overlaid.
3. **Augmented**: the augmented SMPL-X and the points read back off it.

Below them, the readout lists the clip's draw, how far the wrists moved, how many hand points stay
in the frame, and the check that the augmentation at zero strength reproduces the original; the
source frame is beside it. A duration change re-times the clip, so the transport steps through
the augmented frames and each shows the source frame it comes from. The URL keeps the clip,
augmentation and frame (`augment.html#clip=20983&a=combined&f=45`). The same keys play and step.

The page is plain HTML and JavaScript with no build step. It loads three.js from the jsDelivr CDN,
so the viewing device needs an internet connection.

## Quick start: run the viewer for one clip

These steps export one clip and open both pages in a browser. They are written so that a person
or a coding agent can follow them as they stand. Every command runs **from the repository root**
(the folder with `pyproject.toml`), in a shell with the project's virtual environment active.

The example clip is `20983`. Replace it with your clip id. The clip's video must be a file named
`<clip id>.mp4` in one folder: that folder is `--video-dir` below.

### 1. Set up (once)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[smplx,dev]'
```

The licensed model files are not in git. Put them at these paths (where to get them: step 2 of
[Option B](#option-b-export-clips-yourself)):

| File | Needed for |
| --- | --- |
| `models/smplx/SMPLX_NEUTRAL.npz` | the SMPL-X body |
| `models/mano/MANO_LEFT.npz`, `models/mano/MANO_RIGHT.npz` | the hands (made by `python scripts/convert_mano.py`) |
| `models/flame/generic_model.pkl`, `models/flame/mediapipe_landmark_embedding.npz` | the face |
| `models/mediapipe/face_landmarker.task` | extracting a clip's face mesh (`python scripts/download_models.py --face`) |

`models/mano/SMPL-X__FLAME_vertex_ids.npy` is already in git. Check that every file is in place:

```bash
python -c "from pathlib import Path; need = ['models/smplx/SMPLX_NEUTRAL.npz', 'models/mano/MANO_LEFT.npz', 'models/mano/MANO_RIGHT.npz', 'models/mano/SMPL-X__FLAME_vertex_ids.npy', 'models/flame/generic_model.pkl', 'models/flame/mediapipe_landmark_embedding.npz', 'models/mediapipe/face_landmarker.task']; missing = [p for p in need if not Path(p).exists()]; print('missing: ' + ', '.join(missing) if missing else 'all model files present')"
```

It should print `all model files present`.

### 2. Export the clip

```bash
python scripts/export_round_trip_demo.py --clips 20983 --video-dir path/to/videos
```

This runs MediaPipe on the video, fits SMPL-X to every frame, exports the four smoothing settings
and then the seven Tier 1 augmentations. It takes about one to two minutes on a laptop CPU (68 s for
a 41-frame clip), most of it the four SMPL-X fits; adding `--smoothing off` exports only the
unsmoothed setting, in about a quarter of the time. It prints one line per smoothing setting (`off`,
`gauss1`, `gauss2`, `oneeuro`), one line per augmentation (`aug view`, …, `aug combined`), then
`augmentations done`. It writes `demo/round_trip/clips/20983/` and updates
`demo/round_trip/clips.json`; re-running it overwrites that clip.

Check what was exported:

```bash
python -c "import json; c = json.load(open('demo/round_trip/clips.json'))['clips']['20983']; print('smoothing:', sorted(c['variants'])); print('augmentations:', sorted(c.get('augmentations', {})))"
```

Expected output:

```text
smoothing: ['gauss1', 'gauss2', 'off', 'oneeuro']
augmentations: ['arm_joints', 'combined', 'duration', 'mirror', 'scale', 'uncentre', 'view']
```

### 3. Serve the pages

```bash
python3 -m http.server 8765 --directory demo/round_trip
```

The server keeps running in that terminal until you press Ctrl-C. A coding agent should start it
as a background process and then check that it answers:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8765/clips.json
```

`200` means it is serving. If port 8765 is taken (`Address already in use`), use another port,
such as 8777, in both commands and in the addresses below.

### 4. Open the pages

| Page | Address | Shows |
| --- | --- | --- |
| Round trip | <http://localhost:8765/#clip=20983> | the source video, MediaPipe's extraction, extraction vs round trip, the fitted SMPL-X body, error strips |
| Augmentations | <http://localhost:8765/augment.html#clip=20983&a=combined> | what the augmentation changes, then the original, both one over the other, and the augmented body |

On the augmentations page, the **augmentation** menu picks one of the seven (`a=` in the address:
`view`, `mirror`, `duration`, `arm_joints`, `scale`, `uncentre`, `combined`). The links at the top
of each page switch between the two. Space plays and pauses; ← and → step one frame.

### If something goes wrong

| What you see | Why, and what to do |
| --- | --- |
| A blank page, or nothing loads | The page was opened as a file (`file://`). Serve it (step 3) and open the `http://` address. |
| The page loads but no 3-D view appears | three.js comes from the jsDelivr CDN, so the browser needs an internet connection. |
| The augmentations page says no clip has exported augmentations | The clip was exported before augmentations existed, or with `--augment` and no names. Run step 2 again. |
| The clip is missing from the clip menu | `clips.json` is out of date. Run `python scripts/export_round_trip_demo.py --reindex --video-dir path/to/videos`. |
| `decoded N frames, landmarks have M` | The cached MediaPipe extraction does not match the video. Add `--reextract` to step 2. |
| `FileNotFoundError` or `does not exist` naming a file under `models/` | A model file is missing. Run the check in step 1. |
| `Address already in use` | Another server holds the port. Pick another port (step 3). |

## Option A: view a shared bundle (no setup)

You only need **Python 3** (any version; it ships with macOS and most Linux distributions) and a
modern browser.

1. Get `round_trip_demo.tar.gz` from someone who has exported the clips (see
   [Sharing a bundle](#sharing-a-bundle)).
2. Unpack it and start a local web server in the unpacked folder:

   ```bash
   tar -xzf round_trip_demo.tar.gz
   python3 -m http.server 8765 --directory round_trip
   ```

   On Windows, use `py -m http.server 8765 --directory round_trip`.
3. Open <http://localhost:8765> in a browser.

Opening `index.html` directly as a file does **not** work: browsers block the page from reading
its data files over `file://`, so it has to be served.

### Viewing from another device on the same network

Start the server so other machines can reach it:

```bash
python3 -m http.server 8765 --bind 0.0.0.0 --directory round_trip
```

Then open `http://<this-computer's-IP>:8765` on the other device (phone, tablet, laptop). On macOS
the IP is shown by `ipconfig getifaddr en0`; on Linux by `hostname -I`. Anyone on that network can
open the page while the server runs, so stop it (Ctrl-C) when you are done.

## Option B: export clips yourself

This route runs the full pipeline, so it needs the project installed and the licensed model files.

1. Install the package from the repo root (Python 3.9+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e '.[smplx,dev]'
   ```

2. Put the model assets under `models/`; `models/README.md` lists the layout.
   - SMPL-X (`models/smplx/`) and MANO (`models/mano/`, then `python scripts/convert_mano.py`): free
     academic downloads that require registration.
   - FLAME and the MediaPipe face assets:

     ```bash
     python scripts/download_models.py --face
     ```

     This downloads `face_landmarker.task` and `canonical_face_model.obj`, then prints the manual
     FLAME steps (`generic_model.pkl`, `mediapipe_landmark_embedding.npz` in `models/flame/`).
   - Check everything with `python scripts/download_models.py --check`.

3. Export clips from a folder of `<clip id>.mp4` videos. Export writes to `demo/round_trip/clips/`
   and rebuilds `demo/round_trip/clips.json`.

   ```bash
   # one clip, all four smoothing settings (about 100 s on a laptop)
   python scripts/export_round_trip_demo.py --clips 20983 --video-dir path/to/videos

   # the seeded random review set: the first N of 50 clips drawn with seed 20261017
   python scripts/export_round_trip_demo.py --sample 50 --video-dir path/to/videos

   # only some smoothing settings
   python scripts/export_round_trip_demo.py --clips 20983 --smoothing off gauss2 --video-dir path/to/videos

   # only some augmentations, or none; another of the clip's draws (0 is its own)
   python scripts/export_round_trip_demo.py --clips 20983 --augment view mirror --video-dir path/to/videos
   python scripts/export_round_trip_demo.py --clips 20983 --augment --video-dir path/to/videos
   python scripts/export_round_trip_demo.py --clips 20983 --aug-draw 3 --video-dir path/to/videos
   ```

   MediaPipe landmarks are cached per clip (`hands.npz`, `face.npz`) and reused; `--reextract`
   recomputes them. To export faster, run several processes on different `--clips` lists, then
   rebuild the index once they finish:

   ```bash
   python scripts/export_round_trip_demo.py --reindex --video-dir path/to/videos
   ```

4. Serve the page:

   ```bash
   python3 -m http.server 8765 --directory demo/round_trip
   ```

   Then open <http://localhost:8765>.

## Sharing a bundle

The exported data is **not in git**. `demo/round_trip/clips/` and `clips.json` are git-ignored
because they contain frames of the signers from the source dataset. Share bundles only with people
covered by that dataset's licence, through a private channel such as the lab drive. Do not post
them publicly.

From the repo root, pack the page and its data into one archive:

```bash
tar -czf round_trip_demo.tar.gz -C demo round_trip
```

Expect about 30 MB per clip with all four smoothing settings, plus about 40 MB for the seven
augmentations, so the 50-clip review set is about 3.5 GB. Exporting fewer settings (`--smoothing off gauss1`) makes the bundle proportionally
smaller.

## What gets exported

For each clip, `clips/<clip>/` holds:

- `frames/NNNN.jpg`: every decoded frame (the same decode MediaPipe saw).
- `hands.npz`, `face.npz`: the cached raw MediaPipe extraction.
- `v/<smoothing>/clip.json`: frame size and rate, flags, per-frame error series, clip medians, and
  an index of the arrays in `data.bin`.
- `v/<smoothing>/data.bin`: little-endian arrays. These are the 2-D and 3-D landmarks, the
  read-back points, per-point errors and the posed SMPL-X mesh (int16, 0.1 mm).
- `aug/<augmentation>/clip.json`: the clip's Tier 1 draw, the augmentation's title, frame count,
  camera fit and measurements, and an index of its `data.bin`.
- `aug/<augmentation>/data.bin`: the source time of each augmented frame, the augmented mesh
  (int16, 0.1 mm), the augmented read-back points in 3-D and projected into the frame.

### How the numbers are measured

- **Body:** arm landmarks 11–16, in % of the frame's shoulder width.
- **Hands:** mean distance over wrist-relative landmarks 1–20, in mm. These are read off the SMPL-X
  hand; the MANO hand alone is given for reference. Each frame's MediaPipe hand is first rescaled
  about its wrist to the track's median palm size (the mean wrist-to-knuckle distance over landmarks
  5, 9, 13 and 17), which is the input the fit sees, and the errors are measured against that
  rescaled hand.
- **Curl:** at each of the 15 finger joints, the bend angle in degrees between the two bones
  meeting there. The error is the mean over the 15 joints of |fitted angle − observed angle|,
  against the same rescaled hand. It ignores hand size and position, so it shows whether the
  fingers are bent as much as the observed ones. Shown for the SMPL-X hand and the MANO hand.
- **Face:** RMS over the 105 FLAME-embedded points, in mm. The observed face is carried onto the
  fitted head by a similarity transform of its rigid points.
- **Jitter:** mean |second difference| of each stream's points, in mm per frame².
- **Stream gaps and smoothing shift:** in image pixels.

Smoothing is applied to every MediaPipe track before fitting
(`kinetic_augment.utils.smoothing`), and the round-trip error is measured against the smoothed
input.
