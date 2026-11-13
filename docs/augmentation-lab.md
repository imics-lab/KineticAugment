# Augmentation Lab

The Lab is a web page for looking at what each augmentation does to a signing clip. Two panes, A and
B, each show one augmentation at its own settings, drawn over the original body so you can see the
change. Every augmentation and every knob has a plain-English description first and its technical
name second, with the proven range and the range our profiles use marked on each knob bar.

It works on the round-trip bodies: a clip's MediaPipe landmarks are fitted to one SMPL-X body, the
augmentation edits that body, and landmarks are read back off the mesh (see the README section on
the round-trip demo).

## Two ways to run it

| | Share mode (the zip) | Full mode (`kinetic-lab`) |
| --- | --- | --- |
| Needs | a browser | the repo, a Python install, the licensed model files |
| Opens | double-click `index.html` | `http://127.0.0.1:8770` |
| Knobs | snap to stops worked out ahead of time; one knob differs from the profile at a time | any value; the pane updates in about a second |
| Clips | the ones baked into the zip | any clip in the clip folder, plus "＋ add video" |
| Saving | the YAML downloads in the browser | written to `configs/viewer_saved/` |

### Share mode

Unzip `augmentation-lab.zip` and double-click `index.html`. Use Safari 16.4 or newer, or a recent
Chrome. It works offline and holds no model files. The zip can be about 150 MB. Moving a second
knob puts the first back to its profile value, and the page says so. The link at the bottom of the
sidebar, "Want live knobs? Set up full mode", opens the setup steps.

Build the zip on a machine that has the models:

```sh
python scripts/build_lab_share.py --clips 57949 01986 01991 --out dist/augmentation-lab.zip --budget-mb 200
```

Other flags: `--profile` (`default`, `ambitious` or `literature`), `--clips-dir`, `--video-dir`
(a clip that is not in the clip store yet is added from `<video-dir>/<id>.mp4`), `--model-dir`,
`--mesh-vertices` and `--mesh-mm` (mesh size and precision). If the budget does not fit, stops and
rerolls are cut before meshes. It takes about ten minutes for three clips.

### Full mode

Full mode runs the whole pipeline on your machine, so every knob takes any value and you can add
your own videos. It needs Python 3.9 to 3.12 (MediaPipe has no wheels for newer versions) and the
licensed model files.

**1. Get the code and make a virtual environment** (once):

```sh
git clone https://github.com/imics-lab/KineticAugment.git
cd KineticAugment
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[smplx,lab]'
```

**2. Put the model files in `models/`** (once). The Lab needs these seven files, laid out like this
under `models/` (or under any folder with the same layout):

| File | What it is | Where to get it |
| --- | --- | --- |
| `smplx/SMPLX_NEUTRAL.npz` | the body | <https://smpl-x.is.tue.mpg.de/> (free account, accept the licence) |
| `mano/MANO_LEFT.npz`, `mano/MANO_RIGHT.npz` | the hands | comes with the SMPL-X download as `MANO_LEFT.pkl` / `MANO_RIGHT.pkl`; see the MANO note below |
| `mano/SMPL-X__FLAME_vertex_ids.npy` | links the head to the face model | already in the repository |
| `flame/generic_model.pkl` | the face shape | <https://flame.is.tue.mpg.de/> (free account, accept the licence) |
| `flame/mediapipe_landmark_embedding.npz` | ties MediaPipe face points to the face model | <https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe> |
| `mediapipe/face_landmarker.task` | reads the face out of a video | `python scripts/download_models.py --face` |

MANO note: put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `<repo>/models/mano` and run
`python scripts/convert_mano.py` from the repository root (it always reads `models/mano`). If you
keep your models in another folder, convert inside the repo first, then copy the two `.npz` files
over.

The licensed model files are git-ignored, so they are never committed.

**3. Start the Lab** (every time). Open a terminal in the repository folder, switch the virtual
environment on, then run the command:

```sh
cd KineticAugment
source .venv/bin/activate
kinetic-lab
```

It opens `http://127.0.0.1:8770` in your browser; stop it with Ctrl+C. If the terminal says
`command not found: kinetic-lab`, the virtual environment is not switched on (the prompt starts
with `(.venv)` when it is); run `source .venv/bin/activate`, or start it without activating as
`.venv/bin/kinetic-lab`.

**4. Add clips.** A fresh clone has no clips, because the clip folder (`demo/round_trip/clips`) is
git-ignored. Press "＋ add video" in the Lab and pick a signing `.mp4`, or export clips in bulk with
`python scripts/export_round_trip_demo.py --clips <id> ... --video-dir <folder of <id>.mp4 files>`.
Either way each clip is fitted once and cached.

`python -m kinetic_augment.lab.cli` does the same as `kinetic-lab`. Flags: `--port` (default 8770), `--host`
(`127.0.0.1` only), `--clips-dir` (default `demo/round_trip/clips`), `--model-dir`, `--no-browser`.
With no saved folder the server uses the repository's `models/` when it is complete; otherwise it
shows the setup page, where you enter the model folder once (remembered in
`~/.kinetic_lab/config.json`). The model files and where to get them are on that page
(`src/kinetic_augment/lab/static/setup.html`, also reachable at `/setup.html`).

"＋ add video" takes an `.mp4`, runs extraction and the SMPL-X fit once (one to two minutes on a
laptop CPU) and caches it, so reopening the clip is instant.

## Saved settings

"Save" asks which pane (A or B) and a name, then writes a complete profile: the base profile with
only the knobs you moved changed. Each changed line keeps its comment and gains
`# viewer: was <old value>`. In full mode the file lands in `configs/viewer_saved/<name>.yaml`
(listed in that folder's `index.md`); in share mode it downloads. "Load" sets a pane's knobs from a
saved file.

Use one in code:

```python
from kinetic_augment.augmentations import load_profile

profile = load_profile("configs/viewer_saved/<name>.yaml")
```

What a saved file does not hold: the Lab's demo starting values. These are only there so each pane
shows something visible the first time you open it, and they are never written out:

- mirror left and right is always on in the pane;
- hand dropout starts at its top rate;
- time warp and frame drop start with a mode weight on, so there is something to see.

The one exception is a time-warp or frame-drop variant you changed whose mode weight is 0 in the
base profile (for example the varied or segment time warp, or the rate frame drop). Without its
weight the variant would never run in training, so Save also writes that one weight as 1, with
`# viewer: was 0`, and leaves the other weights as the base has them. The Save dialog says so
("Turned on ... in the saved profile so your change is used in training."). Mirror's `p` and hand
dropout's start rate are still never written unless you moved them.

## Where things live

| Path | What |
| --- | --- |
| `src/kinetic_augment/lab/` | the server, pane runner, share build, saved-settings code, `knobs.yaml` (the plain-English text) |
| `src/kinetic_augment/lab/static/` | the page: classic scripts only, no build step, no CDN |
| `scripts/build_lab_share.py` | builds the zip |
| `tests/test_lab_*.py`, `tests/js/` | Python tests, and Node tests for the page's scripts (`node --test tests/js/*.test.js`) |
