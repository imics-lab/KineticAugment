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

```sh
pip install -e '.[smplx,lab]'
kinetic-lab
```

`python -m kinetic_augment.lab.cli` does the same. Flags: `--port` (default 8770), `--host`
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

So if you tweak one time-warp or frame-drop variant, the saved file records that variant's knob but
not its weight. The tweak only takes effect once you turn that variant's weight on in the profile.

## Where things live

| Path | What |
| --- | --- |
| `src/kinetic_augment/lab/` | the server, pane runner, share build, saved-settings code, `knobs.yaml` (the plain-English text) |
| `src/kinetic_augment/lab/static/` | the page: classic scripts only, no build step, no CDN |
| `scripts/build_lab_share.py` | builds the zip |
| `tests/test_lab_*.py`, `tests/js/` | Python tests, and Node tests for the page's scripts (`node --test tests/js/*.test.js`) |
