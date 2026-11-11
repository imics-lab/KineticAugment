# Model Files

Model weights are licensed and too large for git. `models/.gitignore` ignores `*.npz`, `*.pkl`, `*.pt`, `*.pth`, `*.onnx`, `*.h5`, `*.task` and `*.obj`, so every weight file below has to be fetched by hand or by a script. Code looks for them under `models/` relative to the directory you run from (the repository root).

## Layout

```
models/
├── smplx/                              # SMPL-X body (manual download)
│   ├── SMPLX_NEUTRAL.npz               #   required
│   ├── SMPLX_MALE.npz                  #   required by scripts/download_models.py --check
│   └── SMPLX_FEMALE.npz                #   required by scripts/download_models.py --check
├── mano/                               # MANO hands (manual download, then convert)
│   ├── MANO_LEFT.pkl                   #   official download; input to convert_mano.py
│   ├── MANO_RIGHT.pkl                  #   official download; input to convert_mano.py
│   ├── MANO_LEFT.npz                   #   produced by scripts/convert_mano.py; what the code loads
│   ├── MANO_RIGHT.npz                  #   produced by scripts/convert_mano.py; what the code loads
│   └── SMPL-X__FLAME_vertex_ids.npy    #   from the SMPL-X download; face correspondence
├── flame/                              # FLAME head (manual download; face pipeline only)
│   ├── generic_model.pkl               #   FLAME 2020; only its triangle list is read
│   └── mediapipe_landmark_embedding.npz  # from Zielon/metrical-tracker (MPI non-commercial)
├── mediapipe/                          # fetched by scripts/download_models.py --face
│   ├── face_landmarker.task            #   used by scripts/extract_face_landmarks.py
│   └── canonical_face_model.obj        #   used by scripts/evaluate_face.py
└── vposer/                             # vendored, currently unused (see below)
```

## What needs what

| You want | You need |
|----------|----------|
| SMPL-X body (`SMPLXWrapper`, `SMPLXSkeleton`, the intrinsic augmentations, the round-trip viewer) | `smplx/SMPLX_NEUTRAL.npz` |
| MANO hands (hand fitting, round-trip viewer) | `mano/MANO_LEFT.npz` and `mano/MANO_RIGHT.npz` (convert the `.pkl` files, see below) |
| Face and head (`scripts/extract_face_landmarks.py`, `scripts/extract_face_features.py`, `scripts/evaluate_face.py`) | `mano/SMPL-X__FLAME_vertex_ids.npy`, `flame/*`, `mediapipe/*` |
| Landmark-mode augmentations only (`mode='landmark'`) | No model files |

Pass `model_path="models"` (the parent), not `models/smplx`: `smplx.create()` appends the `smplx/` and `mano/` subfolders itself.

## Setup

1. Register at <https://smpl-x.is.tue.mpg.de/>, download SMPL-X and MANO, and put the files as shown above.
2. Convert the MANO pickles to chumpy-free `.npz` files (from the repository root):
   ```bash
   python scripts/convert_mano.py
   ```
   This writes `models/mano/MANO_LEFT.npz` and `MANO_RIGHT.npz`. It also fixes a known upstream defect in the left hand's `shapedirs` ([vchoutas/smplx#48](https://github.com/vchoutas/smplx/issues/48)). The `.pkl` files are not modified.
3. Face pipeline only: register at <https://flame.is.tue.mpg.de/> for FLAME 2020 and put `generic_model.pkl` in `models/flame/`. Download `mediapipe_landmark_embedding.npz` from <https://github.com/Zielon/metrical-tracker/tree/master/flame/mediapipe> into `models/flame/`. Then fetch the MediaPipe assets:
   ```bash
   python scripts/download_models.py --face
   ```

## Verification

```bash
python scripts/download_models.py --check
```

This checks the SMPL-X files (the script exits non-zero if any are missing), and lists MANO and VPoser. It does not check the converted `MANO_*.npz`, `flame/` or `mediapipe/` files. Note that `--setup-dir` writes an older copy of this README (with only the `smplx/`, `mano/` and `vposer/` layout); this file is the current one.

## VPoser

`models/vposer/` holds a vendored copy of VPoser v1.0 code and config (`vposer_smpl.py`, `TR00_004_00_WO_accad.ini`). Nothing in `src/` or `scripts/` loads it, and the checkpoint (`vposer_v1_0/snapshots/TR00_E096.pt`) is not included. It is optional and currently unused.
