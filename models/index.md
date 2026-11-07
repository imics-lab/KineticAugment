# models/

Licensed model weights are not in git (`models/.gitignore` ignores `*.npz`, `*.pkl`, `*.pt`, `*.pth`, `*.onnx`, `*.h5`, `*.task`, `*.obj`). See `README.md` for setup.

| Item | Description |
|---|---|
| `README.md` | Expected layout, what needs what, setup and verification |
| `.gitignore` | Keeps weight files out of git |
| `smplx/` | SMPL-X body model |
| `mano/` | MANO hand models and the SMPL-X/FLAME vertex map |
| `vposer/` | Vendored VPoser code/config (unused, proprietary licence) |
| `flame/` | Not in git, not present by default: FLAME head (`generic_model.pkl`, `mediapipe_landmark_embedding.npz`); face pipeline only |
| `mediapipe/` | Not in git, not present by default: `face_landmarker.task`, `canonical_face_model.obj` (from `scripts/download_models.py --face`) |
