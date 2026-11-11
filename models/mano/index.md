# models/mano/

| File | Description |
|---|---|
| `SMPL-X__FLAME_vertex_ids.npy` | In git: SMPL-X to FLAME head vertex map used for face correspondence |
| `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` | Not in git: official MANO v1.2 download (needs chumpy to unpickle); input to `scripts/convert_mano.py` |
| `MANO_LEFT.npz`, `MANO_RIGHT.npz` | Not in git: converted chumpy-free models, left with the shapedirs x-sign fix; what the code loads (`smplx.create("models", model_type="mano", ext="npz")`) |
| `index.md` | This index |
