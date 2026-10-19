#!/usr/bin/env python3
"""
Convert the official MANO hand models to chumpy-free .npz files.

The MANO release (mano_v1_2.zip, models/MANO_{LEFT,RIGHT}.pkl) pickles some of
its arrays as `chumpy` objects. chumpy does not install on Python 3.11 with a
current numpy, so `smplx` cannot load the .pkl files directly. This script reads
the arrays out of the chumpy objects without importing chumpy and writes .npz
files that `smplx.create("models", model_type="mano", ext="npz")` loads.

It also corrects a known upstream defect (vchoutas/smplx#48): MANO_LEFT's
`shapedirs` x-component equals the right hand's instead of being mirrored, so a
given `betas` vector shapes the two hands differently. The fix negates that
component in the converted left model. The original .pkl files are not modified.

The model files are licensed and are not committed. Place MANO_LEFT.pkl and
MANO_RIGHT.pkl in models/mano/ and run from the project root:

    python scripts/convert_mano.py
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

MANO_DIR = Path("models/mano")

warnings.filterwarnings("ignore", category=DeprecationWarning)


class _ChumpyStub:
    """Stands in for any chumpy class so its pickled state can be read."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


class _ChumpyFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return type(name, (_ChumpyStub,), {})
        return super().find_class(module, name)


def _to_array(value):
    if isinstance(value, _ChumpyStub):
        state = value.__dict__
        if "x" in state:
            return np.asarray(state["x"])
        if type(value).__name__ == "Select":
            # chumpy's a[idxs], stored as flat indices into `a`.
            flat = _to_array(state["a"]).ravel()
            return flat[np.asarray(state["idxs"]).ravel()].reshape(state["preferred_shape"])
        raise TypeError(f"unhandled chumpy object {type(value).__name__}: {sorted(state)}")
    if sp.issparse(value):
        return value.toarray()
    return np.asarray(value)


def load_mano_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        data = _ChumpyFreeUnpickler(f, encoding="latin1").load()
    return {key: _to_array(value) for key, value in data.items()}


def main():
    models = {side: load_mano_pkl(MANO_DIR / f"MANO_{side}.pkl") for side in ("LEFT", "RIGHT")}

    left, right = models["LEFT"]["shapedirs"], models["RIGHT"]["shapedirs"]
    if np.abs(left[:, 0] - right[:, 0]).sum() < 1:
        models["LEFT"]["shapedirs"] = left.copy()
        models["LEFT"]["shapedirs"][:, 0] *= -1
        print("Applied the MANO_LEFT shapedirs x-sign fix (vchoutas/smplx#48).")

    for side, arrays in models.items():
        out = MANO_DIR / f"MANO_{side}.npz"
        np.savez(out, **arrays)
        print(f"Wrote {out}: v_template {arrays['v_template'].shape}, "
              f"J_regressor {arrays['J_regressor'].shape}")


if __name__ == "__main__":
    main()
