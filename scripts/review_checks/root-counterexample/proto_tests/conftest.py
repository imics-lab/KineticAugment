"""Give SMPLXSkeleton the rest_nose attribute the red-phase tests expect (prototype only)."""
import numpy as np
import torch
from kinetic_augment.body_model import skeleton as _sk

_orig_init = _sk.SMPLXSkeleton.__init__

def _init(self, model_path="models", gender="neutral", betas=None):
    _orig_init(self, model_path=model_path, gender=gender, betas=betas)
    b = torch.zeros(1, 10) if betas is None else torch.as_tensor(np.asarray(betas, dtype=np.float32))[None, :10]
    with torch.no_grad():
        j = self.model(betas=b, return_verts=False).joints[0].numpy().astype(np.float64)
    # sanity: the model's rest joints must equal the skeleton's
    assert np.abs(j[:55] - self.rest_joints).max() < 1e-5
    self.rest_nose = j[55]          # SMPL-X 'nose' vertex landmark

_sk.SMPLXSkeleton.__init__ = _init
