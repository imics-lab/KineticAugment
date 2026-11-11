"""
Legacy modules, kept for reference and backwards compatibility.

Everything here is superseded. New code should use the replacements:

    legacy.pipeline           -> kinetic_augment.Pipeline (registry-based, from_yaml / from_preset)
    legacy.augmentations      -> kinetic_augment.augmentations (base / extrinsic / intrinsic /
                                 temporal / tier1); only used by legacy.pipeline
    legacy.mp_to_smplx        -> kinetic_augment.body_model.converter.fit_body
    legacy.smplx_to_mp        -> kinetic_augment.body_model.converter.reconstruct_body
    legacy.canonicalization   -> no replacement; only used by legacy.pipeline

This package is import-light: submodules are not imported here, and mp_to_smplx /
smplx_to_mp only import torch when a model is actually built.
"""
