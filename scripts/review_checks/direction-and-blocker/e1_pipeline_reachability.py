"""E1: Is SMPL-X conversion reachable from the framework's augmentation pipeline at all?
Tests the doc's Blocked claims #1/#2 by exercising Pipeline / AugmentationChain / Dataset."""
import warnings, traceback
import numpy as np
warnings.simplefilter("ignore")
from kinetic_augment.pipeline.chain import AugmentationChain
from kinetic_augment.pipeline.pipeline import Pipeline
from kinetic_augment.pipeline import presets
from kinetic_augment.augmentations.base import AugmentationRegistry, SMPLXAugmentation

print("registered:", sorted(AugmentationRegistry.list_available()))
smplx_augs = [n for n in AugmentationRegistry.list_available()
              if issubclass(AugmentationRegistry.get(n), SMPLXAugmentation)]
print("SMPLXAugmentation subclasses:", smplx_augs)

# 1. landmark-mode chain refuses intrinsic ops
for name in smplx_augs:
    try:
        AugmentationChain(mode="landmark").add(name)
        print(f"landmark chain + {name}: accepted")
    except ValueError as e:
        print(f"landmark chain + {name}: ValueError")

# 2. presets: which modes / which ops
for p in presets.PRESET_NAMES:
    cfg = presets.get_preset(p)
    print(f"preset {p!r}: mode={cfg.mode}, ops={[s['operation'] for s in cfg.plan]}")

# 3. YAML configs shipped in configs/
import glob
for path in sorted(glob.glob("configs/*.yaml")):
    try:
        pl = Pipeline.from_yaml(path)
        print(f"{path}: constructed, mode={pl.mode}, ops={[s['name'] for s in pl.chain.get_step_info()]}")
    except Exception as e:
        print(f"{path}: FAILS -> {type(e).__name__}: {str(e)[:160]}")

# 4. smplx-mode chain silently skips landmark augmentations on a dict
chain = AugmentationChain(mode="smplx", seed=0)
chain.add("GlobalRotation", config={"max_angle_deg": {"x": 30}})
chain.add("JointAnglePerturbation", config={"stddev": 0.1})
params = {"body_pose": np.zeros((4, 63)), "global_orient": np.zeros((4, 3)), "betas": np.zeros((4, 10))}
out = chain(params)
print("smplx chain on dict: GlobalRotation validate(dict) ->",
      chain.steps[0].augmentation.validate(params), "(silently skipped)")
print("  body_pose changed:", not np.allclose(out["body_pose"], 0))

# 5. Dataset wrapper with an smplx-mode pipeline: what does __getitem__ do with a dict sample?
try:
    from kinetic_augment.pipeline.dataset import AugmentedLandmarkDataset
    pl = Pipeline.from_dict({"mode": "smplx", "augmentation_plan": [
        {"operation": "JointAnglePerturbation", "params": {"stddev": 0.1}}]})
    ds = AugmentedLandmarkDataset([params], labels=[0], pipeline=pl)
    ds[0]
    print("dataset with smplx pipeline: OK")
except Exception as e:
    print(f"dataset with smplx pipeline: FAILS -> {type(e).__name__}: {str(e)[:160]}")

# 6. classifier input contract
import inspect
from kinetic_augment.evaluation.models import lstm_classifier
src = inspect.getsource(lstm_classifier)
for kw in ["543", "input_size", "num_landmarks"]:
    lines = [l.strip() for l in src.splitlines() if kw in l][:4]
    print(f"lstm_classifier lines with {kw!r}:", lines)

# 5b. retry dataset with the required smplx_model_path
try:
    from kinetic_augment.pipeline.dataset import AugmentedLandmarkDataset
    pl = Pipeline.from_dict({"mode": "smplx", "smplx_model_path": "models", "augmentation_plan": [
        {"operation": "JointAnglePerturbation", "params": {"stddev": 0.1}}]})
    ds = AugmentedLandmarkDataset([params], labels=[0], pipeline=pl)
    ds[0]
    print("dataset(5b) with smplx pipeline: OK")
except Exception as e:
    print(f"dataset(5b) with smplx pipeline: FAILS -> {type(e).__name__}: {str(e)[:160]}")

# 7. slr_profile.yaml with only the registered ops kept: does it construct?
import yaml
cfg = yaml.safe_load(open("configs/slr_profile.yaml"))
key = [k for k in cfg if isinstance(cfg[k], list)]
print("slr_profile top-level keys:", list(cfg.keys()), "mode:", cfg.get("mode"))
for k in key:
    cfg[k] = [s for s in cfg[k] if isinstance(s, dict) and AugmentationRegistry.get(s.get("operation")) is not None]
try:
    pl = Pipeline.from_dict(cfg)
    print("slr_profile (registered ops only): constructed, mode", pl.mode)
except Exception as e:
    print(f"slr_profile (registered ops only): FAILS -> {type(e).__name__}: {str(e)[:200]}")
