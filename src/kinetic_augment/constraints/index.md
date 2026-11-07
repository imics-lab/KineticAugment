# constraints/

Used by `Pipeline`; only acts in `mode='smplx'` (no-op in landmark mode).

| Item | Description |
|---|---|
| `__init__.py` | Exports the constraint classes |
| `engine.py` | `ConstraintEngine`: validate and enforce all constraint types |
| `joint_limits.py` | Anatomical range-of-motion limits |
| `velocity_limits.py` | Maximum angular velocity limits |
| `collision/` | PyBullet self-collision detection and resolution (experimental, off in all presets) |
