"""Band comparison in measure_noise_floor mixes a 1-axis z band with a 2-axis xy band."""
import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("mnf", "scripts/measure_noise_floor.py")
mnf = importlib.util.module_from_spec(spec); spec.loader.exec_module(mnf)
rng = np.random.default_rng(0)
T, sigma = 5000, 1.0
track = rng.normal(0, sigma, (T, 3))          # stationary, ISOTROPIC iid noise: z exactly as noisy as x or y
ok = np.ones(T, bool)
a_xy, a_z = mnf.method_a(track, ok, [0, 1]), mnf.method_a(track, ok, [2])
b_xy, b_z = mnf.method_b(track, ok, [0, 1]), mnf.method_b(track, ok, [2])
b_xyz = mnf.method_b(track, ok, [0, 1, 2])
print(f"isotropic sigma=1 per axis: method B xy={b_xy:.3f} z={b_z:.3f} xyz={b_xyz:.3f}  -> script's 'z is {b_z/b_xy:.2f}x noisier than x/y' (truth 1.00)")
print(f"                            method A xy={a_xy:.3f} z={a_z:.3f}  -> A ratio z/xy {a_z/a_xy:.2f} (truth 1.00)")
# Method B on pure iid noise WITHOUT the motion gate vs with it (gate conditions on noisy speeds)
print(f"  (method B xyz should be sqrt(3)={np.sqrt(3):.3f} for this noise)")
