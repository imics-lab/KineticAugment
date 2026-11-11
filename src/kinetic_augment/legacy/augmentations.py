
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline

# Import project-specific utilities and mappings
from ..utils.data_formats import POSE_START, POSE_END, FACE_START, FACE_END, LH_START, LH_END, RH_START, RH_END
from ..utils.landmark_maps import POSE_FULL_MAP, FACE_FULL_MAP

# --- Implemented Augmentations ---

def global_rotation(data: np.ndarray, max_angle_deg: dict, **kwargs) -> np.ndarray:
    """Applies a random global rotation to the entire sequence."""
    rx = np.random.uniform(-max_angle_deg.get('x', 0), max_angle_deg.get('x', 0))
    ry = np.random.uniform(-max_angle_deg.get('y', 0), max_angle_deg.get('y', 0))
    rz = np.random.uniform(-max_angle_deg.get('z', 0), max_angle_deg.get('z', 0))

    random_rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
    
    num_frames, num_landmarks, _ = data.shape
    reshaped_data = data.reshape(-1, 3)
    rotated_data = random_rotation.apply(reshaped_data)
    return rotated_data.reshape(num_frames, num_landmarks, 3)

def global_scaling(data: np.ndarray, scale_range: list, **kwargs) -> np.ndarray:
    """Applies a random uniform scaling to the entire sequence."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale_factor

def pose_flipping(data: np.ndarray, **kwargs) -> np.ndarray:
    """Mirrors the pose across the sagittal (Y-Z) plane and swaps landmark labels."""
    flipped_data = data.copy()
    
    # 1. Geometric Flip: Negate the x-coordinate
    flipped_data[:, :, 0] *= -1

    # 2. Semantic Swap: Swap the data for left/right pairs
    # Create a copy of the original data to source from during swaps
    original_copy = data.copy()

    # Swap Pose landmarks
    for left_idx, right_idx in POSE_FULL_MAP.items():
        flipped_data[:, POSE_START + left_idx, :] = original_copy[:, POSE_START + right_idx, :]

    # Swap Face landmarks
    for left_idx, right_idx in FACE_FULL_MAP.items():
        flipped_data[:, FACE_START + left_idx, :] = original_copy[:, FACE_START + right_idx, :]
        
    # Swap Left and Right Hands entirely
    left_hand_data = original_copy[:, LH_START:LH_END, :]
    right_hand_data = original_copy[:, RH_START:RH_END, :]
    flipped_data[:, RH_START:RH_END, :] = left_hand_data
    flipped_data[:, LH_START:LH_END, :] = right_hand_data

    return flipped_data

def time_warping(data: np.ndarray, max_warp_factor: float, **kwargs) -> np.ndarray:
    """Non-linearly warps the time axis of the sequence."""
    num_frames, num_landmarks, num_dims = data.shape
    
    # Original time axis
    t_orig = np.arange(num_frames)
    
    # Create a smooth warping curve using cubic splines
    # More knots = more complex warp. 4-5 is usually sufficient.
    num_knots = 5
    knot_x = np.linspace(0, num_frames - 1, num_knots)
    knot_y_offsets = np.random.uniform(-max_warp_factor, max_warp_factor, size=num_knots) * num_frames
    
    # Ensure start and end points are fixed
    knot_y_offsets[0] = knot_y_offsets[-1] = 0
    
    spline = CubicSpline(knot_x, knot_x + knot_y_offsets)
    t_warped = spline(t_orig)
    
    # Ensure the warped time axis is monotonic
    t_warped = np.clip(t_warped, 0, num_frames - 1)

    # Resample the data at the new warped time points
    warped_data = np.zeros_like(data)
    for i in range(num_landmarks):
        for j in range(num_dims):
            # Use linear interpolation to find values at new time points
            warped_data[:, i, j] = np.interp(t_warped, t_orig, data[:, i, j])
            
    return warped_data

def dynamic_trajectory_jittering(data: np.ndarray, amplitude: float, frequency: float, **kwargs) -> np.ndarray:
    """Adds smooth, low-frequency noise to landmark trajectories."""
    num_frames, num_landmarks, _ = data.shape
    
    # Generate smooth noise using a sum of sine waves
    noise = np.zeros((num_frames, 3))
    for dim in range(3):
        # Generate 3 sine waves with random phases and frequencies
        for _ in range(3):
            freq = np.random.uniform(0.1, frequency)
            phase = np.random.uniform(0, np.pi * 2)
            noise[:, dim] += np.sin(np.linspace(0, freq * np.pi * 2, num_frames) + phase)
    
    # Normalize and scale the noise
    noise /= np.max(np.abs(noise))
    noise *= amplitude

    # Apply the same noise trajectory to all landmarks for a "camera shake" effect
    # or apply to specific joint groups if specified in a more advanced version.
    jittered_data = data + noise[:, np.newaxis, :]
    
    return jittered_data

# --- Placeholder Augmentations (Advanced Implementation Required) ---

def joint_angle_perturbation(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Placeholder for joint angle perturbation.
    NOTE: This requires a full Inverse Kinematics (IK) and Forward Kinematics (FK)
          solver, which is a complex implementation. DynamicTrajectoryJittering
          can be used as a simpler proxy for adding positional noise.
    """
    print("  >> Warning: 'JointAnglePerturbation' is not implemented. Returning data as-is.")
    # raise NotImplementedError("JointAnglePerturbation requires a full IK/FK solver.")
    return data

def joint_coupled_noise(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Placeholder for applying noise while respecting anatomical coupling.
    NOTE: This is an advanced version of joint angle perturbation and also
          requires an IK/FK system with a defined coupling model (e.g., for fingers).
    """
    print("  >> Warning: 'JointCoupledNoise' is not implemented. Returning data as-is.")
    # raise NotImplementedError("JointCoupledNoise requires an advanced IK/FK solver.")
    return data