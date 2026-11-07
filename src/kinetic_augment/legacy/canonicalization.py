
import numpy as np
from scipy.spatial.transform import Rotation
from ..utils.data_formats import LEFT_SHOULDER, RIGHT_SHOULDER, NOSE, POSE_START, POSE_END

def to_canonical_space_frame(frame_landmarks: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Converts a single frame of landmarks to a canonical representation and returns
    the parameters needed to reverse the transformation.

    This version explicitly handles invalid landmarks (all zeros) to prevent
    them from being transformed.

    Returns:
        tuple[np.ndarray, dict]: A tuple containing:
            - The canonicalized (543, 3) landmark array.
            - A dictionary of transformation parameters for inversion.
    """
    # Create a mask to identify landmarks that were not detected (all zeros).
    # These landmarks should not be part of any transformation.
    invalid_mask = np.all(frame_landmarks == 0, axis=1)

    pose_landmarks = frame_landmarks[POSE_START:POSE_END]
    
    # --- 1. Store original translation (centering vector) ---
    translation_vector = (pose_landmarks[LEFT_SHOULDER] + pose_landmarks[RIGHT_SHOULDER]) / 2.0
    # If the reference landmarks are not detected, we cannot process this frame.
    if np.all(translation_vector == 0):
        return frame_landmarks, None

    centered_landmarks = frame_landmarks - translation_vector

    # --- 2. Store original rotation ---
    centered_pose = centered_landmarks[POSE_START:POSE_END]
    x_vec = centered_pose[RIGHT_SHOULDER] - centered_pose[LEFT_SHOULDER]
    if np.linalg.norm(x_vec) < 1e-6: return frame_landmarks, None
    
    x_axis = x_vec / np.linalg.norm(x_vec)
    y_vec = centered_pose[NOSE] - ((centered_pose[LEFT_SHOULDER] + centered_pose[RIGHT_SHOULDER]) / 2.0)
    if np.linalg.norm(y_vec) < 1e-6: return frame_landmarks, None

    z_axis = np.cross(x_axis, y_vec); z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    original_rotation = Rotation.from_matrix(rotation_matrix)
    
    aligning_rotation = original_rotation.inv()
    oriented_landmarks = aligning_rotation.apply(centered_landmarks)

    # --- 3. Store original scale ---
    oriented_pose = oriented_landmarks[POSE_START:POSE_END]
    shoulder_width = np.linalg.norm(oriented_pose[RIGHT_SHOULDER] - oriented_pose[LEFT_SHOULDER])
    if shoulder_width < 1e-6: return frame_landmarks, None
    
    scale_factor = 1.0 / shoulder_width
    canonical_landmarks = oriented_landmarks * scale_factor

    # After all transformations, re-apply the invalid mask to ensure
    # undetected landmarks remain as (0, 0, 0).
    canonical_landmarks[invalid_mask] = 0.0

    transform_params = {
        'translation': translation_vector,
        'rotation': original_rotation,
        'scale': scale_factor
    }
    
    return canonical_landmarks, transform_params

def from_canonical_space_frame(canonical_landmarks: np.ndarray, params: dict) -> np.ndarray:
    """
    Applies the inverse transformation to a frame to convert it back
    from canonical space to its original coordinate space.

    This version also preserves the status of invalid landmarks.
    """
    if params is None:
        return canonical_landmarks

    # Identify invalid landmarks in the canonical space before transforming.
    invalid_mask = np.all(canonical_landmarks == 0, axis=1)

    # Apply inverse transformations in reverse order
    # 1. Inverse Scale
    unscaled_landmarks = canonical_landmarks / params['scale']
    
    # 2. Inverse Rotation (apply the original rotation)
    deoriented_landmarks = params['rotation'].apply(unscaled_landmarks)
    
    # 3. Inverse Translation (add the original centering vector back)
    original_space_landmarks = deoriented_landmarks + params['translation']
    
    # Re-apply the invalid mask to correct for any floating point errors
    # that may have made zero-vectors non-zero.
    original_space_landmarks[invalid_mask] = 0.0

    return original_space_landmarks

def canonicalize_sequence(sequence_data: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Applies canonicalization to each frame and returns the transformation params for each frame.
    """
    num_frames = sequence_data.shape[0]
    canonical_sequence = np.zeros_like(sequence_data)
    params_per_frame = []

    for i in range(num_frames):
        canonical_frame, params = to_canonical_space_frame(sequence_data[i])
        canonical_sequence[i] = canonical_frame
        params_per_frame.append(params)

    return canonical_sequence, params_per_frame

def decanonicalize_sequence(augmented_sequence: np.ndarray, params_per_frame: list) -> np.ndarray:
    """
    Applies de-canonicalization to each augmented frame using the stored parameters.
    """
    num_frames = augmented_sequence.shape[0]
    original_space_sequence = np.zeros_like(augmented_sequence)

    for i in range(num_frames):
        original_space_sequence[i] = from_canonical_space_frame(augmented_sequence[i], params_per_frame[i])

    return original_space_sequence