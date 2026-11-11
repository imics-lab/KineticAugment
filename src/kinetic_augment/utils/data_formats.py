
import json
import numpy as np
from pathlib import Path

# --- Landmark Constants from batch_process_videos.py ---
# This ensures consistency between data extraction and processing.
NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 468
NUM_HAND_LANDMARKS = 21

# The order is fixed: Pose, Face, Left Hand, Right Hand
TOTAL_LANDMARKS = NUM_POSE_LANDMARKS + NUM_FACE_LANDMARKS + NUM_HAND_LANDMARKS + NUM_HAND_LANDMARKS

# Define start and end indices for slicing
POSE_START, POSE_END = 0, NUM_POSE_LANDMARKS
FACE_START, FACE_END = POSE_END, POSE_END + NUM_FACE_LANDMARKS
LH_START, LH_END = FACE_END, FACE_END + NUM_HAND_LANDMARKS
RH_START, RH_END = LH_END, LH_END + NUM_HAND_LANDMARKS

# Define landmark indices for canonicalization
# These are indices within the POSE block (0-32)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
NOSE = 0

def load_and_reshape_json(file_path: Path) -> np.ndarray:
    """
    Loads a JSON file created by batch_process_videos.py and reshapes it.

    The input is a list of frames, where each frame is a flat list of 1629 floats.
    The output is a NumPy array of shape (num_frames, num_landmarks, 3).

    Args:
        file_path (Path): The path to the input JSON file.

    Returns:
        np.ndarray: A 3D NumPy array containing the landmark data.
    """
    try:
        with open(file_path, 'r') as f:
            sequence_data = json.load(f)

        if not isinstance(sequence_data, list) or not all(isinstance(f, list) for f in sequence_data):
            raise ValueError("JSON data is not a list of lists (frames).")

        # Convert list of lists to a 2D NumPy array
        sequence_array = np.array(sequence_data, dtype=np.float32)

        # Reshape from (num_frames, 1629) to (num_frames, 543, 3)
        num_frames = sequence_array.shape[0]
        reshaped_array = sequence_array.reshape(num_frames, TOTAL_LANDMARKS, 3)

        return reshaped_array

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading or reshaping {file_path}: {e}")
        raise

def save_to_json(data: np.ndarray, output_path: Path):
    """
    Saves a NumPy array of landmarks to a JSON file in the original flat format.

    Args:
        data (np.ndarray): The landmark data of shape (num_frames, num_landmarks, 3).
        output_path (Path): The path to save the output JSON file.
    """
    try:
        # Reshape from (num_frames, 543, 3) back to (num_frames, 1629)
        num_frames = data.shape[0]
        flat_data = data.reshape(num_frames, -1)

        # Convert to list for JSON serialization
        data_to_save = flat_data.tolist()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f)

    except Exception as e:
        print(f"An error occurred while saving data to {output_path}: {e}")
        raise