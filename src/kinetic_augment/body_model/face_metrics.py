"""
Metrics for the V1.3 gate (design §7) that need more than a mean: comparing the fitted head
rotation with FaceLandmarker's own head pose, and the correspondence floor.

FaceLandmarker's facial transformation matrix maps its canonical face into an OpenGL-style
camera (x right, y up, z toward the viewer) and includes a uniform scale. The fitted rotation is
in image axes (x right, y down, z away). The two differ by a constant on the face side and by
GL_TO_IMAGE on the camera side, so they are compared as rotations relative to each clip's mean:
Δ_image = F Δ_gl F, which cancels the constant.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .alignment import apply_similarity, similarity_transform
from .face_correspondence import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, RIGID

GL_TO_IMAGE = np.diag([1.0, -1.0, -1.0])


def matrix_rotation(matrix) -> np.ndarray:
    """The proper rotation in a 4x4 similarity, with its scale removed."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64)[:3, :3])
    return u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt


def relative_rotations(rotations) -> np.ndarray:
    """Each rotation relative to the clip's mean rotation."""
    mean = Rotation.from_matrix(rotations).mean().as_matrix()
    return np.asarray(rotations) @ mean.T


def head_pose_errors_deg(fitted, head_matrix, mask) -> np.ndarray:
    """Per-frame angle (degrees) between the two relative head rotations; NaN off `mask`."""
    mask = np.asarray(mask, bool) & np.isfinite(fitted).all(axis=(1, 2)) \
        & np.isfinite(head_matrix).all(axis=(1, 2))
    errors = np.full(len(mask), np.nan)
    if mask.sum() < 2:
        return errors
    ours = relative_rotations(np.asarray(fitted)[mask])
    gl = np.stack([matrix_rotation(m) for m in np.asarray(head_matrix)[mask]])
    theirs = GL_TO_IMAGE @ relative_rotations(gl) @ GL_TO_IMAGE
    errors[mask] = np.degrees(
        Rotation.from_matrix(np.transpose(ours, (0, 2, 1)) @ theirs).magnitude())
    return errors


def load_obj_vertices(path) -> np.ndarray:
    """Vertex positions ('v' lines) of an OBJ file."""
    with open(path) as handle:
        return np.array([[float(x) for x in line.split()[1:4]]
                         for line in handle if line.startswith("v ")])


def correspondence_floor(canonical, rest_rigid) -> float:
    """
    Metric 1's floor: the rigid-set residual of a scaled fit from SMPL-X's rest rigid points to
    MediaPipe's canonical face, as a fraction of the canonical outer-eye-corner distance. This
    error remains even with a perfect fit.
    """
    target = np.asarray(canonical, dtype=np.float64)[list(RIGID)]
    s, r, t = similarity_transform(rest_rigid, target)
    residual = np.linalg.norm(apply_similarity(rest_rigid, s, r, t) - target, axis=1)
    reference = np.linalg.norm(np.asarray(canonical)[RIGHT_EYE_OUTER]
                               - np.asarray(canonical)[LEFT_EYE_OUTER])
    return float(np.sqrt(np.mean(residual ** 2)) / reference)


def gap_components(point, closest) -> tuple:
    """
    Split the gap between a point and its closest surface point into the camera-axis part
    |Δz| and the image-plane part |Δxy| (MediaPipe world: z is the camera axis), in metres.
    """
    delta = np.asarray(point, dtype=np.float64) - np.asarray(closest, dtype=np.float64)
    return float(abs(delta[2])), float(np.linalg.norm(delta[:2]))
