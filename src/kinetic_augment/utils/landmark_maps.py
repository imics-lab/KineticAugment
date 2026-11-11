
import numpy as np
from .data_formats import POSE_START, POSE_END, FACE_START, FACE_END, LH_START, LH_END, RH_START, RH_END

# MediaPipe Pose landmarks have symmetric pairs
# These are indices *within the pose block* (0-32)
# Midline landmarks (0, 9, 10) are not swapped.
POSE_LEFT_TO_RIGHT = {
    1: 2, 3: 4, 5: 6, 7: 8, 11: 12, 13: 14, 15: 16, 17: 18, 19: 20,
    21: 22, 23: 24, 25: 26, 27: 28, 29: 30, 31: 32
}

# MediaPipe Face landmarks also have symmetric pairs and a midline
# These are indices *within the face block* (0-467)
FACE_LEFT_TO_RIGHT = {
    10: 10, 21: 251, 23: 253, 24: 254, 25: 255, 26: 256, 27: 257, 28: 258,
    29: 259, 30: 260, 31: 261, 33: 263, 37: 267, 39: 269, 40: 270, 46: 276,
    52: 282, 53: 283, 54: 284, 55: 285, 58: 288, 61: 291, 63: 293, 65: 295,
    66: 296, 67: 297, 69: 299, 70: 300, 78: 308, 80: 310, 81: 311, 82: 312,
    84: 314, 87: 317, 88: 318, 91: 321, 95: 324, 96: 325, 103: 332, 105: 334,
    107: 336, 109: 338, 122: 351, 132: 361, 133: 362, 136: 365, 144: 373,
    145: 374, 146: 375, 148: 377, 149: 378, 150: 379, 152: 382, 153: 380,
    154: 381, 155: 382, 157: 384, 158: 385, 159: 386, 160: 387, 161: 388,
    163: 390, 172: 397, 173: 466, 176: 399, 177: 400, 178: 402, 181: 405,
    185: 409, 187: 411, 191: 415, 193: 463, 195: 419, 197: 467, 226: 446,
    234: 454, 243: 463, 246: 466, 247: 467,
}

# Add the inverse mappings to the dictionaries
POSE_RIGHT_TO_LEFT = {v: k for k, v in POSE_LEFT_TO_RIGHT.items()}
POSE_FULL_MAP = {**POSE_LEFT_TO_RIGHT, **POSE_RIGHT_TO_LEFT}

FACE_RIGHT_TO_LEFT = {v: k for k, v in FACE_LEFT_TO_RIGHT.items()}
FACE_FULL_MAP = {**FACE_LEFT_TO_RIGHT, **FACE_RIGHT_TO_LEFT}