"""Joint-tracking camera: fixed offset (azimuth/elevation/distance) from a joint, looking at it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackCam:
    joint: int = 0
    azimuth: float = 30.0     # degrees, around world +y
    elevation: float = 10.0   # degrees
    distance: float = 3.0     # meters
    fov: float = 40.0         # vertical fov, degrees
    target_offset: tuple = (0.0, 0.0, 0.0)

    def pose(self, joints: np.ndarray):
        """joints [J,3] -> (position [3], wxyz [4]) in viser's OpenCV camera convention."""
        target = joints[self.joint] + np.asarray(self.target_offset)
        az, el = np.deg2rad(self.azimuth), np.deg2rad(self.elevation)
        offset = self.distance * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        pos = target + offset
        return pos, look_at_wxyz(pos, target)


def look_at_wxyz(pos, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=1)  # columns: camera x (right), y (down), z (forward)
    return mat_to_wxyz(R)


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)
