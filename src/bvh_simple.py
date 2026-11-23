"""Minimal BVH reader + vectorised forward kinematics (any channel order, end sites kept)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Bvh:
    names: list            # joint names (end sites named "<parent>_end")
    parents: np.ndarray    # [J]
    offsets: np.ndarray    # [J, 3]
    is_end: np.ndarray     # [J] bool
    frame_time: float
    positions: np.ndarray  # [N, J, 3] world positions (file units)

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def links(self) -> np.ndarray:
        return np.array([(p, j) for j, p in enumerate(self.parents) if p >= 0])


def _axis_rot(axis: str, deg: np.ndarray) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "Y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.stack([np.stack(r, -1) for r in m], -2)


def load_bvh(path: str) -> Bvh:
    with open(path) as f:
        lines = f.read().split("\n")
    names, parents, offsets, channels, is_end = [], [], [], [], []
    stack = []
    i = 0
    while not lines[i].strip().startswith("MOTION"):
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            names.append(tok[1]); parents.append(stack[-1] if stack else -1)
            offsets.append(None); channels.append([]); is_end.append(False)
        elif tok and tok[0] == "End":
            names.append(names[stack[-1]] + "_end"); parents.append(stack[-1])
            offsets.append(None); channels.append([]); is_end.append(True)
        elif tok and tok[0] == "{":
            stack.append(len(names) - 1)
        elif tok and tok[0] == "}":
            stack.pop()
        elif tok and tok[0] == "OFFSET":
            offsets[-1] = [float(v) for v in tok[1:4]]
        elif tok and tok[0] == "CHANNELS":
            channels[-1] = tok[2:]
        i += 1
    n_frames = int(lines[i + 1].split()[-1])
    frame_time = float(lines[i + 2].split()[-1])
    data = np.loadtxt(lines[i + 3:i + 3 + n_frames], dtype=np.float64, ndmin=2)

    J = len(names)
    off = np.array(offsets, np.float64)
    N = len(data)
    pos = np.zeros((N, J, 3))
    rot = np.zeros((N, J, 3, 3))
    col = 0
    for j in range(J):
        ch = channels[j]
        local_t = np.broadcast_to(off[j], (N, 3)).copy()
        R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
        for c in ch:
            v = data[:, col]
            col += 1
            if c.endswith("position"):
                local_t[:, "XYZ".index(c[0])] = v
            else:
                R = R @ _axis_rot(c[0], v)      # intrinsic order as listed
        p = parents[j]
        if p < 0:
            pos[:, j] = local_t
            rot[:, j] = R
        else:
            pos[:, j] = pos[:, p] + np.einsum("nij,nj->ni", rot[:, p], local_t)
            rot[:, j] = rot[:, p] @ R
    return Bvh(names, np.array(parents), off, np.array(is_end), frame_time, pos)
