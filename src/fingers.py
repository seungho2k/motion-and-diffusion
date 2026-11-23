"""Procedural fingers for LaFAN1 hands (the dataset has none).

Each hand gets 5 fingers x 3 segments in a soft 'ballet hand' rest shape, defined in
the LaFAN1 hand frame (+x along the fingers, -y = palm normal, thumb toward +z on
the left hand and -z on the right; measured from the data). The finger points are
simulated as damped Verlet chains: they are sprung toward the rest shape carried by
the hand, lag behind fast wrist motion (follow-through), and can be pulled by the
same partner gravity that drives the bodies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (name, knuckle x, knuckle s, knuckle palm-depth, segment lengths, flex per joint [deg], splay [deg])
_FINGERS = [
    ("index",  0.085,  0.032, 0.0, (0.040, 0.025, 0.020), (12, 14, 8), -4),
    ("middle", 0.088,  0.011, 0.0, (0.045, 0.028, 0.022), (28, 26, 14), 0),
    ("ring",   0.083, -0.011, 0.0, (0.042, 0.026, 0.021), (18, 18, 10), 4),
    ("pinky",  0.074, -0.030, 0.0, (0.032, 0.020, 0.018), (12, 12, 8), 9),
]
_THUMB = ("thumb", 0.025, 0.028, 0.012, (0.040, 0.032, 0.025))

FINGER_LINKS_PER_HAND = []   # (a, b) indices within one hand's point array


@dataclass
class FingerParams:
    stiffness: float = 220.0     # spring toward the rest shape (1/s^2)
    damping: float = 0.12        # velocity loss per substep
    gravity_gain: float = 1.0    # how strongly partner gravity pulls the fingertips
    curl: float = 1.0            # scales the rest flexion (0 = flat hand)
    substeps: int = 4


def _rest_points(sign: float, curl: float) -> np.ndarray:
    """Rest points of one hand in its local frame: [5 fingers, 4 points (knuckle..tip), 3]."""
    pts = []
    for _, kx, ks, kd, lens, flex, splay in _FINGERS:
        p = np.array([kx, -kd, sign * ks])
        chain = [p.copy()]
        ang = 0.0
        sp = np.deg2rad(splay)
        for L, fl in zip(lens, flex):
            ang += np.deg2rad(fl) * curl
            d = np.array([np.cos(ang) * np.cos(sp), -np.sin(ang), sign * np.cos(ang) * np.sin(-sp)])
            p = p + L * d
            chain.append(p.copy())
        pts.append(chain)
    # thumb: leaves the palm toward the thumb side, curls toward the middle finger
    _, kx, ks, kd, lens = _THUMB
    p = np.array([kx, -kd, sign * ks])
    chain = [p.copy()]
    base = np.array([0.55, -0.45, sign * 0.70])
    base /= np.linalg.norm(base)
    across = np.array([0.35, -0.55, -sign * 0.75])
    across /= np.linalg.norm(across)
    for k, L in enumerate(lens):
        w = min(1.0, 0.25 * curl * (k + 1))
        d = (1 - w) * base + w * across
        d /= np.linalg.norm(d)
        p = p + L * d
        chain.append(p.copy())
    pts.append(chain)
    return np.array(pts)                     # [5,4,3]


def finger_links() -> np.ndarray:
    """Links within one hand's flattened [wrist + 20 finger points] array."""
    links = []
    for f in range(5):
        base = 1 + f * 4
        links.append((0, base))              # wrist -> knuckle
        for k in range(3):
            links.append((base + k, base + k + 1))
    return np.array(links)


def simulate_fingers(hand_pos: np.ndarray, hand_rot: np.ndarray, gravity: np.ndarray,
                     partner_joints: np.ndarray, fp: FingerParams, softening: float = 0.25,
                     fps: float = 30.0) -> np.ndarray:
    """Simulate both hands of one dancer.

    hand_pos       [N, 2, 3]       wrist positions (left, right)
    hand_rot       [N, 2, 3, 3]    world hand orientations (columns = local axes)
    gravity        [N]             partner pull strength per frame
    partner_joints [N, J, 3]       the other dancer's joints (attractors)
    returns        [N, 2, 21, 3]   wrist + 5x4 finger points per hand
    """
    N = len(hand_pos)
    rest = [_rest_points(+1.0, fp.curl), _rest_points(-1.0, fp.curl)]     # left, right
    out = np.zeros((N, 2, 21, 3))
    dt = 1.0 / fps / fp.substeps
    seg_len = [np.linalg.norm(np.diff(r, axis=1), axis=-1) for r in rest]  # [5,3]
    for h in range(2):
        target0 = hand_pos[0, h] + rest[h].reshape(-1, 3) @ hand_rot[0, h].T
        x = target0.reshape(5, 4, 3).copy()
        x_prev = x.copy()
        for i in range(N):
            for sub in range(fp.substeps):
                a = (sub + 1) / fp.substeps
                if i > 0:
                    hp = (1 - a) * hand_pos[i - 1, h] + a * hand_pos[i, h]
                    R = hand_rot[i] [h] if a == 1 else _slerp_mat(hand_rot[i - 1, h], hand_rot[i, h], a)
                else:
                    hp, R = hand_pos[0, h], hand_rot[0, h]
                target = (hp + rest[h].reshape(-1, 3) @ R.T).reshape(5, 4, 3)
                acc = fp.stiffness * (target - x)
                g = gravity[i] * fp.gravity_gain
                if g > 0:
                    d = partner_joints[i][None, None] - x[:, :, None]          # [5,4,J,3]
                    r2 = (d * d).sum(-1)
                    pull = (d / (r2 + softening ** 2)[..., None] ** 1.5).mean(2)
                    weight = np.array([0.0, 0.3, 0.6, 1.0])[None, :, None]   # tips feel it most
                    acc += g * 900.0 * pull * weight
                x_new = x + (x - x_prev) * (1 - fp.damping) + acc * dt * dt
                x_prev, x = x, x_new
                x[:, 0] = target[:, 0]                                       # knuckles ride the hand
                for _ in range(3):                                           # keep segment lengths
                    for k in range(3):
                        v = x[:, k + 1] - x[:, k]
                        L = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
                        corr = v * (1 - seg_len[h][:, k:k + 1] / L)
                        if k == 0:
                            x[:, k + 1] -= corr
                        else:
                            x[:, k] += 0.5 * corr
                            x[:, k + 1] -= 0.5 * corr
            out[i, h, 0] = hand_pos[i, h]
            out[i, h, 1:] = x.reshape(-1, 3)
    return out


def _slerp_mat(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    """Cheap rotation interpolation: blend + re-orthonormalise (fine for sub-frame steps)."""
    M = (1 - a) * R0 + a * R1
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R
