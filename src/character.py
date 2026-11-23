"""Skinned character (glTF) + retargeting from LaFAN1 joint positions.

The character is the Quaternius "Universal Base Characters" (CC0) with a UE-style
rig. Bone poses are produced from the 22 LaFAN1 joint positions only, so the same
code drives the mesh for both decode modes (position / rotation):

  * pelvis and chest get a full orientation from two directions (up + left-right)
  * every other mapped bone gets the minimal swing that points it at its child
    joint, on top of its parent's rotation (twist propagates down the chain)
  * unmapped bones (root, fingers, leaf bones) follow their parent rigidly
  * stretch modes for bone positions:
      "proportional": each mesh segment is scaled by (current joint distance /
                      LaFAN1 rest bone length), so clean poses keep the mesh's own
                      proportions and only noise stretches / tears the body
      "exact":        each mapped bone sits exactly on its target joint (proportion
                      mismatch between the rigs shows even on clean poses)
      "rigid":        rest bone lengths, only rotations follow
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from pygltflib import GLTF2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARACTERS = {
    "male": os.path.join(ROOT, "assets/characters/Superhero_Male_FullBody.gltf"),
    "female": os.path.join(ROOT, "assets/characters/Superhero_Female_FullBody.gltf"),
}

# LaFAN1 joint -> UE-style bone
LAFAN_TO_BONE = {
    "Hips": "pelvis", "Spine": "spine_01", "Spine1": "spine_02", "Spine2": "spine_03",
    "Neck": "neck_01", "Head": "Head",
    "LeftShoulder": "clavicle_l", "LeftArm": "upperarm_l", "LeftForeArm": "lowerarm_l", "LeftHand": "hand_l",
    "RightShoulder": "clavicle_r", "RightArm": "upperarm_r", "RightForeArm": "lowerarm_r", "RightHand": "hand_r",
    "LeftUpLeg": "thigh_l", "LeftLeg": "calf_l", "LeftFoot": "foot_l", "LeftToe": "ball_l",
    "RightUpLeg": "thigh_r", "RightLeg": "calf_r", "RightFoot": "foot_r", "RightToe": "ball_r",
}
# bone -> child bone whose direction it should point along
AIM_CHILD = {
    "spine_01": "spine_02", "spine_02": "spine_03", "neck_01": "Head",
    "clavicle_l": "upperarm_l", "upperarm_l": "lowerarm_l", "lowerarm_l": "hand_l",
    "clavicle_r": "upperarm_r", "upperarm_r": "lowerarm_r", "lowerarm_r": "hand_r",
    "thigh_l": "calf_l", "calf_l": "foot_l", "foot_l": "ball_l",
    "thigh_r": "calf_r", "calf_r": "foot_r", "foot_r": "ball_r",
}
# bones whose full orientation comes from (up, side) frames: bone -> (up_to, side_from, side_to)
FRAMED = {
    "pelvis": ("spine_03", "thigh_r", "thigh_l"),
    "spine_03": ("neck_01", "upperarm_r", "upperarm_l"),
}


@dataclass
class Character:
    vertices: np.ndarray        # [V, 3] bind pose (scaled)
    faces: np.ndarray           # [F, 3]
    skin_weights: np.ndarray    # [V, B]
    bone_names: list
    parents: np.ndarray         # [B] parent bone index or -1
    rest_R: np.ndarray          # [B, 3, 3] bind world rotations
    rest_p: np.ndarray          # [B, 3]    bind world positions (scaled)
    order: np.ndarray           # topological order of bones

    @property
    def rest_wxyz(self) -> np.ndarray:
        return np.stack([mat_to_wxyz(R) for R in self.rest_R])


def load_character(name: str = "male", target_hip_height: float | None = None) -> Character:
    path = CHARACTERS[name]
    g = GLTF2().load(path)
    base = os.path.dirname(path)

    def accessor(i):
        acc = g.accessors[i]
        bv = g.bufferViews[acc.bufferView]
        buf = g.buffers[bv.buffer]
        with open(os.path.join(base, buf.uri), "rb") as f:
            f.seek((bv.byteOffset or 0) + (acc.byteOffset or 0))
            comp = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
                    5125: np.uint32, 5126: np.float32}[acc.componentType]
            ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc.type]
            stride = bv.byteStride or np.dtype(comp).itemsize * ncomp
            raw = f.read(stride * (acc.count - 1) + np.dtype(comp).itemsize * ncomp)
        arr = np.ndarray((acc.count, ncomp), dtype=comp, buffer=raw.ljust(stride * acc.count, b"\0"),
                         strides=(stride, np.dtype(comp).itemsize)).copy()
        if acc.normalized and comp != np.float32:
            arr = arr.astype(np.float32) / np.iinfo(comp).max
        return arr

    skin = g.skins[0]
    joints = skin.joints
    names = [g.nodes[j].name for j in joints]
    node_to_bone = {n: i for i, n in enumerate(joints)}
    parents = -np.ones(len(joints), dtype=int)
    for ni, node in enumerate(g.nodes):
        for c in node.children or []:
            if c in node_to_bone and ni in node_to_bone:
                parents[node_to_bone[c]] = node_to_bone[ni]

    ibm = accessor(skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major
    bind = np.linalg.inv(ibm)
    rest_p = bind[:, :3, 3].copy()
    rest_R = np.stack([_orthonormalize(M[:3, :3]) for M in bind])

    verts, faces, weights = [], [], []
    v_off = 0
    for mi, node in enumerate(g.nodes):
        if node.mesh is None or node.skin is None:
            continue
        for prim in g.meshes[node.mesh].primitives:
            pos = accessor(prim.attributes.POSITION)
            jt = accessor(prim.attributes.JOINTS_0).astype(int)
            wt = accessor(prim.attributes.WEIGHTS_0).astype(np.float32)
            idx = accessor(prim.indices).reshape(-1, 3).astype(np.int64)
            W = np.zeros((len(pos), len(joints)), np.float32)
            np.add.at(W, (np.repeat(np.arange(len(pos)), 4), jt.reshape(-1)), wt.reshape(-1))
            verts.append(pos)
            faces.append(idx + v_off)
            weights.append(W)
            v_off += len(pos)
    V = np.concatenate(verts).astype(np.float64)
    F = np.concatenate(faces)
    W = np.concatenate(weights)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-8)

    if target_hip_height is not None:
        s = target_hip_height / rest_p[names.index("pelvis"), 1]
        V *= s
        rest_p *= s

    order, seen = [], set()

    def visit(b):
        if b in seen:
            return
        if parents[b] >= 0:
            visit(parents[b])
        seen.add(b)
        order.append(b)
    for b in range(len(joints)):
        visit(b)

    return Character(V, F, W, names, parents, rest_R, rest_p, np.array(order))


class Retargeter:
    def __init__(self, ch: Character, lafan_joint_names: list, lafan_parents=None, lafan_offsets=None):
        self.ch = ch
        self.b = {n: i for i, n in enumerate(ch.bone_names)}
        self.target_of = {self.b[bone]: lafan_joint_names.index(j) for j, bone in LAFAN_TO_BONE.items()}
        self.aim = {self.b[k]: self.b[v] for k, v in AIM_CHILD.items()}
        self.framed = {self.b[k]: tuple(self.b[x] for x in v) for k, v in FRAMED.items()}
        # LaFAN1 rest length of the segment ending at each mapped bone's joint
        self.rest_len = {}
        if lafan_parents is not None and lafan_offsets is not None:
            for bone, j in self.target_of.items():
                if lafan_parents[j] >= 0:
                    self.rest_len[bone] = (j, int(lafan_parents[j]), float(np.linalg.norm(lafan_offsets[j])))

    def target_bone(self, lafan_joint: int) -> int:
        """Mesh bone index driven by a LaFAN1 joint index."""
        return next(b for b, j in self.target_of.items() if j == lafan_joint)

    def pose(self, P: np.ndarray, stretch: str = "proportional"):
        """LaFAN joints [22,3] -> (bone wxyz [B,4], bone positions [B,3])."""
        ch = self.ch
        B = len(ch.bone_names)
        R = np.zeros((B, 3, 3))
        p = np.zeros((B, 3))
        delta = np.zeros((B, 3, 3))  # current world rotation * rest^-1
        tgt = lambda bone: P[self.target_of[bone]]
        pelvis = self.b["pelvis"]

        for bi in ch.order:
            par = ch.parents[bi]
            dpar = delta[par] if par >= 0 else None
            if bi in self.framed:
                up_b, sa, sb = self.framed[bi]
                rest_f = _frame(ch.rest_p[up_b] - ch.rest_p[bi], ch.rest_p[sb] - ch.rest_p[sa])
                cur_f = _frame(tgt(up_b) - tgt(bi), tgt(sb) - tgt(sa))
                d = cur_f @ rest_f.T
            elif par < 0:
                d = np.eye(3)  # root: fixed up later
            elif bi in self.aim:
                c = self.aim[bi]
                d_rest = dpar @ (ch.rest_p[c] - ch.rest_p[bi])
                d_tgt = tgt(c) - tgt(bi)
                d = _rot_between(d_rest, d_tgt) @ dpar
            else:
                d = dpar
            delta[bi] = d
            R[bi] = d @ ch.rest_R[bi]
            if bi == pelvis or (bi in self.target_of and stretch == "exact"):
                p[bi] = tgt(bi)
            elif par >= 0 and stretch == "proportional" and bi in self.rest_len:
                j, jp, L = self.rest_len[bi]
                ratio = np.linalg.norm(P[j] - P[jp]) / max(L, 1e-6)
                p[bi] = p[par] + ratio * (delta[par] @ (ch.rest_p[bi] - ch.rest_p[par]))
            elif par >= 0:
                p[bi] = p[par] + delta[par] @ (ch.rest_p[bi] - ch.rest_p[par])
            else:
                p[bi] = ch.rest_p[bi]

        # root sits above the pelvis: carry it rigidly with the pelvis
        for bi in range(B):
            if ch.parents[bi] < 0:
                delta[bi] = delta[pelvis]
                R[bi] = delta[pelvis] @ ch.rest_R[bi]
                p[bi] = p[pelvis] + delta[pelvis] @ (ch.rest_p[bi] - ch.rest_p[pelvis])
        return np.stack([mat_to_wxyz(r) for r in R]), p


def _frame(up, side):
    up = up / (np.linalg.norm(up) + 1e-9)
    side = side - (side @ up) * up
    side = side / (np.linalg.norm(side) + 1e-9)
    fwd = np.cross(side, up)
    return np.stack([side, up, fwd], axis=1)


def _rot_between(a, b):
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _orthonormalize(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    from camera import mat_to_wxyz as _m
    return _m(R)
