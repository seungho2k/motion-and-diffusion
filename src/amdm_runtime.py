"""A-MDM runtime: model loading, per-frame partial denoising, decoding to joints.

Diffusion level convention (k):
    k = T   -> pure noise (no denoising step run)
    k = 0   -> fully denoised
    fractional k is linearly interpolated between neighbouring levels.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMDM_DIR = os.path.join(ROOT, "third_party", "AMDM")
if AMDM_DIR not in sys.path:
    sys.path.insert(0, AMDM_DIR)

import yaml  # noqa: E402

import dataset.dataset_builder as dataset_builder  # noqa: E402
import dataset.util.bvh as bvh_util  # noqa: E402
import model.model_builder as model_builder  # noqa: E402

CHECKPOINTS = {
    "lafan1": "output/base/amdm_lafan1",
    "100style": "output/base/amdm_100style",
}
SAMPLE_BVH = os.path.join(AMDM_DIR, "data/LAFAN1/aiming1_subject4.bvh")


@dataclass
class GenParams:
    num_frames: int = 900
    seed: int = 0
    init_frame: int = 100          # index into the sample bvh used as starting pose
    feed: str = "xt"               # fed back as condition: "xt" | "x0" | "mix" | "final"
    feed_mix: float = 0.5          # for feed == "mix": 0 -> xt, 1 -> x0
    noise_scale: float = 1.0       # multiplier on all injected noise
    noise_coherence: float = 0.0   # AR(1) correlation of noise across frames (0 = white)
    root_source: str = "display"   # root deltas integrated from: "display" | "x0" | "none"
    clip: float = 4.0              # clamp x0-prediction and fed-back condition (normalised units)
    compute_final: bool = False    # also run each frame's chain on to t=0 (same noises) -> x_final


DECODE_MODES = ("position", "rotation")


@dataclass
class GenResult:
    levels: np.ndarray             # [N] diffusion level used per frame
    feats_xt: np.ndarray           # [N, D] denormalised features of the (partially) noisy state
    feats_x0: np.ndarray           # [N, D] denormalised features of the model's x0 prediction
    feats_final: np.ndarray | None # [N, D] same frame's chain continued to t=0 (if computed)
    root_xt: np.ndarray            # [N, 3] (dx, dz, dyaw) integrated for the xt skeleton
    root_x0: np.ndarray            # [N, 3] (dx, dz, dyaw) integrated for the x0 skeleton
    links: np.ndarray
    joint_names: list = field(default_factory=list)
    decode_mode: str = ""
    joints_xt: np.ndarray | None = None   # [N, J, 3] world-space joints (filled by decode_result)
    joints_x0: np.ndarray | None = None
    joints_final: np.ndarray | None = None


class AMDMRuntime:
    def __init__(self, name: str = "lafan1", device: str = "cpu"):
        ckpt_dir = os.path.join(AMDM_DIR, CHECKPOINTS[name])
        cfg_path = os.path.join(ckpt_dir, "config.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["path"] = os.path.join(AMDM_DIR, cfg["data"]["path"])
        tmp_cfg = os.path.join(ROOT, ".cache", f"{name}_config.yaml")
        os.makedirs(os.path.dirname(tmp_cfg), exist_ok=True)
        with open(tmp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        self.device = torch.device(device)
        self.ds = dataset_builder.build_dataset(tmp_cfg, load_full_dataset=False)
        self.model = model_builder.build_model(tmp_cfg, self.ds, device)
        sd = torch.load(os.path.join(ckpt_dir, "model_param.pth"), map_location=device)
        self.model.load_state_dict(sd)
        self.model.eval()
        self.diff = self.model.ema_diffusion if self.model.use_ema else self.model.diffusion
        self.T = self.diff.T
        self.D = int(self.ds.frame_dim)
        self.J = len(self.ds.joint_names)
        self.links = np.asarray(self.ds.links)
        self.joint_names = list(self.ds.joint_names)
        self.jnt_lo, self.jnt_hi = [int(v) for v in self.ds.joint_dim_lst]
        self.ang_lo, self.ang_hi = [int(v) for v in self.ds.angle_dim_lst]
        self.height_index = int(self.ds.height_index)
        self.parents = bvh_util.get_parent_from_link(self.links)
        self.offsets = np.asarray(self.ds.joint_offset, dtype=np.float64)  # [J, 3] meters

        ab = self.diff.alphas_cumprod
        ab_prev = torch.cat([torch.ones(1, device=ab.device), ab[:-1]])
        betas, alphas = self.diff.betas, self.diff.alphas
        self.coef_x0 = betas * ab_prev.sqrt() / (1 - ab)
        self.coef_xt = (1 - ab_prev) * alphas.sqrt() / (1 - ab)

        self.std = torch.tensor(self.ds.std, dtype=torch.float32, device=self.device)
        self.avg = torch.tensor(self.ds.avg, dtype=torch.float32, device=self.device)

        self._sample_frames = None

    # ------------------------------------------------------------------ data
    def sample_frames(self) -> np.ndarray:
        if self._sample_frames is None:
            self._sample_frames = self.ds.load_new_data(SAMPLE_BVH).astype(np.float32)
        return self._sample_frames

    def denorm(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.avg

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def denoise_to(self, cond: torch.Tensor, level: float, noises: torch.Tensor, clip: float = 4.0,
                   final: bool = False):
        """Run the reverse chain for one frame, stopping at (fractional) `level`.

        cond:   [D] normalised condition frame
        noises: [T+1, D] noise for this frame; noises[T] is the initial x_T,
                noises[t] (t<T) is added after denoising step t.
        Returns (x_level, x0_hat, x_final) [D] normalised; x_final is the chain
        continued from level down to 0 with the same noises (None unless final).
        """
        level = float(np.clip(level, 0.0, self.T))
        k_lo = int(np.floor(level))
        k_hi = min(k_lo + 1, self.T)
        frac = level - k_lo

        c = cond[None]
        x = noises[self.T][None]
        states = {self.T: x}
        x0_hat = x  # before any step, "prediction" is undefined: use noise
        x0_at_level = x0_hat
        stop = 0 if final else k_lo
        for t in range(self.T - 1, stop - 1, -1):
            ts = torch.full((1,), t, device=self.device, dtype=torch.long)
            te = self.diff.time_mlp(ts)
            eps = self.diff.model(c, x, te)
            x0_hat = self.diff.get_x0_from_xt(x, ts, eps).clamp(-clip, clip)
            # DDPM posterior mean written with x0 (identical to remove_noise when unclipped)
            x = self.coef_x0[t] * x0_hat + self.coef_xt[t] * x
            if t > 0:
                x = x + self.diff.sigma[t] * noises[t][None]
            states[t] = x
            if t == k_lo:
                x0_at_level = x0_hat
        x_lo = states[k_lo]
        x_hi = states[k_hi]
        x_level = (1 - frac) * x_lo + frac * x_hi
        x_final = states[0][0] if final else None
        return x_level[0], x0_at_level[0], x_final

    @torch.no_grad()
    def generate(self, levels: np.ndarray, p: GenParams, decode_mode: str = "position") -> GenResult:
        N = len(levels)
        g = torch.Generator(device="cpu").manual_seed(p.seed)
        frames = self.sample_frames()
        cond = torch.tensor(frames[p.init_frame % len(frames)], device=self.device)

        rho = float(np.clip(p.noise_coherence, 0.0, 0.999))
        noise_state = torch.randn(self.T + 1, self.D, generator=g)

        xt_seq = torch.zeros(N, self.D)
        x0_seq = torch.zeros(N, self.D)
        fin_seq = torch.zeros(N, self.D)
        want_final = p.compute_final or p.feed == "final"
        for i in range(N):
            if i > 0:
                fresh = torch.randn(self.T + 1, self.D, generator=g)
                noise_state = rho * noise_state + np.sqrt(1 - rho * rho) * fresh
            noises = (noise_state * p.noise_scale).to(self.device)
            x_l, x0, x_fin = self.denoise_to(cond, levels[i], noises, p.clip, final=want_final)
            xt_seq[i] = x_l.cpu()
            x0_seq[i] = x0.cpu()
            if x_fin is not None:
                fin_seq[i] = x_fin.cpu()
            if p.feed == "final":
                cond = x_fin
            elif p.feed == "x0":
                cond = x0
            elif p.feed == "mix":
                cond = (1 - p.feed_mix) * x_l + p.feed_mix * x0
            else:
                cond = x_l
            cond = cond.clamp(-p.clip, p.clip)

        xt_d = self.denorm(xt_seq.to(self.device)).cpu().numpy()
        x0_d = self.denorm(x0_seq.to(self.device)).cpu().numpy()
        fin_d = self.denorm(fin_seq.to(self.device)).cpu().numpy() if want_final else None
        if p.root_source == "x0":
            root_xt = root_x0 = x0_d[:, :3]
        elif p.root_source == "none":
            root_xt = root_x0 = np.zeros_like(xt_d[:, :3])
        else:
            root_xt = root_x0 = xt_d[:, :3]
        res = GenResult(
            levels=np.asarray(levels, dtype=np.float32),
            feats_xt=xt_d, feats_x0=x0_d, feats_final=fin_d, root_xt=root_xt, root_x0=root_x0,
            links=self.links, joint_names=self.joint_names,
        )
        return self.decode_result(res, decode_mode)

    def decode_result(self, res: GenResult, mode: str) -> GenResult:
        """(Re)decode a result's features to joints; cheap, no model calls."""
        if res.decode_mode != mode:
            res.joints_xt = self.decode(res.feats_xt, res.root_xt, mode)
            res.joints_x0 = self.decode(res.feats_x0, res.root_x0, mode)
            # final shares the xt root path so the two skeletons overlay in place
            res.joints_final = (None if res.feats_final is None
                                else self.decode(res.feats_final, res.root_xt, mode))
            res.decode_mode = mode
        return res

    # --------------------------------------------------------------- decode
    def decode(self, x: np.ndarray, root: np.ndarray | None = None, mode: str = "position") -> np.ndarray:
        """Denormalised frames [N, D] -> world joints [N, J, 3] (y up, meters).

        mode="position": joint-position channels used directly. Noise shows up as
                         spheres scattering freely (bone lengths not preserved).
        mode="rotation": forward kinematics from the 6D local rotations + skeleton
                         offsets. Bone lengths always preserved; noise shows up as
                         the skeleton twisting.
        """
        if root is None:
            root = x[:, :3]
        if mode == "rotation":
            jnts = self.fk(x)
        else:
            jnts = x[:, self.jnt_lo:self.jnt_hi].reshape(-1, self.J, 3).copy()
            jnts[:, 0, 0] = 0.0
            jnts[:, 0, 2] = 0.0
        yaws = np.cumsum(root[:, 2])
        out = np.zeros_like(jnts)
        pos = np.zeros(3)
        for i in range(len(x)):
            R = _rot_yaw(yaws[i])
            if i > 0:
                pos = pos + np.array([root[i, 0], 0.0, root[i, 1]]) @ R
            out[i] = jnts[i] @ R + pos
        return out

    def fk(self, x: np.ndarray) -> np.ndarray:
        """Local-frame FK from 6D rotations (root at origin, root height from features)."""
        n = len(x)
        m6d = x[:, self.ang_lo:self.ang_hi].reshape(n, self.J, 6)
        rot = _m6d_to_rotmat(m6d)  # [N, J, 3, 3]
        ori = np.zeros((n, self.J, 3, 3))
        pos = np.zeros((n, self.J, 3))
        for j in range(self.J):
            pj = self.parents[j]
            if pj == -1:
                ori[:, j] = rot[:, j]
            else:
                ori[:, j] = ori[:, pj] @ rot[:, j]
                pos[:, j] = pos[:, pj] + ori[:, pj] @ self.offsets[j]
        pos[..., 1] += x[:, [self.height_index]]
        return pos


def _m6d_to_rotmat(m6d: np.ndarray) -> np.ndarray:
    """Same convention as AMDM geo.m6d_to_rotmat (rows b1, b2, b3)."""
    a1, a2 = m6d[..., :3], m6d[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def _rot_yaw(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
