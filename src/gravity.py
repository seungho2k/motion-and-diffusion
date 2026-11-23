"""Two dancers, one A-MDM prior each, coupled only by a 'gravity' between their joints.

Every frame both dancers are denoised in one batch. At each reverse step the model's
clean-frame prediction x0_hat is nudged so that selected joints of one dancer move
toward the other dancer's joints (softened inverse-square attraction), then the DDPM
posterior step continues from the nudged x0. Later steps (t < guide_stop) run
without the pull, so the prior can re-project the pose onto plausible motion:
the trajectory is dragged, each pose stays human.

The nudge is applied to x0_hat *and* to x_t (scaled by sqrt(alpha_bar_t)), so it
persists through the remaining steps instead of being mostly overwritten by the old
x_t content. The per-frame pull is split evenly over the guided steps, so
`strength` is roughly the metres per frame a joint is dragged at 1 m distance.
With `momentum` > 0 the pull accumulates into a per-joint velocity (an impulse keeps
dragging after the pulse ends, decaying by `momentum` per frame).

The pull has two layers. A-MDM joint positions are relative to the body root and its
root velocity is bounded by the prior, so the prior alone can only reach an arm's
length. `body_pull` of the mean pull on the moving joints is therefore applied as a
rigid translation of the whole dancer outside the model (sliding / optionally
lifting bodies: the impossible part); the remainder goes through the guidance above
(limbs reaching within what the prior allows).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from amdm_runtime import AMDMRuntime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAFAN_DIR = os.path.join(ROOT, "data/lafan1")

HANDS = ("LeftHand", "RightHand")
ARMS = ("LeftForeArm", "RightForeArm", "LeftHand", "RightHand")
CORE = ("Head", "Neck", "Spine2", "Hips")

# preset name -> (joints that feel the pull, partner joints that attract)
PAIR_PRESETS = {
    "hands -> partner (all)": (HANDS, "all"),
    "arms -> partner (all)": (ARMS, "all"),
    "hands -> partner head/chest": (HANDS, CORE),
    "all <-> all": ("all", "all"),
}


@dataclass
class GravityParams:
    num_frames: int = 1200
    seed: int = 0
    clip_a: str = "dance1_subject1"
    clip_b: str = "dance2_subject3"
    frame_a: int = 300
    frame_b: int = 600
    separation: float = 1.6          # initial distance between the dancers (m)
    face_each_other: bool = True
    preset: str = "hands -> partner (all)"
    strength: float = 0.3            # peak pull (m per frame at ~1 m distance)
    base: float = 0.0                # constant pull between pulses
    pulses: int = 3                  # number of pulls over the clip
    pulse_width: float = 4.0         # frames (gaussian sigma)
    softening: float = 0.25          # m; caps the pull at short range
    repulse_radius: float = 0.18     # m; push apart closer than this (0 = off)
    guide_stop: int = 3              # pull applied for diffusion steps t >= guide_stop
    body_pull: float = 0.5           # fraction of the pull that drags the whole body rigidly
    lift: bool = False               # let the body pull lift dancers off the floor
    momentum: float = 0.8            # 0 = no memory; else velocity kept per frame
    noise_scale: float = 1.0
    clip: float = 4.0


@dataclass
class PairResult:
    joints: np.ndarray               # [2, N, J, 3] world joints (position channels)
    gravity: np.ndarray              # [N] pull strength per frame
    links: np.ndarray
    joint_names: list = field(default_factory=list)


def gravity_curve(p: GravityParams) -> np.ndarray:
    f = np.arange(p.num_frames, dtype=np.float64)
    g = np.full(p.num_frames, p.base)
    if p.pulses > 0:
        centers = (np.arange(p.pulses) + 0.5) * p.num_frames / p.pulses
        for c in centers:
            g += p.strength * np.exp(-0.5 * ((f - c) / max(p.pulse_width, 1e-3)) ** 2)
    return g


class PairGenerator:
    def __init__(self, rt: AMDMRuntime):
        self.rt = rt
        self.J = rt.J
        self.names = rt.joint_names
        self.clip_cache: dict[str, np.ndarray] = {}
        jmask = torch.ones(rt.J, 3)
        jmask[0, 0] = jmask[0, 2] = 0.0             # root joint x/z are not used (see decode)
        self.jmask = jmask
        self.std_safe = torch.where(rt.std > 1e-6, rt.std, torch.ones_like(rt.std))

    @staticmethod
    def dance_clips() -> list[str]:
        if not os.path.isdir(LAFAN_DIR):
            return []
        return sorted(f[:-4] for f in os.listdir(LAFAN_DIR) if f.endswith(".bvh"))

    def clip(self, name: str) -> np.ndarray:
        if name not in self.clip_cache:
            self.clip_cache[name] = self.rt.ds.load_new_data(os.path.join(LAFAN_DIR, name + ".bvh")).astype(np.float32)
        return self.clip_cache[name]

    def _idx(self, sel) -> list[int]:
        return list(range(self.J)) if sel == "all" else [self.names.index(n) for n in sel]

    # ------------------------------------------------------------------ world
    def _world(self, x_phys: torch.Tensor, yaw_prev: torch.Tensor, root_prev: torch.Tensor):
        """Denormalised frames [2,D] -> world joints [2,J,3] plus new (yaw, root).

        Mirrors AMDMRuntime.decode for one step (row-vector convention p_w = p_l @ R + root).
        """
        rt = self.rt
        yaw = yaw_prev + x_phys[:, 2]
        c, s = torch.cos(yaw), torch.sin(yaw)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        R = torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1),
                         torch.stack([-s, z, c], -1)], -2)                       # [2,3,3]
        step = torch.stack([x_phys[:, 0], torch.zeros_like(c), x_phys[:, 1]], -1)  # [2,3]
        root = root_prev + torch.einsum("bi,bij->bj", step, R)
        local = x_phys[:, rt.jnt_lo:rt.jnt_hi].reshape(-1, self.J, 3) * self.jmask
        world = torch.einsum("bji,bik->bjk", local, R) + root[:, None]
        return world, yaw, root

    def _pull(self, world: torch.Tensor, g: float, p: GravityParams, movers, sources) -> torch.Tensor:
        """Per-joint displacement [2,J,3] for both dancers (each pulled by the other)."""
        disp = torch.zeros_like(world)
        eps2 = p.softening ** 2
        for a, b in ((0, 1), (1, 0)):
            pm = world[a, movers]                       # [M,3]
            ps = world[b, sources]                      # [S,3]
            d = ps[None] - pm[:, None]                  # [M,S,3]
            r2 = (d * d).sum(-1)
            f = d / (r2 + eps2)[..., None] ** 1.5       # softened inverse square
            pull = g * f.mean(1)
            if p.repulse_radius > 0:
                r = r2.sqrt()
                push = torch.relu(p.repulse_radius - r) / (r + 1e-6)
                pull = pull - 0.5 * (d * push[..., None]).sum(1)
            disp[a, movers] = pull
        return disp

    # --------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, p: GravityParams, progress=None) -> PairResult:
        rt, diff, T, D = self.rt, self.rt.diff, self.rt.T, self.rt.D
        g_curve = gravity_curve(p)
        movers, sources = (self._idx(s) for s in PAIR_PRESETS[p.preset])
        gen = torch.Generator().manual_seed(p.seed)

        ca, cb = self.clip(p.clip_a), self.clip(p.clip_b)
        cond = torch.tensor(np.stack([ca[p.frame_a % len(ca)], cb[p.frame_b % len(cb)]]))
        # place the dancers on the x axis, optionally facing each other
        root = torch.tensor([[-p.separation / 2, 0.0, 0.0], [p.separation / 2, 0.0, 0.0]])
        # world forward angle = local forward angle - yaw (row-vector rotation), so
        # yaw = a - pi/2 makes A face +x and yaw = a + pi/2 makes B face -x
        base = [-np.pi / 2, np.pi / 2] if p.face_each_other else [0.0, 0.0]
        yaw = torch.tensor(base, dtype=torch.float32) + self._facing_correction(cond)

        N = p.num_frames
        joints = np.zeros((2, N, self.J, 3), np.float32)
        root_w = torch.ones(D)
        root_w[:3] = 0.0                                 # whole-body motion is handled rigidly below
        lift = torch.zeros(2, 3)                         # accumulated vertical offset (if lift)

        vel = torch.zeros(2, self.J, 3)
        prev_world = None
        n_guided = max(1, T - p.guide_stop)
        sqrt_ab = diff.sqrt_alphas_cumprod
        for i in range(N):
            g = float(g_curve[i])
            # gravity field from the previous frame's pose, accumulated into a velocity
            if prev_world is not None:
                vel = p.momentum * vel + self._pull(prev_world, g, p, movers, sources)
            body = p.body_pull * vel[:, movers].mean(1)          # [2,3] rigid drag this frame
            if not p.lift:
                body[:, 1] = 0.0
            step_disp = (vel - body[:, None]) / n_guided
            x = torch.randn(2, D, generator=gen) * p.noise_scale
            for t in range(T - 1, -1, -1):
                ts = torch.full((2,), t, dtype=torch.long)
                eps = diff.model(cond, x, diff.time_mlp(ts))
                x0 = diff.get_x0_from_xt(x, ts, eps).clamp(-p.clip, p.clip)
                if t >= p.guide_stop and float(step_disp.abs().max()) > 1e-7:
                    x0_new = self._guide(x0, step_disp, p, yaw, root, root_w)
                    x = x + sqrt_ab[t] * (x0_new - x0)     # keep x_t consistent with the nudged x0
                    x0 = x0_new
                x = rt.coef_x0[t] * x0 + rt.coef_xt[t] * x
                if t > 0:
                    x = x + diff.sigma[t] * torch.randn(2, D, generator=gen) * p.noise_scale
            x = x.clamp(-p.clip, p.clip)
            phys = rt.denorm(x)
            root = root + body * torch.tensor([1.0, 0.0, 1.0])  # slide on the floor plane
            lift = lift + body * torch.tensor([0.0, 1.0, 0.0])
            world, yaw, root = self._world(phys, yaw, root)
            world = world + lift[:, None]
            joints[:, i] = world.numpy()
            prev_world = world
            cond = x
            if progress and i % 30 == 0:
                progress(i, N)
        return PairResult(joints, g_curve.astype(np.float32), rt.links, self.names)

    def _guide(self, x0, disp, p, yaw, root, root_w):
        with torch.enable_grad():
            xp = self.rt.denorm(x0).detach().requires_grad_(True)
            world, _, _ = self._world(xp, yaw, root)
            target = (world + disp).detach()
            # feature change that moves the world joints by the pull (one Gauss-Newton-ish step)
            loss = 0.5 * ((world - target) ** 2).sum()
            (grad,) = torch.autograd.grad(loss, xp)
        delta_phys = -grad * root_w
        return (x0 + delta_phys / self.std_safe).clamp(-p.clip, p.clip)

    def _facing_correction(self, cond: torch.Tensor) -> torch.Tensor:
        """Local forward angle of each dancer's start frame (from the feet)."""
        rt = self.rt
        local = rt.denorm(cond)[:, rt.jnt_lo:rt.jnt_hi].reshape(2, self.J, 3)
        lt, rt_ = self.names.index("LeftToe"), self.names.index("RightToe")
        lf, rf = self.names.index("LeftFoot"), self.names.index("RightFoot")
        fw = (local[:, lt] - local[:, lf]) + (local[:, rt_] - local[:, rf])
        return torch.atan2(fw[:, 0], fw[:, 2])   # forward angle in the local frame
