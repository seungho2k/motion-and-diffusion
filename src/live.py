"""Endless A-MDM generation in a child process (no GIL contention with the viewer).

The worker keeps generating new frames forever for B parallel motions ("lanes"),
batched in one model call. The diffusion level follows a cyclic schedule
(`levels`, repeated), so e.g. a ramp cycle repeats while the motion itself is always
new. Settings and restarts arrive through a command queue; frames leave through a
small bounded queue, so the worker naturally pauses when the viewer does.

Measured on an M5 (fully denoised frames): one motion on one CPU thread ~27 ms;
three motions batched on MPS ~30 ms (MPS cost barely grows with batch size), while
three CPU processes in parallel only reached ~20 frames/s. So a single motion runs on
a CPU worker and several motions run batched on an MPS worker; both workers are
started once and only the one in use generates.
"""
from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class LiveFrame:
    index: int
    level: float
    joints: np.ndarray        # [B,J,3] shown state (partially denoised at `level`)
    joints_final: np.ndarray  # [B,J,3] same frame denoised to t=0 (if requested, else = joints)


def _worker(cmd_q, out_q, device: str = "cpu"):
    import torch

    from amdm_runtime import AMDMRuntime, _rot_yaw

    torch.set_num_threads(1)
    rt = AMDMRuntime("lafan1", device=device)
    s = {"levels": np.zeros(1), "feed": "xt", "root": "display", "decode": "position",
         "noise_scale": 1.0, "noise_coherence": 0.0, "clip": 4.0, "want_final": False, "skip": 1,
         # hand holding: pairs (lane_a, joint_a, lane_b, joint_b), lane slots [B,3], pull per frame (0..1)
         "hand_pairs": [], "slots": None, "hand_pull": 0.0, "guide_stop": 0}
    names = rt.joint_names
    arm_chain = {names.index("LeftHand"): [(names.index("LeftForeArm"), 0.6), (names.index("LeftArm"), 0.25)],
                 names.index("RightHand"): [(names.index("RightForeArm"), 0.6), (names.index("RightArm"), 0.25)]}
    jmask = torch.ones(rt.J, 3, device=device)
    jmask[0, 0] = jmask[0, 2] = 0.0
    std_safe = torch.where(rt.std > 1e-6, rt.std, torch.ones_like(rt.std))
    feat_w = torch.ones(rt.D, device=device)
    feat_w[:3] = 0.0                       # hands reach with the limbs; root motion stays the prior's

    def world_t(xp, yaw_prev, pos_prev):
        """Denormalised [B,D] -> world joints [B,J,3] (torch, one step of the root integration)."""
        yw = yaw_prev + xp[:, 2]
        c, sn = torch.cos(yw), torch.sin(yw)
        z, o = torch.zeros_like(c), torch.ones_like(c)
        R = torch.stack([torch.stack([c, z, sn], -1), torch.stack([z, o, z], -1),
                         torch.stack([-sn, z, c], -1)], -2)
        step = torch.stack([xp[:, 0], z, xp[:, 1]], -1)
        root = pos_prev + torch.einsum("bi,bij->bj", step, R)
        local = xp[:, rt.jnt_lo:rt.jnt_hi].reshape(-1, rt.J, 3) * jmask
        return torch.einsum("bji,bik->bjk", local, R) + root[:, None]

    def make_hand_guide(yaw_prev, pos_prev):
        pairs, slots = s["hand_pairs"], s["slots"]
        if not pairs or s["hand_pull"] <= 0 or slots is None:
            return None
        slots_t = torch.tensor(np.asarray(slots, np.float32), device=device)
        yaw_t = torch.tensor(yaw_prev, dtype=torch.float32, device=device)
        pos_t = torch.tensor(pos_prev, dtype=torch.float32, device=device)
        k = float(s["hand_pull"])          # fraction of the remaining gap closed per diffusion step

        def guide(x0, t):
            with torch.enable_grad():
                xp = rt.denorm(x0).detach().requires_grad_(True)
                w = world_t(xp, yaw_t, pos_t)
                # each body placed at its slot (hips pinned): the frame in which hands should meet
                placed = w - w[:, :1] + slots_t[:, None]
                disp = torch.zeros_like(w)
                for la, ja, lb, jb in pairs:
                    gap = (placed[lb, jb] - placed[la, ja]).detach()
                    for lane, j, sgn in ((la, ja, 1.0), (lb, jb, -1.0)):
                        # move the whole arm (hand, forearm, upper arm) so the pose stays an arm
                        disp[lane, j] += sgn * 0.5 * k * gap
                        for jj, wt in arm_chain.get(j, []):
                            disp[lane, jj] += sgn * 0.5 * k * wt * gap
                loss = 0.5 * ((w - (w + disp).detach()) ** 2).sum()
                (g,) = torch.autograd.grad(loss, xp)
            return x0 - g * feat_w / std_safe
        return guide

    cond = gens = noise_state = None
    epoch, i = 0, 0
    yaw = pos = None

    def local(d):  # d [D] -> [J,3]
        if s["decode"] == "rotation":
            return rt.fk(d[None])[0]
        j = d[rt.jnt_lo:rt.jnt_hi].reshape(rt.J, 3).copy()
        j[0, 0] = j[0, 2] = 0.0
        return j

    pending = None
    while True:
        try:
            while True:
                cmd, arg = cmd_q.get(timeout=0.05) if cond is None else cmd_q.get_nowait()
                if cmd == "stop":
                    return
                if cmd == "set":
                    s.update(arg)
                elif cmd == "idle":
                    cond, pending = None, None
                elif cmd == "reset":
                    starts, seeds, epoch = arg               # starts [B,D], seeds [B]
                    B = len(starts)
                    cond = torch.tensor(np.asarray(starts, np.float32), device=device)
                    gens = [torch.Generator().manual_seed(int(sd)) for sd in seeds]
                    noise_state = torch.stack([torch.randn(rt.T + 1, rt.D, generator=g) for g in gens])
                    i, yaw, pos, pending = 0, np.zeros(B), np.zeros((B, 3)), None
        except queue.Empty:
            pass
        if cond is None:
            continue
        if pending is not None:
            try:
                out_q.put(pending, timeout=0.05)
                pending = None
            except queue.Full:
                continue

        B = cond.shape[0]
        for _ in range(max(1, int(s["skip"]))):
            levels = s["levels"]
            level = float(levels[i % len(levels)])
            rho = float(np.clip(s["noise_coherence"], 0.0, 0.999))
            fresh = torch.stack([torch.randn(rt.T + 1, rt.D, generator=g) for g in gens])
            noise_state = rho * noise_state + np.sqrt(1 - rho * rho) * fresh
            want_final = s["want_final"] or s["feed"] == "final"
            with torch.no_grad():      # (not inference_mode: hand guidance needs autograd)
                x_l, x0, x_f = rt.denoise_to_batch(cond, level, (noise_state * s["noise_scale"]).to(device),
                                                   s["clip"], final=want_final,
                                                   guide=make_hand_guide(yaw, pos), guide_stop=int(s["guide_stop"]))
            nxt = {"x0": x0, "final": x_f}.get(s["feed"], x_l)
            if s["feed"] == "mix":
                nxt = 0.5 * (x_l + x0)
            cond = nxt.clamp(-s["clip"], s["clip"])
            d_l = rt.denorm(x_l).cpu().numpy()
            d_0 = rt.denorm(x0).cpu().numpy()
            d_f = rt.denorm(x_f).cpu().numpy() if x_f is not None else d_l
            J_l = np.zeros((B, rt.J, 3), np.float32)
            J_f = np.zeros((B, rt.J, 3), np.float32)
            for b in range(B):
                root = {"x0": d_0[b, :3], "final": d_f[b, :3], "none": np.zeros(3)}.get(s["root"], d_l[b, :3])
                yaw[b] += float(root[2])
                R = _rot_yaw(yaw[b])
                if i > 0:
                    pos[b] = pos[b] + np.array([root[0], 0.0, root[1]]) @ R
                J_l[b] = local(d_l[b]) @ R + pos[b]
                J_f[b] = local(d_f[b]) @ R + pos[b]
            frame = LiveFrame(i, level, J_l, J_f)
            i += 1
        try:
            out_q.put((epoch, frame), timeout=0.05)
        except queue.Full:
            pending = (epoch, frame)


class LiveGenerator:
    """One motion on a CPU worker, several motions batched on an MPS worker."""

    def __init__(self, lookahead: int = 2):
        import multiprocessing as mp

        import torch
        ctx = mp.get_context("spawn")
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        self.workers = {}
        for dev in devices:
            c, o = ctx.Queue(), ctx.Queue(maxsize=lookahead)
            self.workers[dev] = (c, o, ctx.Process(target=_worker, args=(c, o, dev), daemon=True))
        self.epoch = 0
        self.dev = "cpu"

    def start(self):
        for _, _, p in self.workers.values():
            p.start()

    def configure(self, **kw):
        for c, _, _ in self.workers.values():
            c.put(("set", kw))

    def reset(self, start_frames: np.ndarray, seeds):
        """start_frames [B,D] normalised, seeds [B]."""
        starts = np.asarray(start_frames, np.float32)
        self.dev = "cpu" if len(starts) == 1 or "mps" not in self.workers else "mps"
        self.epoch += 1
        for dev, (c, _, _) in self.workers.items():
            if dev == self.dev:
                c.put(("reset", (starts, [int(s) for s in seeds], self.epoch)))
            else:
                c.put(("idle", None))

    def pop(self, timeout: float = 0.1) -> LiveFrame | None:
        out_q = self.workers[self.dev][1]
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None
            try:
                epoch, f = out_q.get(timeout=left)
            except queue.Empty:
                return None
            if epoch == self.epoch:
                return f

    def stop(self):
        for c, _, _ in self.workers.values():
            c.put(("stop", None))
