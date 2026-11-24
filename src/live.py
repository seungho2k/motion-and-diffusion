"""Endless A-MDM generation in a child process (no GIL contention with the viewer).

The worker keeps generating new frames forever. The diffusion level follows a cyclic
schedule (`levels`, repeated), so e.g. a ramp cycle repeats while the motion itself
is always new. Settings and restarts arrive through a command queue; frames leave
through a small bounded queue, so the worker naturally pauses when the viewer does.
"""
from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np


@dataclass
class LiveFrame:
    index: int
    level: float
    joints: np.ndarray        # [J,3] shown state (partially denoised at `level`)
    joints_final: np.ndarray  # [J,3] same frame denoised to t=0 (only if requested, else = joints)


def _worker(cmd_q, out_q):
    import torch

    from amdm_runtime import AMDMRuntime, _rot_yaw

    torch.set_num_threads(1)
    rt = AMDMRuntime("lafan1")
    s = {"levels": np.zeros(1), "feed": "xt", "root": "display", "decode": "position",
         "noise_scale": 1.0, "noise_coherence": 0.0, "clip": 4.0, "want_final": False, "skip": 1}
    cond = gen = noise_state = None
    epoch, i, yaw, pos = 0, 0, 0.0, np.zeros(3)

    def local(d):
        if s["decode"] == "rotation":
            return rt.fk(d[None])[0]
        j = d[rt.jnt_lo:rt.jnt_hi].reshape(rt.J, 3).copy()
        j[0, 0] = j[0, 2] = 0.0
        return j

    pending = None
    while True:
        # ---- commands (non-blocking unless idle)
        try:
            while True:
                cmd, arg = cmd_q.get(timeout=0.05) if cond is None else cmd_q.get_nowait()
                if cmd == "stop":
                    return
                if cmd == "set":
                    s.update(arg)
                elif cmd == "reset":
                    start, seed, epoch = arg
                    cond = torch.tensor(start)
                    gen = torch.Generator().manual_seed(int(seed))
                    noise_state = torch.randn(rt.T + 1, rt.D, generator=gen)
                    i, yaw, pos, pending = 0, 0.0, np.zeros(3), None
        except queue.Empty:
            pass
        if cond is None:
            continue

        # ---- deliver a frame that did not fit last time
        if pending is not None:
            try:
                out_q.put(pending, timeout=0.05)
                pending = None
            except queue.Full:
                continue

        # ---- generate `skip` frames, send the last one (skip > 1 keeps real speed at lower fps)
        for _ in range(max(1, int(s["skip"]))):
            levels = s["levels"]
            level = float(levels[i % len(levels)])
            rho = float(np.clip(s["noise_coherence"], 0.0, 0.999))
            noise_state = rho * noise_state + np.sqrt(1 - rho * rho) * torch.randn(rt.T + 1, rt.D, generator=gen)
            with torch.inference_mode():
                x_l, x0, x_f = rt.denoise_to(cond, level, noise_state * s["noise_scale"], s["clip"],
                                             final=s["want_final"] or s["feed"] == "final")
            nxt = {"x0": x0, "final": x_f}.get(s["feed"], x_l)
            if s["feed"] == "mix":
                nxt = 0.5 * (x_l + x0)
            cond = nxt.clamp(-s["clip"], s["clip"])
            d_l = rt.denorm(x_l).numpy()
            d_f = rt.denorm(x_f).numpy() if x_f is not None else d_l
            root = {"x0": rt.denorm(x0).numpy()[:3], "final": d_f[:3], "none": np.zeros(3)}.get(s["root"], d_l[:3])
            yaw += float(root[2])
            R = _rot_yaw(yaw)
            if i > 0:
                pos = pos + np.array([root[0], 0.0, root[1]]) @ R
            frame = LiveFrame(i, level, (local(d_l) @ R + pos).astype(np.float32),
                              (local(d_f) @ R + pos).astype(np.float32))
            i += 1
        try:
            out_q.put((epoch, frame), timeout=0.05)
        except queue.Full:
            pending = (epoch, frame)


class LiveGenerator:
    def __init__(self, lookahead: int = 2):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.cmd_q = ctx.Queue()
        self.out_q = ctx.Queue(maxsize=lookahead)
        self.epoch = 0
        self.proc = ctx.Process(target=_worker, args=(self.cmd_q, self.out_q), daemon=True)

    def start(self):
        self.proc.start()

    def configure(self, **kw):
        self.cmd_q.put(("set", kw))

    def reset(self, start_frame: np.ndarray, seed: int):
        self.epoch += 1
        self.cmd_q.put(("reset", (np.asarray(start_frame, np.float32), int(seed), self.epoch)))

    def pop(self, timeout: float = 0.1) -> LiveFrame | None:
        try:
            while True:
                epoch, f = self.out_q.get(timeout=timeout)
                if epoch == self.epoch:
                    return f
        except queue.Empty:
            return None

    def stop(self):
        self.cmd_q.put(("stop", None))
