"""Text-timeline generation with DiP and a per-frame diffusion-level schedule.

DiP plans 40 frames from a 20-frame prefix. To follow a schedule that changes every
few frames (short pulses), we plan, keep only the first `commit` frames, and replan
(receding horizon, as CLoSD does). The kept frames are taken at the scheduled level
tau (partially denoised); what is fed back into the next prefix is selectable
(the shown state, the model's x0 at that level, or the same plan fully denoised).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from dip_runtime import FPS as DIP_FPS
from dip_runtime import DiPRuntime

VIDEO_FPS = 30


def parse_timeline(text: str) -> list[tuple[float, str]]:
    """Lines 'seconds: prompt' (or just 'prompt' for t=0) -> sorted [(t, prompt)]."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, sep, tail = line.partition(":")
        try:
            t = float(head) if sep else 0.0
            prompt = tail.strip() if sep else line
        except ValueError:
            t, prompt = 0.0, line
        if prompt:
            out.append((t, prompt))
    return sorted(out) or [(0.0, "a person stands still")]


def prompt_at(timeline, t: float) -> str:
    cur = timeline[0][1]
    for start, p in timeline:
        if t >= start:
            cur = p
    return cur


@dataclass
class DipGenParams:
    timeline: str = "0: a person dances gracefully"
    seconds: float = 30.0
    seed: int = 0
    commit: int = 4               # frames (20 fps) kept from each 40-frame plan
    guidance: float = 7.5
    feed: str = "xt"              # xt | x0 | final
    root_source: str = "display"  # display | final
    noise_scale: float = 1.0


@dataclass
class DipResult:
    levels: np.ndarray            # [N30] tau per video frame
    joints: np.ndarray            # [N30, 22, 3] shown state
    joints_final: np.ndarray      # [N30, 22, 3] same frames fully denoised
    prompts: list = field(default_factory=list)   # prompt per video frame


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear time resampling of [N, ...] to n_out frames."""
    n = len(x)
    src = np.linspace(0, n - 1, n_out)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (src - i0).reshape(-1, *([1] * (x.ndim - 1)))
    return (1 - w) * x[i0] + w * x[i1]


@torch.no_grad()
def generate(rt: DiPRuntime, p: DipGenParams, levels30: np.ndarray, progress=None) -> DipResult:
    rt.guidance = p.guidance
    d = rt.diffusion
    T, L, P = rt.T, rt.context_len, rt.pred_len
    n30 = len(levels30)
    n20 = int(np.ceil(n30 * DIP_FPS / VIDEO_FPS))
    lv20 = levels30[np.minimum((np.arange(n20) * VIDEO_FPS / DIP_FPS).round().astype(int), n30 - 1)]
    timeline = parse_timeline(p.timeline)
    gen = torch.Generator().manual_seed(p.seed)
    commit = int(np.clip(p.commit, 1, P))

    history = [rt.default_prefix()[0, :, 0].T.numpy()]          # fed-back frames (normalised)
    shown, final = [], []
    i = 0
    while i < n20:
        text = prompt_at(timeline, i / DIP_FPS)
        level = float(np.clip(lv20[i], 0, T))
        k_lo = int(np.floor(level))
        k_hi = min(k_lo + 1, T)
        frac = level - k_lo
        hist = np.concatenate(history)[-L:]
        prefix = torch.tensor(hist.T[None, :, None, :])
        y = {"prefix": prefix, "text_embed": rt.encode(text),
             "mask": torch.ones(1, 1, 1, P, dtype=torch.bool), "lengths": torch.full((1,), P)}
        shape = (1, 263, 1, P)
        x = torch.randn(shape, generator=gen) * p.noise_scale
        states, x0_at = {T: x}, x
        for t in range(T - 1, -1, -1):
            ts = torch.full((1,), t, dtype=torch.long)
            x0 = rt._model_out(x, ts, y)[..., -P:]
            if t == k_lo:
                x0_at = x0
            if t > 0:
                mean, _, logvar = d.q_posterior_mean_variance(x_start=x0, x_t=x, t=ts)
                x = mean + torch.exp(0.5 * logvar) * torch.randn(shape, generator=gen) * p.noise_scale
            else:
                x = x0
            states[t] = x
        x_lvl = (1 - frac) * states[k_lo] + frac * states[k_hi]
        take = lambda z: z[0, :, 0, :commit].T.numpy()
        shown.append(take(x_lvl))
        final.append(take(states[0]))
        fed = {"x0": take(x0_at), "final": take(states[0])}.get(p.feed, take(x_lvl))
        history.append(np.clip(fed, -6, 6))
        i += commit
        if progress:
            progress(i, n20)

    shown = np.concatenate(shown)[:n20]
    final = np.concatenate(final)[:n20]
    if p.root_source == "final":
        shown = shown.copy()
        shown[:, :4] = final[:, :4]          # root rot-vel, xz-vel, height from the clean plan
    j_shown = rt.to_joints(shown)
    j_final = rt.to_joints(final if p.root_source != "display" else np.concatenate(
        [shown[:, :4], final[:, 4:]], axis=1))
    prompts30 = [prompt_at(timeline, f / VIDEO_FPS) for f in range(n30)]
    return DipResult(levels30.astype(np.float32), _resample(j_shown, n30).astype(np.float32),
                     _resample(j_final, n30).astype(np.float32), prompts30)
