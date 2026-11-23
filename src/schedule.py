"""Diffusion-level schedules over motion frames: tau(f)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DipSchedule:
    """Noise everywhere except a 'recognition window' where the level dips.

    tau(f) = high - (high - low) * bump(f)
    bump   = exp(-|(f - center) / half_width| ** sharpness)   (flat-top for large sharpness)
    """
    high: float = 15.0        # level outside the window (T = pure noise)
    low: float = 0.0          # level at the bottom of the dip (0 = clean)
    center: float = 0.5       # window centre as a fraction of the clip
    half_width: float = 600.0 # frames
    sharpness: float = 2.0    # 2 = gaussian, larger = flatter top, steeper edges

    def levels(self, num_frames: int) -> np.ndarray:
        f = np.arange(num_frames, dtype=np.float64)
        c = self.center * (num_frames - 1)
        bump = np.exp(-np.abs((f - c) / max(self.half_width, 1e-3)) ** self.sharpness)
        return self.high - (self.high - self.low) * bump


@dataclass
class PulseSchedule:
    """Clean generation (tau = base) with short pulses back toward noise.

    Each pulse rises to `peak` over `attack` frames and falls back over `decay`
    frames (smoothstep). Pulses start at the listed frames, or are spaced evenly.
    """
    frames: str = ""          # comma-separated start frames; empty = `count` evenly spaced
    count: int = 3
    peak: float = 12.0
    attack: int = 2
    decay: int = 6
    base: float = 0.0

    def starts(self, num_frames: int) -> list[int]:
        txt = self.frames.replace(";", ",").strip()
        if txt:
            out = []
            for tok in txt.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        out.append(int(float(tok)))
                    except ValueError:
                        pass
            return [s for s in out if 0 <= s < num_frames]
        if self.count <= 0:
            return []
        return [int((k + 0.5) * num_frames / self.count) for k in range(self.count)]

    def levels(self, num_frames: int) -> np.ndarray:
        lv = np.full(num_frames, float(self.base))
        f = np.arange(num_frames, dtype=np.float64)
        for s in self.starts(num_frames):
            age = f - s
            up = (age >= 0) & (age < self.attack)
            shape = np.zeros(num_frames)
            shape[up] = (age[up] + 1) / (self.attack + 1)
            a = (age - self.attack) / max(self.decay, 1)
            down = (age >= self.attack) & (a < 1)
            r = 1 - a[down]
            shape[down] = r * r * (3 - 2 * r)
            lv = np.maximum(lv, self.base + (self.peak - self.base) * shape)
        return lv
