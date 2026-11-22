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
    half_width: float = 45.0  # frames
    sharpness: float = 2.0    # 2 = gaussian, larger = flatter top, steeper edges

    def levels(self, num_frames: int) -> np.ndarray:
        f = np.arange(num_frames, dtype=np.float64)
        c = self.center * (num_frames - 1)
        bump = np.exp(-np.abs((f - c) / max(self.half_width, 1e-3)) ** self.sharpness)
        return self.high - (self.high - self.low) * bump
