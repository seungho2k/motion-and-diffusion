"""DiP (CLoSD's autoregressive text-to-motion diffusion planner) without the CLoSD stack.

Only the diffusion planner is used: transformer decoder, DistilBERT text encoder,
10 diffusion steps, 20-frame prefix -> 40 predicted frames, HumanML3D 263-d features
at 20 fps (22 SMPL joints). Assets are fetched individually from the `guytevet/CLoSD`
Hugging Face repo (the full snapshot is ~19 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOSD_DIR = os.path.join(ROOT, "third_party", "CLoSD")
for p in (CLOSD_DIR, os.path.join(ROOT, "src", "shims")):
    if p not in sys.path:
        sys.path.insert(0, p)

from huggingface_hub import hf_hub_download  # noqa: E402

from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric  # noqa: E402
from closd.diffusion_planner.utils.model_util import create_model_and_diffusion, load_saved_model  # noqa: E402

HF_REPO = "guytevet/CLoSD"
CKPT_DIR = "checkpoints/dip/DiP_no-target_10steps_context20_predict40"
CKPT = "model000600343.pt"
SAMPLE_MOTION = "data/dataset/HumanML3D/new_joint_vecs/012314.npy"
FPS = 20

# HumanML3D / SMPL 22-joint skeleton
JOINT_NAMES = ["Pelvis", "LeftHip", "RightHip", "Spine1", "LeftKnee", "RightKnee", "Spine2",
               "LeftAnkle", "RightAnkle", "Spine3", "LeftFoot", "RightFoot", "Neck", "LeftCollar",
               "RightCollar", "Head", "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
               "LeftWrist", "RightWrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
LINKS = np.array([(p, j) for j, p in enumerate(PARENTS) if p >= 0])


def _hf(path: str) -> str:
    return hf_hub_download(HF_REPO, path)


class _DummyData:
    class dataset:  # noqa: N801  (create_model_and_diffusion only probes data.dataset)
        pass


class DiPRuntime:
    def __init__(self, device: str = "cpu", guidance: float = 7.5):
        with open(_hf(f"{CKPT_DIR}/args.json")) as f:
            args = argparse.Namespace(**json.load(f))
        self.args = args
        self.device = torch.device(device)
        self.model, self.diffusion = create_model_and_diffusion(args, _DummyData())
        load_saved_model(self.model, _hf(f"{CKPT_DIR}/{CKPT}"), use_avg=args.use_ema)
        self.model.to(self.device).eval()
        self.guidance = guidance
        self.context_len, self.pred_len = args.context_len, args.pred_len
        self.T = args.diffusion_steps
        self.mean = np.load(_hf("data/dataset/HumanML3D/Mean.npy")).astype(np.float32)
        self.std = np.load(_hf("data/dataset/HumanML3D/Std.npy")).astype(np.float32)
        self.sample_motion = np.load(_hf(SAMPLE_MOTION)).astype(np.float32)   # [N, 263] raw features
        self._text_cache: dict = {}

    # ------------------------------------------------------------- helpers
    def norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denorm(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def default_prefix(self) -> torch.Tensor:
        """First `context_len` frames of the bundled sample motion, normalised, [1,263,1,L]."""
        p = self.norm(self.sample_motion[: self.context_len])
        return torch.tensor(p.T[None, :, None, :], device=self.device)

    def encode(self, text: str):
        if text not in self._text_cache:
            with torch.no_grad():
                self._text_cache[text] = self.model.encode_text([text])
        return self._text_cache[text]

    def _model_out(self, x, t, y):
        """Classifier-free guided x0 prediction."""
        out = self.model(x, t, dict(y))
        if self.guidance != 1.0:
            out_u = self.model(x, t, dict(y, text_uncond=True))
            out = out_u + self.guidance * (out - out_u)
        return out

    @torch.no_grad()
    def plan(self, prefix: torch.Tensor, text: str, gen: torch.Generator | None = None,
             stop_level: int = 0, noise_scale: float = 1.0):
        """Denoise one 40-frame plan given a 20-frame prefix.

        stop_level = 0 gives the fully denoised plan; > 0 stops early and returns the
        partially noisy x_t at that level (plus the model's x0 prediction there).
        Returns (x_level, x0_hat), each [1,263,1,pred_len] normalised.
        """
        d = self.diffusion
        bs = prefix.shape[0]
        shape = (bs, 263, 1, self.pred_len)
        x = torch.randn(shape, generator=gen).to(self.device) * noise_scale
        y = {"prefix": prefix, "text_embed": self.encode(text),
             "mask": torch.ones(bs, 1, 1, self.pred_len, dtype=torch.bool, device=self.device),
             "lengths": torch.full((bs,), self.pred_len, device=self.device)}
        x0 = x
        for t in range(self.T - 1, stop_level - 1, -1):
            ts = torch.full((bs,), t, dtype=torch.long, device=self.device)
            x0 = self._model_out(x, ts, y)[..., -self.pred_len:]
            if t > 0:
                mean, _, logvar = d.q_posterior_mean_variance(x_start=x0, x_t=x, t=ts)
                z = torch.randn(shape, generator=gen).to(self.device) * noise_scale
                x = mean + torch.exp(0.5 * logvar) * z
            else:
                x = x0
        return x, x0

    def to_joints(self, feats_norm: np.ndarray) -> np.ndarray:
        """Normalised features [N,263] -> global joints [N,22,3] (y up, metres)."""
        raw = torch.tensor(self.denorm(feats_norm), dtype=torch.float32)[None]
        return recover_from_ric(raw, 22)[0].numpy()
