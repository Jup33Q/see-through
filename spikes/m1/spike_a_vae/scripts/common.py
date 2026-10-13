"""Shared paths / wrappers / metrics for M1 Spike A (VAE CoreML feasibility)."""
import os
import sys

import numpy as np
import torch

REPO = "/Users/jup33q/see-through"
SPIKE = os.path.join(REPO, "spikes", "m1", "spike_a_vae")
DATA = os.path.join(SPIKE, "data")
MODELS = os.path.join(SPIKE, "models")
LOGS = os.path.join(SPIKE, "logs")
RESULTS_JSON = os.path.join(SPIKE, "results.json")

SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--layerdifforg--seethroughv0.0.2_layerdiff3d/"
    "snapshots/4477e6ce529bc6a141732e1a55a4932176db3b89"
)

GOLDEN_DIR = os.path.join(REPO, "tests", "golden", "test_image")

LAT = 160          # latent H=W -> 1280px image
IMG = LAT * 8      # 1280
SEED = 1234

sys.path.insert(0, os.path.join(REPO, "common"))

for d in (DATA, MODELS, LOGS):
    os.makedirs(d, exist_ok=True)


def load_torch_models(dtype=torch.float32, device="mps"):
    from diffusers import AutoencoderKL
    from modules.layerdiffuse.vae import TransparentVAE

    vae = AutoencoderKL.from_pretrained(os.path.join(SNAP, "vae"), torch_dtype=dtype)
    vae = vae.to(device).eval()
    trans = TransparentVAE.from_pretrained(SNAP, subfolder="trans_vae", torch_dtype=dtype)
    trans = trans.to(device).eval()
    return vae, trans


class SDXLVaeDecoderWrapper(torch.nn.Module):
    """z (already unscaled by 1/scaling_factor) -> image in [-1, 1]."""

    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        out = self.decoder(self.post_quant_conv(z))
        return out.sample if hasattr(out, "sample") else out


class SDXLVaeEncoderWrapper(torch.nn.Module):
    """image in [-1, 1] -> moments (mean+logvar, 8ch)."""

    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        return self.quant_conv(self.encoder(x))


class TransparentDecoderWrapper(torch.nn.Module):
    """(pixel in [0,1], latent) -> 4ch (alpha+rgb) estimate."""

    def __init__(self, trans_vae):
        super().__init__()
        self.model = trans_vae.decoder.model

    def forward(self, pixel, latent):
        return self.model(pixel, latent)


class TransparentEncoderWrapper(torch.nn.Module):
    """argb in [0,1] (4ch, alpha first) -> latent offset (already * alpha=300)."""

    def __init__(self, trans_vae):
        super().__init__()
        self.model = trans_vae.encoder.model
        self.alpha = trans_vae.encoder.alpha

    def forward(self, x):
        return self.model(x) * self.alpha


def max_abs_diff(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr01(a, b):
    """PSNR with both arrays clamped to the [0,1] image domain."""
    a = np.clip(a.astype(np.float64), 0.0, 1.0)
    b = np.clip(b.astype(np.float64), 0.0, 1.0)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)
