"""Shared paths / wrappers / metrics / compute-plan helpers for M2
(embedding staticization + small-model CoreML migration).

Wrappers and reference tensors are inherited from M1 Spike A
(`spikes/m1/spike_a_vae`); see docs/macos-native-roadmap.md M2.
"""
import asyncio
import glob
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
import torch

REPO = "/Users/jup33q/see-through"
M2 = os.path.join(REPO, "migration", "m2")
OUT = os.path.join(M2, "out")
EMB_DIR = os.path.join(OUT, "embeddings")
MODELS = os.path.join(OUT, "models")
LOGS = os.path.join(OUT, "logs")
RESULTS_JSON = os.path.join(M2, "results.json")

SPIKE_A = os.path.join(REPO, "spikes", "m1", "spike_a_vae")
SPIKE_A_DATA = os.path.join(SPIKE_A, "data")  # alignment inputs + fp32 refs (golden-derived)

GOLDEN_DIR = os.path.join(REPO, "tests", "golden", "test_image")

LAYERDIFF_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--layerdifforg--seethroughv0.0.2_layerdiff3d/"
    "snapshots/4477e6ce529bc6a141732e1a55a4932176db3b89"
)
MARIGOLD_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--24yearsold--seethroughv0.0.1_marigold/"
    "snapshots/4f4ffc46050b6feb764b628859968a26e76e6b6a"
)

# xcrun points at CommandLineTools which lacks coremlcompiler (M1 spike A finding).
COREMLCOMPILER = (
    "/Applications/Xcode.app/Contents/Developer/Toolchains/"
    "XcodeDefault.xctoolchain/usr/bin/coremlcompiler"
)

# v3 tag lists, hardcoded in diffusers_kdiffusion_sdxl.py:cache_tag_embeds (242-273)
BODY_TAG_LIST = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear',
                 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
HEAD_TAG_LIST = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear',
                 'ears', 'earwear', 'nose', 'mouth']
NEGATIVE_PROMPT = ""  # negative_prompt='' at all three call sites in inference_utils.py

sys.path.insert(0, os.path.join(REPO, "common"))

for d in (OUT, EMB_DIR, MODELS, LOGS):
    os.makedirs(d, exist_ok=True)


def load_torch_models(dtype=torch.float32, device="mps"):
    from diffusers import AutoencoderKL
    from modules.layerdiffuse.vae import TransparentVAE

    vae = AutoencoderKL.from_pretrained(os.path.join(LAYERDIFF_SNAP, "vae"), torch_dtype=dtype)
    vae = vae.to(device).eval()
    trans = TransparentVAE.from_pretrained(LAYERDIFF_SNAP, subfolder="trans_vae", torch_dtype=dtype)
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


# name -> ([input npy files], ref npy file)
MODULE_SPEC = {
    "sdxl_vae_decoder": (["sdxl_dec_in.npy"], "sdxl_dec_out.npy"),
    "sdxl_vae_encoder": (["sdxl_enc_in.npy"], "sdxl_enc_out.npy"),
    "trans_vae_decoder": (["trans_dec_in_pixel.npy", "trans_dec_in_latent.npy"], "trans_dec_out.npy"),
    "trans_vae_encoder": (["trans_enc_in.npy"], "trans_enc_out.npy"),
}

INPUT_NAMES = {
    "sdxl_vae_decoder": ["z"],
    "sdxl_vae_encoder": ["x"],
    "trans_vae_decoder": ["pixel", "latent"],
    "trans_vae_encoder": ["argb"],
}

# modules whose output is an image-domain tensor (SSIM meaningful)
IMAGE_DOMAIN_MODULES = {"sdxl_vae_decoder", "trans_vae_decoder"}

# acceptance thresholds (M1 gate reused; recorded in results.json)
THRESHOLDS = {"psnr01_db": 35.0, "psnr_dynamic_db": 40.0, "ssim": 0.99}


def build_wrappers():
    vae, trans = load_torch_models(dtype=torch.float32, device="cpu")
    return {
        "sdxl_vae_decoder": SDXLVaeDecoderWrapper(vae),
        "sdxl_vae_encoder": SDXLVaeEncoderWrapper(vae),
        "trans_vae_decoder": TransparentDecoderWrapper(trans),
        "trans_vae_encoder": TransparentEncoderWrapper(trans),
    }


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


def psnr_dynamic(a, b):
    """PSNR with peak = max|a|, for latent-domain outputs whose range exceeds [0,1]."""
    peak = float(np.max(np.abs(a)))
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def ssim_channels_last(a, b):
    """Mean per-channel SSIM on NCHW float arrays; data_range from reference."""
    from skimage.metrics import structural_similarity

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    data_range = float(a.max() - a.min()) or 1.0
    vals = []
    for c in range(a.shape[1]):
        vals.append(structural_similarity(a[0, c], b[0, c], data_range=data_range))
    return float(np.mean(vals))


def compile_mlpackage(pkg_path, dst_dir):
    """coremlcompiler compile -> .mlmodelc path inside dst_dir."""
    subprocess.run([COREMLCOMPILER, "compile", pkg_path, dst_dir],
                   check=True, capture_output=True, text=True)
    hits = glob.glob(os.path.join(dst_dir, "*.mlmodelc"))
    assert hits, f"coremlcompiler produced no .mlmodelc in {dst_dir}"
    return hits[0]


async def _compute_plan_stats_async(mlmodelc_path):
    from coremltools import ComputeUnit
    from coremltools.models.ml_program.experimental.compute_plan_utils import (
        load_compute_plan_from_path_on_device,
    )

    cp = await load_compute_plan_from_path_on_device(mlmodelc_path, ComputeUnit.ALL)
    ops = cp.model_structure.program.functions["main"].block.operations
    counter = Counter()
    for op in ops:
        usage = cp.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            counter["const_or_no_usage"] += 1
            continue
        dev = usage.preferred_compute_device
        counter[type(dev).__name__ if dev is not None else "none"] += 1
    return {"total_ops": len(ops), "preferred_device": dict(counter)}


def compute_plan_device_stats(pkg_path):
    """Per-op preferred-device attribution under computeUnits=ALL (M1 leftover:
    record device placement for every small module in M2)."""
    with tempfile.TemporaryDirectory() as td:
        mlmodelc = compile_mlpackage(pkg_path, td)
        return asyncio.run(_compute_plan_stats_async(mlmodelc))


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / 1e6, 1)
