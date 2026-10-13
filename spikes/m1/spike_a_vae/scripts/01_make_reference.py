"""Step 1: build PyTorch fp32 reference inputs/outputs on MPS and save .npy.

Modules covered:
  - sdxl_vae_decoder : z (4x160x160, unscaled) -> image [-1,1] (3x1280x1280)
  - sdxl_vae_encoder : image [-1,1] (3x1280x1280, golden reconstruction) -> moments 8x160x160
  - trans_vae_decoder: (pixel [0,1], latent) -> 4ch alpha+rgb (UNet1024)
  - trans_vae_encoder: argb [0,1] (4ch, alpha first, golden face.png) -> latent offset *300

Also records an fp16-vs-fp32 PyTorch gap per module (fp16 error floor).
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from common import (DATA, GOLDEN_DIR, IMG, LAT, RESULTS_JSON, SEED,
                    SDXLVaeDecoderWrapper, SDXLVaeEncoderWrapper,
                    TransparentDecoderWrapper, TransparentEncoderWrapper,
                    load_torch_models, max_abs_diff, psnr01)


def load_golden_rgb(name="reconstruction.png"):
    p = os.path.join(GOLDEN_DIR, name)
    img = Image.open(p).convert("RGB").resize((IMG, IMG), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]  # 1,3,H,W in [0,1]


def load_golden_argb(name="face.png"):
    p = os.path.join(GOLDEN_DIR, name)
    img = Image.open(p).convert("RGBA").resize((IMG, IMG), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC rgba
    argb = np.concatenate([arr[..., 3:4], arr[..., 0:3]], axis=-1)
    return torch.from_numpy(argb).permute(2, 0, 1)[None]  # 1,4,H,W in [0,1]


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    device = "mps"
    results = {"torch_ref": {}, "fp16_floor": {}}

    vae, trans = load_torch_models(dtype=torch.float32, device=device)

    # ---- fixed-seed inputs ----
    g = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, 4, LAT, LAT, generator=g)  # scaled latent space
    z = latent / vae.config.scaling_factor             # wrapper input (unscaled)
    enc_img = load_golden_rgb() * 2.0 - 1.0            # [-1,1]

    np.save(os.path.join(DATA, "sdxl_dec_in.npy"), z.numpy())
    np.save(os.path.join(DATA, "sdxl_enc_in.npy"), enc_img.numpy())

    dec_mod = SDXLVaeDecoderWrapper(vae)
    enc_mod = SDXLVaeEncoderWrapper(vae)
    tdec_mod = TransparentDecoderWrapper(trans)
    tenc_mod = TransparentEncoderWrapper(trans)

    z_d = z.to(device)
    enc_img_d = enc_img.to(device)

    with torch.no_grad():
        dec_out = dec_mod(z_d)
        enc_out = enc_mod(enc_img_d)

    np.save(os.path.join(DATA, "sdxl_dec_out.npy"), dec_out.cpu().numpy())
    np.save(os.path.join(DATA, "sdxl_enc_out.npy"), enc_out.cpu().numpy())

    # trans decoder inputs: pixel = decoded image in [0,1], latent = same latent
    pixel = (dec_out * 0.5 + 0.5).clamp(0, 1).contiguous()
    lat_in = latent.to(device)
    np.save(os.path.join(DATA, "trans_dec_in_pixel.npy"), pixel.cpu().numpy())
    np.save(os.path.join(DATA, "trans_dec_in_latent.npy"), latent.numpy())

    argb = load_golden_argb().to(device)
    np.save(os.path.join(DATA, "trans_enc_in.npy"), argb.cpu().numpy())

    with torch.no_grad():
        tdec_out = tdec_mod(pixel, lat_in)
        tenc_out = tenc_mod(argb)

    np.save(os.path.join(DATA, "trans_dec_out.npy"), tdec_out.cpu().numpy())
    np.save(os.path.join(DATA, "trans_enc_out.npy"), tenc_out.cpu().numpy())

    # ---- PyTorch fp32 timings (MPS) ----
    with torch.no_grad():
        results["torch_ref"]["sdxl_vae_decoder_s"] = bench(lambda: dec_mod(z_d))
        results["torch_ref"]["sdxl_vae_encoder_s"] = bench(lambda: enc_mod(enc_img_d))
        results["torch_ref"]["trans_vae_decoder_s"] = bench(lambda: tdec_mod(pixel, lat_in))
        results["torch_ref"]["trans_vae_encoder_s"] = bench(lambda: tenc_mod(argb))

    del vae, trans, dec_mod, enc_mod, tdec_mod, tenc_mod
    torch.mps.empty_cache()

    # ---- fp16 error floor (PyTorch fp16 on MPS vs fp32 refs) ----
    vae16, trans16 = load_torch_models(dtype=torch.float16, device=device)
    dec16 = SDXLVaeDecoderWrapper(vae16)
    enc16 = SDXLVaeEncoderWrapper(vae16)
    tdec16 = TransparentDecoderWrapper(trans16)
    tenc16 = TransparentEncoderWrapper(trans16)
    with torch.no_grad():
        d16 = dec16(z_d.half()).float().cpu().numpy()
        e16 = enc16(enc_img_d.half()).float().cpu().numpy()
        td16 = tdec16(pixel.half(), lat_in.half()).float().cpu().numpy()
        te16 = tenc16(argb.half()).float().cpu().numpy()

    refs = {
        "sdxl_vae_decoder": ("sdxl_dec_out.npy", d16),
        "sdxl_vae_encoder": ("sdxl_enc_out.npy", e16),
        "trans_vae_decoder": ("trans_dec_out.npy", td16),
        "trans_vae_encoder": ("trans_enc_out.npy", te16),
    }
    for name, (ref_file, out16) in refs.items():
        ref = np.load(os.path.join(DATA, ref_file))
        results["fp16_floor"][name] = {
            "max_abs_diff": max_abs_diff(ref, out16),
            "psnr01_db": psnr01(ref, out16),
        }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print("reference tensors saved to", DATA)


if __name__ == "__main__":
    main()
