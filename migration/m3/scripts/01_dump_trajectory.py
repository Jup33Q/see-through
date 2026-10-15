"""M3-2: dump golden denoising trajectories (body/head passes) as alignment anchors.

Replicates the M0 golden run exactly (same code path, seed=42 defaults of
inference_psd.py, mps/bf16) and records everything the Swift side needs:

  per pass (body F=13 group 0 / head F=11 group 1) under <out>/<pass>/:
    meta.json             frames/group/tags/timesteps/sigmas/init_noise_sigma/seed/
                          scaling_factor + head-pass compositing params
    noise.npy             (1,1,4,160,160)  initial noise (pre frame-expand)
    latent_init.npy       (1,F,4,160,160)  noise * init_noise_sigma, frame-expanded
    argb_in.npy           (1,4,1280,1280)  vae_encode input (A first, [0,1])
    vae_moments.npy       (1,8,160,160)    SDXL VAE encoder moments (mean|logvar)
    c_concat.npy          (1,4,160,160)    scaled latent fed to UNet (pre expand)
    prompt_embeds.npy     (F,77,2048)      as passed to UNet (pre group2 aug)
    pooled_embeds.npy     (F,1280)         pre group aug
    time_ids.npy          (F,6)
    page_alpha.npy        (1280,1280,1)    decode mask
    step_XX_unet_in.npy   (1,F,8,160,160)  UNet input (latent|c_concat concatenated)
    step_XX_unet_out.npy  (1,F,4,160,160)  noise_pred
    step_XX_latents_next.npy (1,F,4,160,160)
    decode_in.npy         (F,4,160,160)    final latents / scaling_factor

Also verifies reproduction of the M0 golden PNGs (per-tag PSNR) and writes
results.json at <out>/.

Run: python 01_dump_trajectory.py   (conda env see_through, ~10 min on M5 Max)
"""
import json
import os
import sys
import time
import types

import numpy as np
import torch
from PIL import Image

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))

from diffusers import AutoencoderKL  # noqa: E402
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import (  # noqa: E402
    KDiffusionStableDiffusionXLPipeline,
    UNetFrameConditionModel,
)
import modules.layerdiffuse.diffusers_kdiffusion_sdxl as kd  # noqa: E402
from modules.layerdiffuse.vae import TransparentVAE  # noqa: E402
from utils.cv import center_square_pad_resize, smart_resize  # noqa: E402
from utils.torch_utils import seed_everything  # noqa: E402
import cv2  # noqa: E402

LAYERDIFF_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
GOLDEN = os.path.join(REPO, "tests", "golden", "test_image")
OUT_DEFAULT = os.path.join(REPO, "migration", "m3", "golden_trajectory")
SEED = 42
STEPS = 30
RES = 1280

BODY_TAG_LIST = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear',
                 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
HEAD_TAG_LIST = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear',
                 'ears', 'earwear', 'nose', 'mouth']


def f32(t):
    return t.detach().to(torch.float32).cpu().numpy()


class Recorder:
    """Hooks installed on the live pipeline; buffers per pass, flushed to disk."""

    def __init__(self):
        self.unet_calls = []   # (group_index, t, sample, noise_pred)
        self.step_calls = []   # (t, latents_in, latents_next)
        self.noise = None      # randn_tensor product (1,1,4,h,w)
        self.argb_in = None    # vae_encode input
        self.c_concat = None   # vae_encode output (scaled)
        self.moments = None    # (mean, logvar)
        self.prompt_embeds = None
        self.pooled = None


def install_hooks(pipeline, rec: Recorder):
    unet = pipeline.unet
    orig_forward = unet.forward

    def hooked_forward(self, sample, timestep, *args, **kwargs):
        out = orig_forward(sample, timestep, *args, **kwargs)
        noise_pred = out[0] if isinstance(out, tuple) else out.sample
        rec.unet_calls.append((kwargs.get("group_index", None),
                               float(f32(timestep).flat[0]),
                               f32(sample), f32(noise_pred)))
        return out

    unet.forward = types.MethodType(hooked_forward, unet)

    sched = pipeline.scheduler
    orig_step = sched.step

    def hooked_step(self, model_output, timestep, sample, *args, **kwargs):
        out = orig_step(model_output, timestep, sample, *args, **kwargs)
        prev = out[0] if isinstance(out, tuple) else out.prev_sample
        rec.step_calls.append((float(f32(timestep).flat[0]), f32(sample), f32(prev)))
        return out

    sched.step = types.MethodType(hooked_step, sched)

    orig_randn = kd.randn_tensor

    def hooked_randn(shape, generator=None, device=None, dtype=None, **kw):
        t = orig_randn(shape, generator=generator, device=device, dtype=dtype, **kw)
        rec.noise = f32(t)
        return t

    kd.randn_tensor = hooked_randn

    orig_vae_encode = kd.vae_encode

    def hooked_vae_encode(vae, trans_enc, argb_tensor, use_offset=True, scale=True):
        rec.argb_in = f32(argb_tensor)
        out = orig_vae_encode(vae, trans_enc, argb_tensor, use_offset=use_offset, scale=scale)
        rec.c_concat = f32(out)
        return out

    kd.vae_encode = hooked_vae_encode

    orig_encode = pipeline.vae.encode

    def hooked_encode(self, x, *args, **kwargs):
        out = orig_encode(x, *args, **kwargs)
        rec.moments = (f32(out.latent_dist.mean), f32(out.latent_dist.logvar))
        return out

    pipeline.vae.encode = types.MethodType(hooked_encode, pipeline.vae)

    orig_cached = pipeline.encode_cropped_prompt_77tokens_cached

    def hooked_cached(prompt):
        pe, ppe = orig_cached(prompt)
        rec.prompt_embeds = f32(pe)
        rec.pooled = f32(ppe)
        return pe, ppe

    pipeline.encode_cropped_prompt_77tokens_cached = hooked_cached


def save_pass(out_dir, rec: Recorder, tags, meta_extra):
    os.makedirs(out_dir, exist_ok=True)
    assert len(rec.unet_calls) == STEPS, f"expect {STEPS} unet calls, got {len(rec.unet_calls)}"
    assert len(rec.step_calls) == STEPS

    group_index = rec.unet_calls[0][0]
    frames = rec.unet_calls[0][2].shape[1]

    np.save(os.path.join(out_dir, "noise.npy"), rec.noise)
    np.save(os.path.join(out_dir, "latent_init.npy"), rec.step_calls[0][1])
    np.save(os.path.join(out_dir, "argb_in.npy"), rec.argb_in)
    np.save(os.path.join(out_dir, "c_concat.npy"), rec.c_concat)
    np.save(os.path.join(out_dir, "vae_moments.npy"),
            np.concatenate([rec.moments[0], rec.moments[1]], axis=1))
    np.save(os.path.join(out_dir, "prompt_embeds.npy"), rec.prompt_embeds)
    np.save(os.path.join(out_dir, "pooled_embeds.npy"), rec.pooled)

    for i, ((gi, t_u, s_in, s_out), (t_s, l_in, l_next)) in enumerate(
            zip(rec.unet_calls, rec.step_calls)):
        assert gi == group_index and abs(t_u - t_s) < 1e-3
        np.save(os.path.join(out_dir, f"step_{i:02d}_unet_in.npy"), s_in)
        np.save(os.path.join(out_dir, f"step_{i:02d}_unet_out.npy"), s_out)
        np.save(os.path.join(out_dir, f"step_{i:02d}_latents_next.npy"), l_next)

    final_latents = rec.step_calls[-1][2][0]  # (F,4,h,w)
    scaling_factor = 0.13025
    np.save(os.path.join(out_dir, "decode_in.npy"), final_latents / scaling_factor)

    meta = dict(
        frames=frames, group_index=group_index, tags=tags, seed=SEED,
        num_inference_steps=STEPS, resolution=RES, scaling_factor=scaling_factor,
        timesteps=[c[1] for c in rec.unet_calls],
        device="mps", dtype="bfloat16", guidance_scale=1.0,
        **meta_extra,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def psnr_u8(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def compare_golden(images, tags, results, pass_name):
    for img, tag in zip(images, tags):
        gp = os.path.join(GOLDEN, f"{tag}.png")
        g = np.array(Image.open(gp).convert("RGBA"))
        p = psnr_u8(img, g)
        exact = bool(np.array_equal(img, g))
        results.setdefault("golden_reproduction", {})[f"{pass_name}/{tag}"] = {
            "psnr_db": p, "exact": exact}


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT
    os.makedirs(out_root, exist_ok=True)
    results = {"seed": SEED, "steps": STEPS}
    t_all = time.time()

    device, dtype = torch.device("mps"), torch.bfloat16
    trans_vae = TransparentVAE.from_pretrained(LAYERDIFF_REPO, subfolder="trans_vae")
    unet = UNetFrameConditionModel.from_pretrained(LAYERDIFF_REPO, subfolder="unet")
    pipeline = KDiffusionStableDiffusionXLPipeline.from_pretrained(
        LAYERDIFF_REPO, trans_vae=trans_vae, unet=unet, scheduler=None)
    pipeline.vae.to(dtype=dtype, device=device)
    pipeline.trans_vae.to(dtype=dtype, device=device)
    pipeline.unet.to(dtype=dtype, device=device)
    pipeline.text_encoder.to(dtype=dtype, device=device)
    pipeline.text_encoder_2.to(dtype=dtype, device=device)
    pipeline.set_progress_bar_config(disable=True)

    seed_everything(SEED)
    pipeline.cache_tag_embeds()

    rec = Recorder()
    install_hooks(pipeline, rec)

    input_img = np.array(Image.open(os.path.join(REPO, "assets", "test_image.png")).convert("RGBA"))
    fullpage, pad_size, pad_pos = center_square_pad_resize(input_img, RES, return_pad_info=True)
    scale = pad_size[0] / RES
    assert np.array_equal(fullpage, np.array(Image.open(
        os.path.join(GOLDEN, "src_img.png")).convert("RGBA"))), "src_img reproduction mismatch"

    rng = torch.Generator(device=device).manual_seed(SEED)

    # ---------------- body pass (group 0, 13 frames) ----------------
    t0 = time.time()
    out = pipeline(strength=1.0, num_inference_steps=STEPS, batch_size=1, generator=rng,
                   guidance_scale=1.0, prompt=BODY_TAG_LIST, negative_prompt='',
                   fullpage=fullpage, group_index=0)
    body_s = time.time() - t0
    page_alpha = (fullpage[..., -1] / 255.0)[..., None].astype(np.float32)
    np.save(os.path.join(out_root, "tmp_body_page_alpha.npy"), page_alpha)
    save_pass(os.path.join(out_root, "body"), rec, BODY_TAG_LIST,
              {"wall_s": round(body_s, 2)})
    os.rename(os.path.join(out_root, "tmp_body_page_alpha.npy"),
              os.path.join(out_root, "body", "page_alpha.npy"))
    compare_golden(out.images, BODY_TAG_LIST, results, "body")
    results["body_wall_s"] = round(body_s, 2)
    print(f"[body] done in {body_s:.1f}s", flush=True)

    # ---------------- head crop (verbatim from inference_utils.apply_layerdiff) ----------------
    def _crop_head(img, xywh):
        x, y, w, h = xywh
        ih, iw = img.shape[:2]
        x1, y1, x2, y2 = x, y, x + w, y + h
        if w < iw // 2:
            px = min(iw - x - w, x, w // 5)
            x1 = min(max(x - px, 0), iw)
            x2 = min(max(x + w + px, 0), iw)
        if h < ih // 2:
            py = min(ih - y - h, y, h // 5)
            y2 = min(max(y + h + py, 0), ih)
            y1 = min(max(y - py, 0), ih)
        return img[y1: y2, x1: x2], (x1, y1, x2, y2)

    head_img = out.images[2]
    hx0, hy0, hw, hh = cv2.boundingRect(cv2.findNonZero((head_img[..., -1] > 15).astype(np.uint8)))
    hx = int(hx0 * scale) - pad_pos[0]
    hy = int(hy0 * scale) - pad_pos[1]
    hw = int(hw * scale)
    hh = int(hh * scale)
    input_head, (cx1, cy1, cx2, cy2) = _crop_head(input_img, [hx, hy, hw, hh])
    px1f = int(cx1 / scale + pad_pos[0] / scale)
    py1f = int(cy1 / scale + pad_pos[1] / scale)
    ih, iw = input_head.shape[:2]
    input_head_sq, pad_size2, pad_pos2 = center_square_pad_resize(input_head, RES, return_pad_info=True)

    golden_head_src = np.array(Image.open(os.path.join(GOLDEN, "src_head.png")).convert("RGBA"))
    src_head_match = bool(np.array_equal(input_head_sq, golden_head_src))
    results["src_head_exact"] = src_head_match
    if not src_head_match:
        print("[head] WARNING: src_head reproduction mismatch, falling back to golden src_head",
              flush=True)
        input_head_sq = golden_head_src

    # ---------------- head pass (group 1, 11 frames) ----------------
    rec2 = Recorder()
    install_hooks(pipeline, rec2)
    t0 = time.time()
    out2 = pipeline(strength=1.0, num_inference_steps=STEPS, batch_size=1, generator=rng,
                    guidance_scale=1.0, prompt=HEAD_TAG_LIST, negative_prompt='',
                    fullpage=input_head_sq, group_index=1)
    head_s = time.time() - t0

    canvas = np.zeros((RES, RES, 4), dtype=np.uint8)
    py1, py2, px1, px2 = (np.array([pad_pos2[1], pad_pos2[1] + ih, pad_pos2[0],
                                    pad_pos2[0] + iw]) / scale).astype(np.int64)
    scale_size = (int(pad_size2[0] / scale), int(pad_size2[1] / scale))
    head_images = []
    for rst, tag in zip(out2.images, HEAD_TAG_LIST):
        rst_r = smart_resize(rst, scale_size)[py1:py2, px1:px2]
        full = canvas.copy()
        full[py1f:py1f + rst_r.shape[0], px1f:px1f + rst_r.shape[1]] = rst_r
        head_images.append(full)

    page_alpha2 = (input_head_sq[..., -1] / 255.0)[..., None].astype(np.float32)
    save_pass(os.path.join(out_root, "head"), rec2, HEAD_TAG_LIST, {
        "wall_s": round(head_s, 2),
        "composite": {
            "bbox_head_alpha": [int(hx0), int(hy0), int(hw), int(hh)],
            "full_scale": float(scale), "full_pad_pos": [int(pad_pos[0]), int(pad_pos[1])],
            "crop_input_xyxy": [int(cx1), int(cy1), int(cx2), int(cy2)],
            "paste_xy": [px1f, py1f],
            "head_hw": [int(ih), int(iw)],
            "head_pad_size": [int(pad_size2[0]), int(pad_size2[1])],
            "head_pad_pos": [int(pad_pos2[0]), int(pad_pos2[1])],
            "scale_size_wh": [int(scale_size[0]), int(scale_size[1])],
            "crop_after_resize": {"py1": int(py1), "py2": int(py2),
                                  "px1": int(px1), "px2": int(px2)},
        }})
    np.save(os.path.join(out_root, "head", "page_alpha.npy"), page_alpha2)
    compare_golden(head_images, HEAD_TAG_LIST, results, "head")
    results["head_wall_s"] = round(head_s, 2)
    print(f"[head] done in {head_s:.1f}s", flush=True)

    results["total_wall_s"] = round(time.time() - t_all, 2)
    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    bad = {k: v for k, v in results["golden_reproduction"].items() if not v["exact"]}
    print(f"golden reproduction: {len(results['golden_reproduction']) - len(bad)} exact, "
          f"{len(bad)} non-exact")
    for k, v in bad.items():
        print(f"  {k}: psnr={v['psnr_db']:.2f}dB")
    print("done")


if __name__ == "__main__":
    main()
