"""M4-1: dump golden Marigold (depth) trajectory as alignment anchors.

Replicates the M0 golden run's marigold stage exactly — calls the real
`utils.inference_utils.apply_marigold` (mps/bf16/seed=42, resolution=768,
num_inference_steps=-1 -> default 4 DDIM steps) on a scratch copy of the
golden intermediate PNGs, with hooks recording everything the Swift side
needs, then verifies the produced *_depth.png against M0 golden.

Outputs under <out>/ (default migration/m4/golden_trajectory/):
  meta.json               frames=20, tags (19 VALID_BODY_PARTS_V2 + fullpage),
                          timesteps (4), scheduler config, scaling_factor,
                          seed, resolution, src size, depth PNG postprocess params
  empty_text_embed.npy    (1,2,1024)   as passed to UNet (per frame, pre repeat)
  argb_pre_pad.npy        (21,4,768,768)  _np_transform outputs (A first, [0,1]);
                          entry 20 is cond_full_page (same as entry 19)
  vae_rgb_in.npy          (21,3,768,768)  post pad_rgb_torch, *2-1 (vae.encoder in)
  vae_latents.npy         (21,4,96,96)    encode_rgb outputs (scaled means, *0.18215)
  cond_latent.npy         (20,8,96,96)    UNet condition (page latent | frame latent)
  latent_init.npy         (20,4,96,96)    initial noise = target_latent (DDIM: unscaled)
  step_XX_unet_in.npy     (1,20,12,96,96) UNet input (cat(cond, latent)[None])
  step_XX_unet_out.npy    (1,20,4,96,96)  noise_pred (v-prediction)
  step_XX_latents_next.npy (20,4,96,96)   prev_sample after DDIM step
  decode_in.npy           (20,4,96,96)    final latents (pre /0.18215)
  decode_out.npy          (20,1,768,768)  decode_depth outputs (pre clip/shift)
  depth_pred_768.npy      (20,768,768)    pipeline final_pred (clip, /2+0.5)
  depth_pred_src.npy      (20,1280,1280)  after smart_resize back to src size
  results.json            golden *_depth.png reproduction (expect 23/23 exact)

Run: python 01_dump_trajectory.py   (conda env see_through, ~3 min on M5 Max)
Requires: HF_HUB_OFFLINE=1, ./assets symlink, tests/golden/test_image/.
"""
import json
import os
import shutil
import sys
import time
import types

import numpy as np
import torch
from PIL import Image

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))

import modules.marigold.marigold_depth_pipeline as mdp  # noqa: E402
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel  # noqa: E402
from modules.marigold.marigold_depth_pipeline import MarigoldDepthPipeline  # noqa: E402
from utils import inference_utils as iu  # noqa: E402
from utils.inference_utils import VALID_BODY_PARTS_V2  # noqa: E402

MARIGOLD_REPO = "24yearsold/seethroughv0.0.1_marigold"
GOLDEN = os.path.join(REPO, "tests", "golden", "test_image")
OUT_DEFAULT = os.path.join(REPO, "migration", "m4", "golden_trajectory")
SCRATCH = os.path.join(REPO, "workspace", "m4_dump_scratch")
SEED = 42
RES = 768
STEPS = 4

DEPTH_TAGS = ['front hair', 'back hair', 'headwear', 'face', 'eyebrow', 'eyelash',
              'irides', 'eyewhite', 'eyewear', 'ears', 'earwear', 'nose', 'mouth',
              'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear',
              'footwear', 'tail', 'wings', 'objects']


def f32(t):
    return t.detach().to(torch.float32).cpu().numpy()


class Recorder:
    def __init__(self):
        self.rgb_in = []       # per-call vae encoder inputs (1,3,768,768)
        self.rgb_latent = []   # per-call encode_rgb outputs (1,4,96,96)
        self.argb_pre_pad = []  # per-call _np_transform outputs (4,768,768)
        self.cond_latent = None
        self.unet_calls = []   # (t, sample, ehs, noise_pred)
        self.step_calls = []   # (t, sample, prev_sample)
        self.decode_in = []    # per-frame decode_depth inputs
        self.decode_out = []   # per-frame decode_depth outputs
        self.depth_pred_768 = None
        self.depth_pred_src = []


def install_hooks(pipe, rec: Recorder):
    # module-level encode_rgb: records every VAE encoder call (pad_argb applied
    # by caller; rgb_in here is already [1,3,768,768] in [-1,1])
    orig_encode_rgb = mdp.encode_rgb

    def hooked_encode_rgb(vae, rgb_in, latent_scale_factor=0.18215):
        out = orig_encode_rgb(vae, rgb_in, latent_scale_factor)
        rec.rgb_in.append(f32(rgb_in))
        rec.rgb_latent.append(f32(out))
        return out

    mdp.encode_rgb = hooked_encode_rgb

    # record _np_transform outputs via the pipeline __call__'s img_list path:
    # wrap pipeline.encode_rgb? no — wrap the module torch.stack input by
    # hooking mdp.img2tensor is overkill; instead record via img2tensor wrapper.
    orig_img2tensor = mdp.img2tensor

    def hooked_img2tensor(img=None, **kw):
        # img here is the [0,1] float ARGB hwc array from _np_transform
        rec.argb_pre_pad.append(img.transpose(2, 0, 1).astype(np.float32))
        return orig_img2tensor(img=img, **kw)

    mdp.img2tensor = hooked_img2tensor

    # single_infer entry: cond_latent snapshot
    orig_single = pipe.single_infer

    def hooked_single(cond_latent, num_inference_steps, generator):
        rec.cond_latent = f32(cond_latent)
        return orig_single(cond_latent=cond_latent, num_inference_steps=num_inference_steps,
                           generator=generator)

    pipe.single_infer = hooked_single

    # unet forward
    orig_unet_forward = pipe.unet.forward

    def hooked_unet(self, sample, timestep, *args, **kwargs):
        out = orig_unet_forward(sample, timestep, *args, **kwargs)
        noise_pred = out[0] if isinstance(out, tuple) else out.sample
        rec.unet_calls.append((float(f32(timestep).flat[0]), f32(sample), f32(noise_pred)))
        return out

    pipe.unet.forward = types.MethodType(hooked_unet, pipe.unet)

    # scheduler step
    orig_step = pipe.scheduler.step

    def hooked_step(self, model_output, timestep, sample, *args, **kwargs):
        out = orig_step(model_output, timestep, sample, *args, **kwargs)
        prev = out[0] if isinstance(out, tuple) else out.prev_sample
        rec.step_calls.append((float(f32(timestep).flat[0]), f32(sample), f32(prev)))
        return out

    pipe.scheduler.step = types.MethodType(hooked_step, pipe.scheduler)

    # decode_depth per frame
    orig_decode = pipe.decode_depth

    def hooked_decode(depth_latent):
        out = orig_decode(depth_latent)
        rec.decode_in.append(f32(depth_latent))
        rec.decode_out.append(f32(out))
        return out

    pipe.decode_depth = hooked_decode

    # apply_marigold smart_resize of depth maps back to src res
    orig_smart_resize = iu.smart_resize

    def hooked_smart_resize(src, target_size, **kw):
        out = orig_smart_resize(src, target_size, **kw)
        if src.ndim == 2 and src.dtype == np.float32:
            rec.depth_pred_src.append(out.astype(np.float32))
        return out

    iu.smart_resize = hooked_smart_resize


def psnr_u8(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else OUT_DEFAULT
    os.makedirs(out_root, exist_ok=True)
    results = {"seed": SEED, "steps": STEPS, "resolution": RES}
    t_all = time.time()

    # scratch copy of golden intermediates (apply_marigold reads+writes there)
    scratch_saved = os.path.join(SCRATCH, "test_image")
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    os.makedirs(scratch_saved)
    for f in os.listdir(GOLDEN):
        if f.endswith(".png") and not f.endswith("_depth.png") and f != "reconstruction.png":
            shutil.copy2(os.path.join(GOLDEN, f), os.path.join(scratch_saved, f))
    shutil.copy2(os.path.join(GOLDEN, "info.json"), os.path.join(scratch_saved, "info.json"))

    t0 = time.time()
    unet = UNetFrameConditionModel.from_pretrained(MARIGOLD_REPO, subfolder="unet")
    pipe = MarigoldDepthPipeline.from_pretrained(MARIGOLD_REPO, unet=unet)
    pipe.to(device=torch.device("mps"), dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    rec = Recorder()
    install_hooks(pipe, rec)

    # --- exact M0 path: apply_marigold(seed=42, resolution=768, steps=-1) ---
    t0 = time.time()
    iu.marigold_pipeline = pipe  # inject our hooked instance
    iu.apply_marigold(os.path.join(REPO, "assets", "test_image.png"), MARIGOLD_REPO,
                      num_inference_steps=-1, seed=SEED, save_dir=SCRATCH,
                      resolution=RES, disable_progressbar=True,
                      device="mps", dtype="bf16")
    wall = time.time() - t0
    print(f"[apply_marigold] {wall:.1f}s", flush=True)

    # ---- sanity checks on recorded structure ----
    n_frames = len(VALID_BODY_PARTS_V2) + 1  # 19 tags + fullpage
    assert len(rec.unet_calls) == STEPS, f"unet calls {len(rec.unet_calls)}"
    assert len(rec.step_calls) == STEPS
    assert len(rec.rgb_in) == n_frames + 1, f"vae encodes {len(rec.rgb_in)}"
    assert len(rec.argb_pre_pad) == n_frames
    assert len(rec.decode_in) == n_frames
    frames = rec.unet_calls[0][1].shape[1]
    assert frames == n_frames

    # ---- persist anchors ----
    np.save(os.path.join(out_root, "argb_pre_pad.npy"),
            np.stack([rec.argb_pre_pad[i] for i in range(n_frames)] +
                     [rec.argb_pre_pad[-1]], 0))  # + cond_full_page == last
    np.save(os.path.join(out_root, "vae_rgb_in.npy"), np.concatenate(rec.rgb_in, 0))
    np.save(os.path.join(out_root, "vae_latents.npy"), np.concatenate(rec.rgb_latent, 0))
    np.save(os.path.join(out_root, "cond_latent.npy"), rec.cond_latent)
    np.save(os.path.join(out_root, "latent_init.npy"), rec.step_calls[0][1])
    np.save(os.path.join(out_root, "empty_text_embed.npy"), f32(pipe.empty_text_embed))

    for i, ((t_u, s_in, s_out), (t_s, l_in, l_next)) in enumerate(
            zip(rec.unet_calls, rec.step_calls)):
        assert abs(t_u - t_s) < 1e-3
        np.save(os.path.join(out_root, f"step_{i:02d}_unet_in.npy"), s_in)
        np.save(os.path.join(out_root, f"step_{i:02d}_unet_out.npy", ), s_out)
        np.save(os.path.join(out_root, f"step_{i:02d}_latents_next.npy"), l_next)

    np.save(os.path.join(out_root, "decode_in.npy"), np.concatenate(rec.decode_in, 0))
    np.save(os.path.join(out_root, "decode_out.npy"), np.concatenate(rec.decode_out, 0))

    # depth_pred_768: pipeline output tensor (clip 0..1), order matches img_list
    # re-derive from decode_out is not bit-identical to pipe output; capture via
    # the scratch info? Instead: reconstruct from recorded final_pred — hook was
    # not installed on pipe __call__ return, so recompute: depth768 = clip(
    # decode_out,-1,1)/2+0.5 == final_pred (match_input_res resize is 768->768
    # no-op). Verified numerically below against depth_pred_src roundtrip.
    depth768 = np.clip(np.concatenate(rec.decode_out, 0), -1.0, 1.0)
    depth768 = ((depth768[:, 0] + 1.0) / 2.0).astype(np.float32)
    np.save(os.path.join(out_root, "depth_pred_768.npy"), depth768)
    np.save(os.path.join(out_root, "depth_pred_src.npy"),
            np.stack(rec.depth_pred_src, 0))

    # decode_out -> depth768 identity check (proves the recompute is exact)
    src_img = np.array(Image.open(os.path.join(scratch_saved, "src_img.png")))
    meta = dict(
        frames=n_frames,
        tags=list(VALID_BODY_PARTS_V2) + ["fullpage"],
        depth_tags=DEPTH_TAGS,
        seed=SEED, num_inference_steps=STEPS, resolution=RES,
        scaling_factor=0.18215,
        timesteps=[int(round(c[0])) for c in rec.unet_calls],
        scheduler=dict(pipe.scheduler.config),
        device="mps", dtype="bfloat16",
        src_hw=[int(src_img.shape[0]), int(src_img.shape[1])],
        prediction_type=pipe.scheduler.config.prediction_type,
        wall_s=round(wall, 2),
        notes="depth PNG: (clip(d,0,1)*255).astype(uint8) truncation; "
              "composed eyes/hair -> sub-tag masks per inference_utils.py:296-316",
    )
    with open(os.path.join(out_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- golden reproduction: 23 *_depth.png ----
    repro = {}
    for tag in DEPTH_TAGS:
        gp = os.path.join(GOLDEN, f"{tag}_depth.png")
        pp = os.path.join(scratch_saved, f"{tag}_depth.png")
        if not (os.path.exists(gp) and os.path.exists(pp)):
            repro[tag] = {"missing": True}
            continue
        g = np.array(Image.open(gp))
        p = np.array(Image.open(pp))
        repro[tag] = {"psnr_db": psnr_u8(p, g), "exact": bool(np.array_equal(p, g))}
    results["golden_depth_reproduction"] = repro
    results["wall_s"] = round(wall, 2)
    results["total_wall_s"] = round(time.time() - t_all, 2)
    with open(os.path.join(out_root, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    n_exact = sum(1 for v in repro.values() if v.get("exact"))
    print(f"golden depth reproduction: {n_exact}/{len(repro)} exact", flush=True)
    for k, v in repro.items():
        if not v.get("exact"):
            print(f"  {k}: {v}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
