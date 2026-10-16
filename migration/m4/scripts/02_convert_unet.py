"""M4-2: CoreML conversion of the Marigold UNet (UNetFrameConditionModel).

Unlike M3 there is nothing to merge: config has no `group_embedding`
(both group branches are skipped), and the frame count is fixed
(19 VALID_BODY_PARTS_V2 slots + fullpage = 20), so a single static shape
suffices — no EnumeratedShapes, hence neither of the M3 symbolic-shape
patches (repeat_interleave surgery / _group_norm_impl) is needed.

  wrapper inputs:  sample (1,20,12,96,96)  cat(cond_latent, target_latent)[None]
                   timestep (1,)
                   encoder_hidden_states (20,2,1024)  empty prompt, repeated

Checkpoints:
  0. eager fp32 reference on golden-trajectory anchor inputs (steps 00/02)
  1. torch.jit.trace (static shapes)
  2. ct.convert fp16 mlprogram, computeUnits=ALL
  3. load + predict vs anchors (PSNR) + bench + compute plan

Run: python 02_convert_unet.py [--from-traced]  (conda env see_through)
Requires migration/m4/golden_trajectory/ (01_dump_trajectory.py).
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, os.path.join(REPO, "migration", "m2", "scripts"))

from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel  # noqa: E402
from common import compute_plan_device_stats, dir_size_mb  # noqa: E402  (m2 helpers)

MARIGOLD_REPO = "24yearsold/seethroughv0.0.1_marigold"
M4 = os.path.join(REPO, "migration", "m4")
TRAJ = os.path.join(M4, "golden_trajectory")
OUT_MODELS = os.path.join(M4, "out", "models")
LOG = os.path.join(M4, "out", "logs", "convert_unet.log")
RESULTS = os.path.join(M4, "out", "convert_unet_results.json")
PKG = os.path.join(OUT_MODELS, "marigold_unet_fp16.mlpackage")
TRACED = os.path.join(M4, "out", "traced_unet.pt")
CHECK_STEPS = (0, 2)
PSNR_THRESHOLD = 60.0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def rss_gb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


class UNetWrapper(nn.Module):
    """UNetFrameConditionModel without added_cond_kwargs / group_index
    (marigold config: addition_embed_type=None, group_embedding absent)."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep,
                         encoder_hidden_states=encoder_hidden_states,
                         return_dict=False)[0]


def anchor_feed(step):
    sample = torch.from_numpy(np.load(os.path.join(TRAJ, f"step_{step:02d}_unet_in.npy")))
    ehs = torch.from_numpy(np.load(os.path.join(TRAJ, "empty_text_embed.npy")))
    meta = json.load(open(os.path.join(TRAJ, "meta.json")))
    t = torch.tensor([float(meta["timesteps"][step])], dtype=torch.float32)
    ehs = ehs.repeat(sample.shape[1], 1, 1)
    ref = np.load(os.path.join(TRAJ, f"step_{step:02d}_unet_out.npy"))
    return {"sample": sample, "timestep": t, "encoder_hidden_states": ehs}, ref


def psnr_vs(ref, out):
    mse = float(((ref.astype(np.float64) - out.astype(np.float64)) ** 2).mean())
    peak = float(np.abs(ref).max())
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


def main():
    from_traced = "--from-traced" in sys.argv
    for d in (OUT_MODELS, os.path.dirname(LOG)):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(LOG) and not from_traced:
        os.remove(LOG)
    results = {"package": PKG, "check_steps": CHECK_STEPS,
               "psnr_threshold": PSNR_THRESHOLD}
    try:
        t0 = time.time()
        unet = UNetFrameConditionModel.from_pretrained(
            MARIGOLD_REPO, subfolder="unet", torch_dtype=torch.float32)
        unet.eval().to("cpu")
        assert not unet.config.group_embedding, "unexpected group_embedding in config"
        n_params = sum(p.numel() for p in unet.parameters())
        log(f"unet loaded in {time.time()-t0:.1f}s params={n_params/1e9:.2f}B rss={rss_gb():.1f}GB")
        results["params_b"] = round(n_params / 1e9, 3)

        feeds = {s: anchor_feed(s) for s in CHECK_STEPS}
        wrapper = UNetWrapper(unet).eval()

        # ---- checkpoint 0: eager fp32 reference ----
        for s in CHECK_STEPS:
            feed, ref = feeds[s]
            with torch.inference_mode():
                t0 = time.time()
                out = wrapper(**feed).numpy()
            p = psnr_vs(ref, out)
            results.setdefault("eager_fp32_vs_anchor_bf16", {})[f"step_{s:02d}"] = {
                "psnr_db": p, "max_abs": float(np.abs(out - ref).max()),
                "forward_s": round(time.time() - t0, 2)}
            log(f"eager[step{s}]: psnr={p:.2f}dB (fp32 vs bf16 anchor) "
                f"({time.time()-t0:.1f}s cpu)")

        # ---- checkpoint 1: trace ----
        if from_traced and os.path.exists(TRACED):
            traced = torch.jit.load(TRACED)
            log(f"traced graph reloaded from {TRACED}")
        else:
            ex = feeds[CHECK_STEPS[0]][0]
            example = (ex["sample"], ex["timestep"], ex["encoder_hidden_states"])
            with torch.inference_mode():
                t0 = time.time()
                traced = torch.jit.trace(wrapper, example, strict=True)
            results["trace_s"] = round(time.time() - t0, 2)
            log(f"TRACE OK in {results['trace_s']}s rss={rss_gb():.1f}GB")
            torch.jit.save(traced, TRACED)
            log(f"traced graph saved: {TRACED}")
        del wrapper, unet
        import gc
        gc.collect()

        # ---- checkpoint 2: convert (static shapes) ----
        import coremltools as ct
        log(f"coremltools {ct.__version__}")
        inputs = [
            ct.TensorType(name="sample", shape=(1, 20, 12, 96, 96), dtype=np.float32),
            ct.TensorType(name="timestep", shape=(1,), dtype=np.float32),
            ct.TensorType(name="encoder_hidden_states", shape=(20, 2, 1024),
                          dtype=np.float32),
        ]
        t0 = time.time()
        mlmodel = ct.convert(
            traced, inputs=inputs,
            outputs=[ct.TensorType(name="noise_pred", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        results["convert_s"] = round(time.time() - t0, 2)
        log(f"CONVERT OK in {results['convert_s']}s rss={rss_gb():.1f}GB")
        if os.path.exists(PKG):
            import shutil
            shutil.rmtree(PKG)
        mlmodel.save(PKG)
        results["mlpackage_mb"] = dir_size_mb(PKG)
        log(f"mlpackage saved: {PKG} ({results['mlpackage_mb']}MB)")
        del mlmodel, traced

        # ---- checkpoint 3: load + predict + bench + compute plan ----
        gc.collect()
        model = ct.models.MLModel(PKG, compute_units=ct.ComputeUnit.ALL)
        log(f"mlpackage loaded rss={rss_gb():.1f}GB")
        for s in CHECK_STEPS:
            feed, ref = feeds[s]
            feed_np = {k: v.numpy() for k, v in feed.items()}
            t0 = time.time()
            pred = model.predict(feed_np)["noise_pred"]
            first = time.time() - t0
            times = []
            for _ in range(3):
                t0 = time.time()
                pred = model.predict(feed_np)["noise_pred"]
                times.append(time.time() - t0)
            p = psnr_vs(ref, pred)
            results.setdefault("coreml", {})[f"step_{s:02d}"] = {
                "psnr_db": p, "max_abs": float(np.abs(pred - ref).max()),
                "predict_first_s": round(first, 2),
                "predict_times_s": [round(t, 2) for t in times]}
            log(f"coreml[step{s}]: psnr={p:.2f}dB first={first:.1f}s "
                f"next3={[f'{t:.1f}' for t in times]}s rss={rss_gb():.1f}GB")
        results["compute_plan"] = compute_plan_device_stats(PKG)
        log(f"compute plan: {results['compute_plan']}")
        results["pass"] = all(v["psnr_db"] >= PSNR_THRESHOLD
                              for v in results["coreml"].values())
    except Exception:
        results["error"] = traceback.format_exc()
        results["pass"] = False
        log("FAILED:\n" + results["error"])
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    log(f"done pass={results['pass']}")


if __name__ == "__main__":
    main()
