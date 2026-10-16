"""M4-3: CoreML conversion of the Marigold VAE encoder/decoder (SD2-style,
scaling_factor 0.18215 — distinct from the M2 SDXL VAE).

Wrappers (mirror marigold_depth_pipeline.encode_rgb / decode_depth exactly):
  marigold_vae_encoder: rgb [1,3,768,768] in [-1,1] (post pad_rgb_torch)
      -> encoder -> quant_conv -> chunk mean -> *0.18215  => [1,4,96,96]
     (Marigold never samples the cond latent: mean-only path.)
  marigold_vae_decoder: latent [1,4,96,96] (scaled)
      -> /0.18215 -> post_quant_conv -> decoder -> mean(dim=1)  => [1,1,768,768]

Alignment anchors come from migration/m4/golden_trajectory/ (bf16 MPS compute,
stored fp32): vae_rgb_in/vae_latents frames {0, 19}, decode_in/decode_out
frames {0, 19}. Thresholds follow M2 (dyn PSNR >= 40 latent domain).
Also records bench + compute-plan device attribution + ANE-path runnability.

Run: python 03_convert_vae.py   (conda env see_through; a few minutes)
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import coremltools as ct

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, os.path.join(REPO, "migration", "m2", "scripts"))

from common import compute_plan_device_stats, dir_size_mb  # noqa: E402

M4 = os.path.join(REPO, "migration", "m4")
TRAJ = os.path.join(M4, "golden_trajectory")
OUT_MODELS = os.path.join(M4, "out", "models")
LOG = os.path.join(M4, "out", "logs", "convert_vae.log")
RESULTS = os.path.join(M4, "out", "convert_vae_results.json")
SCALING = 0.18215
FRAMES = (0, 19)  # 0 = 'hair' slot, 19 = fullpage
DYN_THRESHOLD = 40.0


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


class MarigoldVaeEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, x):
        moments = self.quant_conv(self.encoder(x))
        mean, _ = torch.chunk(moments, 2, dim=1)
        return mean * SCALING


class MarigoldVaeDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, z):
        z = z / SCALING
        stacked = self.decoder(self.post_quant_conv(z))
        return stacked.mean(dim=1, keepdim=True)


def psnr_dynamic(a, b):
    peak = float(np.max(np.abs(a)))
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


def bench(model, feed, n=5, warmup=2):
    for _ in range(warmup):
        model.predict(feed)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(feed)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def convert_and_check(name, wrapper, input_name, in_shape, feeds_refs, results):
    pkg = os.path.join(OUT_MODELS, f"{name}_fp16.mlpackage")
    entry = {"package": pkg, "threshold_dyn_db": DYN_THRESHOLD}
    wrapper.eval()

    with torch.no_grad():
        example = torch.from_numpy(feeds_refs[0][1][0])
        traced = torch.jit.trace(wrapper, example)

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=in_shape, dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    entry["convert_s"] = round(time.time() - t0, 2)
    if os.path.exists(pkg):
        import shutil
        shutil.rmtree(pkg)
    mlmodel.save(pkg)
    entry["mlpackage_mb"] = dir_size_mb(pkg)
    del mlmodel, traced

    model = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)
    checks = {}
    for tag, (inp, ref) in feeds_refs:
        feed = {input_name: inp}
        out = list(model.predict(feed).values())[0]
        p = psnr_dynamic(ref, out)
        entry.setdefault("align_all", {})[tag] = {
            "psnr_dynamic_db": p, "max_abs": float(np.abs(out - ref).max())}
        checks[tag] = bool(p >= DYN_THRESHOLD)
        log(f"[{name}] {tag}: dyn_psnr={p:.2f}dB max_abs={entry['align_all'][tag]['max_abs']:.3e}")
    entry["coreml_forward_s"] = round(bench(model, {input_name: feeds_refs[0][1][0]}), 4)
    del model

    # ANE path runnability (M2 convention)
    model_ane = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out_ane = list(model_ane.predict({input_name: feeds_refs[0][1][0]}).values())[0]
    p_ane = psnr_dynamic(feeds_refs[0][1][1], out_ane)
    entry["ane_path"] = {"runnable": True, "psnr_dynamic_db": p_ane}
    checks["ane"] = bool(p_ane >= DYN_THRESHOLD)
    del model_ane

    entry["compute_plan"] = compute_plan_device_stats(pkg)
    entry["pass"] = bool(all(checks.values()))
    entry["checks"] = checks
    results["modules"][name] = entry
    log(f"[{name}] fwd={entry['coreml_forward_s']}s plan={entry['compute_plan']['preferred_device']} "
        f"ane={p_ane:.2f}dB pass={entry['pass']}")


def main():
    from diffusers import AutoencoderKL
    for d in (OUT_MODELS, os.path.dirname(LOG)):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(LOG):
        os.remove(LOG)
    results = {"modules": {}, "scaling_factor": SCALING}
    try:
        vae = AutoencoderKL.from_pretrained(
            "24yearsold/seethroughv0.0.1_marigold", subfolder="vae",
            torch_dtype=torch.float32).to("cpu").eval()
        log("marigold vae loaded (fp32 cpu)")

        rgb_in = np.load(os.path.join(TRAJ, "vae_rgb_in.npy"))
        lat = np.load(os.path.join(TRAJ, "vae_latents.npy"))
        dec_in = np.load(os.path.join(TRAJ, "decode_in.npy"))
        dec_out = np.load(os.path.join(TRAJ, "decode_out.npy"))

        enc_feeds = [(f"frame_{i:02d}", (rgb_in[i:i + 1], lat[i:i + 1])) for i in FRAMES]
        dec_feeds = [(f"frame_{i:02d}", (dec_in[i:i + 1], dec_out[i:i + 1])) for i in FRAMES]

        convert_and_check("marigold_vae_encoder", MarigoldVaeEncoderWrapper(vae),
                          "rgb", (1, 3, 768, 768), enc_feeds, results)
        convert_and_check("marigold_vae_decoder", MarigoldVaeDecoderWrapper(vae),
                          "z", (1, 4, 96, 96), dec_feeds, results)
        results["pass"] = all(m["pass"] for m in results["modules"].values())
    except Exception:
        results["error"] = traceback.format_exc()
        results["pass"] = False
        log("FAILED:\n" + results["error"])
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    log(f"done pass={results['pass']}")


if __name__ == "__main__":
    main()
