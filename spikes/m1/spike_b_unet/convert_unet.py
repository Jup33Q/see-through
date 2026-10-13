"""Spike B step 2: torch.jit.trace -> coremltools convert for UNetFrameConditionModel.

Wrapper fixes group_index and added_cond_kwargs structure so trace sees pure tensors.
Logs every checkpoint; on trace/convert failure dumps full stack to the log.
"""
import argparse
import os
import sys
import time
import json
import traceback

import numpy as np
import torch
import torch.nn as nn

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))

from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel  # noqa: E402

REPO_ID = "layerdifforg/seethroughv0.0.2_layerdiff3d"
HERE = os.path.dirname(os.path.abspath(__file__))


class UNetTraceWrapper(nn.Module):
    """Fixed group_index; flat tensor args only. timestep shape (1,)."""

    def __init__(self, unet, group_index: int):
        super().__init__()
        self.unet = unet
        self.group_index = group_index

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        out = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
            group_index=self.group_index,
        )
        return out[0]


def rss_gb():
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--tag", default="body13_160")
    ap.add_argument("--skip-predict", action="store_true")
    args = ap.parse_args()

    result = {"tag": args.tag, "frames": args.frames, "group": args.group, "size": args.size}

    data = np.load(os.path.join(HERE, f"ref_in_{args.tag}.npz"))
    sample = torch.from_numpy(data["sample"])
    timestep = torch.from_numpy(np.atleast_1d(data["timestep"]))
    ehs = torch.from_numpy(data["encoder_hidden_states"])
    text_embeds = torch.from_numpy(data["text_embeds"])
    time_ids = torch.from_numpy(data["time_ids"])
    ref_out = np.load(os.path.join(HERE, f"ref_out_{args.tag}.npy"))
    log(f"loaded inputs: sample{tuple(sample.shape)} ehs{tuple(ehs.shape)}")

    t0 = time.time()
    unet = UNetFrameConditionModel.from_pretrained(REPO_ID, subfolder="unet", torch_dtype=torch.float32)
    unet.eval().to("cpu")
    wrapper = UNetTraceWrapper(unet, args.group).eval()
    log(f"model loaded in {time.time()-t0:.1f}s rss={rss_gb():.1f}GB")

    # ---- checkpoint 1: torch.jit.trace ----
    traced = None
    try:
        with torch.inference_mode():
            t0 = time.time()
            traced = torch.jit.trace(
                wrapper, (sample, timestep, ehs, text_embeds, time_ids), strict=True
            )
            log(f"TRACE OK in {time.time()-t0:.1f}s rss={rss_gb():.1f}GB")
            # sanity: traced forward matches eager
            t0 = time.time()
            tout = traced(sample, timestep, ehs, text_embeds, time_ids).numpy()
            log(f"traced forward {time.time()-t0:.1f}s")
            d = np.abs(tout - ref_out)
            denom = np.maximum(np.abs(ref_out), 1e-6)
            log(f"trace-vs-eager: max_abs={d.max():.3e} mean_abs={d.mean():.3e} "
                f"max_rel={(d/denom).max():.3e}")
            result["trace_ok"] = True
            result["trace_vs_eager_max_abs"] = float(d.max())
    except Exception:
        result["trace_ok"] = False
        result["trace_error"] = traceback.format_exc()
        log("TRACE FAILED:\n" + result["trace_error"])
        with open(os.path.join(HERE, f"convert_result_{args.tag}.json"), "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(2)

    torch.jit.save(traced, os.path.join(HERE, f"traced_{args.tag}.pt"))
    log(f"traced graph saved ({os.path.getsize(os.path.join(HERE, f'traced_{args.tag}.pt'))/1e9:.2f}GB)")

    # ---- checkpoint 2: ct.convert ----
    mlmodel = None
    try:
        import coremltools as ct

        log(f"coremltools {ct.__version__}")
        inputs = [
            ct.TensorType(name="sample", shape=sample.shape, dtype=np.float32),
            ct.TensorType(name="timestep", shape=timestep.shape, dtype=np.float32),
            ct.TensorType(name="encoder_hidden_states", shape=ehs.shape, dtype=np.float32),
            ct.TensorType(name="text_embeds", shape=text_embeds.shape, dtype=np.float32),
            ct.TensorType(name="time_ids", shape=time_ids.shape, dtype=np.float32),
        ]
        t0 = time.time()
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=[ct.TensorType(name="noise_pred", dtype=np.float32)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        log(f"CONVERT OK in {time.time()-t0:.1f}s rss={rss_gb():.1f}GB")
        result["convert_ok"] = True
    except Exception:
        result["convert_ok"] = False
        result["convert_error"] = traceback.format_exc()
        log("CONVERT FAILED:\n" + result["convert_error"])
        with open(os.path.join(HERE, f"convert_result_{args.tag}.json"), "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(3)

    pkg = os.path.join(HERE, f"unet_{args.tag}.mlpackage")
    mlmodel.save(pkg)
    import subprocess

    sz = subprocess.run(["du", "-sh", pkg], capture_output=True, text=True).stdout.split()[0]
    log(f"mlpackage saved: {pkg} ({sz})")
    result["mlpackage"] = pkg
    result["mlpackage_size"] = sz

    # ---- checkpoint 3: load + predict + compare ----
    if not args.skip_predict:
        try:
            import coremltools as ct

            t0 = time.time()
            mlmodel = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)
            log(f"mlpackage load {time.time()-t0:.1f}s rss={rss_gb():.1f}GB")
            feed = {
                "sample": sample.numpy(),
                "timestep": timestep.numpy(),
                "encoder_hidden_states": ehs.numpy(),
                "text_embeds": text_embeds.numpy(),
                "time_ids": time_ids.numpy(),
            }
            # warmup + timed runs
            t0 = time.time()
            pred = mlmodel.predict(feed)["noise_pred"]
            t_first = time.time() - t0
            times = []
            for _ in range(3):
                t0 = time.time()
                pred = mlmodel.predict(feed)["noise_pred"]
                times.append(time.time() - t0)
            log(f"predict first={t_first:.1f}s, next3={[f'{t:.1f}' for t in times]}s rss={rss_gb():.1f}GB")
            result["predict_first_sec"] = t_first
            result["predict_times_sec"] = times
            result["rss_gb"] = rss_gb()

            d = np.abs(pred - ref_out)
            mse = float(((pred - ref_out) ** 2).mean())
            peak = float(np.abs(ref_out).max())
            psnr = 10 * np.log10(peak * peak / mse) if mse > 0 else float("inf")
            log(f"coreml-vs-ref: max_abs={d.max():.4e} mean_abs={d.mean():.4e} "
                f"mse={mse:.4e} PSNR(peak={peak:.3f})={psnr:.2f}dB")
            result.update(max_abs=float(d.max()), mean_abs=float(d.mean()), mse=mse, psnr=psnr)
        except Exception:
            result["predict_error"] = traceback.format_exc()
            log("PREDICT FAILED:\n" + result["predict_error"])

    with open(os.path.join(HERE, f"convert_result_{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
