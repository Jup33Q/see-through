"""Spike B step 1: PyTorch reference forward for UNetFrameConditionModel.

Fixed-shape single forward: (B=1, num_frames, 8, S, S) latent @ 1280px -> S=160.
Saves inputs (npz) + reference output (npy) for later CoreML comparison.
"""
import argparse
import os
import sys
import time
import json

import numpy as np
import torch

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))

from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel  # noqa: E402

REPO_ID = "layerdifforg/seethroughv0.0.2_layerdiff3d"


def make_inputs(num_frames: int, size: int, device: str, dtype: torch.dtype, seed: int = 1234):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn(1, num_frames, 8, size, size, generator=g).to(device=device, dtype=dtype)
    timestep = torch.tensor(500.0, device=device, dtype=torch.float32)
    encoder_hidden_states = torch.randn(num_frames, 77, 2048, generator=g).to(device=device, dtype=dtype)
    text_embeds = torch.randn(num_frames, 1280, generator=g).to(device=device, dtype=dtype)
    time_ids = torch.randn(num_frames, 6, generator=g).to(device=device, dtype=dtype)
    return sample, timestep, encoder_hidden_states, text_embeds, time_ids


def rss_gb():
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--tag", default="body13_160")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    t0 = time.time()
    unet = UNetFrameConditionModel.from_pretrained(REPO_ID, subfolder="unet", torch_dtype=dtype)
    unet.eval().to(args.device)
    print(f"[load] {time.time()-t0:.1f}s, params={sum(p.numel() for p in unet.parameters())/1e9:.2f}B, rss={rss_gb():.1f}GB")

    inputs = make_inputs(args.frames, args.size, args.device, dtype)
    sample, timestep, ehs, text_embeds, time_ids = inputs

    with torch.inference_mode():
        t0 = time.time()
        out = unet(
            sample,
            timestep,
            encoder_hidden_states=ehs,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
            group_index=args.group,
        )[0]
        if args.device == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
    print(f"[forward] {dt:.1f}s, out.shape={tuple(out.shape)}, rss={rss_gb():.1f}GB")

    out_np = out.float().cpu().numpy()
    np.save(os.path.join(args.outdir, f"ref_out_{args.tag}.npy"), out_np)
    np.savez(
        os.path.join(args.outdir, f"ref_in_{args.tag}.npz"),
        sample=sample.float().cpu().numpy(),
        timestep=np.float32(500.0),
        encoder_hidden_states=ehs.float().cpu().numpy(),
        text_embeds=text_embeds.float().cpu().numpy(),
        time_ids=time_ids.float().cpu().numpy(),
    )
    meta = dict(tag=args.tag, frames=args.frames, group=args.group, size=args.size,
                device=args.device, dtype=args.dtype, forward_sec=dt,
                out_mean=float(out_np.mean()), out_std=float(out_np.std()))
    with open(os.path.join(args.outdir, f"ref_meta_{args.tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
