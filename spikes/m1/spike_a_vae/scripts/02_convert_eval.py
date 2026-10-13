"""Step 2: convert each module to CoreML (jit.trace -> fp16 mlprogram, compute_units=ALL),
run prediction, compare against PyTorch fp32 reference .npy, and time single forwards.

Usage: python 02_convert_eval.py [module ...]   (default: all four)
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import coremltools as ct

from common import (DATA, LOGS, MODELS, RESULTS_JSON,
                    SDXLVaeDecoderWrapper, SDXLVaeEncoderWrapper,
                    TransparentDecoderWrapper, TransparentEncoderWrapper,
                    load_torch_models, max_abs_diff, psnr01)

MODULES = ["sdxl_vae_decoder", "sdxl_vae_encoder", "trans_vae_decoder", "trans_vae_encoder"]

# name -> (wrapper ctor key, [input npy files], ref npy file)
SPEC = {
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


def build_wrappers():
    vae, trans = load_torch_models(dtype=torch.float32, device="cpu")
    return {
        "sdxl_vae_decoder": SDXLVaeDecoderWrapper(vae),
        "sdxl_vae_encoder": SDXLVaeEncoderWrapper(vae),
        "trans_vae_decoder": TransparentDecoderWrapper(trans),
        "trans_vae_encoder": TransparentEncoderWrapper(trans),
    }


def bench_coreml(model, feed, n=5, warmup=2):
    for _ in range(warmup):
        model.predict(feed)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(feed)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run_module(name, wrappers, results):
    in_files, ref_file = SPEC[name]
    inputs = [np.load(os.path.join(DATA, f)) for f in in_files]
    ref = np.load(os.path.join(DATA, ref_file))
    pkg_path = os.path.join(MODELS, f"{name}_fp16.mlpackage")
    entry = {"ref_shape": list(ref.shape)}

    mod = wrappers[name]
    mod.eval()

    # ---- trace + convert ----
    t0 = time.time()
    with torch.no_grad():
        example = [torch.from_numpy(a) for a in inputs]
        traced = torch.jit.trace(mod, tuple(example) if len(example) > 1 else example[0])
    entry["trace_s"] = time.time() - t0

    ct_inputs = [
        ct.TensorType(name=n, shape=a.shape, dtype=np.float32)
        for n, a in zip(INPUT_NAMES[name], inputs)
    ]
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        source="pytorch",
        inputs=ct_inputs,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
    )
    entry["convert_s"] = time.time() - t0
    mlmodel.save(pkg_path)
    entry["mlpackage_mb"] = None  # filled below
    total = 0
    for root, _, files in os.walk(pkg_path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    entry["mlpackage_mb"] = round(total / 1e6, 1)
    entry["spec_type"] = mlmodel.get_spec().WhichOneof("Type")

    # ---- load with ALL and predict ----
    del mlmodel
    model = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.ALL)
    feed = {n: a for n, a in zip(INPUT_NAMES[name], inputs)}
    t0 = time.time()
    out = model.predict(feed)
    entry["first_predict_s"] = time.time() - t0
    out = list(out.values())[0]
    entry["coreml_forward_s"] = bench_coreml(model, feed)

    entry["max_abs_diff"] = max_abs_diff(ref, out)
    entry["psnr01_db"] = psnr01(ref, out)
    peak = float(np.max(np.abs(ref)))
    mse = float(np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2))
    entry["psnr_dynamic_db"] = float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)
    entry["ref_absmax"] = peak
    results["coreml"][name] = entry
    print(f"[{name}] OK max_abs={entry['max_abs_diff']:.5f} psnr01={entry['psnr01_db']:.2f}dB "
          f"fwd={entry['coreml_forward_s']*1e3:.1f}ms", flush=True)


def main():
    names = sys.argv[1:] or MODULES
    results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
    results.setdefault("coreml", {})

    wrappers = build_wrappers()
    for name in names:
        try:
            run_module(name, wrappers, results)
        except Exception:
            err = traceback.format_exc()
            results["coreml"][name] = {"error": err}
            print(f"[{name}] FAILED:\n{err}", flush=True)
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results.get("coreml", {}), indent=2))


if __name__ == "__main__":
    main()
