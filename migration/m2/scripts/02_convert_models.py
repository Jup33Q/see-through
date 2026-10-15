"""M2-2/3/4: formal CoreML conversion of the four small modules + golden alignment
+ compute-plan device attribution + ANE-path (CPU_AND_NEURAL_ENGINE) runnability.

Evolved from spikes/m1/spike_a_vae/scripts/02_convert_eval.py. Differences:
- artifacts land in migration/m2/out/models/ (formal M2 packages);
- alignment adds SSIM for image-domain outputs and explicit pass/fail thresholds;
- every module gets a compute-plan device attribution (M1 leftover requirement);
- every module is additionally loaded with computeUnits=CPU_AND_NEURAL_ENGINE and
  predicted once to prove the non-GPU path works.

Usage: python 02_convert_models.py [module ...]   (default: all four)
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

import coremltools as ct

from common import (IMAGE_DOMAIN_MODULES, INPUT_NAMES, LOGS, MODELS, MODULE_SPEC,
                    RESULTS_JSON, SPIKE_A_DATA, THRESHOLDS, build_wrappers,
                    compute_plan_device_stats, dir_size_mb, max_abs_diff, psnr01,
                    psnr_dynamic, ssim_channels_last)

MODULES = list(MODULE_SPEC.keys())


def bench_coreml(model, feed, n=5, warmup=2):
    for _ in range(warmup):
        model.predict(feed)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(feed)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def metrics_against_ref(ref, out):
    entry = {
        "max_abs_diff": max_abs_diff(ref, out),
        "psnr01_db": float(psnr01(ref, out)),
        "psnr_dynamic_db": float(psnr_dynamic(ref, out)),
    }
    return entry


def evaluate(name, ref, out, entry, key):
    m = metrics_against_ref(ref, out)
    entry[key] = m
    checks = {}
    if name in IMAGE_DOMAIN_MODULES:
        m["ssim"] = float(ssim_channels_last(ref, out))
        checks["psnr01_db"] = bool(m["psnr01_db"] >= THRESHOLDS["psnr01_db"])
        checks["ssim"] = bool(m["ssim"] >= THRESHOLDS["ssim"])
    else:
        checks["psnr_dynamic_db"] = bool(m["psnr_dynamic_db"] >= THRESHOLDS["psnr_dynamic_db"])
    return checks


def run_module(name, wrappers, results):
    in_files, ref_file = MODULE_SPEC[name]
    inputs = [np.load(os.path.join(SPIKE_A_DATA, f)) for f in in_files]
    ref = np.load(os.path.join(SPIKE_A_DATA, ref_file))
    pkg_path = os.path.join(MODELS, f"{name}_fp16.mlpackage")
    entry = {"ref_shape": list(ref.shape),
             "ref_source": os.path.relpath(os.path.join(SPIKE_A_DATA, ref_file)),
             "thresholds": THRESHOLDS}

    mod = wrappers[name]
    mod.eval()

    # ---- trace + convert (fp16 mlprogram, computeUnits=ALL) ----
    t0 = time.time()
    with torch.no_grad():
        example = [torch.from_numpy(a) for a in inputs]
        traced = torch.jit.trace(mod, tuple(example) if len(example) > 1 else example[0])
    entry["trace_s"] = round(time.time() - t0, 2)

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
    entry["convert_s"] = round(time.time() - t0, 2)
    if os.path.exists(pkg_path):
        import shutil
        shutil.rmtree(pkg_path)
    mlmodel.save(pkg_path)
    entry["mlpackage_mb"] = dir_size_mb(pkg_path)
    del mlmodel

    feed = {n: a for n, a in zip(INPUT_NAMES[name], inputs)}

    # ---- load ALL, align vs fp32 reference, bench ----
    model = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.ALL)
    out = list(model.predict(feed).values())[0]
    entry["coreml_forward_s"] = round(bench_coreml(model, feed), 4)
    checks = evaluate(name, ref, out, entry, "align_all")
    del model

    # ---- ANE path runnability: CPU_AND_NEURAL_ENGINE load + predict ----
    model_ane = ct.models.MLModel(pkg_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out_ane = list(model_ane.predict(feed).values())[0]
    checks_ane = evaluate(name, ref, out_ane, entry, "align_cpu_and_neural_engine")
    entry["ane_path_runnable"] = True
    checks.update({f"ane_{k}": v for k, v in checks_ane.items()})
    del model_ane

    # ---- compute-plan device attribution (M1 leftover) ----
    entry["compute_plan"] = compute_plan_device_stats(pkg_path)

    entry["pass"] = bool(all(checks.values()))
    entry["checks"] = checks
    results["modules"][name] = entry
    al = entry["align_all"]
    print(f"[{name}] psnr01={al['psnr01_db']:.2f}dB dyn={al['psnr_dynamic_db']:.2f}dB "
          f"ssim={al.get('ssim', float('nan')):.4f} plan={entry['compute_plan']['preferred_device']} "
          f"pass={entry['pass']}", flush=True)


def main():
    names = sys.argv[1:] or MODULES
    results = {}
    if os.path.exists(RESULTS_JSON):
        with open(RESULTS_JSON) as f:
            results = json.load(f)
    results.setdefault("modules", {})
    results["thresholds"] = THRESHOLDS

    wrappers = build_wrappers()
    for name in names:
        try:
            run_module(name, wrappers, results)
        except Exception:
            err = traceback.format_exc()
            results["modules"][name] = {"error": err, "pass": False}
            print(f"[{name}] FAILED:\n{err}", flush=True)
        with open(RESULTS_JSON, "w") as f:
            json.dump(results, f, indent=2)

    summary = {n: r.get("pass") for n, r in results["modules"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
