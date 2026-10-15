"""M3-1: merged single-package CoreML conversion of UNetFrameConditionModel.

M1 Spike B converted body13/head11 as two 7.6 GB packages whose weights are 99%
identical — the only difference is the Python-level `group_index` branch
(layerdiff3d.py:1628 group_embeds2 on encoder_hidden_states, :1689 group_embeds
on text_embeds). Both group applications act on *static* inputs (prompt
embeddings are fixed per tag set, M2), so this script pre-applies them and
converts a single model with `config.group_embedding=False`:

  wrapper inputs:  sample (1,F,8,160,160), timestep (1,),
                   encoder_hidden_states (F,77,2048)  [pre group2-augmented],
                   text_embeds (F,1280)               [pre group-augmented],
                   time_ids (F,6)
  F enumerated over {13 (body, group 0), 11 (head, group 1)} -> one mlpackage.

Pipeline glue pre-computation (ehs + group_embeds2[g](ehs), te + group_embeds[g](te))
is pure static math; this script emits those tensors to out/statics/{body,head}/
from the M2 embedding package + the UNet's group weights, so the app needs no
group-embedding code at all.

Checkpoints:
  0. eager equivalence: wrapper(augmented spike-B inputs) == spike-B ref outputs
  1. torch.jit.trace (F=13 example)
  2. ct.convert fp16 mlprogram with EnumeratedShapes
  3. load + predict both shapes vs spike-B refs (PSNR) + bench + compute plan

Run: python 02_convert_unet_merged.py   (conda env see_through; tens of minutes)
"""
import json
import os
import subprocess
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

LAYERDIFF_REPO = "layerdifforg/seethroughv0.0.2_layerdiff3d"
SPIKE_B = os.path.join(REPO, "spikes", "m1", "spike_b_unet")
EMB = os.path.join(REPO, "migration", "m2", "out", "embeddings", "layerdiff")
M3 = os.path.join(REPO, "migration", "m3")
OUT_MODELS = os.path.join(M3, "out", "models")
OUT_STATICS = os.path.join(M3, "out", "statics")
LOG = os.path.join(M3, "out", "logs", "convert_unet_merged.log")
RESULTS = os.path.join(M3, "out", "convert_unet_merged_results.json")
PKG = os.path.join(OUT_MODELS, "layerdiff3d_unet_merged_fp16.mlpackage")

BODY_TAG_LIST = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear',
                 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
HEAD_TAG_LIST = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear',
                 'ears', 'earwear', 'nose', 'mouth']
TIME_IDS = [1280, 1280, 0, 0, 1280, 1280]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def rss_gb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


class UNetMergedWrapper(nn.Module):
    """UNetFrameConditionModel with group_embedding disabled; callers feed
    pre-augmented encoder_hidden_states / text_embeds (static per pass)."""

    def __init__(self, unet):
        super().__init__()
        unet.config.group_embedding = False
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        out = self.unet(
            sample, timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            return_dict=False,
        )
        return out[0]


def load_tag_embeds(tags):
    ehs, pooled = [], []
    for t in tags:
        key = t.replace(" ", "_")
        ehs.append(np.load(os.path.join(EMB, f"{key}.prompt_embeds.npy")))
        pooled.append(np.load(os.path.join(EMB, f"{key}.pooled_prompt_embeds.npy")))
    return (torch.from_numpy(np.concatenate(ehs, 0)),
            torch.from_numpy(np.concatenate(pooled, 0)))


def compute_statics(unet):
    """ehs_cond = ehs + group_embeds2[g](ehs); text_aug = te + group_embeds[g](te)."""
    statics = {}
    with torch.no_grad():
        for g, name, tags in ((0, "body", BODY_TAG_LIST), (1, "head", HEAD_TAG_LIST)):
            ehs, pooled = load_tag_embeds(tags)  # fp32 (exact upcast of bf16 compute)
            ehs_cond = ehs + unet.group_embeds2[g](ehs)
            text_aug = pooled + unet.group_embeds[g](pooled)
            time_ids = torch.tensor([TIME_IDS] * ehs.shape[0], dtype=torch.float32)
            d = os.path.join(OUT_STATICS, name)
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "encoder_hidden_states.npy"), ehs_cond.numpy())
            np.save(os.path.join(d, "text_embeds.npy"), text_aug.numpy())
            np.save(os.path.join(d, "time_ids.npy"), time_ids.numpy())
            statics[name] = (ehs_cond, text_aug, time_ids)
            log(f"statics[{name}]: ehs_cond{tuple(ehs_cond.shape)} "
                f"text_aug{tuple(text_aug.shape)}")
    return statics


def psnr_vs(ref, out):
    mse = float(((ref.astype(np.float64) - out.astype(np.float64)) ** 2).mean())
    peak = float(np.abs(ref).max())
    return float("inf") if mse == 0 else 10 * np.log10(peak * peak / mse)


def spike_b_feed(unet, tag, group):
    """Spike B ref inputs (synthetic, seed 1234) with the same group augmentation
    the original UNet applies internally for `group_index=group`."""
    d = np.load(os.path.join(SPIKE_B, f"ref_in_{tag}.npz"))
    ehs = torch.from_numpy(d["encoder_hidden_states"])
    te = torch.from_numpy(d["text_embeds"])
    with torch.no_grad():
        ehs_cond = ehs + unet.group_embeds2[group](ehs)
        text_aug = te + unet.group_embeds[group](te)
    return {
        "sample": torch.from_numpy(d["sample"]),
        "timestep": torch.from_numpy(np.atleast_1d(d["timestep"])),
        "encoder_hidden_states": ehs_cond,
        "text_embeds": text_aug,
        "time_ids": torch.from_numpy(d["time_ids"]),
    }, np.load(os.path.join(SPIKE_B, f"ref_out_{tag}.npy"))


def patch_coremltools_symbolic_group_norm():
    """coremltools 9.0 `_group_norm_impl` builds [n, groups, c/groups, *spatial] as a
    python list; with a symbolic batch n (our flexible frames dim) mb.reshape rejects
    the mixed list ("Cannot add const [is0, np.int32(32), ...]"). Rebuild the shape
    via mb.concat so symbolic batch works. Same ops after the reshape — numerics
    unchanged."""
    from coremltools.converters.mil.frontend.torch import ops as _tops
    from coremltools.converters.mil.mil import Builder as _mb
    from coremltools.converters.mil.mil.types.symbolic import any_symbolic as _any
    import numpy as _np

    def impl(x, num_groups, weight, bias, eps):
        c = x.shape[1]
        num_groups = min(num_groups, c)
        cpg = c // num_groups
        if not _any(x.shape):
            new_shape = _np.array([x.shape[0], num_groups, cpg] +
                                  [int(d) for d in x.shape[2:]], dtype=_np.int32)
            restore_shape = _np.array([int(d) for d in x.shape], dtype=_np.int32)
        else:
            st = _mb.shape(x=x)
            n_v = _mb.slice_by_size(x=st, begin=[0], size=[1])
            g_v = _mb.const(val=_np.array([num_groups, cpg], dtype=_np.int32))
            head = _mb.concat(values=[n_v, g_v], axis=0)
            if not _any(x.shape[2:]):
                tail = _mb.const(val=_np.array([int(d) for d in x.shape[2:]],
                                               dtype=_np.int32))
            else:
                tail = _mb.slice_by_size(x=st, begin=[2], size=[-1])
            new_shape = _mb.concat(values=[head, tail], axis=0)
            restore_shape = st

        num_extra_axes = len(x.shape[2:])
        axes_ = [int(i) for i in range(2, 2 + num_extra_axes + 1)]
        weight_shape = [1, c] + [1] * num_extra_axes
        bias_shape = [1, c] + [1] * num_extra_axes

        x = _mb.reshape(x=x, shape=new_shape)
        mean = _mb.reduce_mean(x=x, axes=axes_, keep_dims=True)
        var = _tops._var(x, axes=axes_, keep_dims=True, unbiased=False)
        x = _mb.real_div(x=_mb.sub(x=x, y=mean), y=_mb.sqrt(x=_mb.add(x=var, y=eps)))
        x = _mb.reshape(x=x, shape=restore_shape)
        if weight is not None:
            x = _mb.mul(x=x, y=_mb.reshape(x=weight, shape=weight_shape))
        if bias is not None:
            x = _mb.add(x=x, y=_mb.reshape(x=bias, shape=bias_shape))
        return x

    _tops._group_norm_impl = impl
    log("patched coremltools _group_norm_impl for symbolic batch")


def fix_repeat_interleave(traced):
    """Graph surgery on traced.unet.forward.graph: the forward does
    `timestep.repeat_interleave(num_frames)` (layerdiff3d.py:1673) where num_frames
    is symbolic under EnumeratedShapes, and coremltools' repeat_interleave requires
    constant repeats ("Cannot add const [None]"). Replace with
    aten::expand(timestep, [Int(repeats)]) — value-identical (timestep is a single
    element broadcast to F frames) and symbolic-safe. Verified on a minimal repro:
    EnumeratedShapes convert + predict both frame counts, torch output bit-exact
    pre/post surgery."""
    g = traced.unet.forward.graph
    targets = [n for n in g.nodes() if n.kind() == "aten::repeat_interleave"]
    assert len(targets) == 1, f"expected 1 repeat_interleave, got {len(targets)}"
    target = targets[0]
    timestep_v, repeats_v = list(target.inputs())[0], list(target.inputs())[1]
    int_node = g.create("aten::Int", [repeats_v], 1).insertBefore(target)
    list_node = g.create("prim::ListConstruct", [int_node.output()], 1).insertBefore(target)
    list_node.output().setType(torch.ListType.ofInts())
    false_v = g.insertConstant(False)
    false_v.node().moveBefore(target)
    ex_node = g.create("aten::expand", [timestep_v, list_node.output(), false_v], 1)
    ex_node.insertBefore(target)
    ex_node.output().setType(target.output().type())
    target.output().replaceAllUsesWith(ex_node.output())
    target.destroy()
    g.lint()
    log("graph surgery: repeat_interleave -> expand (symbolic frames) OK")


def main():
    from_traced = "--from-traced" in sys.argv
    for d in (OUT_MODELS, OUT_STATICS, os.path.dirname(LOG)):
        os.makedirs(d, exist_ok=True)
    if os.path.exists(LOG) and not from_traced:
        os.remove(LOG)
    results = {"package": PKG}
    try:
        t0 = time.time()
        unet = UNetFrameConditionModel.from_pretrained(
            LAYERDIFF_REPO, subfolder="unet", torch_dtype=torch.float32)
        unet.eval().to("cpu")
        log(f"unet loaded in {time.time()-t0:.1f}s rss={rss_gb():.1f}GB")

        statics = compute_statics(unet)
        wrapper = UNetMergedWrapper(unet).eval()
        # augmented spike-B feeds, computed once (unet is deleted before checkpoint 3)
        feeds = {"body": spike_b_feed(unet, "body13_160", 0),
                 "head": spike_b_feed(unet, "head11_160", 1)}

        # ---- checkpoint 0: eager equivalence vs spike-B refs (both groups) ----
        if not from_traced:
            for name in ("body", "head"):
                feed, ref = feeds[name]
                with torch.inference_mode():
                    t0 = time.time()
                    out = wrapper(**feed).numpy()
                d = np.abs(out - ref)
                results.setdefault("eager_equivalence", {})[name] = {
                    "max_abs": float(d.max()), "psnr_db": psnr_vs(ref, out),
                    "forward_s": round(time.time() - t0, 2)}
                log(f"eager[{name}]: max_abs={d.max():.3e} psnr={results['eager_equivalence'][name]['psnr_db']:.2f}dB "
                    f"({time.time()-t0:.1f}s cpu)")
                assert d.max() < 1e-4, "wrapper semantics diverge from spike-B reference"

        # ---- checkpoint 1: trace (F=13 example) ----
        traced_path = os.path.join(M3, "out", "traced_unet_merged.pt")
        if from_traced and os.path.exists(traced_path):
            traced = torch.jit.load(traced_path)
            log(f"traced graph reloaded from {traced_path}")
        else:
            ex = feeds["body"][0]
            example = (ex["sample"], ex["timestep"], ex["encoder_hidden_states"],
                       ex["text_embeds"], ex["time_ids"])
            with torch.inference_mode():
                t0 = time.time()
                traced = torch.jit.trace(wrapper, example, strict=True)
            log(f"TRACE OK in {time.time()-t0:.1f}s rss={rss_gb():.1f}GB")
            results["trace_s"] = round(time.time() - t0, 2)
            torch.jit.save(traced, traced_path)
            log(f"traced graph saved: {traced_path}")
        fix_repeat_interleave(traced)
        del wrapper, unet
        import gc
        gc.collect()

        # ---- checkpoint 2: convert, EnumeratedShapes over frames ----
        import coremltools as ct
        patch_coremltools_symbolic_group_norm()
        log(f"coremltools {ct.__version__}")
        frames_shapes = [("sample", [(1, 13, 8, 160, 160), (1, 11, 8, 160, 160)]),
                         ("encoder_hidden_states", [(13, 77, 2048), (11, 77, 2048)]),
                         ("text_embeds", [(13, 1280), (11, 1280)]),
                         ("time_ids", [(13, 6), (11, 6)])]
        inputs = [ct.TensorType(name="sample",
                                shape=ct.EnumeratedShapes(shapes=frames_shapes[0][1]),
                                dtype=np.float32),
                  ct.TensorType(name="timestep", shape=(1,), dtype=np.float32)]
        for name, shapes in frames_shapes[1:]:
            inputs.append(ct.TensorType(
                name=name, shape=ct.EnumeratedShapes(shapes=shapes), dtype=np.float32))
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

        # ---- checkpoint 3: load + predict both shapes + bench + compute plan ----
        gc.collect()
        model = ct.models.MLModel(PKG, compute_units=ct.ComputeUnit.ALL)
        log(f"mlpackage loaded rss={rss_gb():.1f}GB")
        for name in ("body", "head"):
            feed, ref = feeds[name]
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
            results.setdefault("coreml", {})[name] = {
                "psnr_db": p, "max_abs": float(np.abs(pred - ref).max()),
                "predict_first_s": round(first, 2),
                "predict_times_s": [round(t, 2) for t in times]}
            log(f"coreml[{name}]: psnr={p:.2f}dB first={first:.1f}s "
                f"next3={[f'{t:.1f}' for t in times]}s rss={rss_gb():.1f}GB")
        results["compute_plan"] = compute_plan_device_stats(PKG)
        log(f"compute plan: {results['compute_plan']}")
        results["pass"] = all(v["psnr_db"] >= 60 for v in results["coreml"].values())
    except Exception:
        results["error"] = traceback.format_exc()
        results["pass"] = False
        log("FAILED:\n" + results["error"])
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    log(f"done pass={results['pass']}")


if __name__ == "__main__":
    main()
