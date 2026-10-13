"""Spike B: minimal-repro probes for the risky structures.

  sdpa     - bare F.scaled_dot_product_attention module -> trace -> ct.convert
  group    - GroupEmbedding -> trace -> ct.convert
  t3d      - one real Transformer3DModel (pulled from the loaded UNet, mid block attn)
             -> trace -> ct.convert  (exercises frame reshape + temporal block + SDPA)

Each probe prints OK/FAIL + error tail and appends a line to min_repro_results.log.
"""
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = "/Users/jup33q/see-through"
sys.path.insert(0, os.path.join(REPO, "common"))
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "min_repro_results.log")


def record(name, ok, detail):
    line = f"=== {name}: {'OK' if ok else 'FAIL'} @ {time.strftime('%H:%M:%S')}\n{detail}\n"
    with open(LOG, "a") as f:
        f.write(line)
    print(line[-2000:])


class SDPAProbe(nn.Module):
    def __init__(self, heads=20, dim_head=64):
        super().__init__()
        self.h, self.d = heads, dim_head
        self.q = nn.Linear(heads * dim_head, heads * dim_head)
        self.k = nn.Linear(heads * dim_head, heads * dim_head)
        self.v = nn.Linear(heads * dim_head, heads * dim_head)
        self.o = nn.Linear(heads * dim_head, heads * dim_head)

    def forward(self, x):
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.h, self.d).transpose(1, 2)
        k = self.k(x).reshape(b, n, self.h, self.d).transpose(1, 2)
        v = self.v(x).reshape(b, n, self.h, self.d).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, n, c)
        return self.o(o)


def try_convert(traced, shapes, name, dtypes=None):
    import coremltools as ct

    dtypes = dtypes or [np.float32] * len(shapes)
    inputs = [ct.TensorType(shape=s, dtype=d) for s, d in zip(shapes, dtypes)]
    return ct.convert(
        traced, inputs=inputs, compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL, minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )


def probe_sdpa():
    name = "sdpa"
    try:
        m = SDPAProbe().eval()
        x = torch.randn(2, 1024, 1280)
        with torch.inference_mode():
            ref = m(x).numpy()
            traced = torch.jit.trace(m, (x,))
        ml = try_convert(traced, [x.shape], name)
        pred = ml.predict({ml.input_description._fd_spec[0].name if hasattr(ml.input_description, '_fd_spec') else list(ml.input_description)[0]: x.numpy()})
        out = list(pred.values())[0]
        d = np.abs(out - ref)
        record(name, True, f"trace+convert+predict ok; max_abs={d.max():.3e} mean_abs={d.mean():.3e}")
    except Exception:
        record(name, False, traceback.format_exc())


def probe_group():
    name = "group_embedding"
    try:
        from modules.layerdiffuse.layerdiff3d import GroupEmbedding

        ge = GroupEmbedding(13, 2048).eval()
        x3 = torch.randn(13, 77, 2048)
        x2 = torch.randn(13, 1280) if False else None
        with torch.inference_mode():
            ref = ge(x3).numpy()
            traced = torch.jit.trace(ge, (x3,))
        ml = try_convert(traced, [x3.shape], name)
        pred = ml.predict({list(ml.input_description)[0]: x3.numpy()})
        out = list(pred.values())[0]
        d = np.abs(out - ref)
        record(name, True, f"3D-input trace+convert+predict ok; max_abs={d.max():.3e}")
    except Exception:
        record(name, False, traceback.format_exc())


def probe_t3d():
    name = "transformer3d_midblock_attn0"
    try:
        from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel

        unet = UNetFrameConditionModel.from_pretrained(
            "layerdifforg/seethroughv0.0.2_layerdiff3d", subfolder="unet", torch_dtype=torch.float32
        )
        unet.eval()
        t3d = unet.mid_block.attentions[0].eval()

        class W(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, hidden_states, encoder_hidden_states):
                return self.m(hidden_states, encoder_hidden_states=encoder_hidden_states,
                              return_dict=False, num_frames=13)[0]

        w = W(t3d)
        # mid block: 13 frames * (1280ch, 40x40)
        hs = torch.randn(13, 1280, 40, 40)
        ehs = torch.randn(13, 77, 2048)
        with torch.inference_mode():
            ref = w(hs, ehs).numpy()
            t0 = time.time()
            traced = torch.jit.trace(w, (hs, ehs))
            trace_t = time.time() - t0
        t0 = time.time()
        ml = try_convert(traced, [hs.shape, ehs.shape], name)
        conv_t = time.time() - t0
        names = list(ml.input_description)
        pred = ml.predict({names[0]: hs.numpy(), names[1]: ehs.numpy()})
        out = list(pred.values())[0]
        d = np.abs(out - ref)
        record(name, True, f"trace {trace_t:.1f}s, convert {conv_t:.1f}s; max_abs={d.max():.3e} mean_abs={d.mean():.3e}")
    except Exception:
        record(name, False, traceback.format_exc())


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("sdpa", "all"):
        probe_sdpa()
    if which in ("group", "all"):
        probe_group()
    if which in ("t3d", "all"):
        probe_t3d()
