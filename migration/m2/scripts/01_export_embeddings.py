"""M2-1: export the static embedding package.

Runs the REAL upstream encode paths once (PyTorch, bf16 on MPS — same dtype/device
as the M0 golden run) and freezes the results as .npy files + manifest.json:

- LayerDiff (v3): 24 tags (body 13 + head 11) x {prompt_embeds, pooled_prompt_embeds}
  via KDiffusionStableDiffusionXLPipeline.encode_cropped_prompt_77tokens
  (diffusers_kdiffusion_sdxl.py:183-214), plus the negative prompt '' embeddings.
- Marigold: empty-prompt embedding via MarigoldDepthPipeline.encode_empty_text
  (marigold_depth_pipeline.py:516-529) using the seethrough marigold snapshot's own
  text_encoder/tokenizer (NOT the prs-eth fallback, which needs network).
- Derived quantities (add_time_ids etc.) are deterministic from resolution and are
  recorded in the manifest, not stored as tensors.

The pipeline classes are used through their real methods (uninitialized-instance
harness) so the exported values are bit-identical to what `cache_tag_embeds()` would
cache at inference time. Stored as fp32 (exact upcast from bf16 compute).

Usage: python 01_export_embeddings.py
"""
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from common import (BODY_TAG_LIST, EMB_DIR, HEAD_TAG_LIST, LAYERDIFF_SNAP,
                    MARIGOLD_SNAP, NEGATIVE_PROMPT)

DEVICE = "mps"
DTYPE = torch.bfloat16  # M0 golden run dtype


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_tensor(arr_f32, rel_path):
    path = os.path.join(EMB_DIR, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.ascontiguousarray(arr_f32.astype(np.float32))
    np.save(path, arr)
    return {"file": rel_path, "shape": list(arr.shape), "dtype": str(arr.dtype),
            "md5": md5_file(path)}


def encode_layerdiff_tags(tags):
    """Call the real pipeline method on an uninitialized instance."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from modules.layerdiffuse.diffusers_kdiffusion_sdxl import (
        KDiffusionStableDiffusionXLPipeline as Pipe,
    )

    pipe = Pipe.__new__(Pipe)
    pipe.tokenizer = CLIPTokenizer.from_pretrained(LAYERDIFF_SNAP, subfolder="tokenizer")
    pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(LAYERDIFF_SNAP, subfolder="tokenizer_2")
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        LAYERDIFF_SNAP, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE).eval()
    pipe.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        LAYERDIFF_SNAP, subfolder="text_encoder_2", torch_dtype=DTYPE).to(DEVICE).eval()
    pipe.unet = SimpleNamespace(dtype=DTYPE)  # only .dtype is read by the encode path
    pipe._cached_prompt_embeds = {}

    prompt_embeds, pooled_prompt_embeds = Pipe.encode_cropped_prompt_77tokens(pipe, tags)
    return prompt_embeds.cpu().float().numpy(), pooled_prompt_embeds.cpu().float().numpy()


def encode_marigold_empty():
    from transformers import CLIPTextModel, CLIPTokenizer
    from modules.marigold.marigold_depth_pipeline import MarigoldDepthPipeline as MP

    # DiffusionPipeline.dtype is a read-only property derived from registered
    # modules; the uninitialized-instance harness has none, so override it.
    class _MP(MP):
        @property
        def dtype(self):
            return DTYPE

    mp = _MP.__new__(_MP)
    from diffusers.configuration_utils import FrozenDict
    mp._internal_dict = FrozenDict({})  # DiffusionPipeline.__setattr__ touches self.config
    mp.tokenizer = CLIPTokenizer.from_pretrained(MARIGOLD_SNAP, subfolder="tokenizer")
    mp.text_encoder = CLIPTextModel.from_pretrained(
        MARIGOLD_SNAP, subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE).eval()
    mp.empty_text_embed = None
    MP.encode_empty_text(mp)
    return mp.empty_text_embed.detach().cpu().float().numpy()


def main():
    manifest = {
        "format": "manifest.json + one .npy per tensor (npy headers carry shape/dtype)",
        "stage": "M2 staticization",
        "source": {
            "layerdiff_snapshot": LAYERDIFF_SNAP,
            "marigold_snapshot": MARIGOLD_SNAP,
            "compute_dtype": str(DTYPE),
            "compute_device": DEVICE,
            "storage_dtype": "float32 (exact upcast from bf16 compute; cast to app dtype at load)",
            "encode_path": "KDiffusionStableDiffusionXLPipeline.encode_cropped_prompt_77tokens "
                           "(diffusers_kdiffusion_sdxl.py:183-214) / "
                           "MarigoldDepthPipeline.encode_empty_text (marigold_depth_pipeline.py:516)",
        },
        "layerdiff": {
            "tag_version": "v3",
            "body_tag_list": BODY_TAG_LIST,
            "head_tag_list": HEAD_TAG_LIST,
            "negative_prompt": NEGATIVE_PROMPT,
            "derived": {
                "add_time_ids": [1280, 1280, 0, 0, 1280, 1280],
                "add_neg_time_ids": [1280, 1280, 0, 0, 1280, 1280],
                "resolution": 1280,
                "guidance_scale": 1.0,
                "num_inference_steps": 30,
                "note": "add_time_ids = (h, w, 0, 0, h, w), deterministic from resolution; "
                        "guidance_scale=1.0 disables CFG in the M0 golden run "
                        "(do_classifier_free_guidance = scale > 1), negative embeddings are "
                        "exported for M3+ flexibility.",
            },
            "tensors": {},
        },
        "marigold": {},
    }

    all_tags = BODY_TAG_LIST + HEAD_TAG_LIST
    print(f"[layerdiff] encoding {len(all_tags)} tags + negative ...", flush=True)
    pe, ppe = encode_layerdiff_tags(all_tags + [NEGATIVE_PROMPT])
    for i, tag in enumerate(all_tags + [NEGATIVE_PROMPT]):
        slug = tag.replace(" ", "_") if tag else "__negative__"
        manifest["layerdiff"]["tensors"][tag] = {
            "prompt_embeds": save_tensor(pe[i:i + 1], f"layerdiff/{slug}.prompt_embeds.npy"),
            "pooled_prompt_embeds": save_tensor(ppe[i:i + 1], f"layerdiff/{slug}.pooled_prompt_embeds.npy"),
        }
        print(f"  {tag or '<negative>'}: embeds{list(pe[i:i+1].shape)} pooled{list(ppe[i:i+1].shape)}",
              flush=True)

    print("[marigold] encoding empty prompt ...", flush=True)
    emb = encode_marigold_empty()
    manifest["marigold"] = {
        "empty_text_embed": save_tensor(emb, "marigold/empty_text_embed.npy"),
        "resolution": 768,
        "scheduler": "DDIM, 4 steps",
        "note": "MarigoldDepthPipeline.encode_empty_text; consumed as the only text condition.",
    }
    print(f"  empty_text_embed: {list(emb.shape)}", flush=True)

    mpath = os.path.join(EMB_DIR, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ---- round-trip verification: reload every file, check md5/shape/dtype ----
    print("[verify] reloading package ...", flush=True)
    n_checked = 0
    for group in (manifest["layerdiff"]["tensors"],):
        for tag, entry in group.items():
            for kind, meta in entry.items():
                arr = np.load(os.path.join(EMB_DIR, meta["file"]))
                assert list(arr.shape) == meta["shape"], (tag, kind, arr.shape)
                assert str(arr.dtype) == meta["dtype"]
                assert md5_file(os.path.join(EMB_DIR, meta["file"])) == meta["md5"]
                assert np.isfinite(arr).all(), (tag, kind)
                n_checked += 1
    meta = manifest["marigold"]["empty_text_embed"]
    arr = np.load(os.path.join(EMB_DIR, meta["file"]))
    assert list(arr.shape) == meta["shape"] and str(arr.dtype) == meta["dtype"]
    assert md5_file(os.path.join(EMB_DIR, meta["file"])) == meta["md5"]
    assert np.isfinite(arr).all()
    n_checked += 1
    print(f"[verify] OK: {n_checked} tensors, manifest={mpath}", flush=True)


if __name__ == "__main__":
    main()
