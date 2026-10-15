"""M2 acceptance helper: pure-numpy load check of the static embedding package.

Simulates the app-side consumer: no torch, no transformers — just manifest.json +
.npy headers. Validates file presence, md5, shape, dtype, finiteness, and the
expected tag coverage (24 v3 tags + negative + marigold empty embed).

Usage: python 03_load_check.py
Exit code 0 = package loadable and complete.
"""
import hashlib
import json
import os
import sys

import numpy as np

from common import BODY_TAG_LIST, EMB_DIR, HEAD_TAG_LIST


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_tensor(root, meta, label, errors):
    path = os.path.join(root, meta["file"])
    if not os.path.exists(path):
        errors.append(f"{label}: missing {meta['file']}")
        return
    if md5_file(path) != meta["md5"]:
        errors.append(f"{label}: md5 mismatch {meta['file']}")
        return
    arr = np.load(path)
    if list(arr.shape) != meta["shape"]:
        errors.append(f"{label}: shape {list(arr.shape)} != {meta['shape']}")
    if str(arr.dtype) != meta["dtype"]:
        errors.append(f"{label}: dtype {arr.dtype} != {meta['dtype']}")
    if not np.isfinite(arr).all():
        errors.append(f"{label}: non-finite values")


def main():
    errors = []
    mpath = os.path.join(EMB_DIR, "manifest.json")
    with open(mpath) as f:
        manifest = json.load(f)

    ld = manifest["layerdiff"]
    expected_tags = BODY_TAG_LIST + HEAD_TAG_LIST + [ld["negative_prompt"]]
    if set(ld["tensors"].keys()) != set(expected_tags):
        errors.append(f"tag coverage mismatch: {sorted(set(expected_tags) - set(ld['tensors'].keys()))}")
    for tag, entry in ld["tensors"].items():
        for kind in ("prompt_embeds", "pooled_prompt_embeds"):
            check_tensor(EMB_DIR, entry[kind], f"layerdiff[{tag!r}].{kind}", errors)
    check_tensor(EMB_DIR, manifest["marigold"]["empty_text_embed"],
                 "marigold.empty_text_embed", errors)

    # spot-check semantic sanity: distinct tags must not share identical embeddings
    sample = [np.load(os.path.join(EMB_DIR, ld["tensors"][t]["prompt_embeds"]["file"]))
              for t in ("front hair", "face", "objects")]
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if np.array_equal(sample[i], sample[j]):
                errors.append("distinct tags share identical prompt_embeds")

    if errors:
        print("LOAD CHECK FAILED:")
        for e in errors:
            print(" -", e)
        sys.exit(1)
    n = len(expected_tags) * 2 + 1
    print(f"LOAD CHECK OK: {n} tensors verified "
          f"({len(BODY_TAG_LIST)} body + {len(HEAD_TAG_LIST)} head tags, negative, marigold empty)")


if __name__ == "__main__":
    main()
