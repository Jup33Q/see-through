#!/usr/bin/env python3
"""Build a small upstream oracle that exercises eye and left/right splitting."""

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image

from common.utils.inference_utils import further_extr


def rgba_canvas(size=128):
    yy, xx = np.mgrid[:size, :size]
    out = np.zeros((size, size, 4), dtype=np.uint8)
    out[..., 0] = xx
    out[..., 1] = yy
    out[..., 2] = (xx + yy) % 256
    out[..., 3] = 255
    return out


def add_component(image, depth, box, value):
    x, y, w, h = box
    image[y:y + h, x:x + w, 3] = 255
    depth[y:y + h, x:x + w] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="migration/m5/out/split_fixture")
    args = parser.parse_args()
    root = Path(args.root)
    if root.exists():
        shutil.rmtree(root)
    src = root / "input"
    src.mkdir(parents=True)

    Image.fromarray(rgba_canvas()).save(src / "src_img.png")
    specs = {
        "eyes": [
            ((20, 20, 8, 8), 40),
            ((80, 20, 10, 6), 80),
            ((20, 50, 6, 6), 120),
            ((80, 50, 4, 4), 160),
        ],
        "handwear": [
            ((10, 95, 12, 8), 60),
            ((90, 95, 10, 10), 100),
        ],
    }
    for tag, components in specs.items():
        image = np.zeros((128, 128, 4), dtype=np.uint8)
        image[..., 0] = 90
        image[..., 1] = 140
        image[..., 2] = 190
        depth = np.full((128, 128), 255, dtype=np.uint8)
        for box, value in components:
            add_component(image, depth, box, value)
        Image.fromarray(image).save(src / f"{tag}.png")
        Image.fromarray(depth, mode="L").save(src / f"{tag}_depth.png")

    (src / "info.json").write_text(
        json.dumps({"parts": {"eyes": {}, "handwear": {}}}), encoding="utf-8"
    )
    further_extr(str(src), rotate=False, save_to_psd=True, tblr_split=True)
    print(f"fixture input: {src}")
    print(f"upstream oracle: {root / 'input.psd'}")
    print(f"run: ldrunner m5-post --src {src} --psd {root / 'swift.psd'} --tblr-split true")


if __name__ == "__main__":
    main()
