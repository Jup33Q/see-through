#!/usr/bin/env python3
"""Read a native Swift PSD with psd-tools and compare it with an upstream PSD."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from psd_tools import PSDImage


def pixels(layer, channel=None):
    value = layer.numpy(channel=channel)
    if value is None:
        return None
    return np.rint(value * 255).astype(np.uint8)


def metrics(a, b):
    if a is None or b is None:
        return {"exact": a is b}
    if a.shape != b.shape:
        return {"exact": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "exact": bool(np.array_equal(a, b)),
        "differing_values": int(np.count_nonzero(diff)),
        "max_abs_u8": int(diff.max()) if diff.size else 0,
        "shape": list(a.shape),
    }


def layer_row(reference, actual):
    rgba = metrics(pixels(reference), pixels(actual))
    mask = metrics(pixels(reference, "mask"), pixels(actual, "mask"))
    ref_channels = [int(c.id) for c in reference._record.channel_info]
    actual_channels = [int(c.id) for c in actual._record.channel_info]
    ref_rle = [int(c.compression) for c in reference._channels]
    actual_rle = [int(c.compression) for c in actual._channels]
    structure = {
        "name": reference.name == actual.name,
        "bbox": tuple(reference.bbox) == tuple(actual.bbox),
        "opacity": reference.opacity == actual.opacity,
        "visible": reference.visible == actual.visible,
        "channel_ids": ref_channels == actual_channels,
        "compression": ref_rle == actual_rle,
    }
    return {
        "name": actual.name,
        "bbox": list(actual.bbox),
        "channel_ids": actual_channels,
        "compression": actual_rle,
        "structure": structure,
        "rgba": rgba,
        "mask": mask,
        "pass": all(structure.values()) and rgba["exact"] and mask["exact"],
    }


def compare(reference, actual, include_rows=True):
    document = {
        "size": reference.size == actual.size,
        "depth": reference.depth == actual.depth,
        "color_mode": reference.color_mode == actual.color_mode,
        "channels": reference.channels == actual.channels,
        "layer_count": len(reference) == len(actual),
    }
    rows = []
    layers_ok = document["layer_count"]
    if layers_ok:
        rows = [layer_row(r, a) for r, a in zip(reference, actual)]
        layers_ok = all(row["pass"] for row in rows)
    result = {
        "document": document,
        "layer_count": len(actual),
        "exact_layers": sum(row["pass"] for row in rows),
        "layers_pass": bool(all(document.values()) and layers_ok),
        "merged": metrics(pixels(reference, channel="color"), pixels(actual, channel="color")),
    }
    if include_rows:
        result["layers"] = rows
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--actual-json")
    parser.add_argument("--reference-json")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    reference = PSDImage.open(args.reference)
    actual = PSDImage.open(args.actual)
    result = compare(reference, actual)
    result["actual"] = str(Path(args.actual).resolve())
    result["reference"] = str(Path(args.reference).resolve())
    result["psd_tools_version"] = __import__("psd_tools").__version__

    json_ok = True
    if args.actual_json or args.reference_json:
        if not (args.actual_json and args.reference_json):
            raise ValueError("both JSON paths are required when comparing sidecars")
        actual_json = json.loads(Path(args.actual_json).read_text(encoding="utf-8"))
        reference_json = json.loads(Path(args.reference_json).read_text(encoding="utf-8"))
        json_ok = actual_json == reference_json
        result["sidecar_json_exact"] = json_ok

    with tempfile.TemporaryDirectory(prefix="m5-psd-roundtrip-") as tmp:
        roundtrip_path = Path(tmp) / "roundtrip.psd"
        actual.save(roundtrip_path)
        roundtrip = PSDImage.open(roundtrip_path)
        roundtrip_result = compare(actual, roundtrip, include_rows=False)
        result["roundtrip"] = roundtrip_result
        result["roundtrip_pass"] = roundtrip_result["layers_pass"]

    result["pass"] = bool(result["layers_pass"] and json_ok and result["roundtrip_pass"])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"layers {result['exact_layers']}/{result['layer_count']} exact; "
        f"sidecar={json_ok}; roundtrip={result['roundtrip_pass']}; "
        f"merged_max={result['merged'].get('max_abs_u8')} -> {'PASS' if result['pass'] else 'FAIL'}"
    )
    print(f"JSON -> {report}")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
