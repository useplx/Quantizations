"""Compare a compiled tiled Q8 projection against its decoded NumPy reference."""

import argparse
import json
from pathlib import Path
import struct
import subprocess

import numpy as np


def parameters(path, tensors):

    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        data = tensor.tobytes()
        header[name] = {
            "dtype": {"int32": "I32", "float16": "F16"}[str(tensor.dtype)],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("module", type=Path)
    parser.add_argument("--device", default="local-task")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--rounding", choices=("nearest", "toward-zero"), default="nearest")
    args = parser.parse_args()
    prefix = args.module.with_suffix("")
    rng = np.random.default_rng(193)
    codes = rng.integers(0, 256, size=(args.rows, 64, 8, 4), dtype=np.uint32)
    shifts = np.array([0, 8, 16, 24], dtype=np.int32)
    packed = np.bitwise_or.reduce(codes << shifts.astype(np.uint32), axis=-1).view(np.int32)
    scale = np.full((args.rows, 64), 1 / 256, dtype=np.float16)
    zero = np.full((args.rows, 64), 128, dtype=np.float16)
    weights = prefix.with_suffix(".safetensors")
    parameters(weights, {
        "projection.weight.q": packed,
        "projection.weight.shifts": shifts,
        "projection.weight.scale": scale,
        "projection.weight.zero": zero,
    })
    x = (rng.integers(-3, 4, size=(args.batch, 2048)) / 8).astype(np.float16)
    decoded = ((codes.astype(np.float16) - zero[..., None, None]) *
               scale[..., None, None]).reshape(args.rows, 2048)
    accumulated = x.astype(np.float32) @ decoded.astype(np.float32).T
    expected = accumulated.astype(np.float16)
    if args.rounding == "toward-zero":


        away = np.abs(expected.astype(np.float32)) > np.abs(accumulated)
        expected[away] = np.nextafter(expected[away], np.float16(0))
    input_path = prefix.with_suffix(".input.npy")
    output_path = prefix.with_suffix(".output.npy")
    expected_path = prefix.with_suffix(".expected.npy")
    np.save(input_path, x)
    np.save(expected_path, expected)
    if args.prepare_only:
        print(f"Prepared {weights}, {input_path}, {expected_path}")
        return
    if not args.check_only:
        subprocess.run([
            str(args.runner.resolve()), f"--module={args.module.resolve()}",
            f"--device={args.device}", f"--parameters={weights.resolve()}",
            "--function=tiling_projection", f"--input=@{input_path.resolve()}",
            f"--output=@{output_path.resolve()}",
        ], check=True, timeout=30)
    actual = np.load(output_path)
    np.testing.assert_array_equal(actual, expected)
    print(f"{args.device}: all {actual.size} outputs match the decoded reference "
          f"exactly ({args.rounding} FP16 conversion)")


if __name__ == "__main__":
    main()
