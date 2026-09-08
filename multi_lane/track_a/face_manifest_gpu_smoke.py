"""Real-manifest Face crop to frozen CLIP GPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from torchvision.transforms import functional as transform_functional

from .face_manifest import letterbox_square
from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual
from .runner import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.samples <= 0:
        parser.error("samples must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["valid_face"] and not row["ambiguous_match"] and row["face_crop_bbox"]:
                rows.append(row)
    if len(rows) < args.samples:
        raise RuntimeError("Manifest does not contain enough unambiguous valid faces")
    # Exercise both tiny and large crops rather than selecting only easy inputs.
    rows.sort(key=lambda row: (float(row["face_short_side"]), str(row["sample_id"])))
    positions = [round(index * (len(rows) - 1) / max(1, args.samples - 1)) for index in range(args.samples)]
    selected = [rows[position] for position in positions]
    tensors = []
    for row in selected:
        image = Image.open(row["image_absolute_path"]).convert("RGB")
        crop = image.crop(tuple(row["face_crop_bbox"]))
        crop = letterbox_square(crop, size=224)
        tensor = transform_functional.pil_to_tensor(crop).float().div_(255.0)
        tensor = transform_functional.normalize(tensor, CLIP_IMAGE_MEAN, CLIP_IMAGE_STD)
        if tensor.shape != (3, 224, 224) or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid Face tensor for {row['sample_id']}")
        tensors.append(tensor)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Face GPU smoke requires CUDA")
    visual = load_openai_clip_visual(args.clip_checkpoint).to(device).eval()
    batch = torch.stack(tensors).to(device)
    with torch.inference_mode():
        features = visual(batch)
    if features.ndim != 2 or features.shape[0] != args.samples or not torch.isfinite(features).all():
        raise RuntimeError(f"Invalid CLIP Face features: shape={tuple(features.shape)}")
    result = {
        "status": "passed",
        "manifest": str(args.manifest.resolve()),
        "clip_checkpoint": str(args.clip_checkpoint.resolve()),
        "clip_checkpoint_sha256": OPENAI_VIT_B16_SHA256,
        "device": str(device),
        "samples": [row["sample_id"] for row in selected],
        "face_short_sides": [float(row["face_short_side"]) for row in selected],
        "input_shape": list(batch.shape),
        "feature_shape": list(features.shape),
        "input_finite": True,
        "feature_finite": True,
        "training_started": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"FACE_MANIFEST_GPU_SMOKE_PASSED output={args.output}", flush=True)


if __name__ == "__main__":
    main()
