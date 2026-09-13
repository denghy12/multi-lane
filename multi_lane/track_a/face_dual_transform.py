"""Shared-geometry CLIP/AffectNet transforms for a target Face."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .face_alignment import align_face_five_points
from .runner import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD, PadToSquare


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _margin_crop(
    image: Image.Image, record: Mapping[str, object], margin: float,
) -> Image.Image:
    values = np.asarray(record.get("face_bbox"), dtype=np.float64).ravel()
    if values.shape != (4,) or not np.isfinite(values).all():
        raise RuntimeError("Dual Face transform received an invalid bbox")
    x1, y1, x2, y2 = values.tolist()
    width, height = image.size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise RuntimeError("Dual Face bbox is outside the source image")
    box_width, box_height = x2 - x1, y2 - y1
    box = (
        int(np.floor(max(0.0, x1 - margin * box_width))),
        int(np.floor(max(0.0, y1 - margin * box_height))),
        int(np.ceil(min(float(width), x2 + margin * box_width))),
        int(np.ceil(min(float(height), y2 + margin * box_height))),
    )
    return image.crop(box)


class DualFaceTransform:
    def __init__(self, train: bool, margin: float = 0.15) -> None:
        if margin != 0.15:
            raise ValueError("Expression residual validation locks Face margin0.15")
        self.train = bool(train)
        self.margin = float(margin)
        self.clip_transform = transforms.Compose([
            PadToSquare(tuple(int(round(x * 255)) for x in CLIP_IMAGE_MEAN)),
            transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ])
        self.expression_transform = transforms.Compose([
            PadToSquare(tuple(int(round(x * 255)) for x in IMAGENET_MEAN)),
            transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __call__(
        self, image: Image.Image, record: Mapping[str, object], valid: bool,
    ) -> dict:
        if valid:
            clip_face = _margin_crop(image, record, self.margin)
            expression_face = align_face_five_points(
                image, record, size=224, margin=self.margin,
                fill=tuple(int(round(x * 255)) for x in IMAGENET_MEAN),
            )
            if expression_face is None:
                expression_face = clip_face
        else:
            clip_face = Image.new("RGB", (1, 1), (123, 117, 104))
            expression_face = Image.new(
                "RGB", (1, 1),
                tuple(int(round(x * 255)) for x in IMAGENET_MEAN),
            )
        if self.train and bool(torch.rand(()).item() < 0.5):
            clip_face = TF.hflip(clip_face)
            expression_face = TF.hflip(expression_face)
        return {
            "clip": self.clip_transform(clip_face),
            "expression": self.expression_transform(expression_face),
        }
