"""Paired Full/Person inputs with crop-aware geometry and isolated RNG.

The Full pixels and RNG consumption match the existing Track-A transforms.
Person uses the original target crop, letterboxing, and the SAME flip as Full.
Its optional color jitter runs in a forked RNG so it cannot move the Full
augmentation/shuffle stream. Geometry describes the unexpanded target bbox
inside the actual Full crop, never an assumed 224x224 coordinate system.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class PairedFullPersonTransform:
    def __init__(self, train: bool, normalization: str = "clip",
                 crop_scale=(0.05, 1.0), margin: float = 0.15,
                 jitter_strength: float = 0.1, jitter_probability: float = 0.2):
        # Lazy import avoids a runner import cycle.
        from .runner import build_transforms, PadToSquare, CLIP_IMAGE_MEAN
        if not math.isfinite(margin) or not 0 <= margin <= 1:
            raise ValueError("Person margin must be finite and in [0, 1]")
        full_train, full_eval = build_transforms(normalization, crop_scale)
        _, person_eval = build_transforms(
            normalization, crop_scale, "person_crop", "letterbox",
            jitter_strength, jitter_probability,
        )
        self.train = train
        self.margin = margin
        self.crop_scale = tuple(crop_scale)
        self.full_eval = full_eval
        self.person_eval = person_eval
        self.to_tensor = transforms.Compose(full_train.transforms[2:])
        fill = tuple(round(x * 255) for x in CLIP_IMAGE_MEAN) if normalization == "clip" else 0
        self.pad = PadToSquare(fill)
        self.jitter = transforms.RandomApply([
            transforms.ColorJitter(jitter_strength, jitter_strength,
                                   jitter_strength, jitter_strength * 0.2)
        ], p=jitter_probability)
        self.use_jitter = bool(jitter_strength and jitter_probability)

    @staticmethod
    def target_box(bbox, width: int, height: int):
        try:
            box = np.asarray(bbox, dtype=np.float64).ravel()[:4]
        except (TypeError, ValueError):
            return None
        if len(box) != 4 or not np.isfinite(box).all():
            return None
        x1, y1, x2, y2 = box
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            return None
        return float(x1), float(y1), float(x2), float(y2)

    @staticmethod
    def geometry(box, crop, flip: bool) -> torch.Tensor:
        """[clipped x1,y1,x2,y2, visible fraction, bbox-valid], normalized."""
        if box is None:
            return torch.zeros(6)
        top, left, height, width = crop
        x1, y1, x2, y2 = box
        a, c = np.clip([x1 - left, x2 - left], 0, width)
        b, d = np.clip([y1 - top, y2 - top], 0, height)
        visible = (c - a) * (d - b) / ((x2 - x1) * (y2 - y1))
        a, c, b, d = a / width, c / width, b / height, d / height
        if flip:
            a, c = 1 - c, 1 - a
        return torch.tensor([a, b, c, d, visible, 1.0], dtype=torch.float32)

    def __call__(self, image, bbox):
        width, height = image.size
        box = self.target_box(bbox, width, height)
        if self.train:
            crop = transforms.RandomResizedCrop.get_params(
                image, self.crop_scale, (3 / 4, 4 / 3)
            )
            full = TF.resized_crop(image, *crop, (224, 224),
                                   transforms.InterpolationMode.BILINEAR)
            flip = bool(torch.rand(1) < 0.5)
            if flip:
                full = TF.hflip(full)
            full = self.to_tensor(full)
            geometry = self.geometry(box, crop, flip)
        else:
            # Match Resize(256, bicubic) -> CenterCrop(224) exactly, including
            # torchvision's rounding for odd sizes. Geometry follows that path.
            resized = self.full_eval.transforms[0](image)
            rw, rh = resized.size
            left, top = round((rw - 224) / 2), round((rh - 224) / 2)
            scaled_box = (None if box is None else
                          (box[0] * rw / width, box[1] * rh / height,
                           box[2] * rw / width, box[3] * rh / height))
            geometry = self.geometry(scaled_box, (top, left, 224, 224), False)
            full = self.to_tensor(TF.center_crop(resized, (224, 224)))
            flip = False

        # Reuse the legacy crop implementation, including its exact rounding
        # and margin behavior. Invalid boxes fall back to Full but are masked.
        from multi_lane.continual_datasets.continual_datasets import EMOTIC
        person = EMOTIC.crop_person(image, bbox, self.margin)
        if self.train:
            person = TF.resize(self.pad(person), (224, 224),
                               transforms.InterpolationMode.BICUBIC)
            if flip:
                person = TF.hflip(person)
            if self.use_jitter:
                with torch.random.fork_rng(devices=[]):
                    person = self.jitter(person)
            person = self.to_tensor(person)
        else:
            person = self.person_eval(person)
        return {"full": full, "person": person, "bbox": geometry,
                "condition_valid": (geometry[4] > 0).float() * geometry[5]}


def move_model_inputs(images, device):
    if isinstance(images, dict):
        return {key: value.to(device, non_blocking=True).float()
                for key, value in images.items()}
    return images.to(device, non_blocking=True).float()
