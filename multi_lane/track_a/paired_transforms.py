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
                 jitter_strength: float = 0.1, jitter_probability: float = 0.2,
                 full_crop_mode: str = "legacy", patch_grid_size: int = 14):
        # Lazy import avoids a runner import cycle.
        from .runner import build_transforms, PadToSquare, CLIP_IMAGE_MEAN
        if not math.isfinite(margin) or not 0 <= margin <= 1:
            raise ValueError("Person margin must be finite and in [0, 1]")
        if full_crop_mode not in {"legacy", "target_aware"}:
            raise ValueError("Full crop mode must be legacy or target_aware")
        if patch_grid_size <= 0:
            raise ValueError("Person patch grid size must be positive")
        full_train, full_eval = build_transforms(normalization, crop_scale)
        _, person_eval = build_transforms(
            normalization, crop_scale, "person_crop", "letterbox",
            jitter_strength, jitter_probability,
        )
        self.train = train
        self.margin = margin
        self.crop_scale = tuple(crop_scale)
        self.full_crop_mode = full_crop_mode
        self.patch_grid_size = int(patch_grid_size)
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

    @staticmethod
    def _randint(low: int, high: int) -> int:
        """Sample from the inclusive integer interval without Python RNG."""
        if low == high:
            return low
        return int(torch.randint(low, high + 1, (1,)).item())

    def target_aware_train_crop(self, image, box):
        """RandomResizedCrop-style context crop that contains the target box."""
        if box is None:
            return transforms.RandomResizedCrop.get_params(
                image, self.crop_scale, (3 / 4, 4 / 3)
            )
        width, height = image.size
        x1, y1, x2, y2 = box
        log_ratio = torch.log(torch.tensor((3 / 4, 4 / 3), dtype=torch.float64))
        for _ in range(10):
            target_area = height * width * float(
                torch.empty(1).uniform_(self.crop_scale[0], self.crop_scale[1]).item()
            )
            aspect = math.exp(float(torch.empty(1).uniform_(
                float(log_ratio[0]), float(log_ratio[1])
            ).item()))
            crop_width = int(round(math.sqrt(target_area * aspect)))
            crop_height = int(round(math.sqrt(target_area / aspect)))
            if not (0 < crop_width <= width and 0 < crop_height <= height):
                continue
            left_low = max(0, int(math.ceil(x2 - crop_width)))
            left_high = min(int(math.floor(x1)), width - crop_width)
            top_low = max(0, int(math.ceil(y2 - crop_height)))
            top_high = min(int(math.floor(y1)), height - crop_height)
            if left_low <= left_high and top_low <= top_high:
                return (
                    self._randint(top_low, top_high),
                    self._randint(left_low, left_high),
                    crop_height,
                    crop_width,
                )
        # A full-image fallback is deterministic, contains every valid target,
        # and preserves context when the sampled scale/aspect constraints are
        # incompatible with a large or extreme target box.
        return 0, 0, height, width

    @staticmethod
    def _target_aware_axis_start(length: int, crop_size: int,
                                 box_start: float, box_stop: float) -> int:
        centered = round((length - crop_size) / 2)
        low = max(0, int(math.ceil(box_stop - crop_size)))
        high = min(length - crop_size, int(math.floor(box_start)))
        if low <= high:
            return min(max(centered, low), high)
        # The scaled target itself is wider/taller than the evaluation crop.
        # Centering on it maximizes retained target content.
        return min(max(round((box_start + box_stop - crop_size) / 2), 0),
                   length - crop_size)

    def target_aware_eval_crop(self, resized, scaled_box):
        rw, rh = resized.size
        if scaled_box is None:
            return round((rh - 224) / 2), round((rw - 224) / 2), 224, 224
        x1, y1, x2, y2 = scaled_box
        left = self._target_aware_axis_start(rw, 224, x1, x2)
        top = self._target_aware_axis_start(rh, 224, y1, y2)
        return top, left, 224, 224

    def person_patch_mask(self, person, flip: bool) -> torch.Tensor:
        """Mask CLIP patches whose centers fall inside non-padding pixels."""
        width, height = person.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        grid = self.patch_grid_size
        centers = (torch.arange(grid, dtype=torch.float32) + 0.5) / grid
        keep_x = (centers >= left / side) & (centers < (left + width) / side)
        keep_y = (centers >= top / side) & (centers < (top + height) / side)
        mask = keep_y[:, None] & keep_x[None, :]
        if flip:
            mask = torch.flip(mask, dims=(1,))
        return mask.reshape(-1)

    def __call__(self, image, bbox):
        width, height = image.size
        box = self.target_box(bbox, width, height)
        if self.train:
            crop = (
                self.target_aware_train_crop(image, box)
                if self.full_crop_mode == "target_aware"
                else transforms.RandomResizedCrop.get_params(
                    image, self.crop_scale, (3 / 4, 4 / 3)
                )
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
            crop = (
                self.target_aware_eval_crop(resized, scaled_box)
                if self.full_crop_mode == "target_aware"
                else (top, left, 224, 224)
            )
            geometry = self.geometry(scaled_box, crop, False)
            full = self.to_tensor(TF.crop(resized, *crop))
            flip = False

        # Reuse the legacy crop implementation, including its exact rounding
        # and margin behavior. Invalid boxes fall back to Full but are masked.
        from multi_lane.continual_datasets.continual_datasets import EMOTIC
        person = EMOTIC.crop_person(image, bbox, self.margin)
        person_patch_mask = self.person_patch_mask(person, flip)
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
                "person_patch_mask": person_patch_mask,
                "condition_valid": (geometry[4] > 0).float() * geometry[5]}


def move_model_inputs(images, device):
    if isinstance(images, dict):
        return {key: value.to(device, non_blocking=True).float()
                for key, value in images.items()}
    return images.to(device, non_blocking=True).float()
