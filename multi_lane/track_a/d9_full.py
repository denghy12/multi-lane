"""Target-person spatial bias for the native Full lane, with unchanged Full pixels."""
import copy
import math
from collections import defaultdict

import torch
from PIL import Image

from .paired_transforms import PairedFullPersonTransform


def patch_coverage(geometry, grid=14):
    """Fraction of each patch occupied by the visible target, row-major like CLIP."""
    if not (geometry[4] > 0 and geometry[5] > 0):
        return geometry.new_zeros(grid * grid)
    left = torch.arange(grid, device=geometry.device, dtype=geometry.dtype)
    right = left + 1
    x1, y1, x2, y2 = geometry[:4] * grid
    width = (right.clamp(max=x2) - left.clamp(min=x1)).clamp(0, 1)
    height = (right.clamp(max=y2) - left.clamp(min=y1)).clamp(0, 1)
    return (height[:, None] * width[None, :]).flatten()


def bias_selector_scores(scores, coverage):
    """Add at most log(2) to spatial-key scores of the last two selectors."""
    if scores.ndim != 4 or scores.shape[2] < 2 or coverage.shape != (scores.shape[1], scores.shape[-1] - 1):
        raise ValueError("Expected [tasks,batch,selectors,CLS+patches] and [batch,patches]")
    result = scores.clone()
    result[:, :, -2:, 1:] += (coverage.to(scores) * math.log(2))[None, :, None, :]
    return result


class FullBoxTransform(PairedFullPersonTransform):
    def __call__(self, image, bbox):
        full, geometry, _ = self.full_with_geometry(image, bbox)
        return {"full": full, "patch_coverage": patch_coverage(geometry)}


def visible_eval_box(bbox, width, height):
    box = PairedFullPersonTransform.target_box(bbox, width, height)
    if box is None:
        return False
    rw, rh = ((256, int(256 * height / width)) if width <= height
              else (int(256 * width / height), 256))
    scaled = (box[0] * rw / width, box[1] * rh / height,
              box[2] * rw / width, box[3] * rh / height)
    crop = (round((rh - 224) / 2), round((rw - 224) / 2), 224, 224)
    return bool(PairedFullPersonTransform.geometry(scaled, crop, False)[4] > 0)


def swapped_box_indices(source):
    """Rotate boxes among visible labelled persons within each evaluation image."""
    groups = defaultdict(list)
    for i, path in enumerate(source.file_paths):
        groups[path].append(i)
    indices = list(range(len(source.file_paths)))
    for path, people in groups.items():
        if len(people) < 2:
            continue
        with Image.open(path) as image:
            visible = [i for i in people if visible_eval_box(source.body_bboxes[i], *image.size)]
        if len(visible) >= 2:
            for i, j in zip(visible, visible[1:] + visible[:1]):
                indices[i] = j
    return indices


def attach_full_box(source, normalization, crop_scale, *, training, mode="true"):
    if source.input_mode != "full" or source.paired_transform is not None:
        raise ValueError("Full-box hints require unmodified full-image inputs")
    if mode not in ("true", "swapped") or (training and mode != "true"):
        raise ValueError("Box swaps are evaluation-only")
    wrapped = copy.copy(source)
    wrapped.paired_transform = FullBoxTransform(training, normalization, crop_scale,
                                                jitter_strength=0, jitter_probability=0)
    wrapped.d9_swap_source_ids = {}
    if mode == "swapped":
        indices = swapped_box_indices(source)
        wrapped.body_bboxes = [source.body_bboxes[j] for j in indices]
        wrapped.d9_swap_source_ids = {source.sample_ids[i]: source.sample_ids[j]
                                      for i, j in enumerate(indices) if i != j}
    return wrapped
