"""Export leakage-free low-dimensional Full/Person/Face visual descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual
from .runner import (
    CLASS_ORDER,
    build_transforms,
    fit_calibration_indices,
    git_metadata,
    load_face_manifest_provenance,
    resolve_dataset_parent,
    set_seed,
    validate_classes,
)


CALIBRATION_FRACTION = 0.10
FEATURE_NAMES = ("clip_cos_full_person", "clip_cos_full_face", "clip_cos_person_face")


class ThreeViewDescriptorDataset(Dataset):
    def __init__(
        self,
        full: EMOTIC,
        person: EMOTIC,
        face: EMOTIC,
        indices: Sequence[int],
    ) -> None:
        validate_classes(full, person, face)
        targets_aligned = all(
            tuple(int(value) for value in full_target)
            == tuple(int(value) for value in person_target)
            == tuple(int(value) for value in face_target)
            for full_target, person_target, face_target in zip(
                full.targets, person.targets, face.targets
            )
        )
        if not (
            full.sample_ids == person.sample_ids == face.sample_ids
            and len(full.targets) == len(person.targets) == len(face.targets)
            and targets_aligned
        ):
            raise ValueError("Three-view descriptor datasets are not aligned")
        self.full = full
        self.person = person
        self.face = face
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        full_image, _ = self.full[index]
        person_image, _ = self.person[index]
        face_image, _ = self.face[index]
        return (
            full_image,
            person_image,
            face_image,
            self.full.sample_ids[index],
            bool(self.face.face_reliable[index]),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_sources(
    dataset_parent: Path, face_manifest_root: Path, split: str
) -> tuple[EMOTIC, EMOTIC, EMOTIC]:
    _, full_transform = build_transforms(
        "clip", (0.05, 1.0), input_mode="full"
    )
    _, person_transform = build_transforms(
        "clip",
        (0.70, 1.0),
        input_mode="person_crop",
        person_transform_mode="letterbox",
        person_color_jitter_strength=0.10,
        person_color_jitter_probability=0.20,
    )
    _, face_transform = build_transforms(
        "clip", (0.05, 1.0), input_mode="face_crop"
    )
    common = {"root": str(dataset_parent), "download": False}
    if split == "train":
        split_args = {"train": True}
    elif split in ("val", "test"):
        split_args = {"train": False, "eval_splits": (split,)}
    else:
        raise ValueError("Descriptor export split must be train, val, or test")
    full = EMOTIC(**common, **split_args, transform=full_transform, input_mode="full")
    person = EMOTIC(
        **common,
        **split_args,
        transform=person_transform,
        input_mode="person_crop",
        person_crop_margin=0.15,
    )
    face = EMOTIC(
        **common,
        **split_args,
        transform=face_transform,
        input_mode="face_crop",
        face_manifest_root=face_manifest_root,
    )
    return full, person, face


def extract_descriptors(
    visual: torch.nn.Module,
    dataset: ThreeViewDescriptorDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    ids: List[str] = []
    reliable_rows: List[torch.Tensor] = []
    feature_rows: List[torch.Tensor] = []
    visual.eval()
    with torch.no_grad():
        for full, person, face, sample_ids, reliable in loader:
            batch = torch.cat((full, person, face), dim=0).to(device)
            with torch.cuda.amp.autocast(enabled=True):
                encoded = visual(batch)
            encoded = F.normalize(encoded.float(), dim=-1).cpu()
            full_feature, person_feature, face_feature = encoded.chunk(3, dim=0)
            features = torch.stack(
                (
                    torch.sum(full_feature * person_feature, dim=-1),
                    torch.sum(full_feature * face_feature, dim=-1),
                    torch.sum(person_feature * face_feature, dim=-1),
                ),
                dim=-1,
            )
            reliable_tensor = torch.as_tensor(reliable, dtype=torch.bool)
            features[~reliable_tensor, 1:] = 0.0
            if not torch.isfinite(features).all():
                raise FloatingPointError("Non-finite three-view visual descriptor")
            ids.extend(str(value) for value in sample_ids)
            reliable_rows.append(reliable_tensor)
            feature_rows.append(features)
    if len(set(ids)) != len(ids) or len(ids) != len(dataset):
        raise ValueError("Descriptor export produced invalid sample IDs")
    return {
        "sample_ids": np.asarray(ids),
        "features": torch.cat(feature_rows).numpy().astype(np.float32),
        "face_reliable": torch.cat(reliable_rows).numpy(),
    }


def export_descriptor_set(
    data_root: Path,
    face_manifest_root: Path,
    clip_checkpoint: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    workers: int,
    train_scope: str = "calibration",
) -> Dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(clip_checkpoint)
    if _sha256(clip_checkpoint) != OPENAI_VIT_B16_SHA256:
        raise ValueError("Unexpected CLIP checkpoint")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Visual descriptor export requires CUDA")
    source_root = Path(__file__).resolve().parents[2]
    source_git = git_metadata(source_root)
    if source_git["dirty"]:
        raise RuntimeError("Descriptor export requires a clean Git worktree")
    face_provenance = load_face_manifest_provenance(face_manifest_root)
    dataset_parent = resolve_dataset_parent(data_root)
    set_seed(0, tf32=True)
    visual = load_openai_clip_visual(clip_checkpoint).float().to(device)
    visual.requires_grad_(False)
    if train_scope not in ("calibration", "all"):
        raise ValueError("train_scope must be calibration or all")
    output_dir.mkdir(parents=True)
    split_records = {}
    for split in ("train", "val"):
        full, person, face = _build_sources(dataset_parent, face_manifest_root, split)
        if split == "train":
            if train_scope == "calibration":
                _, indices = fit_calibration_indices(
                    full, tuple(range(len(CLASS_ORDER))), CALIBRATION_FRACTION
                )
                purpose = "stable_sha256_image_group_v1_calibration_only"
            else:
                indices = list(range(len(full)))
                purpose = "complete_train_pool_for_oof_router"
        else:
            indices = list(range(len(full)))
            purpose = "complete_validation_pool"
        dataset = ThreeViewDescriptorDataset(full, person, face, indices)
        arrays = extract_descriptors(visual, dataset, device, batch_size, workers)
        artifact = output_dir / f"{split}_descriptors.npz"
        np.savez_compressed(
            artifact,
            schema_version=np.asarray(1, dtype=np.int64),
            split=np.asarray(split),
            purpose=np.asarray(purpose),
            sample_ids=arrays["sample_ids"],
            feature_names=np.asarray(FEATURE_NAMES),
            features=arrays["features"],
            face_reliable=arrays["face_reliable"],
        )
        split_records[split] = {
            "purpose": purpose,
            "samples": len(dataset),
            "reliable_face_samples": int(arrays["face_reliable"].sum()),
            "sha256": _sha256(artifact),
        }
    result = {
        "schema_version": 1,
        "selection_only": True,
        "test_accessed": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_definition": "cosine_similarity_of_frozen_clip_final_embeddings",
        "invalid_face_values": "face-related_cosines_zero_and_masked",
        "train_scope": train_scope,
        "calibration_fraction": (
            CALIBRATION_FRACTION if train_scope == "calibration" else None
        ),
        "calibration_split": (
            "stable_sha256_image_group_v1"
            if train_scope == "calibration" else None
        ),
        "source_git": source_git,
        "clip_checkpoint": str(clip_checkpoint.resolve()),
        "clip_checkpoint_sha256": OPENAI_VIT_B16_SHA256,
        "face_manifest": face_provenance,
        "splits": split_records,
    }
    manifest = output_dir / "descriptor_manifest.json"
    manifest.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(split_records, indent=2), flush=True)
    print("THREE_VIEW_DESCRIPTOR_EXPORT_COMPLETE", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--train-scope", choices=("calibration", "all"), default="calibration")
    args = parser.parse_args()
    export_descriptor_set(
        args.data_root,
        args.face_manifest_root,
        args.clip_checkpoint,
        args.output_dir,
        args.device,
        args.batch_size,
        args.workers,
        args.train_scope,
    )


if __name__ == "__main__":
    main()
