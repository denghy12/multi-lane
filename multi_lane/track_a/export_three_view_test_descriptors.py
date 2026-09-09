"""Export deterministic Full/Person/Face descriptors for diagnostic test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .export_three_view_descriptors import (
    FEATURE_NAMES,
    ThreeViewDescriptorDataset,
    _build_sources,
    _sha256,
    extract_descriptors,
)
from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual
from .runner import git_metadata, load_face_manifest_provenance, resolve_dataset_parent, set_seed


def export_test_descriptors(
    data_root: Path,
    face_manifest_root: Path,
    clip_checkpoint: Path,
    output_dir: Path,
    device_name: str,
    batch_size: int,
    workers: int,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not clip_checkpoint.is_file() or _sha256(clip_checkpoint) != OPENAI_VIT_B16_SHA256:
        raise ValueError("Unexpected CLIP checkpoint")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Test descriptor export requires CUDA")
    source_git = git_metadata(Path(__file__).resolve().parents[2])
    if source_git["dirty"]:
        raise RuntimeError("Test descriptor export requires a clean Git worktree")
    provenance = load_face_manifest_provenance(face_manifest_root)
    if "test_manifest" not in provenance["artifact_sha256"]:
        raise ValueError("Face manifest provenance does not include test")

    set_seed(0, tf32=True)
    visual = load_openai_clip_visual(clip_checkpoint).float().to(device)
    visual.requires_grad_(False)
    full, person, face = _build_sources(
        resolve_dataset_parent(data_root), face_manifest_root, "test"
    )
    dataset = ThreeViewDescriptorDataset(full, person, face, range(len(full)))
    arrays = extract_descriptors(visual, dataset, device, batch_size, workers)
    output_dir.mkdir(parents=True)
    artifact = output_dir / "test_descriptors.npz"
    np.savez_compressed(
        artifact,
        schema_version=np.asarray(1, dtype=np.int64),
        split=np.asarray("test"),
        purpose=np.asarray("diagnostic_test_evaluation_only"),
        sample_ids=arrays["sample_ids"],
        feature_names=np.asarray(FEATURE_NAMES),
        features=arrays["features"],
        face_reliable=arrays["face_reliable"],
    )
    result = {
        "schema_version": 1,
        "evaluation_split": "test",
        "used_for_selection": False,
        "test_weight_search": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_definition": "cosine_similarity_of_frozen_clip_final_embeddings",
        "invalid_face_values": "face-related_cosines_zero_and_masked",
        "source_git": source_git,
        "clip_checkpoint": str(clip_checkpoint.resolve()),
        "clip_checkpoint_sha256": OPENAI_VIT_B16_SHA256,
        "face_manifest": provenance,
        "samples": len(dataset),
        "reliable_face_samples": int(arrays["face_reliable"].sum()),
        "artifact_sha256": _sha256(artifact),
    }
    (output_dir / "descriptor_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "samples": result["samples"],
        "reliable_face_samples": result["reliable_face_samples"],
        "artifact_sha256": result["artifact_sha256"],
    }, indent=2), flush=True)
    print("THREE_VIEW_TEST_DESCRIPTOR_EXPORT_COMPLETE", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    export_test_descriptors(
        args.data_root,
        args.face_manifest_root,
        args.clip_checkpoint,
        args.output_dir,
        args.device,
        args.batch_size,
        args.workers,
    )


if __name__ == "__main__":
    main()
