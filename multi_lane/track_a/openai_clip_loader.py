"""Load the frozen OpenAI CLIP visual tower from a local checkpoint.

The model definition is derived from OpenAI CLIP (MIT license).  Only the
checkpoint's visual tower is returned; no tokenizer or text encoder is used by
the Track-A MULTI-LANE reproduction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import torch

from .openai_clip_model import build_model


OPENAI_VIT_B16_SHA256 = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_openai_clip_visual(
    checkpoint: Union[str, Path], *, verify_sha256: bool = True
) -> torch.nn.Module:
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OpenAI CLIP checkpoint not found: {checkpoint}")
    if verify_sha256:
        actual = sha256_file(checkpoint)
        if actual != OPENAI_VIT_B16_SHA256:
            raise RuntimeError(
                "OpenAI CLIP ViT-B/16 checkpoint SHA-256 differs: "
                f"expected {OPENAI_VIT_B16_SHA256}, got {actual}"
            )

    try:
        scripted = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        state_dict = scripted.state_dict()
    except RuntimeError:
        payload = torch.load(str(checkpoint), map_location="cpu")
        state_dict = payload.state_dict() if hasattr(payload, "state_dict") else payload

    model = build_model(state_dict).float().eval()
    visual = model.visual
    visual.requires_grad_(False)
    return visual
