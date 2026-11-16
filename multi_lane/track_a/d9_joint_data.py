"""One EMOTIC person per training row, with native and sender transforms."""
from PIL import Image
import torch

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .r1_crossvit_experiment import read, source_runs
from .runner import build_transforms


class JointEMOTIC(EMOTIC):
    """Decode one original and apply all crop operations to the same person id."""

    def __init__(self, inputs, split):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unexpected split {split}")
        configs = {v: read(p / "config.json") for v, p in source_runs(inputs.reference_root).items()}
        parent = inputs.data_root.parent if inputs.data_root.name == "EMOTIC" else inputs.data_root
        super().__init__(str(parent), train=split == "train", eval_splits=(split,),
                         input_mode="face_crop", face_manifest_root=inputs.face_manifest_root)
        self.person_crop_margin = configs["person"]["person_crop_margin"]
        self.transforms_by_view = {}
        for view, config in configs.items():
            training, evaluation = build_transforms(
                config["input_normalization"], config["train_crop_scale"],
                input_mode=config["input_mode"], person_transform_mode=config["person_transform_mode"],
                person_color_jitter_strength=config["person_color_jitter_strength"],
                person_color_jitter_probability=config["person_color_jitter_probability"],
            )
            self.transforms_by_view[view] = (training if self.train else evaluation, evaluation)

    def __getitem__(self, index):
        with Image.open(self.file_paths[index]) as original:
            full = original.convert("RGB")
        person = self._crop_person(full, self.body_bboxes[index])
        face = self._crop_face(full, index)
        data = {}
        for name, crop in (("full", full), ("person", person), ("face", face)):
            own, evidence = self.transforms_by_view[name]
            data[name] = own(crop)
            data["evidence_" + name] = evidence(crop) if self.train else data[name]
        data["evidence_face_reliable"] = torch.tensor(self.face_reliable[index], dtype=torch.bool)
        data["face_training_valid"] = torch.tensor(self.face_training_valid[index], dtype=torch.bool)
        target = torch.zeros(len(self.classes), dtype=torch.long)
        target[self.targets[index]] = 1
        return data, target


def build_joint_source(inputs, split):
    return JointEMOTIC(inputs, split)
