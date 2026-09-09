from __future__ import annotations

import unittest

from PIL import Image

from multi_lane.track_a.face_manifest import (
    expand_box,
    letterbox_square,
    match_faces_to_people,
    person_face_score,
    parse_args,
    select_visual_records,
    summarize_records,
)


class FaceManifestTest(unittest.TestCase):
    def test_cli_accepts_test_without_changing_default_splits(self) -> None:
        base = [
            "--data-root", "/tmp/data",
            "--detector-checkpoint", "/tmp/detector.onnx",
            "--output-root", "/tmp/output",
        ]
        self.assertEqual(tuple(parse_args(base).splits), ("train", "val"))
        self.assertEqual(
            parse_args(base + ["--splits", "train", "val", "test"]).splits,
            ["train", "val", "test"],
        )

    def test_global_matching_never_reuses_a_face(self) -> None:
        bodies = [[0, 0, 100, 200], [80, 0, 180, 200]]
        faces = [
            {"bbox": [85, 20, 105, 45], "score": 0.99},
            {"bbox": [130, 25, 155, 55], "score": 0.90},
        ]
        matches, diagnostics = match_faces_to_people(bodies, faces)
        assigned = [match.face_index for match in matches if match.face_index is not None]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(diagnostics["duplicate_face_assignments"], 0)
        self.assertEqual(set(assigned), {0, 1})

    def test_ambiguous_candidates_are_flagged(self) -> None:
        bodies = [[0, 0, 100, 200]]
        faces = [
            {"bbox": [35, 20, 55, 40], "score": 0.90},
            {"bbox": [45, 22, 65, 42], "score": 0.90},
        ]
        matches, _ = match_faces_to_people(bodies, faces, ambiguity_gap=0.08)
        self.assertTrue(matches[0].ambiguous)
        self.assertEqual(matches[0].candidate_count, 2)

    def test_outside_face_is_not_eligible(self) -> None:
        self.assertIsNone(person_face_score([0, 0, 100, 200], [120, 20, 150, 50], 0.9))
        matches, _ = match_faces_to_people(
            [[0, 0, 100, 200]], [{"bbox": [120, 20, 150, 50], "score": 0.9}]
        )
        self.assertIsNone(matches[0].face_index)

    def test_face_crop_expands_clamps_and_letterboxes(self) -> None:
        self.assertEqual(expand_box([5, 10, 25, 30], 100, 80, 0.5), [0, 0, 35, 40])
        crop = Image.new("RGB", (20, 40), "white")
        transformed = letterbox_square(crop, size=224)
        self.assertEqual(transformed.size, (224, 224))

    def test_task_summary_counts_current_class_valid_positives(self) -> None:
        base = {
            "image_path": "cvpr_emotic/a.jpg",
            "people_in_image": 1,
            "ambiguous_match": False,
            "invalid_reason": None,
            "face_width": 20.0,
            "face_height": 20.0,
            "face_short_side": 20.0,
            "face_person_area_ratio": 0.05,
        }
        records = [
            dict(base, sample_id="train:a#person=0", target_indices=[0, 1], valid_face=True),
            dict(base, sample_id="train:b#person=0", image_path="cvpr_emotic/b.jpg",
                 target_indices=[0], valid_face=False, invalid_reason="no_face_detected"),
            dict(base, sample_id="train:c#person=0", image_path="cvpr_emotic/c.jpg",
                 target_indices=[5], valid_face=True),
        ]
        summary = summarize_records(records, {}, partial=False)
        task0 = summary["tasks"][0]
        self.assertEqual(task0["eligible_samples"], 2)
        self.assertEqual(task0["valid_face_samples"], 1)
        affection = task0["per_current_class"][0]
        self.assertEqual(affection["positive_samples"], 2)
        self.assertEqual(affection["valid_face_positive_samples"], 1)

    def test_visual_selection_includes_failure_modes(self) -> None:
        records = []
        for index in range(20):
            records.append({
                "sample_id": f"train:{index}#person=0",
                "ambiguous_match": index == 0,
                "people_in_image": 2 if index == 1 else 1,
                "valid_face": index != 2,
                "face_short_side": 20.0 if index == 3 else 64.0,
            })
        selected = select_visual_records(records, 10)
        selected_ids = {record["sample_id"] for record in selected}
        self.assertIn("train:0#person=0", selected_ids)
        self.assertIn("train:1#person=0", selected_ids)
        self.assertIn("train:2#person=0", selected_ids)
        self.assertIn("train:3#person=0", selected_ids)


if __name__ == "__main__":
    unittest.main()
