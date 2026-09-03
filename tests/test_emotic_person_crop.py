from __future__ import annotations

import unittest

from PIL import Image

from multi_lane.continual_datasets.continual_datasets import EMOTIC


class EmoticPersonCropTest(unittest.TestCase):
    @staticmethod
    def _dataset(margin: float) -> EMOTIC:
        dataset = object.__new__(EMOTIC)
        dataset.person_crop_margin = margin
        return dataset

    def test_person_crop_without_margin_matches_body_box(self) -> None:
        image = Image.new("RGB", (100, 80))
        cropped = self._dataset(0.0)._crop_person(image, [20, 10, 60, 50])
        self.assertEqual(cropped.size, (40, 40))

    def test_person_crop_margin_expands_and_clamps_to_image(self) -> None:
        image = Image.new("RGB", (100, 80))
        cropped = self._dataset(0.25)._crop_person(image, [20, 10, 60, 50])
        self.assertEqual(cropped.size, (60, 60))

    def test_invalid_person_box_falls_back_to_full_image(self) -> None:
        image = Image.new("RGB", (100, 80))
        cropped = self._dataset(0.15)._crop_person(image, [60, 50, 20, 10])
        self.assertIs(cropped, image)


if __name__ == "__main__":
    unittest.main()
