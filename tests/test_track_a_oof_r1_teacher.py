from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_lane.track_a.compare_oof_r1_teacher_validation import compare


class OOFR1TeacherComparisonTest(unittest.TestCase):
    @patch("multi_lane.track_a.compare_oof_r1_teacher_validation._row")
    def test_locked_gate_requires_all_three_metrics(self, row) -> None:
        row.side_effect = [
            {
                "name": "D0", "distillation": None,
                "full_metrics": {"final_mAP": 10.0, "average_mAP": 20.0},
                "locked_R1_metrics": {"final_mAP": 30.0},
            },
            {
                "name": "D3", "distillation": {"mode": "r1", "mix": 0.2},
                "full_metrics": {"final_mAP": 10.1, "average_mAP": 19.9},
                "locked_R1_metrics": {"final_mAP": 30.1},
            },
        ]
        locked = {"seed": 0}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("d0", "d3"):
                (root / name).mkdir()
                (root / name / "config.json").write_text(json.dumps(locked))
            result = compare(
                ("D0", root / "d0"), ("D3", root / "d3"),
                Path("person"), Path("face"), Path("manifest"),
            )
        self.assertFalse(result["candidate"]["eligible"])
        self.assertFalse(result["decision"]["advance_to_seed1_seed2_validation"])


if __name__ == "__main__":
    unittest.main()
