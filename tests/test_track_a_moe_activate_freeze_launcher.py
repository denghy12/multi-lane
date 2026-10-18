"""Exercise split concurrency/failure handling without loading real datasets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class ActivateFreezeLauncherTest(unittest.TestCase):
    def run_launcher(self, fail_test=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            script = repo / "scripts/emotic/run_moe_activate_freeze.sh"
            script.parent.mkdir(parents=True)
            shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/emotic/run_moe_activate_freeze.sh", script)
            for args in (("init", "-q"), ("add", "."),
                         ("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                          "commit", "-qm", "fixture")):
                subprocess.run(["git", *args], cwd=repo, check=True)
            data = root / "data"
            data.mkdir()
            (data / "CVPR17_Annotations.mat").touch()
            clip = root / "clip.pt"
            clip.touch()
            manifests = root / "face/manifests"
            manifests.mkdir(parents=True)
            for split in ("train", "val", "test"):
                (manifests / (split + ".jsonl")).touch()
            fake = root / "fake_python"
            fake.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys, time
if 'multi_lane.track_a.evaluate_moe_task_ffn' not in sys.argv:
    sys.exit(0)
split = sys.argv[sys.argv.index('--split') + 1]
output = sys.argv[sys.argv.index('--output') + 1]
label = 'off' if output.endswith('full_val_specialist_off') else split
def event(stage):
    with open(os.environ['EVENT_PATH'], 'a') as handle:
        handle.write(json.dumps({'label':label, 'stage':stage, 'time':time.time_ns()}) + '\\n')
event('start')
time.sleep(0.25)
event('end')
sys.exit(7 if os.environ.get('FAIL_SPLIT') == split and label != 'off' else 0)
''')
            fake.chmod(0o755)
            events = root / "events.jsonl"
            env = dict(os.environ, GPU="0", SEED="0", PYTHON=str(fake), DATA_ROOT=str(data),
                       CLIP_CHECKPOINT=str(clip), FACE_MANIFEST_ROOT=str(manifests.parent),
                       RUN_ID="fixture", ROUTE_MODE="dynamic", EVENT_PATH=str(events))
            env.pop("SMOKE_UPDATES", None)
            if fail_test:
                env["FAIL_SPLIT"] = "test"
            else:
                env.pop("FAIL_SPLIT", None)
            result = subprocess.run(["bash", str(script)], env=env,
                                    capture_output=True, text=True, timeout=15)
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            status = (repo / "logs/moe_activate_freeze/fixture/evaluation_status.txt").read_text()
            return result.returncode, rows, status

    def test_val_test_overlap_after_training(self):
        code, rows, status = self.run_launcher()
        self.assertEqual(code, 0)
        times = {(r["label"], r["stage"]):r["time"] for r in rows}
        self.assertLess(max(times[v, "start"] for v in ("val", "test")),
                        min(times[v, "end"] for v in ("val", "test")))
        self.assertEqual({r["label"] for r in rows}, {"val", "test"})
        self.assertEqual(status.strip(), "val_exit=0 test_exit=0")

    def test_failed_test_still_waits_for_val_and_stops_batch(self):
        code, rows, status = self.run_launcher(fail_test=True)
        self.assertEqual(code, 1)
        self.assertEqual({r["label"] for r in rows if r["stage"] == "end"}, {"val", "test"})
        self.assertEqual(status.strip(), "val_exit=0 test_exit=7")


if __name__ == "__main__":
    unittest.main()
