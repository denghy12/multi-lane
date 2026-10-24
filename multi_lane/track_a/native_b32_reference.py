"""Two-task reference executed from the unchanged September Git checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-root', type=Path, required=True)
    p.add_argument('--arguments-json', type=Path, required=True)
    args = p.parse_args()
    from .historical_b32_search import LEGACY_COMMIT
    legacy = args.legacy_root.resolve()
    flags = json.loads(args.arguments_json.read_text())
    if subprocess.check_output(['git', '-C', str(legacy), 'rev-parse', 'HEAD'], text=True).strip() != LEGACY_COMMIT:
        raise ValueError('Wrong native b32 source revision')
    if subprocess.check_output(['git', '-C', str(legacy), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Native b32 checkout must remain clean')
    if '--historical-feature-fusion' in flags or '--historical-b32-profile' in flags:
        raise ValueError('New-profile switches cannot enter the native runner')
    # Resolve only input/output paths; changing directories must not change data.
    for flag in ('--data-root', '--clip-checkpoint', '--output-root', '--face-manifest-root'):
        if flag in flags:
            i = flags.index(flag) + 1
            flags[i] = str(Path(flags[i]).resolve())
    for name in list(sys.modules):
        if name == 'multi_lane' or name.startswith('multi_lane.'):
            del sys.modules[name]
    sys.path.insert(0, str(legacy)); os.chdir(legacy)
    from multi_lane.track_a import runner
    sys.argv = ['native_b32_runner', *flags]
    runner.main()


if __name__ == '__main__':
    main()
