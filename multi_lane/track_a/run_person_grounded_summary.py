"""GPU0 adaptive queue for all six predeclared person-grounded summary controls."""
import argparse
from pathlib import Path

from .person_grounded_summary import GROUNDED_ARMS
from .run_protected_joint_attention import run
from .run_historical_b32_test_attribution import read, write
from .export_compact_test_scores import _sha256


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run', 'data-root', 'clip-checkpoint', 'face-manifest-root', 'output', 'logs', 'census'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--maximum-jobs', type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-updates', type=int, default=4)
    args = p.parse_args(); census = read(args.census)
    assert census['status']=='complete' and not census['evaluation_data_read'] and census['split']=='train'
    assert census['tasks']==list(range(8)) and all(r['eligible_loc']>0 for r in census['records'].values())
    assert census['source_configuration_sha256']==_sha256(args.source_run/'config.json')
    write(args.output/'preflight_gate.json', dict(census=str(args.census), census_sha256=_sha256(args.census),
        localization_supervision_nonempty_all_tasks=True, tasks=list(range(8))))
    run(args, arms=GROUNDED_ARMS, anchor='PERSON_CONST_POOL',
        calibration_module='multi_lane.track_a.person_grounded_evaluation')
