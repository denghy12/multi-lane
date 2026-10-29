"""GPU0 queue: four 2x2 models, TRAIN calibration then complete Val/Test."""
import argparse
from pathlib import Path
from .private_summarize_attention import SUMMARY_ARMS
from .run_protected_joint_attention import run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','data-root','clip-checkpoint','face-manifest-root','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--maximum-jobs',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-updates',type=int,default=4)
    run(p.parse_args(),arms=SUMMARY_ARMS,anchor='SUMMARY_FIXED_POOL',
        calibration_module='multi_lane.track_a.private_summary_evaluation')
