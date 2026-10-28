"""GPU0 four-control queue, historical eight-task training and complete evaluation."""
import argparse
from pathlib import Path

from .protected_attention_dose import DOSE_ARMS
from .run_protected_joint_attention import run


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','data-root','clip-checkpoint','face-manifest-root','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--maximum-jobs',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-updates',type=int,default=4)
    args=p.parse_args()
    run(args, arms=DOSE_ARMS, anchor='DOSE_MLP')
