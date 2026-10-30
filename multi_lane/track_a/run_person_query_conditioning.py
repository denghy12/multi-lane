"""Prepare independent projection/gain families; run only explicitly invoked jobs."""
import argparse
from pathlib import Path

from .person_query_conditioning import QUERY_ARMS
from .run_protected_joint_attention import run
from .run_historical_b32_test_attribution import read,write
from .export_compact_test_scores import _sha256


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','data-root','clip-checkpoint','face-manifest-root','census','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--family',choices=('projection','gain','both'),default='projection')
    p.add_argument('--maximum-jobs',type=int,choices=(1,2,3,4),default=1)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-updates',type=int,default=4)
    args=p.parse_args();census=read(args.census)
    assert census['status']=='complete' and census['split']=='train' and not census['evaluation_data_read']
    assert census['source_configuration_sha256']==_sha256(args.source_run/'config.json')
    assert census['tasks']==list(range(8)) and all(x['eligible_loc']>0 for x in census['records'].values())
    arms=QUERY_ARMS if args.family=='both' else tuple(a for a in QUERY_ARMS if a.startswith('QPROJ_' if args.family=='projection' else 'QGAIN_'))
    write(args.output/'preflight_gate.json',dict(status='verified',census_sha256=_sha256(args.census),
        family=args.family,mutually_exclusive_repairs=True,full_training_only_when_explicitly_invoked=not args.smoke))
    run(args,arms=arms,anchor=arms[0],calibration_module='multi_lane.track_a.person_grounded_evaluation')
