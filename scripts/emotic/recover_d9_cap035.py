"""Diagnose D9 cap035 at its last valid epoch and run an isolated low-scale repeat."""
import argparse
import os
from pathlib import Path

import torch

from multi_lane.track_a.r1_crossvit_experiment import read, write
from scripts.emotic.diagnose_d9_cap035 import scan
from scripts.emotic.run_d9_branch_refinement import ANCHOR, ROOT, JOBS, train_flags, execute
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths
from scripts.emotic.run_r1_frozen_late_stable import copy_prefix
from scripts.emotic.run_r1_patch_receiver_depth import REFERENCE
from scripts.emotic.summarize_d9_refinement import run

SOURCE = Path('../multi-lane-main-d9-branch-refinement-20261115') / ROOT / 'cap035/attention/person/train'
OUTPUT = Path('output/d9_cap035_stable/seed1_20261115')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Run the numerical repeat on GPU0')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp32', action='store_true')
    fp32 = parser.parse_args().fp32
    output = OUTPUT / 'fp32' if fp32 else OUTPUT
    job = next(j for j in JOBS if j.name == 'cap035_person')
    path = job.path(output)
    train = path / 'train'
    inputs = input_paths()
    checkpoint = SOURCE / 'epoch_recovery/task3/epoch14.pth'
    audit = OUTPUT / 'diagnostic'
    if not (audit / 'diagnostic.json').exists():
        scan(checkpoint, audit, batch_limit=None)
    audit_result = read(audit / 'diagnostic.json')
    if (audit_result['status'] != 'complete' or not audit_result['all_expected_batches_scanned']
            or audit_result['mode_nonfinite_batches']['amp32'] or audit_result['mode_nonfinite_batches']['fp32']):
        raise RuntimeError('Low-scale/FP32 diagnostic did not clear the full fixed-parameter TRAIN epoch')
    if fp32:
        low_scale = read(job.path(OUTPUT) / 'train/progress.json')
        if (low_scale['task'], low_scale['epoch'], low_scale['skipped_steps']) != (3, 15, 2):
            raise ValueError('Expected the isolated low-scale repeat to have failed at epoch15')
    copy_prefix(SOURCE, train)
    args = train_flags(job, output, False, inputs)
    scale_flag = args.index('--r1-fixed-loss-scale')
    if fp32:
        args[scale_flag:scale_flag + 2] = ['--r1-training-fp32']
    else:
        args[scale_flag + 1] = '32'
    args.extend(['--r1-recovery-source', SOURCE])
    execute('multi_lane.track_a.runner', args, threads=72)
    counts = check_training(train, REFERENCE, job.view, 8, False)
    execute('multi_lane.track_a.r1_crossvit_experiment', [
        'val', '--training-root', train, '--data-root', inputs.data_root,
        '--clip-checkpoint', inputs.clip_checkpoint, '--output', path / 'val',
        '--face-manifest-root', inputs.face_manifest_root])
    torch.set_num_threads(2)
    results = run(output, ANCHOR, inputs.face_manifest_root, group_names=('D', 'cap035_person'))
    write(output / 'validation_test.json', results)
    write(path / 'completion.json', dict(status='complete', seed=1, training=counts,
          changed_scale_after='task3_epoch14', original_scale=4096, new_scale=None if fp32 else 32, training_precision='fp32' if fp32 else 'amp'))


if __name__ == '__main__':
    main()
