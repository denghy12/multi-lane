"""Cross-arm audits, including base tensors and complete OFF/view predictions."""
from pathlib import Path
import numpy as np
import torch

from .export_compact_test_scores import _torch_load, _sha256
from .protected_joint_attention import ARMS
from .run_historical_b32_test_attribution import read, write
from .runner import tensor_state_digest


def verify_outputs(root, tasks, smoke, arms=ARMS, anchor='C0'):
    root = Path(root)
    records = []
    for task in tasks:
        baseline_path = root/anchor/'train/compact_checkpoints'/f'task{task}.pth'
        baseline = _torch_load(baseline_path)['model']
        baseline = {k:v for k,v in baseline.items() if not k.startswith('interaction_bank.')}
        history = read(root/anchor/'train/training_history.json')[str(task)]
        audit = read(root/anchor/'train/audits.json')[str(task)]
        for arm in arms:
            path = root/arm/'train/compact_checkpoints'/f'task{task}.pth'
            actual = _torch_load(path)['model']
            base = {k:v for k,v in actual.items() if not k.startswith('interaction_bank.')}
            if set(base)!=set(baseline) or any(not torch.equal(v,base[k]) for k,v in baseline.items()):
                raise AssertionError(('Base tensor mismatch',arm,task))
            rows = read(root/arm/'train/training_history.json')[str(task)]
            for old,new in zip(history,rows):
                for key in ('current_loss','adapter_objective_loss','optimizer_steps','skipped_optimizer_steps'):
                    if old[key]!=new[key]:
                        raise AssertionError(('Original training objective changed',arm,task,key))
            aa = read(root/arm/'train/audits.json')[str(task)]
            if audit['batch_traces'] != aa['batch_traces']:
                raise AssertionError(('Augmented training stream changed',arm,task))
            if arm!='C0' and not any(r['interaction_gradient_norm']>0 for r in rows):
                raise AssertionError(('Private branch never received gradient',arm,task))
            record = dict(task=task,arm=arm,base_tensors_exact=True,training_losses_exact=True,
                augmented_training_batches_exact=True,base_sha256=tensor_state_digest(base),
                checkpoint_sha256=_sha256(path))
            for split in ('val',) if smoke else ('val','test'):
                with np.load(root/anchor/split/'off/view_scores'/f'task{task}.npz') as expected, \
                     np.load(root/arm/split/'off/view_scores'/f'task{task}.npz') as actual:
                    for key in ('sample_ids','targets','face_reliable','class_indices','batch_lengths',
                        'fused_logits','fused_probabilities','full_logits','full_probabilities',
                        'person_logits','person_probabilities','face_logits','face_probabilities'):
                        if not np.array_equal(expected[key],actual[key]):
                            raise AssertionError(('OFF/standalone prediction mismatch',arm,task,split,key))
                record[split+'_off_all_person_predictions_exact'] = True
            records.append(record)
    write(root/'cross_arm_audit.json',dict(status='verified',tasks=list(tasks),records=records,
        anchor=anchor,arms=list(arms),
        base_parameters_losses_batches_and_complete_off_predictions_exact=True))
    return records
