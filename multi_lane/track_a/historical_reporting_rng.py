"""Reproduce historical reporting iterator seeds without reading evaluation data."""
import hashlib
import pickle
from pathlib import Path

import torch

from .export_compact_test_scores import _sha256
from .runner import tensor_state_digest
from .task_recovery import rng_state


def cpu_digest(state):
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()


def replay_reporting_rng(task):
    """One base seed per historical reporting/diagnostic DataLoader iterator.

    Called exactly once before the FINAL recovery boundary is saved. Warmup
    and trained-before-audit boundaries remain before reporting; loading a
    completed final boundary skips this operation, preventing double replay.
    """
    before=rng_state()
    seeds=[int(torch.empty((),dtype=torch.int64,device='cpu').random_()) for _ in range(2)]
    after=rng_state()
    assert pickle.dumps((before['python'],before['numpy']))==pickle.dumps((after['python'],after['numpy']))
    assert all(torch.equal(a,b) for a,b in zip(before['torch_cuda'],after['torch_cuda']))
    return dict(task=task,policy='historical_two_reporting_iterator_base_seeds',
        int64_draws=2,base_seeds=seeds,cpu_rng_before=cpu_digest(before['torch_cpu']),
        cpu_rng_after=cpu_digest(after['torch_cpu']),python_numpy_cuda_rng_unchanged=True,
        evaluation_images_read=False)


def verify_historical_boundary(model,history,reference,task):
    path=Path(reference)/'recovery_checkpoints'/f'task{task}_before_reporting.pth'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    state=model.state_dict()
    assert set(saved['model'])=={k for k in state if not k.startswith('visual_encoder.')}
    unequal=[k for k,v in saved['model'].items() if k not in state or not torch.equal(v,state[k].detach().cpu())]
    assert not unequal,('historical_model_tensor_mismatch',task,unequal)
    expected=saved['provenance']['training_history'][str(task)]
    assert len(history)==len(expected)==30
    for index,(a,b) in enumerate(zip(history,expected)):
        for key in ('current_loss','adapter_objective_loss','optimizer_steps','skipped_optimizer_steps'):
            assert a[key]==b[key],('historical_history_mismatch',task,index,key,a[key],b[key])
    actual=rng_state();old=saved['rng']
    assert torch.equal(actual['torch_cpu'],old['torch_cpu']),('historical_cpu_rng_mismatch',task)
    assert all(torch.equal(a,b) for a,b in zip(actual['torch_cuda'],old['torch_cuda'])),('historical_cuda_rng_mismatch',task)
    assert pickle.dumps((actual['python'],actual['numpy']))==pickle.dumps((old['python'],old['numpy']))
    return dict(status='verified',task=task,all_model_tensors_exact=True,
        all_30_epoch_losses_and_update_counts_exact=True,all_rng_states_exact=True,
        reference=str(path),reference_sha256=_sha256(path),
        compact_model_sha256=tensor_state_digest(saved['model']),no_reference_parameters_loaded_into_student=True)
