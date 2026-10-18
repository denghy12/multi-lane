"""Atomic task-boundary recovery snapshots, saved before reporting inference."""
from __future__ import annotations
import os
import random
import tempfile
from pathlib import Path
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=len(state['torch_cuda']):
            raise ValueError('Recovery requires the same visible CUDA device count')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def optimizer_snapshot(model, optimizer, scheduler, scaler):
    names={id(p):name for name,p in model.named_parameters()}
    return dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(), optimizer_parameter_names=[
                    [names[id(p)] for p in group['params']] for group in optimizer.param_groups])


def atomic_save(payload, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',delete=False) as handle:
            temporary=Path(handle.name)
            torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,path)
        directory=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


def save_boundary(model, model_state, task, config, state, generator, provenance, path):
    payload=dict(schema_version=1,kind='task_boundary_before_reporting',task_id=task,
                 next_action='report_completed_task_then_activate_next',source_git=config['git'],
                 config=config,model=model_state,model_training=model.training,
                 requires_grad={name:p.requires_grad for name,p in model.named_parameters()},
                 training_state=state,rng=rng_state(),
                 sampler_generator=generator.get_state() if generator is not None else None,
                 provenance=provenance)
    atomic_save(payload,path)
    return payload
