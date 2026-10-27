"""Verified continuation of complete checkpoints from the SAME student run."""
import ast
from pathlib import Path
import subprocess

import torch

from .export_compact_test_scores import _sha256
from .run_historical_b32_test_attribution import read, write
from .runner import tensor_state_digest
from .task_recovery import atomic_save

CORE_FILES=(
    'multi_lane/track_a/runner.py','multi_lane/track_a/model.py',
    'multi_lane/track_a/fusion_function_constraint.py','multi_lane/track_a/adapter.py',
    'multi_lane/track_a/cross_view_attention.py','multi_lane/track_a/view_fusion.py',
    'multi_lane/track_a/paired_transforms.py','multi_lane/track_a/head_anchored_pilot.py',
    'multi_lane/continual_datasets/continual_datasets.py',
)


def compatible_config(old,new):
    ignored={'git','resume_origin'}
    a={k:v for k,v in old.items() if k not in ignored}
    b={k:v for k,v in new.items() if k not in ignored}
    if a!=b:
        raise ValueError('Recovery changes training configuration')
    if old['arm']!=new['arm']:
        raise ValueError('Cannot recover from a different student arm')


def training_ast(source):
    tree=ast.parse(source)
    stable=[]
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in (
                'configure_phase','student_template','construct_teachers','initial_core'):
            stable.append(ast.dump(node,include_attributes=False))
        if isinstance(node,ast.FunctionDef) and node.name=='train':
            for child in ast.walk(node):
                if isinstance(child,ast.Call) and isinstance(child.func,ast.Name) and child.func.id=='train_task':
                    stable.append(ast.dump(child,include_attributes=False))
                if isinstance(child,ast.FunctionDef) and child.name=='predict':
                    stable.append(ast.dump(child,include_attributes=False))
    return stable


def verify_training_code(old_git):
    root=Path(__file__).resolve().parents[2]
    changed=subprocess.check_output(['git','diff','--name-only',old_git,'--','multi_lane/'],
        cwd=root,text=True).splitlines()
    allowed={'multi_lane/track_a/self_trained_view_transfer.py',
             'multi_lane/track_a/run_self_trained_view_transfer.py',
             'multi_lane/track_a/self_trained_recovery.py'}
    if not set(changed)<=allowed:
        raise ValueError('Recovery changes additional model/training code: '+str(changed))
    rows={}
    for name in CORE_FILES:
        old=subprocess.check_output(['git','show',old_git+':'+name],cwd=root)
        if old!=(root/name).read_bytes():
            raise ValueError('Training computation changed: '+name)
        rows[name]='byte_identical'
    name='multi_lane/track_a/self_trained_view_transfer.py'
    old=subprocess.check_output(['git','show',old_git+':'+name],cwd=root,text=True)
    if training_ast(old)!=training_ast((root/name).read_text()):
        raise ValueError('Student construction, optimization calls, or teacher target changed')
    return dict(core_files=rows,changed_package_files=changed,
        student_training_calls_and_initialization_AST_exact=True)


def import_own_recovery(source,output,config):
    """Repackage metadata, preserving complete same-arm tensors/RNG/optimizer.

    source_git denotes the current checkpoint packaging/runtime; the original
    weight-training git and source file SHA remain explicit in recovery_origin.
    No view, layer, or head is selected from another trained model.
    """
    source=Path(source);output=Path(output)
    old=read(source/'config.json');compatible_config(old,config)
    code=verify_training_code(old['git']['commit'])
    marker=output/'resume_import.json'
    if marker.exists():
        audit=read(marker)
        assert audit['source_config_sha256']==_sha256(source/'config.json')
        return audit
    rows=[]
    for directory in ('compact_checkpoints','recovery_checkpoints','warmup_checkpoints',
                      'warmup_recovery','trained_checkpoints','trained_recovery'):
        for src in sorted((source/directory).glob('task*.pth')):
            payload=torch.load(src,map_location='cpu',weights_only=False)
            assert payload['source_git']==old['git']
            before=tensor_state_digest(payload['model'])
            if 'config' in payload:
                assert payload['config']==old
            changed=dict(payload,source_git=config['git'],recovery_origin=dict(
                path=str(src),sha256=_sha256(src),weight_training_git=payload['source_git'],
                model_sha256=before,complete_own_student_state=True))
            if 'config' in changed:changed['config']=config
            dst=output/directory/src.name
            if not dst.exists():atomic_save(changed,dst)
            actual=torch.load(dst,map_location='cpu',weights_only=False)
            assert actual['source_git']==config['git'] and tensor_state_digest(actual['model'])==before
            if 'rng' in payload:
                assert torch.equal(actual['rng']['torch_cpu'],payload['rng']['torch_cpu'])
                assert all(torch.equal(a,b) for a,b in zip(actual['rng']['torch_cuda'],payload['rng']['torch_cuda']))
            rows.append(dict(source=str(src),source_sha256=_sha256(src),target=str(dst),
                model_sha256=before,complete_state_preserved=True))
    assert rows and (output/'recovery_checkpoints/task0.pth').exists()
    audit=dict(status='verified',source_config_sha256=_sha256(source/'config.json'),
        original_training_git=old['git'],recovery_runtime_git=config['git'],
        identical_hyperparameters=True,training_code=code,files=rows,
        external_teacher_parameters_imported=False,full_same_student_states_only=True)
    write(marker,audit)
    return audit
