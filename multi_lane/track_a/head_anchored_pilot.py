"""Extra task-local adaptation from archived, independently trained B0 stages.

Only validation is opened. Teacher coordinates and random streams are isolated
from student updates. Old student lanes are retained when the next B0 stage is
loaded; the next teacher is never initialized from a student.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .historical_b32_backbone_attribution import build, component, method_state
from .historical_b32_modules import modes, validate_config
from .paired_transforms import ThreeViewTransform, move_model_inputs
from .run_historical_b32_test_attribution import MODEL_HEAD, read, write
from .runner import (build_optimizer_groups, build_transforms, compact_model_state_dict,
                     dataset_view, git_metadata, resolve_dataset_parent, seen_indices,
                     set_seed, tensor_state_digest, train_task, validate_classes)
from .strong_full_evaluation import evaluate_fused_and_views
from .task_recovery import atomic_save, rng_state, restore_rng, save_boundary

ARMS = ('HEAD_FIXED_NO_ATTENTION', 'HEAD_FIXED_ATTENTION',
        'BASE_FIXED_ATTENTION', 'HEAD_TRAINABLE_ATTENTION')
TASKS = (0, 1, 2)


def rng_digest(state):
    digest = hashlib.sha256(pickle.dumps((state['python'], state['numpy'])))
    for value in (state['torch_cpu'], *state['torch_cuda']):
        digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def adaptation_rng(seed):
    """Extra training/reporting cannot advance the independent B0 RNG."""
    saved = rng_state()
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng(saved)
        assert rng_digest(rng_state()) == rng_digest(saved)


def load_initial_teacher(model, teacher):
    result = model.load_state_dict(teacher, strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys and all(k.startswith(('visual_encoder.', 'cross_view_bank.'))
                                      for k in result.missing_keys)


def task_slices(state, task, sizes, current):
    """Views of only current or strictly earlier task-owned coordinates."""
    start, stop = sum(sizes[:task]), sum(sizes[:task+1])
    selected = {}
    for key, value in state.items():
        group = component(key)
        if group == 'S': value = value[task:task+1] if current else value[:task]
        elif group == 'P': value = value[:, task:task+1] if current else value[:, :task]
        elif group == 'H': value = value[start:stop] if current else value[:start]
        elif group == 'A' or key.startswith('cross_view_bank.tasks.'):
            index = int(key.split('.')[2])
            if (index != task if current else index >= task): continue
        else: continue
        selected[key] = value
    return selected


def install_current_teacher(model, teacher, task):
    """Do not call activate_task: its previous-student S/P copy is forbidden."""
    state = method_state(model)
    old_before = tensor_state_digest(task_slices(state, task, model._task_sizes, False))
    own = task_slices(state, task, model._task_sizes, True)
    reference = task_slices(teacher, task, model._task_sizes, True)
    with torch.no_grad():
        for key, value in reference.items(): own[key].copy_(value)
    assert old_before == tensor_state_digest(task_slices(state, task, model._task_sizes, False))
    assert all(torch.equal(own[k].cpu(), v) for k, v in reference.items())
    model.restore_task(task)
    return dict(current_teacher_coordinates_exact=True, old_student_coordinates_preserved=True,
                teacher_current_sha256=tensor_state_digest(reference),
                previous_student_sha256=old_before)


def configure_student(model, arm, task):
    if arm not in ARMS: raise ValueError(arm)
    model.restore_task(task)
    model.requires_grad_(False)
    if arm != 'BASE_FIXED_ATTENTION':
        model.selectors.requires_grad_(True)
        for prompt in model.prompts: prompt.requires_grad_(True)
        model.adapter_bank.task_adapters[task].requires_grad_(True)
    if arm != 'HEAD_FIXED_NO_ATTENTION':
        model.cross_view_bank.tasks[task].requires_grad_(True)
    if arm == 'HEAD_TRAINABLE_ATTENTION': model.head.requires_grad_(True)
    model.assert_visual_frozen()


def student_optimizer_groups(**kwargs):
    main, adapter, groups = build_optimizer_groups(**kwargs)
    main = [p for p in main if p.requires_grad]
    adapter = [p for p in adapter if p.requires_grad]
    groups = [dict(g, params=[p for p in g['params'] if p.requires_grad]) for g in groups]
    groups = [g for g in groups if g['params']]
    assert groups and {id(p) for p in main+adapter} == {
        id(p) for g in groups for p in g['params']}
    assert len({id(p) for g in groups for p in g['params']}) == sum(len(g['params']) for g in groups)
    return main, adapter, groups


class AuditedLoader:
    """Record actual consumed CPU batches without an extra sampler iteration."""
    def __init__(self, loader):
        self.loader, self.generator, self.epochs = loader, loader.generator, []

    def __len__(self): return len(self.loader)

    def __iter__(self):
        trace = []; self.epochs.append(trace)
        for index, batch in enumerate(self.loader):
            if index < 2:
                images, targets, identifiers = batch
                trace.append(dict(sample_ids=list(identifiers), targets_sha256=tensor_state_digest({'y':targets}),
                                  pixels_sha256=tensor_state_digest(images)))
            yield batch[:2]


def source_train(config, args):
    transform, _ = build_transforms(config['input_normalization'], config['train_crop_scale'])
    paired = ThreeViewTransform(train=True, normalization=config['input_normalization'],
        crop_scale=config['train_crop_scale'], margin=config['person_crop_margin'],
        jitter_strength=config['person_color_jitter_strength'],
        jitter_probability=config['person_color_jitter_probability'],
        full_crop_mode=config['full_crop_mode'])
    return EMOTIC(str(resolve_dataset_parent(args.data_root)), train=True, transform=transform,
                  input_mode='full', person_crop_margin=config['person_crop_margin'],
                  multi_view_transform=paired, face_manifest_root=args.face_manifest_root)


def source_checkpoint(args, config, task):
    path = args.source_run/'compact_checkpoints'/f'task{task}.pth'
    payload = _torch_load(path)
    assert payload['task_id'] == task and payload['source_git'] == config['git']
    audit = read(args.source_run.parent/'training_verification.json')
    assert _sha256(path) == audit['checkpoint_sha256'][str(task)]
    return payload['model']


def assert_scores_equal(actual, reference, current=None):
    with np.load(actual) as a, np.load(reference) as b:
        for key in ('sample_ids', 'targets', 'class_indices'):
            assert np.array_equal(a[key], b[key]), key
        for key in ('logits', 'probabilities'):
            x, y = a[key], b[key]
            if current is not None: x, y = x[:64, current], y[:64, current]
            assert np.array_equal(x, y), (key, float(np.abs(x-y).max()))


def evaluate_stage(model, validation, args, task, output, mode):
    loader = DataLoader(dataset_view(validation, seen_indices(task), include_sample_id=True),
                        batch_size=64, shuffle=False, num_workers=2, pin_memory=False,
                        generator=torch.Generator().manual_seed(87001+task))
    model.eval()
    with torch.no_grad():
        metric, views = evaluate_fused_and_views(model, loader, torch.device('cuda'), task, .5, True,
            output/mode/'scores'/f'task{task}.npz', output/mode/'view_scores'/f'task{task}.npz')
    write(output/mode/f'task{task}.json', dict(metrics=asdict(metric), views=views))
    return asdict(metric)


def worker(args):
    start = time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert torch.get_num_threads() == 72 and torch.cuda.is_available()
    git = git_metadata(Path(__file__).resolve().parents[2]); assert not git['dirty']
    baseline_config = read(args.source_run/'config.json')
    assert validate_config(baseline_config) == 'BASE'
    assert baseline_config['git']['commit'] == MODEL_HEAD and not baseline_config['git']['dirty']
    assert read(args.source_run/'seed_summary.json')['status'] == 'complete'
    set_seed(1, True)
    architecture = dict(baseline_config)
    if args.arm not in ('BASE', 'HEAD_FIXED_NO_ATTENTION'):
        architecture.update(historical_b32_module_profile='ATTENTION_BOUND',
            view_fusion_learning_rate=.0004, **modes('ATTENTION_BOUND'))
    model = build(architecture, args.clip_checkpoint)
    load_initial_teacher(model, source_checkpoint(args, baseline_config, 0))
    # Store actual construction arguments, not B0's derived inactive-M counts
    # or its 30-epoch optimizer metadata as if they described the student.
    architecture_keys = ('num_selectors', 'num_prompts', 'num_prompt_layers',
        'selector_mode', 'prompt_mode', 'normalize', 'adapter_mode',
        'adapter_bottleneck_dim', 'adapter_layer_indices', 'adapter_residual_scale',
        'adapter_activation', 'adapter_task_initialization', 'view_fusion',
        'view_classifier_mode', 'cross_view_mode', 'cross_view_layer',
        'cross_view_dim', 'cross_view_heads', 'cross_view_scale', 'cross_view_residual_cap')
    actual_architecture = {k:architecture[k] for k in architecture_keys}
    actual_architecture.update(task_sizes=list(model._task_sizes), spatial_summary_mode='disabled',
        parax_mode='disabled', attention_parameters_per_task=(
            model.cross_view_bank.parameter_count_per_task() if model.cross_view_bank is not None else 0))
    config = dict(schema_version=1, experiment='head_anchored_extra_adaptation', arm=args.arm,
        git=git, teacher_git=baseline_config['git'], teacher_source=str(args.source_run),
        teacher_architecture=baseline_config, student_architecture=actual_architecture,
        seed=1, tasks=list(TASKS), split='val', test_loaded=False, historical_test_exploration=True,
        epochs=5, learning_rate=.0004, batch_size=64, workers=2, amp=True, tf32=True,
        optimizer='Adam_reset_per_task', scheduler='cosine', weight_decay=0.,
        view_auxiliary_loss_weight=.1, evaluation_scope='samples_intersect_seen_classes',
        adaptation_rng_seeds={str(t):20261025+1009*t for t in TASKS},
        train_sampler_seeds={str(t):317001+t for t in TASKS},
        amp_initial_scale=65536., amp_growth_interval=2000, training_loss_mode='legacy_full_zero',
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                     cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(),
                     omp_threads=os.environ.get('OMP_NUM_THREADS'), mkl_threads=os.environ.get('MKL_NUM_THREADS')),
        smoke=args.smoke, smoke_updates=2 if args.smoke else None,
        teacher_checkpoint_sha256={str(t):_sha256(args.source_run/'compact_checkpoints'/f'task{t}.pth') for t in TASKS},
        data_root=str(args.data_root), clip_checkpoint=str(args.clip_checkpoint),
        face_manifest_root=str(args.face_manifest_root))
    output = args.output
    if (output/'config.json').exists(): assert read(output/'config.json') == config
    else: write(output/'config.json', config)
    # Only TRAIN and VAL datasets are instantiated. No TEST paths are opened.
    train = None if args.arm == 'BASE' else source_train(baseline_config, args)
    validation = make_source(baseline_config, args.data_root, args.face_manifest_root, 'val')
    validate_classes(validation)
    teacher_history = read(args.source_run/'training_history.json')
    frozen_visual = tensor_state_digest(model.visual_encoder.state_dict())
    history, audits, metrics, off_metrics = {}, {}, [], []
    for task in TASKS:
        boundary = output/'recovery_checkpoints'/f'task{task}.pth'
        before_rng = rng_digest(rng_state())
        with adaptation_rng(20261025+1009*task):
            if boundary.exists():
                # Trusted, locally generated recovery contains NumPy/Python RNG.
                recovery = torch.load(boundary, map_location='cpu', weights_only=False)
                assert recovery['config'] == config and recovery['task_id'] == task
                result = model.load_state_dict(recovery['model'], strict=False)
                assert not result.unexpected_keys and all(k.startswith('visual_encoder.') for k in result.missing_keys)
                model.restore_task(task); restore_rng(recovery['rng'])
                history = recovery['provenance']['history']; audits = recovery['provenance']['audits']
                print(f'RESUME task={task} repeated_updates=0', flush=True)
            else:
                teacher = source_checkpoint(args, baseline_config, task)
                audits[str(task)] = install_current_teacher(model, teacher, task)
                # Same cumulative-lane GEMM shapes as the archived validation.
                probe_loader = DataLoader(dataset_view(validation, seen_indices(task), True),
                    batch_size=64, num_workers=0, shuffle=False,
                    generator=torch.Generator().manual_seed(87001+task))
                images, labels, ids = next(iter(probe_loader))
                model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast():
                    probe, _ = model.seen_logits_with_views(move_model_inputs(images, torch.device('cuda')))
                indices = list(range(sum(model._task_sizes[:task]), sum(model._task_sizes[:task+1])))
                with np.load(args.source_run.parent/'historical_val/scores'/f'task{task}.npz') as saved:
                    assert np.array_equal(np.asarray(ids), saved['sample_ids'][:64])
                    assert np.array_equal(probe.float().cpu().numpy()[:, indices], saved['logits'][:64, indices])
                audits[str(task)]['initial_current_predictions_exact'] = True
                state = method_state(model)
                previous_hash = tensor_state_digest(task_slices(state, task, model._task_sizes, False))
                fixed_head_hash = tensor_state_digest(model.head.state_dict())
                fixed_base_hash = tensor_state_digest({k:v for k,v in state.items() if component(k)})
                training_state = {}
                if args.arm != 'BASE':
                    configure_student(model, args.arm, task)
                    view = dataset_view(train, range(sum(model._task_sizes[:task]), sum(model._task_sizes[:task+1])), True)
                    loader = AuditedLoader(DataLoader(view, batch_size=64, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=False, generator=torch.Generator().manual_seed(317001+task)))
                    assert len(loader) == int(teacher_history[str(task)][0]['optimizer_steps'])
                    history[str(task)] = train_task(model, loader, None, torch.device('cuda'), task,
                        epochs=5, learning_rate=.0004, weight_decay=0., temperature=1., amp=True,
                        adapter_learning_rate=.0004, adapter_weight_decay=0.,
                        view_fusion_learning_rate=.0004, view_auxiliary_loss_weight=.1,
                        loss_routing='joint_bce' if args.arm == 'BASE_FIXED_ATTENTION' else 'adapter_asl',
                        optimizer_updates_per_task=2 if args.smoke else None,
                        recovery_state=training_state, optimizer_group_builder=student_optimizer_groups)
                    audits[str(task)].update(train_instances=len(view), batches_per_epoch=len(loader),
                        batch_trace=loader.epochs,
                        optimizer_parameter_names=training_state['optimizer_parameter_names'])
                    successful = sum(int(r['optimizer_steps']) for r in history[str(task)])
                    skipped = sum(int(r['skipped_optimizer_steps']) for r in history[str(task)])
                    assert successful == (2 if args.smoke else 5*len(loader)) and skipped == 0
                    if args.arm != 'HEAD_TRAINABLE_ATTENTION':
                        assert fixed_head_hash == tensor_state_digest(model.head.state_dict())
                    if args.arm == 'BASE_FIXED_ATTENTION':
                        assert fixed_base_hash == tensor_state_digest({k:v for k,v in state.items() if component(k)})
                    sampler = loader.generator
                else: history[str(task)] = []; sampler = None
                assert previous_hash == tensor_state_digest(task_slices(state, task, model._task_sizes, False))
                assert frozen_visual == tensor_state_digest(model.visual_encoder.state_dict())
                audits[str(task)].update(old_student_parameters_unchanged=True, frozen_visual_unchanged=True,
                    head_frozen=args.arm != 'HEAD_TRAINABLE_ATTENTION',
                    source_checkpoint_unchanged=_sha256(args.source_run/'compact_checkpoints'/f'task{task}.pth') == config['teacher_checkpoint_sha256'][str(task)])
                compact = compact_model_state_dict(model)
                atomic_save(dict(schema_version=1, task_id=task, source_git=git, model=compact),
                            output/'compact_checkpoints'/f'task{task}.pth')
                save_boundary(model, compact, task, config, training_state, sampler,
                              dict(history=history, audits=audits), boundary)
                print(f'CHECKPOINT_BEFORE_VALIDATION task={task}', flush=True)
            metrics.append(evaluate_stage(model, validation, args, task, output, 'on'))
            if args.arm == 'BASE':
                assert_scores_equal(output/'on/scores'/f'task{task}.npz',
                                    args.source_run.parent/'historical_val/scores'/f'task{task}.npz')
            elif model.cross_view_bank is not None:
                model.cross_view_runtime_enabled = False
                try: off_metrics.append(evaluate_stage(model, validation, args, task, output, 'off'))
                finally: model.cross_view_runtime_enabled = True
                if args.arm == 'BASE_FIXED_ATTENTION':
                    assert_scores_equal(output/'off/scores'/f'task{task}.npz',
                                        args.source_run.parent/'historical_val/scores'/f'task{task}.npz')
            write(output/'training_history.json', history); write(output/'audits.json', audits)
        assert before_rng == rng_digest(rng_state())
        print(f'TASK_COMPLETE arm={args.arm} task={task} validation_mAP={metrics[-1]["mAP"]:.6f}', flush=True)
    assert frozen_visual == tensor_state_digest(model.visual_encoder.state_dict())
    assert all(_sha256(args.source_run/'compact_checkpoints'/f'task{t}.pth') == config['teacher_checkpoint_sha256'][str(t)] for t in TASKS)
    write(output/'result.json', dict(status='complete', config=config, task_metrics=metrics,
        off_task_metrics=off_metrics, pilot_final_mAP=metrics[-1]['mAP'],
        pilot_average_mAP=float(np.mean([m['mAP'] for m in metrics])), full_eight_stage_average=False,
        rng_isolation_verified=True, elapsed_seconds=time.time()-start,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--clip-checkpoint', type=Path, required=True)
    p.add_argument('--face-manifest-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=('BASE', *ARMS), required=True)
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__': worker(parser().parse_args())
