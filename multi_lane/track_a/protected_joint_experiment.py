"""Fresh single-model protected interaction training and historical evaluation."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .historical_b32_backbone_attribution import build as historical_build, component, method_state
from .protected_joint_attention import (ARMS, ProtectedJointModel, PrivateInteractionTraining)
from .historical_b32_modules import validate_config
from .historical_reporting_rng import replay_reporting_rng, verify_historical_boundary
from .paired_transforms import ThreeViewTransform, move_model_inputs
from .run_historical_b32_test_attribution import read, write
from .runner import (TaskMetrics, build_optimizer_groups, build_transforms,
    compact_model_state_dict, dataset_view, git_metadata, resolve_dataset_parent,
    seen_indices, set_seed, summarize_tasks, tensor_state_digest, train_task)
from .task_recovery import atomic_save, restore_rng, rng_state, save_boundary

def build(config, clip, arm):
    from .person_grounded_summary import GROUNDED_ARMS, PersonGroundedModel
    from .protected_attention_dose import DOSE_ARMS, DoseMatchedModel
    from .private_summarize_attention import SUMMARY_ARMS, PrivateSummaryModel
    model = historical_build(config, clip,
        model_class=PersonGroundedModel if arm in GROUNDED_ARMS else PrivateSummaryModel if arm in SUMMARY_ARMS else DoseMatchedModel if arm in DOSE_ARMS else ProtectedJointModel)
    model.initialize_interaction(arm)
    return model


def verify_base_boundary(model, history, reference, task):
    from types import SimpleNamespace
    proxy = SimpleNamespace(state_dict=lambda: {
        k:v for k,v in model.state_dict().items() if not k.startswith('interaction_bank.')})
    return verify_historical_boundary(proxy, history, reference, task)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('Value must be a positive integer')
    return result


def smoke_update_budget(args):
    if not args.smoke:
        return None
    updates = int(args.smoke_updates)
    if updates <= 0:
        raise ValueError('Smoke update count must be positive')
    return updates


def runtime():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.get_num_threads() != 72:
        raise ValueError('Historical initialization requires CUDA_VISIBLE_DEVICES=0 and 72 model threads')
    if not torch.cuda.is_available():
        raise RuntimeError('This entry point requires the remote CUDA runtime')
    result = git_metadata(Path(__file__).resolve().parents[2])
    if result['dirty']:
        raise ValueError('Run only from a clean, committed experiment worktree')
    return result


def validate_target_tasks(tasks, smoke=False):
    if smoke:
        return [0, 1]
    if tasks not in (1, 3, 8):
        raise ValueError('Formal stages are task0, task0-2 or all eight tasks')
    return list(range(tasks))


@contextmanager
def preserved_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def task_slices(state, task, sizes, current=False):
    start, stop = sum(sizes[:task]), sum(sizes[:task+1])
    result = {}
    for key, value in state.items():
        group = component(key)
        if group == 'S':
            value = value[task:task+1] if current else value[:task]
        elif group == 'P':
            value = value[:, task:task+1] if current else value[:, :task]
        elif group == 'H':
            value = value[start:stop] if current else value[:start]
        elif group == 'A':
            index = int(key.split('.')[2])
            if (index != task if current else index >= task):
                continue
        elif key.startswith('interaction_bank.'):
            index = int(key.split('.')[1])
            if (index != task if current else index >= task):
                continue
        else:
            continue
        result[key] = value
    return result


def configure_current(model, task):
    model.restore_task(task)
    model.requires_grad_(False)
    for parameter in model.parameters():
        parameter.grad = None
    model.selectors.requires_grad_(True)
    for parameter in model.prompts:
        parameter.requires_grad_(True)
    model.adapter_bank.task_adapters[task].requires_grad_(True)
    model.head.requires_grad_(True)
    if model.cross_view_bank is not None or model.spatial_summary_bank is not None:
        raise ValueError('Teacher-free first version does not add Attention or spatial summary')
    model.cross_view_runtime_enabled = False
    model.configure_interaction()
    model.assert_visual_frozen()


def optimizer_groups(**kwargs):
    main, adapter, groups = build_optimizer_groups(**kwargs)
    main = [p for p in main if p.requires_grad]
    adapter = [p for p in adapter if p.requires_grad]
    groups = [dict(g, params=[p for p in g['params'] if p.requires_grad]) for g in groups]
    groups = [g for g in groups if g['params']]
    if {id(p) for p in main + adapter} != {id(p) for g in groups for p in g['params']}:
        raise AssertionError('Trainable parameter routing is incomplete')
    return main, adapter, groups


class AuditedLoader:
    def __init__(self, loader, epochs=None):
        self.loader = loader
        self.generator = loader.generator
        self.epochs = list(epochs or [])

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        trace = []
        self.epochs.append(trace)
        for index, batch in enumerate(self.loader):
            if index < 2:
                images, targets, identifiers = batch
                trace.append(dict(sample_ids=list(identifiers),
                    targets_sha256=tensor_state_digest({'y': targets}),
                    pixels_sha256=tensor_state_digest(images)))
            yield batch[:2]


def training_source(base, args):
    transform, _ = build_transforms(base['input_normalization'], base['train_crop_scale'])
    paired = ThreeViewTransform(train=True, normalization=base['input_normalization'],
        crop_scale=base['train_crop_scale'], margin=base['person_crop_margin'],
        jitter_strength=base['person_color_jitter_strength'],
        jitter_probability=base['person_color_jitter_probability'],
        full_crop_mode=base['full_crop_mode'], emit_spatial_geometry=True)
    return EMOTIC(str(resolve_dataset_parent(args.data_root)), train=True, transform=transform,
        input_mode='full', person_crop_margin=base['person_crop_margin'],
        multi_view_transform=paired, face_manifest_root=args.face_manifest_root)


def single_thread_worker(_):
    torch.set_num_threads(1)


def evaluation_loader(view, task, workers=2):
    options = dict(batch_size=64, shuffle=False, num_workers=workers,
        generator=torch.Generator().manual_seed(87001 + task))
    if workers:
        options.update(prefetch_factor=1, persistent_workers=True, pin_memory=True,
            multiprocessing_context='spawn', worker_init_fn=single_thread_worker)
    return DataLoader(view, **options)


def load_own(model, path, cfg):
    # Our atomic recovery contains Python/NumPy RNG and Adam, not just tensors.
    # Only paths produced by this own-run recovery protocol are accepted here.
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['config'] != cfg:
        raise ValueError('Own recovery configuration changed')
    missing = model.load_state_dict(saved['model'], strict=False)
    if missing.unexpected_keys or any(not k.startswith('visual_encoder.') for k in missing.missing_keys):
        raise ValueError('Own recovery omitted task parameters')
    model.restore_task(saved['task_id'])
    restore_rng(saved['rng'])
    return saved


def training_config(args, git, base, initial):
    clip_sha256 = _sha256(args.clip_checkpoint)
    if clip_sha256 != base['clip_checkpoint_sha256']:
        raise ValueError('CLIP visual checkpoint differs from the strong baseline recipe')
    config = dict(kind='protected_joint_summary_interaction', git=git,
        arm=args.arm, seed=1, smoke=args.smoke, tasks=list(range(8)),
        stages_allowed=[1, 3, 8], model_template=base,
        source_configuration=str(args.source_run/'config.json'),
        source_configuration_sha256=_sha256(args.source_run/'config.json'),
        student_initialization='standard_CLIP_visual_and_fresh_task_parameters_and_classifier',
        initial_selector_prompt_classifier_sha256=initial, external_task_parameter_loads=0,
        teacher_sources=[], teacher_outputs_used=False, classifier_policy='trainable_all_30_epochs',
        epochs_per_task=30, extra_adaptation_epochs=0, batch_size=64, learning_rate=.0125,
        adapter_learning_rate=.0004, weight_decay=0., optimizer='Adam_reset_per_task',
        scheduler='cosine_reset_per_task', loss_routing='adapter_asl',
        training_loss_mode='legacy_full_zero', view_auxiliary_loss_weight=.1,
        asl_gamma_neg=9.8, asl_gamma_pos=0., asl_clip=.05, amp=True, tf32=True,
        amp_initial_scale=65536., threads=72, workers=2, prefetch_factor=1,
        interaction=dict(layer=5, dim=64, heads=4, residual_cap=.1, fixed_gate=.1,
            learning_rate=.0004, independent_optimizer_scaler_rng=True,
            source_only='full' if args.arm=='FULL_ONLY' else 'three_views'),
        historical_reporting_rng='two_CPU_int64_draws_before_final_boundary',
        data_root=str(args.data_root), clip_checkpoint=str(args.clip_checkpoint),
        clip_checkpoint_sha256=clip_sha256,
        face_manifest_root=str(args.face_manifest_root),
        historical_reference_root=str(args.historical_reference_root) if args.historical_reference_root else None,
        evaluation_scope='samples_intersect_seen_classes', test_used_for_development=True)
    if args.smoke:
        config['smoke_updates'] = smoke_update_budget(args)
    from .protected_attention_dose import DOSE_ARMS, dose_configuration
    if args.arm in DOSE_ARMS:
        config['kind'] = 'protected_attention_dose_matched_controls'
        config['interaction'] = dose_configuration(args.arm)
    from .private_summarize_attention import SUMMARY_ARMS, summary_configuration
    if args.arm in SUMMARY_ARMS:
        config['kind']='protected_private_summary_2x2'
        config['interaction']=summary_configuration(args.arm)
    from .person_grounded_summary import GROUNDED_ARMS, grounded_configuration
    if args.arm in GROUNDED_ARMS:
        config['kind']='protected_person_grounded_summary'
        config['interaction']=grounded_configuration(args.arm)
    return config


def old_probe(model, probe_source, task):
    if task == 0:
        return None
    with preserved_rng():
        view = dataset_view(probe_source, seen_indices(task-1), True)
        view = Subset(view, range(min(32, len(view))))
        x, _, _ = next(iter(DataLoader(view, batch_size=32, num_workers=0,
            generator=torch.Generator().manual_seed(881001+task))))
        x = move_model_inputs(x, torch.device('cuda'))
        model.eval()
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                enable_mem_efficient=False), torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            before_tf32 = torch.backends.cuda.matmul.allow_tf32
            before_cudnn = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                z, vs = model.seen_logits_with_views(x)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = before_tf32
                torch.backends.cudnn.allow_tf32 = before_cudnn
        stop = sum(model._task_sizes[:task])
        return {name:value[:, :stop].float().cpu() for name,value in dict(fused=z, **vs).items()}


def compare_old_probe(before, after):
    if before is None:
        return dict(checked=False, reason='first_task')
    if set(before) != set(after):
        raise AssertionError('Old-view probe keys changed')
    differences = {k:float((v-after[k]).abs().max()) for k,v in before.items()}
    if any(v > 1e-5 for v in differences.values()):
        raise AssertionError(('old_path_prediction_drift', differences))
    return dict(checked=True, max_abs_difference=differences, absolute_tolerance=1e-5)


def train(args):
    started = time.time()
    git = runtime()
    if args.epochs != 30:
        raise ValueError('Original 30-epoch budget must be preserved')
    targets = validate_target_tasks(args.tasks, args.smoke)
    if args.historical_reference_root and args.smoke:
        raise ValueError('Historical comparisons are formal U0 only')
    base = read(args.source_run/'config.json')
    if validate_config(base) != 'BASE':
        raise ValueError('Start from the historical strong BASE recipe')
    set_seed(1, True)
    model = build(base, args.clip_checkpoint, args.arm)
    initial = tensor_state_digest({k:v for k,v in method_state(model).items()
        if k == 'selectors' or k.startswith(('prompts.', 'head.'))})
    if initial != base['initial_selector_prompt_classifier_sha256']:
        raise AssertionError('Fresh initialization did not reproduce the historical recipe')
    cfg = training_config(args, git, base, initial)
    if cfg['kind'] in ('protected_attention_dose_matched_controls','protected_private_summary_2x2','protected_person_grounded_summary'):
        cfg['interaction']['parameters_per_task'] = [sum(p.numel() for p in m.parameters())
                                                   for m in model.interaction_bank]
        cfg['method_parameter_count'] = sum(p.numel() for n,p in model.named_parameters()
                                            if not n.startswith('visual_encoder.'))
    if (args.output/'training_complete.json').exists():
        previous=read(args.output/'training_complete.json')
        if len(previous['tasks']) > len(targets):
            raise ValueError('Cannot replace a longer completed prefix with a shorter stage')
    config_path = args.output/'config.json'
    if config_path.exists():
        if read(config_path) != cfg:
            raise ValueError('Continuation changed the predeclared training configuration')
    else:
        write(config_path, cfg)
    source = training_source(base, args)
    probe_source = make_source(base, args.data_root, args.face_manifest_root, 'train')
    probe_source.multi_view_transform.emit_spatial_geometry = True
    visual = tensor_state_digest(model.visual_encoder.state_dict())
    history, audits = {}, {}
    for task in targets:
        final = args.output/'recovery_checkpoints'/f'task{task}.pth'
        trained = args.output/'before_reporting_recovery'/f'task{task}.pth'
        epoch_root = args.output/'epoch_recovery'/f'task{task}'
        if final.exists():
            saved = load_own(model, final, cfg)
            history, audits = saved['provenance']['history'], saved['provenance']['audits']
            print('RESUME_OWN_COMPLETED_TASK', args.arm, task, flush=True)
            continue
        epoch_paths = sorted(epoch_root.glob('epoch*.pth'),
            key=lambda p:int(p.stem.removeprefix('epoch')))
        resume = None
        if trained.exists():
            saved = load_own(model, trained, cfg)
            history, audits = saved['provenance']['history'], saved['provenance']['audits']
            state = saved['training_state']
            old_before = saved['provenance']['old_parameter_sha256']
            probe_before = saved['provenance']['old_probe']
        else:
            if epoch_paths:
                saved = load_own(model, epoch_paths[-1], cfg)
                history, audits = saved['provenance']['history'], saved['provenance']['audits']
                resume = saved['training_state']
                old_before = saved['provenance']['old_parameter_sha256']
                probe_before = saved['provenance']['old_probe']
                prior_batches = saved['provenance']['batch_traces']
                print('RESUME_OWN_EPOCH', args.arm, task, resume['next_epoch'], flush=True)
            else:
                model.activate_task(task)
                old_before = tensor_state_digest(task_slices(method_state(model), task, model._task_sizes))
                probe_before = old_probe(model, probe_source, task)
                prior_batches = []
            configure_current(model, task)
            current = range(sum(model._task_sizes[:task]), sum(model._task_sizes[:task+1]))
            loader = AuditedLoader(DataLoader(dataset_view(source, current, True),
                batch_size=64, shuffle=True, num_workers=2, prefetch_factor=1,
                pin_memory=True, drop_last=False), prior_batches)
            state = {}
            private = (PrivateInteractionTraining(model, amp=True)
                       if model.interaction_bank else None)
            def epoch_callback(next_epoch, rows, training_state):
                with preserved_rng():
                    provenance = dict(history=dict(history, **{str(task):rows}), audits=audits,
                        old_parameter_sha256=old_before, old_probe=probe_before,
                        batch_traces=loader.epochs, next_epoch=next_epoch)
                    save_boundary(model, compact_model_state_dict(model), task, cfg, training_state,
                        None, provenance, epoch_root/f'epoch{next_epoch:02d}.pth')
                    write(args.output/'training_history.json', dict(history, **{str(task):rows}))
                    write(args.output/'progress.json', dict(status='training', arm=args.arm,
                        task=task, completed_epochs=next_epoch, target_epochs=30,
                        epoch_checkpoint=str(epoch_root/f'epoch{next_epoch:02d}.pth')))
            task_training_started = time.monotonic()
            rows = train_task(model, loader, None, torch.device('cuda'), task, 30, .0125, 0., 1., True,
                adapter_learning_rate=.0004, adapter_weight_decay=0.,
                view_fusion_learning_rate=.0004, view_auxiliary_loss_weight=.1,
                loss_routing='adapter_asl', optimizer_updates_per_task=smoke_update_budget(args),
                optimizer_group_builder=optimizer_groups, recovery_state=state,
                private_interaction=private,
                epoch_callback=epoch_callback, resume_training_state=resume)
            if any(row['skipped_optimizer_steps'] != 0 or row.get('interaction_skipped_steps',0) != 0 for row in rows):
                raise AssertionError('Nonzero AMP skipped steps')
            session_training_seconds = time.monotonic()-task_training_started
            expected = smoke_update_budget(args) if args.smoke else 30*len(loader)
            if sum(row['optimizer_steps'] for row in rows) != expected:
                raise AssertionError('Training budget changed')
            history[str(task)] = rows
            audits[str(task)] = dict(old_parameter_sha256=old_before, batch_traces=loader.epochs,
                train_instances=len(loader.loader.dataset), optimizer_parameter_names=state['optimizer_parameter_names'],
                external_task_parameter_loads=0, head_trainable_all_epochs=True,
                training_elapsed_seconds=sum(float(row['elapsed_seconds']) for row in rows),
                current_session_training_wall_seconds=session_training_seconds,
                resumed_from_epoch=int(resume['next_epoch']) if resume is not None else 0,
                completed_optimizer_updates=int(sum(row['optimizer_steps'] for row in rows)),
                epoch_recovery_count=len(list(epoch_root.glob('epoch*.pth'))))
            save_boundary(model, compact_model_state_dict(model), task, cfg, state, None,
                dict(history=history, audits=audits, old_parameter_sha256=old_before,
                    old_probe=probe_before), trained)
            print('TRAINING_SAVED_BEFORE_REPORTING', args.arm, task, flush=True)
        if args.historical_reference_root:
            audits[str(task)]['historical_reference_verification'] = verify_base_boundary(
                model, history[str(task)], args.historical_reference_root, task)
        if old_before != tensor_state_digest(task_slices(method_state(model), task, model._task_sizes)):
            raise AssertionError('Old task parameters changed')
        if visual != tensor_state_digest(model.visual_encoder.state_dict()):
            raise AssertionError('Frozen CLIP changed')
        probe_after = old_probe(model, probe_source, task)
        audits[str(task)].update(old_task_parameters_exact=True, frozen_visual_exact=True,
            old_TRAIN_probe=compare_old_probe(probe_before,probe_after),
            head_source='this_single_jointly_trained_model',
            historical_reporting_rng=replay_reporting_rng(task))
        compact = compact_model_state_dict(model)
        path = args.output/'compact_checkpoints'/f'task{task}.pth'
        if path.exists():
            old = _torch_load(path)
            if not all(torch.equal(v,old['model'][k]) for k,v in compact.items()):
                raise AssertionError('Existing compact checkpoint differs')
        else:
            atomic_save(dict(schema_version=1, task_id=task, source_git=git, model=compact), path)
        save_boundary(model, compact, task, cfg, state, None,
            dict(history=history, audits=audits), final)
        write(args.output/'training_history.json', history)
        write(args.output/'audits.json', audits)
        print('PROTECTED_INTERACTION_TASK_SAVED', args.arm, task, flush=True)
    write(args.output/'training_history.json', history)
    write(args.output/'audits.json', audits)
    write(args.output/'training_complete.json', dict(status='complete', config=cfg,
        tasks=targets, checkpoints={str(t):dict(sha256=_sha256(args.output/'compact_checkpoints'/f'task{t}.pth'))
            for t in targets}, elapsed_seconds=time.time()-started,
        completed_optimizer_updates=sum(row['optimizer_steps'] for t in targets for row in history[str(t)]),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        external_task_parameter_loads=0, teacher_required=False))


def evaluate(args):
    start = time.time()
    git = runtime()
    set_seed(1, True)
    cfg = read(args.training_root/'config.json')
    complete = read(args.training_root/'training_complete.json')
    if complete['status'] != 'complete' or cfg != complete['config'] or cfg['git'] != git:
        raise ValueError('Incomplete or changed own training source')
    if cfg['smoke'] and args.split == 'test':
        raise ValueError('Smoke is implementation validation, not Test evaluation')
    constants=None
    if cfg['kind'] in ('protected_private_summary_2x2','protected_person_grounded_summary'):
        from .private_summary_evaluation import locked_constants
        constants=locked_constants(args.training_root)
    model = build(cfg['model_template'], Path(cfg['clip_checkpoint']), cfg['arm'])
    source = make_source(cfg['model_template'], Path(cfg['data_root']),
        Path(cfg['face_manifest_root']), args.split)
    source.multi_view_transform.emit_spatial_geometry = True
    results, off_results, audits = [], [], []
    tasks = complete['tasks']
    for task in tasks:
        path = args.training_root/'compact_checkpoints'/f'task{task}.pth'
        if _sha256(path) != complete['checkpoints'][str(task)]['sha256']:
            raise AssertionError('Own checkpoint SHA changed')
        payload = _torch_load(path)
        if payload['source_git'] != cfg['git'] or payload['task_id'] != task:
            raise AssertionError('Checkpoint provenance mismatch')
        missing = model.load_state_dict(payload['model'], strict=False)
        if missing.unexpected_keys or any(not k.startswith('visual_encoder.') for k in missing.missing_keys):
            raise AssertionError('Checkpoint task state missing')
        model.restore_task(task)
        model.requires_grad_(False).eval()
        model.cross_view_runtime_enabled = False
        before = tensor_state_digest(method_state(model))
        view = dataset_view(source, seen_indices(task), True)
        if cfg['smoke']:
            view = Subset(view, range(min(64, len(view))))
        loader = evaluation_loader(view, task, args.workers)
        from .protected_joint_evaluation import evaluate_cached_modes
        if cfg['kind']=='protected_person_grounded_summary':
            from .person_grounded_evaluation import evaluate_grounded_modes
            values=evaluate_grounded_modes(model,loader,task,args.output,constants,args.workers,cfg['smoke'])
        elif constants is None:
            values = evaluate_cached_modes(model, loader, task, args.output)
        else:
            from .private_summary_evaluation import evaluate_private_modes
            values=evaluate_private_modes(model,loader,task,args.output,constants,args.workers,cfg['smoke'])
        value, views = values['on']
        off_results.append(asdict(values['off'][0]))
        results.append(asdict(value))
        write(args.output/f'task{task}.json', dict(metrics=asdict(value), views=views))
        if before != tensor_state_digest(method_state(model)):
            raise AssertionError('Read-only evaluation changed task state')
        audits.append(dict(task=task, own_checkpoint_sha256=_sha256(path),
            samples=len(view), parameters_unchanged=True))
        write(args.output/'progress.json', dict(status='running', tasks=tasks, completed=audits))
        print('PROTECTED_INTERACTION_EVALUATE', cfg['arm'], args.split, task, value.mAP, flush=True)
        del loader
    write(args.output/'result.json', dict(status='complete', git=git, training_git=cfg['git'],
        arm=cfg['arm'], split=args.split, seed=1, smoke=cfg['smoke'], tasks=tasks,
        metrics=results, summary=summarize_tasks([TaskMetrics(**v) for v in results]),
        partial_sequence=len(tasks)<8, stage_average_is_not_eight_task_average=len(tasks)<8,
        teacher_loaded=False, parameter_assembly=False, audits=audits,
        evaluation_scope='samples_intersect_seen_classes', workers=args.workers,
        batch_size=64, worker_threads=1, elapsed_seconds=time.time()-start,
        off_summary=summarize_tasks([TaskMetrics(**v) for v in off_results]),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        test_used_for_development=True))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('train','eval'))
    p.add_argument('--output', type=Path, required=True)
    from .protected_attention_dose import DOSE_ARMS
    from .private_summarize_attention import SUMMARY_ARMS
    from .person_grounded_summary import GROUNDED_ARMS
    p.add_argument('--arm', choices=ARMS+DOSE_ARMS+SUMMARY_ARMS+GROUNDED_ARMS)
    p.add_argument('--source-run', type=Path)
    p.add_argument('--historical-reference-root', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--clip-checkpoint', type=Path)
    p.add_argument('--face-manifest-root', type=Path)
    p.add_argument('--training-root', type=Path)
    p.add_argument('--tasks', type=int, default=1)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--workers', type=int, choices=(0,2), default=2)
    p.add_argument('--split', choices=('val','test'), default='val')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-updates', type=positive_int, default=2,
        help='Diagnostic updates per task, only applied and recorded with --smoke')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.action == 'train':
        if args.arm is None or any(getattr(args,k) is None for k in
                ('source_run','data_root','clip_checkpoint','face_manifest_root')):
            raise ValueError('Training needs --arm and historical recipe/data/CLIP/Face paths')
        train(args)
    else:
        if args.training_root is None:
            raise ValueError('Evaluation needs --training-root')
        evaluate(args)
