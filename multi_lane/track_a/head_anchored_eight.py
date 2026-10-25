"""Continue the unchanged three-task pilot and export eight-stage Val/Test."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .head_anchored_pilot import (ARMS, AuditedLoader, adaptation_rng, configure_student,
    install_current_teacher, load_initial_teacher, source_checkpoint, source_train,
    student_optimizer_groups, task_slices, assert_scores_equal)
from .historical_b32_backbone_attribution import build, component, method_state, restore_components
from .historical_b32_modules import modes, validate_config
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read, write
from .runner import (compact_model_state_dict, dataset_view, git_metadata, seen_indices,
    set_seed, summarize_tasks, tensor_state_digest, train_task)
from .strong_full_evaluation import evaluate_fused_and_views
from .task_recovery import atomic_save, save_boundary


def runtime():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.get_num_threads() == 72
    assert torch.cuda.is_available()
    git = git_metadata(Path(__file__).resolve().parents[2]); assert not git['dirty']
    return git


def require_test_ready(split, smoke, files):
    if split=='test' and (smoke or set(map(int,files))!=set(range(8))):
        raise ValueError('Test requires complete eight-task training')


def validate_prefix(root, arm, source):
    result = read(root/'result.json'); cfg = result['config']
    assert result['status'] == 'complete' and cfg['arm'] == arm and cfg['tasks'] == [0,1,2]
    assert not cfg['smoke'] and cfg['seed'] == 1 and cfg['epochs'] == 5
    assert cfg['learning_rate'] == .0004 and cfg['batch_size'] == 64 and not cfg['git']['dirty']
    assert cfg['teacher_git'] == source['git']
    history = read(root/'training_history.json'); audit = read(root/'audits.json')
    for task in range(3):
        rows = history[str(task)]
        assert len(rows) == 5 and all(row['skipped_optimizer_steps'] == 0 for row in rows)
        assert sum(row['optimizer_steps'] for row in rows) == 5*audit[str(task)]['batches_per_epoch']
        payload = _torch_load(root/'compact_checkpoints'/f'task{task}.pth')
        assert payload['task_id'] == task and payload['source_git'] == cfg['git']
    return cfg, history, audit


def continuation(args):
    start = time.time(); git = runtime(); set_seed(1, True)
    base = read(args.source_run/'config.json'); assert validate_config(base) == 'BASE'
    prefix = args.pilot_root/args.arm
    prefix_cfg, history, audits = validate_prefix(prefix, args.arm, base)
    template = dict(base)
    if args.arm != 'HEAD_FIXED_NO_ATTENTION':
        template.update(historical_b32_module_profile='ATTENTION_BOUND',
                        view_fusion_learning_rate=.0004, **modes('ATTENTION_BOUND'))
    model = build(template, args.clip_checkpoint)
    payload = _torch_load(prefix/'compact_checkpoints/task2.pth')
    missing = model.load_state_dict(payload['model'], strict=False)
    assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
    model.restore_task(2)
    prefix_model_hash = tensor_state_digest(method_state(model))
    assert prefix_model_hash == tensor_state_digest(payload['model'])
    prefix_files = {str(t):dict(path=str(prefix/'compact_checkpoints'/f'task{t}.pth'),
        sha256=_sha256(prefix/'compact_checkpoints'/f'task{t}.pth'), source_git=prefix_cfg['git']) for t in range(3)}
    cfg = dict(kind='continued_head_anchored', arm=args.arm, seed=1, git=git,
        tasks=list(range(8)), epochs_per_task=5, learning_rate=.0004, train_batch_size=64,
        workers=2, optimizer='Adam_reset_per_task', scheduler='cosine', weight_decay=0.,
        amp=True, tf32=True, teacher_architecture=base, model_template=template,
        student_architecture=dict(prefix_cfg['student_architecture'], cross_view_face_attention=False),
        adapter_learning_rate=.0004,attention_learning_rate=.0004,view_auxiliary_loss_weight=.1,
        amp_initial_scale=65536.,amp_growth_interval=2000,
        evaluation_scope='samples_intersect_seen_classes', continued_from=str(prefix),
        prefix_git=prefix_cfg['git'], prefix_result_sha256=_sha256(prefix/'result.json'),
        prefix_checkpoints=prefix_files, prefix_task2_model_sha256=prefix_model_hash,
        adaptation_rng_seeds={str(t):20261025+1009*t for t in range(8)},
        sampler_seeds={str(t):317001+t for t in range(8)}, smoke=args.smoke,
        historical_test_exploration=True, test_used_for_parameter_search=True,
        repeated_prefix_updates=0, new_training_tasks=[3,4,5,6,7])
    if (args.output/'config.json').exists(): assert read(args.output/'config.json') == cfg
    else: write(args.output/'config.json', cfg)
    train = source_train(base, args); val = make_source(base,args.data_root,args.face_manifest_root,'val')
    baseline_history = read(args.source_run/'training_history.json')
    frozen_visual = tensor_state_digest(model.visual_encoder.state_dict())
    for task in range(3,8):
        checkpoint = args.output/'compact_checkpoints'/f'task{task}.pth'
        recovery = args.output/'recovery_checkpoints'/f'task{task}.pth'
        with adaptation_rng(20261025+1009*task):
            if recovery.exists():
                saved = torch.load(recovery,map_location='cpu',weights_only=False)
                assert saved['config'] == cfg and saved['task_id'] == task
                missing = model.load_state_dict(saved['model'],strict=False)
                assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
                model.restore_task(task)
                history, audits = saved['provenance']['history'], saved['provenance']['audits']
                print(f'RESUME_COMPLETED_TASK task={task} repeated_updates=0',flush=True)
                continue
            teacher = source_checkpoint(args,base,task)
            audits[str(task)] = install_current_teacher(model,teacher,task)
            probe = DataLoader(dataset_view(val,seen_indices(task),True),batch_size=64,
                num_workers=0,shuffle=False,generator=torch.Generator().manual_seed(87001+task))
            images,_,ids = next(iter(probe)); model.eval()
            with torch.no_grad(),torch.cuda.amp.autocast():
                scores,_ = model.seen_logits_with_views(move_model_inputs(images,torch.device('cuda')))
            indices = list(range(sum(model._task_sizes[:task]),sum(model._task_sizes[:task+1])))
            with np.load(args.source_run.parent/'historical_val/scores'/f'task{task}.npz') as reference:
                assert np.array_equal(ids,reference['sample_ids'][:64])
                assert np.array_equal(scores.float().cpu().numpy()[:,indices],reference['logits'][:64,indices])
            before_old = tensor_state_digest(task_slices(method_state(model),task,model._task_sizes,False))
            head_before = tensor_state_digest(model.head.state_dict())
            asp_before = tensor_state_digest({k:v for k,v in method_state(model).items() if component(k)})
            configure_student(model,args.arm,task)
            view = dataset_view(train,indices,True)
            loader = AuditedLoader(DataLoader(view,batch_size=64,shuffle=True,num_workers=2,
                pin_memory=True,drop_last=False,generator=torch.Generator().manual_seed(317001+task)))
            assert len(loader) == baseline_history[str(task)][0]['optimizer_steps']
            state = {}
            history[str(task)] = train_task(model,loader,None,torch.device('cuda'),task,5,.0004,0.,1.,True,
                adapter_learning_rate=.0004,adapter_weight_decay=0.,view_fusion_learning_rate=.0004,
                view_auxiliary_loss_weight=.1,
                loss_routing='joint_bce' if args.arm=='BASE_FIXED_ATTENTION' else 'adapter_asl',
                optimizer_updates_per_task=2 if args.smoke else None,
                optimizer_group_builder=student_optimizer_groups,recovery_state=state)
            rows = history[str(task)]
            assert sum(r['optimizer_steps'] for r in rows) == (2 if args.smoke else 5*len(loader))
            assert all(r['skipped_optimizer_steps']==0 for r in rows)
            assert before_old == tensor_state_digest(task_slices(method_state(model),task,model._task_sizes,False))
            if args.arm != 'HEAD_TRAINABLE_ATTENTION': assert head_before == tensor_state_digest(model.head.state_dict())
            if args.arm == 'BASE_FIXED_ATTENTION':
                assert asp_before == tensor_state_digest({k:v for k,v in method_state(model).items() if component(k)})
            assert frozen_visual == tensor_state_digest(model.visual_encoder.state_dict())
            audits[str(task)].update(initial_current_predictions_exact=True,old_student_parameters_unchanged=True,
                frozen_visual_unchanged=True,train_instances=len(view),batches_per_epoch=len(loader),
                batch_trace=loader.epochs,optimizer_parameter_names=state['optimizer_parameter_names'])
            compact = compact_model_state_dict(model)
            atomic_save(dict(schema_version=1,task_id=task,source_git=git,model=compact),checkpoint)
            save_boundary(model,compact,task,cfg,state,loader.generator,dict(history=history,audits=audits),recovery)
            write(args.output/'training_history.json',history);write(args.output/'audits.json',audits)
            print(f'TASK_SAVED task={task} no_reporting_rng_consumed=1',flush=True)
    files = dict(prefix_files)
    for t in range(3,8):
        p=args.output/'compact_checkpoints'/f'task{t}.pth'
        files[str(t)]=dict(path=str(p),sha256=_sha256(p),source_git=git)
    assert all(_sha256(Path(r['path']))==r['sha256'] for r in prefix_files.values())
    write(args.output/'training_history.json',history);write(args.output/'audits.json',audits)
    write(args.output/'training_complete.json',dict(status='complete',config=cfg,checkpoints=files,
        prefix_preserved_exact=True,repeated_prefix_updates=0,smoke=args.smoke,
        new_completed_optimizer_updates=sum(r['optimizer_steps'] for t in range(3,8) for r in history[str(t)]),
        elapsed_seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def evaluate(args):
    start=time.time();git=runtime();set_seed(1,True)
    if args.kind=='student':
        training=read(args.trained_run/'training_complete.json');cfg=training['config']
        assert training['status']=='complete' and training['prefix_preserved_exact']
        template=cfg['model_template'];files=training['checkpoints'];smoke=cfg['smoke']
    else:
        template=read(args.trained_run/'config.json');validate_config(template)
        summary=read(args.trained_run/'seed_summary.json');assert summary['status']=='complete'
        smoke=template['max_tasks']!=8
        files={str(t):dict(path=str(args.trained_run/'compact_checkpoints'/f'task{t}.pth'),
            source_git=template['git'],sha256=_sha256(args.trained_run/'compact_checkpoints'/f'task{t}.pth'))
            for t in range(template['max_tasks'])}
        cfg=template
    require_test_ready(args.split,smoke,files)
    tasks=sorted(map(int,files)) if args.checkpoint_tasks is None else args.checkpoint_tasks
    assert tasks and all(str(t) in files for t in tasks)
    baseline=read(args.source_run/'config.json');assert validate_config(baseline)=='BASE'
    model=build(template,args.clip_checkpoint)
    data=make_source(baseline,args.data_root,args.face_manifest_root,args.split)
    metrics={mode:[] for mode in args.conditions};audits=[]
    for task in tasks:
        record=files[str(task)];payload=_torch_load(Path(record['path']))
        assert _sha256(Path(record['path']))==record['sha256']
        assert payload['task_id']==task and payload['source_git']==record['source_git']
        missing=model.load_state_dict(payload['model'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
        model.restore_task(task);model.requires_grad_(False).eval()
        original=tensor_state_digest(method_state(model))
        teacher=source_checkpoint(args,baseline,task)
        context=restore_components(model,teacher,'H') if args.head=='b0' else nullcontext(dict(condition='native'))
        with context as head_audit:
            for mode in args.conditions:
                if mode=='face_off' and (model.cross_view_bank is None or model.cross_view_bank.face_tasks is None):
                    raise ValueError('Face-only bypass requires a Face attention model')
                model.cross_view_runtime_enabled=model.cross_view_bank is not None and mode!='off'
                if model.cross_view_bank is not None: model.cross_view_bank.face_runtime_enabled=mode!='face_off'
                loader=DataLoader(dataset_view(data,seen_indices(task),True),batch_size=64,shuffle=False,
                    num_workers=2,pin_memory=False,generator=torch.Generator().manual_seed(87001+task))
                metric,views=evaluate_fused_and_views(model,loader,torch.device('cuda'),task,.5,True,
                    args.output/mode/'scores'/f'task{task}.npz',args.output/mode/'view_scores'/f'task{task}.npz')
                metrics[mode].append(metric)
                write(args.output/mode/f'task{task}.json',dict(metrics=asdict(metric),views=views))
                if args.kind=='student' and args.split=='val' and task<3 and mode=='on':
                    assert_scores_equal(args.output/mode/'scores'/f'task{task}.npz',
                        Path(cfg['continued_from'])/'on/scores'/f'task{task}.npz')
                if args.kind=='student' and cfg['arm']=='BASE_FIXED_ATTENTION' and mode=='off':
                    assert_scores_equal(args.output/mode/'scores'/f'task{task}.npz',
                        args.source_run.parent/('historical_'+args.split)/'scores'/f'task{task}.npz')
                print(f'EVALUATE split={args.split} task={task} mode={mode} head={args.head} mAP={metric.mAP:.6f}',flush=True)
        assert tensor_state_digest(method_state(model))==original
        audits.append(dict(task=task,checkpoint_sha256=record['sha256'],head=head_audit,
            b0_checkpoint_sha256=_sha256(args.source_run/'compact_checkpoints'/f'task{task}.pth')))
    summaries={mode:summarize_tasks(rows) for mode,rows in metrics.items()}
    if args.kind=='person_reference' and 7 in tasks and args.head=='b0':
        expected=43.07406218346569 if args.split=='val' else 32.886421356337166
        assert abs(next(m.mAP for m in metrics['on'] if m.task_id==7)-expected)<1e-10
    if 'face_off' in args.conditions:
        for task in tasks:
            with np.load(args.output/'on/view_scores'/f'task{task}.npz') as a, np.load(args.output/'face_off/view_scores'/f'task{task}.npz') as b:
                assert all(np.array_equal(a[k+'_logits'],b[k+'_logits']) for k in ('full','person'))
                invalid=~a['face_reliable'].astype(bool)
                assert np.array_equal(a['face_logits'][invalid],b['face_logits'][invalid])
                assert np.array_equal(a['fused_logits'][invalid],b['fused_logits'][invalid])
    write(args.output/'result.json',dict(status='complete',kind=args.kind,source_config=cfg,evaluation_git=git,
        b0_source_run=str(args.source_run),b0_git=baseline['git'],
        split=args.split,head=args.head,checkpoint_tasks=tasks,evaluation_scope='samples_intersect_seen_classes',
        test_used_for_parameter_search=True,full_eight_stage_average=tasks==list(range(8)),
        task_metrics={mode:[asdict(m) for m in rows] for mode,rows in metrics.items()},summaries=summaries,
        audits=audits,elapsed_seconds=time.time()-start))


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('train','eval'):
        q=sub.add_parser(action)
        for key in ('source-run','data-root','clip-checkpoint','face-manifest-root','output'):
            q.add_argument('--'+key,type=Path,required=True)
        if action=='train':
            q.add_argument('--pilot-root',type=Path,required=True);q.add_argument('--arm',choices=ARMS,required=True)
            q.add_argument('--smoke',action='store_true')
        else:
            q.add_argument('--trained-run',type=Path,required=True)
            q.add_argument('--kind',choices=('student','joint','person_reference'),required=True)
            q.add_argument('--head',choices=('native','b0'),required=True)
            q.add_argument('--split',choices=('val','test'),required=True)
            q.add_argument('--conditions',nargs='+',choices=('on','off','face_off'),default=['on'])
            q.add_argument('--checkpoint-tasks',nargs='+',type=int)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    (continuation if args.action=='train' else evaluate)(args)
