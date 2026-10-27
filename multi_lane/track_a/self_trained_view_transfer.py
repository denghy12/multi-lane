"""A predefined training flow; students never import trained teacher parameters."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .fusion_function_constraint import FusionFunctionConstraint
from .head_anchored_eight import runtime
from .head_anchored_pilot import (AuditedLoader, adaptation_rng, source_train,
    student_optimizer_groups, task_slices)
from .historical_b32_backbone_attribution import build, method_state
from .historical_b32_modules import modes, validate_config
from .paired_transforms import move_model_inputs
from .protected_view_forward import install, models
from .run_historical_b32_test_attribution import read, write
from .runner import (TaskMetrics, compact_model_state_dict, dataset_view, seen_indices,
    set_seed, summarize_tasks, tensor_state_digest, train_task)
from .strong_full_evaluation import evaluate_fused_and_views
from .task_recovery import atomic_save, restore_rng, save_boundary

ARMS=('SELF_BASE','SELF_NO_CONSTRAINT','SELF_MAIN_CONSTRAINT','SELF_MAIN_ADAPTER_CONSTRAINT')


def student_template(base, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    template=dict(base)
    if arm!='SELF_BASE':
        template.update(historical_b32_module_profile='ATTENTION_BOUND',
            view_fusion_learning_rate=.0004,**modes('ATTENTION_BOUND'))
    return template


def initial_core(model):
    return {k:v for k,v in method_state(model).items()
            if k=='selectors' or k.startswith(('prompts.','head.'))}


def configure_phase(model,task,adapt):
    """Current task only. Phase II freezes the head learned by THIS student."""
    model.restore_task(task); model.requires_grad_(False)
    for p in model.parameters(): p.grad=None
    model.selectors.requires_grad_(True)
    for p in model.prompts: p.requires_grad_(True)
    model.adapter_bank.task_adapters[task].requires_grad_(True)
    model.head.requires_grad_(not adapt)
    if model.cross_view_bank is not None:
        model.cross_view_bank.tasks[task].requires_grad_(adapt)
    model.cross_view_runtime_enabled=bool(adapt and model.cross_view_bank is not None)
    model.assert_visual_frozen()


def prediction_probe(model,probe,task):
    if probe is None:
        return None
    images,_=probe
    was=model.cross_view_runtime_enabled
    model.cross_view_runtime_enabled=model.cross_view_bank is not None
    model.eval()
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=False):
        z,views=model.seen_logits_with_views(images)
    model.cross_view_runtime_enabled=was
    stop=sum(model._task_sizes[:task])
    return {k:v[:,:stop].float().cpu() for k,v in dict(fused=z,**views).items()}


def load_own_boundary(model,path,config):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['config']==config
    missing=model.load_state_dict(saved['model'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
    model.restore_task(saved['task_id']); restore_rng(saved['rng'])
    return saved['provenance']['history'],saved['provenance']['audits']


def construct_teachers(args,student):
    # Isolate construction from the student's initialization, sampler, and augmentation RNG.
    before=tensor_state_digest(method_state(student))
    with adaptation_rng(927001):
        a,b=models(args)
        assert all(torch.equal(v,a.visual_encoder.state_dict()[k])
                   for k,v in student.visual_encoder.state_dict().items())
        a.visual_encoder=student.visual_encoder; b.visual_encoder=student.visual_encoder
    assert before==tensor_state_digest(method_state(student))
    assert a.head is not student.head and b.head is not student.head
    return a,b


def train(args):
    start=time.time(); git=runtime(); set_seed(1,True)
    base=read(args.source_run/'config.json'); assert validate_config(base)=='BASE'
    template=student_template(base,args.arm)
    # Only the standard pretrained CLIP visual weights are loaded. All task parameters
    # and the classifier are freshly initialized by MultiLaneModel, not by load_state_dict.
    student=build(template,args.clip_checkpoint)
    initial=tensor_state_digest(initial_core(student))
    assert initial==base['initial_selector_prompt_classifier_sha256']
    tasks=list(range(2 if args.smoke else 8))
    cfg=dict(kind='self_trained_single_three_view_model',git=git,arm=args.arm,seed=1,
        tasks=tasks,smoke=args.smoke,model_template=template,
        student_initialization='CLIP_visual_only_and_fresh_task_parameters_and_head',
        initial_selector_prompt_classifier_sha256=initial,external_student_parameter_loads=0,
        head_policy='own_hard_supervised_30_epochs_then_freeze_for_5',
        next_task_initialization='standard_activate_task_from_own_previous_student',
        warmup_epochs=30,adaptation_epochs=0 if args.arm=='SELF_BASE' else 5,
        warmup_main_lr=.0125,adapter_lr=.0004,adaptation_lr=.0004,
        optimizer='Adam_reset_each_task_and_phase',scheduler='cosine',weight_decay=0.,
        loss_routing='adapter_asl',training_loss_mode='legacy_full_zero',
        asl_gamma_neg=9.8,asl_gamma_pos=0.,asl_clip=.05,view_auxiliary_loss_weight=.1,
        batch_size=64,workers=2,prefetch_factor=1,threads=72,amp=True,tf32=True,
        warmup_sampler='historical_global_RNG',adaptation_sampler_seeds={str(t):317001+t for t in tasks},
        adaptation_rng_seeds={str(t):20261027+1009*t for t in tasks},
        teacher_constraint=args.arm in ARMS[2:],constraint_ratio=.25,constraint_maximum_beta=1.,
        constraint_calibration='first_actual_augmented_TRAIN_batch_per_task',
        teacher_current_task_only=True,teacher_required_for_inference=False,
        teacher_sources=dict(b0=str(args.source_run),person=str(args.person_run),dual=str(args.dual_run)),
        data_root=str(args.data_root),clip_checkpoint=str(args.clip_checkpoint),face_manifest_root=str(args.face_manifest_root),
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True)
    if (args.output/'config.json').exists(): assert read(args.output/'config.json')==cfg
    else: write(args.output/'config.json',cfg)
    training=source_train(base,args)
    # This is TRAIN with deterministic evaluation transforms, never Val/Test supervision.
    probe_source=make_source(base,args.data_root,args.face_manifest_root,'train')
    teachers=construct_teachers(args,student) if cfg['teacher_constraint'] else None
    visual_before=tensor_state_digest(student.visual_encoder.state_dict())
    history={}; audits={}
    for task in tasks:
        final=args.output/'recovery_checkpoints'/f'task{task}.pth'
        warm=args.output/'warmup_recovery'/f'task{task}.pth'
        if final.exists():
            history,audits=load_own_boundary(student,final,cfg)
            print('RESUME_OWN_COMPLETED_TASK',args.arm,task,flush=True); continue
        if warm.exists():
            history,audits=load_own_boundary(student,warm,cfg)
            print('RESUME_OWN_WARMUP',args.arm,task,flush=True)
        else:
            student.activate_task(task)
        current=range(sum(student._task_sizes[:task]),sum(student._task_sizes[:task+1]))
        old_before=tensor_state_digest(task_slices(method_state(student),task,student._task_sizes,False))
        probe=None
        with adaptation_rng(881001+task):
            if task:
                probe_data=Subset(dataset_view(probe_source,seen_indices(task-1),True),range(32))
                images,y,_=next(iter(DataLoader(probe_data,batch_size=32,num_workers=0,
                    generator=torch.Generator().manual_seed(881001+task))))
                probe=(move_model_inputs(images,torch.device('cuda')),y)
            old_prediction=prediction_probe(student,probe,task)
        if not warm.exists():
            configure_phase(student,task,False)
            loader=AuditedLoader(DataLoader(dataset_view(training,current,True),batch_size=64,
                shuffle=True,num_workers=2,prefetch_factor=1,pin_memory=True,drop_last=False))
            state={}
            h=train_task(student,loader,None,torch.device('cuda'),task,30,.0125,0.,1.,True,
                adapter_learning_rate=.0004,adapter_weight_decay=0.,view_fusion_learning_rate=.0004,
                view_auxiliary_loss_weight=.1,loss_routing='adapter_asl',
                optimizer_updates_per_task=2 if args.smoke else None,
                optimizer_group_builder=student_optimizer_groups,recovery_state=state)
            assert all(r['skipped_optimizer_steps']==0 for r in h)
            assert sum(r['optimizer_steps'] for r in h)==(2 if args.smoke else 30*len(loader))
            assert old_before==tensor_state_digest(task_slices(method_state(student),task,student._task_sizes,False))
            assert visual_before==tensor_state_digest(student.visual_encoder.state_dict())
            history[str(task)]=dict(warmup=h,adaptation=[])
            audits[str(task)]=dict(warmup_batches=loader.epochs,train_instances=len(loader.loader.dataset),
                warmup_optimizer_parameter_names=state['optimizer_parameter_names'],
                student_external_parameter_loads=0,own_previous_task_inheritance=True,
                old_task_parameter_sha256=old_before,old_parameters_preserved_by_warmup=True)
            compact=compact_model_state_dict(student)
            atomic_save(dict(task_id=task,source_git=git,model=compact),args.output/'warmup_checkpoints'/f'task{task}.pth')
            save_boundary(student,compact,task,cfg,state,None,dict(history=history,audits=audits),warm)
            write(args.output/'training_history.json',history); write(args.output/'audits.json',audits)
            print('OWN_WARMUP_SAVED',args.arm,task,flush=True)
        fixed_head=tensor_state_digest(student.head.state_dict())
        constraint=None; phase_state={}; phase_generator=None
        if args.arm!='SELF_BASE':
            with adaptation_rng(20261027+1009*task):
                configure_phase(student,task,True)
                if teachers is not None:
                    before=tensor_state_digest(method_state(student))
                    install(args,*teachers,task)
                    assert before==tensor_state_digest(method_state(student))
                    teacher_digest=[tensor_state_digest(method_state(m)) for m in teachers]
                    def predict(images,task_id):
                        assert task_id==task
                        values=[]
                        for teacher in teachers:
                            teacher.eval()
                            z,_,_=teacher.current_all_logits_with_view_features(images)
                            values.append(z.float())
                        return .5*values[0]+.5*values[1]
                    constraint=FusionFunctionConstraint(predict,current,
                        include_adapter=args.arm=='SELF_MAIN_ADAPTER_CONSTRAINT')
                phase_generator=torch.Generator().manual_seed(317001+task)
                loader=AuditedLoader(DataLoader(dataset_view(training,current,True),batch_size=64,
                    shuffle=True,num_workers=2,prefetch_factor=1,pin_memory=True,drop_last=False,
                    generator=phase_generator))
                h=train_task(student,loader,None,torch.device('cuda'),task,5,.0004,0.,1.,True,
                    adapter_learning_rate=.0004,adapter_weight_decay=0.,view_fusion_learning_rate=.0004,
                    view_auxiliary_loss_weight=.1,loss_routing='adapter_asl',
                    optimizer_updates_per_task=2 if args.smoke else None,
                    optimizer_group_builder=student_optimizer_groups,recovery_state=phase_state,
                    functional_constraint=constraint)
                assert all(r['skipped_optimizer_steps']==0 for r in h)
                assert sum(r['optimizer_steps'] for r in h)==(2 if args.smoke else 5*len(loader))
                assert fixed_head==tensor_state_digest(student.head.state_dict())
                history[str(task)]['adaptation']=h
                audits[str(task)].update(adaptation_batches=loader.epochs,
                    adaptation_optimizer_parameter_names=phase_state['optimizer_parameter_names'],
                    own_warmup_classifier_preserved=True,
                    functional_calibration=constraint.calibration if constraint else None)
                if teachers is not None:
                    assert teacher_digest==[tensor_state_digest(method_state(m)) for m in teachers]
                    assert all(p.grad is None for m in teachers for p in m.parameters())
                    audits[str(task)]['teacher_parameters_and_gradients_unchanged']=True
        # Re-enable the trained old attention paths for the same-shaped protection probe.
        with adaptation_rng(881001+task):
            after_prediction=prediction_probe(student,probe,task)
            if probe is not None:
                assert all(torch.equal(v,after_prediction[k]) for k,v in old_prediction.items())
        assert old_before==tensor_state_digest(task_slices(method_state(student),task,student._task_sizes,False))
        assert visual_before==tensor_state_digest(student.visual_encoder.state_dict())
        student.cross_view_runtime_enabled=student.cross_view_bank is not None
        audits[str(task)].update(old_task_parameters_exact=True,old_TRAIN_probe_predictions_exact=True,
            frozen_visual_exact=True,final_head_source='this_student_own_supervised_warmup',
            inference_teacher_dependencies=False)
        compact=compact_model_state_dict(student)
        atomic_save(dict(schema_version=1,task_id=task,source_git=git,model=compact),
                    args.output/'compact_checkpoints'/f'task{task}.pth')
        save_boundary(student,compact,task,cfg,phase_state,phase_generator,
            dict(history=history,audits=audits),final)
        write(args.output/'training_history.json',history); write(args.output/'audits.json',audits)
        print('SELF_TRAINED_TASK_SAVED',args.arm,task,'external_student_parameter_loads=0',flush=True)
    checkpoints={str(t):dict(sha256=_sha256(args.output/'compact_checkpoints'/f'task{t}.pth')) for t in tasks}
    write(args.output/'training_complete.json',dict(status='complete',config=cfg,checkpoints=checkpoints,
        elapsed_seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        completed_optimizer_updates=sum(r['optimizer_steps'] for h in history.values() for phase in h.values() for r in phase),
        student_external_parameter_loads=0,teacher_required_for_inference=False))


def evaluate(args):
    start=time.time();git=runtime();set_seed(1,True)
    cfg=read(args.training_root/'config.json');complete=read(args.training_root/'training_complete.json')
    assert complete['status']=='complete' and cfg==complete['config'] and cfg['git']==git
    if args.split=='test' and (cfg['smoke'] or cfg['tasks']!=list(range(8))):
        raise ValueError('Test requires completed eight-task training')
    # No teacher model or teacher parameter file is loaded in this inference process.
    model=build(cfg['model_template'],args.clip_checkpoint)
    data=make_source(cfg['model_template'],args.data_root,args.face_manifest_root,args.split)
    results=[];off=[];audits=[]
    for task in cfg['tasks']:
        path=args.training_root/'compact_checkpoints'/f'task{task}.pth'
        assert _sha256(path)==complete['checkpoints'][str(task)]['sha256']
        payload=_torch_load(path);assert payload['source_git']==git and payload['task_id']==task
        missing=model.load_state_dict(payload['model'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
        model.restore_task(task);model.requires_grad_(False).eval()
        model.cross_view_runtime_enabled=model.cross_view_bank is not None
        before=tensor_state_digest(method_state(model))
        for mode in ('on','off') if model.cross_view_bank is not None else ('on',):
            model.cross_view_runtime_enabled=mode=='on' and model.cross_view_bank is not None
            view=dataset_view(data,seen_indices(task),True)
            if cfg['smoke']:view=Subset(view,range(64))
            loader=DataLoader(view,batch_size=64,shuffle=False,num_workers=0,
                generator=torch.Generator().manual_seed(87001+task))
            with torch.no_grad():
                value,views=evaluate_fused_and_views(model,loader,torch.device('cuda'),task,.5,True,
                    args.output/mode/'scores'/f'task{task}.npz',args.output/mode/'view_scores'/f'task{task}.npz')
            (results if mode=='on' else off).append(asdict(value))
            write(args.output/mode/f'task{task}.json',dict(metrics=asdict(value),views=views))
            print('SELF_EVALUATE',cfg['arm'],args.split,task,mode,value.mAP,flush=True)
        assert before==tensor_state_digest(method_state(model))
        audits.append(dict(task=task,own_checkpoint_sha256=_sha256(path),parameters_unchanged=True))
    write(args.output/'result.json',dict(status='complete',git=git,split=args.split,smoke=cfg['smoke'],
        seed=1,arm=cfg['arm'],tasks=cfg['tasks'],metrics=results,off_metrics=off,
        summary=summarize_tasks([TaskMetrics(**v) for v in results]) if not cfg['smoke'] else {},
        off_summary=summarize_tasks([TaskMetrics(**v) for v in off]) if off and not cfg['smoke'] else {},
        teacher_loaded_for_inference=False,external_parameter_assembly=False,audits=audits,
        elapsed_seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def audit_initialization(args):
    git=runtime();base=read(args.source_run/'config.json');set_seed(1,True)
    a=build(student_template(base,'SELF_BASE'),args.clip_checkpoint)
    rng=torch.get_rng_state().clone(); cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
    set_seed(1,True); b=build(student_template(base,'SELF_NO_CONSTRAINT'),args.clip_checkpoint)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    state=method_state(a);other=method_state(b)
    assert all(torch.equal(v,other[k]) for k,v in state.items())
    assert tensor_state_digest(initial_core(a))==base['initial_selector_prompt_classifier_sha256']
    a.activate_task(0);b.activate_task(0);configure_phase(a,0,False);configure_phase(b,0,False)
    data=make_source(base,args.data_root,args.face_manifest_root,'train')
    x,_,_=next(iter(DataLoader(Subset(dataset_view(data,seen_indices(0),True),range(64)),batch_size=64,
        num_workers=0,generator=torch.Generator().manual_seed(87001))))
    x=move_model_inputs(x,torch.device('cuda'));a.eval();b.eval()
    with torch.no_grad(),torch.cuda.amp.autocast():
        za,_=a.seen_logits_with_views(x);zb,_=b.seen_logits_with_views(x)
    assert torch.equal(za,zb)
    write(args.output,dict(status='verified',git=git,common_initial_parameters_exact=True,
        global_rng_exact=True,initial_TRAIN_predictions_exact=True,student_external_parameter_loads=0,
        own_head_random_initialization_verified=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('train','eval','audit'))
    for name in ('data-root','clip-checkpoint','face-manifest-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('source-run','person-run','dual-run','reference-root','training-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--split',choices=('val','test'),default='val')
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    {'train':train,'eval':evaluate,'audit':audit_initialization}[args.action](args)
