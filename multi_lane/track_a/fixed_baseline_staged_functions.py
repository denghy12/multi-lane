"""Fit-only common donors; old-only and old-then-new task-static routing."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .complete_eight_task_test import require_eight, require_formal_budget, restore_cached, model_digest
from .evaluate_joint26_checkpoint import audit_face_manifest
from .export_compact_test_scores import _read_json, _sha256, _torch_load
from .fixed_baseline_routes import (cached_identity, compare_source_scores,
                                    loader, restore_base, source_dataset, validate_source, write)
from .openai_clip_loader import load_openai_clip_visual
from .post_task_calibration import parameter_hash
from .runner import (compact_model_state_dict, evaluate, evaluate_view_diagnostics,
                     git_metadata, resolve_dataset_parent, seen_indices, set_seed,
                     summarize_tasks, task_indices, validate_classes)

from .staged_functional_experts import (ARMS, COMBINATION, StagedFunctionBank,
                                         StagedRouteAudit, split_cache, image_group, held_out)
from .staged_functional_fit import fit_phase, cache_metrics
from .fixed_baseline_functional_experts import validate_cache, train_metadata

POLICIES = ARMS

def build_expanding_model(config, visual, arm):
    from .fixed_baseline_expert_reuse import build_expanding_model as original_builder
    policy = 'fresh_only' if arm == 'fresh_only' else 'reuse_old'
    model = original_builder(config, visual, policy)
    with torch.random.fork_rng(devices=[]):
        model.parax_bank = StagedFunctionBank(model.output_dim, 32, 2*len(model.task_sizes), (0,),
            arm=arm, policy=policy, num_tasks=len(model.task_sizes), router_hidden=16,
            task_local_router=True, task_local_projection=True, initialization='zero_output',
            output_scale_mode='fixed', residual_scale=1., smooth_ratio_bound=.02)
    return model


def paired_banks(bank):
    banks = {'fresh_only': bank}
    for arm in ('old_only', 'old_then_new'):
        other = copy.deepcopy(bank);other.arm=arm;other.policy='reuse_old'
        for t in range(other.num_tasks):other.expert_access[t,:2*t+2]=True
        banks[arm]=other
    if len({parameter_hash(b.named_parameters()) for b in banks.values()})!=1:
        raise RuntimeError('Unpaired parameter initializations')
    return banks


def checkpoint_payload(model, task, git, arm):
    return dict(schema_version=1,task_id=task,source_git=git,arm=arm,
                expert_policy=model.parax_bank.policy,router_input_mode='constant',combination=COMBINATION,
                model=compact_model_state_dict(model))


def restore_fitted(model,payload,original,task,git,arm):
    bank=model.parax_bank
    if payload.get('arm')!=arm or bank.arm!=arm or payload.get('combination')!=COMBINATION:
        raise ValueError('Wrong staged checkpoint')
    expected=bank.expert_access.detach().cpu().clone()
    restore_cached(model,payload,original,task,git,'static')
    if not torch.equal(bank.expert_access.cpu(),expected):raise ValueError('Access mask changed')
    if not torch.equal(bank.expert_owners.cpu(),torch.arange(bank.num_experts)//2):raise ValueError('Owner changed')
    if not bool(bank.sealed_tasks[:task+1].all()) or bool(bank.sealed_tasks[task+1:].any()):raise ValueError('Bad sealing')
    if not bool(bank.donor_loaded[:task].all()) or bool(bank.donor_loaded[task:].any()):raise ValueError('Missing/future donor')
    bank.set_phase('off');model.requires_grad_(False)


def capture_functions(bank,task,features):
    with torch.no_grad():
        values={f'own:{i}:{v}':bank.expert_output(i,x).clone() for i in range(2*task) for v,x in features.items()}
        values.update({f'donor:{i}:{v}':bank.donor_output(i,x,task).clone()
                       for i in range(2*task) for v,x in features.items()})
        return values


def experiment_policies(args, config):
    """Keep the recheck validation-only, with a separate fit-only donor control."""
    validation_only = bool(getattr(args, 'validation_only', False))
    recheck = bool(getattr(args, 'old_only_recheck', False))
    if recheck and not validation_only:
        raise ValueError('Old-only recheck requires --validation-only; Test is forbidden')
    if recheck:
        if config.get('seed') not in (0, 1, 2):
            raise ValueError('Declared recheck seeds are 0, 1, 2')
        return ('fresh_only', 'old_only')
    if config.get('seed') != 0:
        raise ValueError('Original four-arm experiment is seed0 only')
    return POLICIES


def finish_validation_only(output, records, source, cached, source_hashes, cache_hashes, smoke):
    # This function must complete before any Test dataset/score/checkpoint evaluation.
    if any(len(rows) != 8 for rows in records.values()):
        raise ValueError('Validation-only completion requires all eight tasks for every recorded arm')
    if source_hashes != {str(t): _sha256(source / 'compact_checkpoints' / f'task{t}.pth') for t in range(8)}:
        raise RuntimeError('Source checkpoints changed')
    if cache_hashes != {str(t): _sha256(cached / 'train_cache' / f'task{t}.pth') for t in range(8)}:
        raise RuntimeError('Source TRAIN caches changed')
    write(output / 'summary.json', dict(status='smoke_complete' if smoke else 'complete', completed_tasks=8,
          source_checkpoints_unchanged=True, source_caches_unchanged=True,
          validation={k:dict(task_metrics=[asdict(r) for r in rows], **summarize_tasks(rows))
                      for k, rows in records.items()},
          test_accessed=False, test_model_selection=False, holdout_used_for_selection=False,
          candidate_arms=['baseline', 'old_only'] if 'old_then_new' not in records else list(records),
          donor_control='fresh_only'))


def run(args):
    source, cached, output = map(Path, (args.source, args.cache_source, args.output))
    if output.exists():
        raise FileExistsError(output)
    config, source_summary = _read_json(source / "config.json"), _read_json(source / "seed_summary.json")
    validate_source(config, source_summary, True)
    require_eight(config, source_summary)
    require_formal_budget(config)
    policies = experiment_policies(args, config)
    validation_only = bool(getattr(args, "validation_only", False))
    if not validation_only and (source_summary.get("test_metrics") is None or len(source_summary.get("test_task_metrics", [])) != 8):
        raise ValueError("Same-baseline eight-task test reference is required")
    cache_config, cache_summary = _read_json(cached / "config.json"), _read_json(cached / "summary.json")
    if (cache_config.get("protocol") != "fixed_checkpoint_train_cache_v1"
            or cache_config.get("source_config") != config or cache_config.get("cache_split") != "train"
            or cache_config.get("smoke") or cache_summary.get("status") != "complete"
            or cache_summary.get("completed_tasks") != 8):
        raise ValueError("Complete same-baseline TRAIN cache source required")
    git = git_metadata(Path(__file__).resolve().parents[2])
    if git["dirty"]:
        raise RuntimeError("Clean experiment worktree required")
    clip = Path(args.clip_checkpoint)
    if _sha256(clip) != config["clip_checkpoint_sha256"]:
        raise ValueError("Pretrained CLIP changed")
    if (resolve_dataset_parent(Path(args.data_root)) / "EMOTIC").resolve() != Path(config["data_root"]).resolve():
        raise ValueError("Dataset root changed")
    faces = audit_face_manifest(config, Path(args.face_manifest_root))
    source_hashes = {str(t): _sha256(source / "compact_checkpoints" / f"task{t}.pth") for t in range(8)}
    cache_hashes = {str(t): _sha256(cached / "train_cache" / f"task{t}.pth") for t in range(8)}
    device = torch.device("cuda:0")
    set_seed(config["seed"], config["tf32"])
    model = build_expanding_model(config, load_openai_clip_visual(clip), 'fresh_only').float().to(device)
    model.requires_grad_(False);banks={p:b for p,b in paired_banks(model.parax_bank).items() if p in policies}
    datasets={s:source_dataset(config,args.data_root,args.face_manifest_root,s) for s in ('train','val')}
    validate_classes(*datasets.values())
    groups={image_group(s) for s in datasets['train'].sample_ids}
    fit_groups={g for g in groups if not held_out(g+'#person=0')}
    held_groups=groups-fit_groups
    write(output/'image_group_split.json',dict(source_split='train',seed=0,model_seed=config['seed'],rule='sha256 module-holdout-seed0-v1 modulo5==0',
          fit_groups=sorted(fit_groups),holdout_groups=sorted(held_groups),
          original_baseline_may_have_seen_holdout=True,extra_modules_exclude_holdout_from_task0=True))
    audits={p:StagedRouteAudit(output/p,device) for p in policies}
    records={p:[] for p in ('baseline',*policies)}
    for p in policies:(output/p).mkdir(parents=True,exist_ok=True)
    write(output/'config.json',dict(protocol=COMBINATION,source=str(source),source_config=config,cache_source=str(cached),
          git=git,seed=config['seed'],tasks=8,phase_epochs=5,batch_size=64,optimizer='Adam',learning_rate=.0004,weight_decay=0.,
          scheduler='cosine reset each five-epoch phase',gradient_clip_norm=1.,auxiliary_weight=.1,consistency_weight=0.,
          rank=32,new_experts_per_task=2,expert_slots=16,residual_bound=.02,residual_scale=1.,
          combination=COMBINATION,donors='same sealed fit-only fresh-control birth functions for all arms',
          gate_records='task-static signed coefficients; no probability entropy interpretation',
          budgets={'fresh_only':'new5+new5','old_only':'old5; task0 identity',
                   'old_then_new':'old5+new5; task0 new5+new5'},
          donor_training_cost='shared fresh-control training; included in reported provenance, not free standalone compute',
          holdout_fraction_target=.2,holdout_used_for_selection=False,baseline_retrained=False,
          source_checkpoint_hashes=source_hashes,train_cache_hashes=cache_hashes,
          face_manifest_hashes=faces,smoke=args.smoke,test_after_all_fitting=not validation_only,validation_only=validation_only,
          old_only_recheck=bool(getattr(args,'old_only_recheck',False)),
          candidate_arms=['baseline','old_only'] if getattr(args,'old_only_recheck',False) else ['baseline',*policies],
          donor_control='fresh_only',partition_seed=0,model_selection=False))
    previous_base=None
    for task in range(8):
        payload=_torch_load(source/'compact_checkpoints'/f'task{task}.pth')
        restored=restore_base(model,payload,task,config['git'])
        if previous_base is not None:
            now=audits['fresh_only']._protected_state(model,task)
            if any(not torch.equal(now[k],v) for k,v in previous_base.items()):raise RuntimeError('Baseline changed old state')
        model.set_parax_runtime_enabled(False)
        val_loader=loader(datasets['val'],seen_indices(task),config,args.smoke)
        score=output/'baseline_scores'/f'task{task}.npz'
        records['baseline'].append(evaluate(model,val_loader,device,task,config['threshold'],config['amp'],score))
        recovery={'smoke':True} if args.smoke else compare_source_scores(score,source/'val_scores'/f'task{task}.npz')
        cache=_torch_load(cached/'train_cache'/f'task{task}.pth')
        ids,labels=train_metadata(datasets['train'],task)
        validate_cache(cache,task,source_hashes[str(task)],ids,labels,model.output_dim)
        parts=split_cache(cache,args.smoke)
        if any(image_group(s) not in fit_groups for s in parts['fit']['sample_ids']):raise RuntimeError('Fit leakage')
        if any(image_group(s) not in held_groups for s in parts['holdout']['sample_ids']):raise RuntimeError('Holdout leakage')
        write(output/f'task{task}_partition.json',dict(fit_ids=parts['fit']['sample_ids'],holdout_ids=parts['holdout']['sample_ids'],
              fit_positive_counts=parts['fit']['targets'].sum(0).tolist(),holdout_positive_counts=parts['holdout']['targets'].sum(0).tolist(),
              cache_sha256=cache_hashes[str(task)],smoke=args.smoke))
        initial=[]
        for bank in banks.values():
            names=[(n,p) for n,p in bank.named_parameters() if n.startswith((f'expert_a.{2*task}',f'expert_a.{2*task+1}',
                f'expert_b.{2*task}',f'expert_b.{2*task+1}',f'task_projections.{task}.',f'new_logits.{task}'))]
            initial.append(parameter_hash(names))
        if len(set(initial))!=1:raise RuntimeError('New expert initialization mismatch')
        old_stage_coeff=None
        for arm,bank in banks.items():
            model.parax_bank=bank;model.restore_task(task);model.requires_grad_(False);bank.set_phase('off')
            identity=cached_identity(model,parts['fit']);audits[arm].begin(model,val_loader,task)
            probe={v:x[:16].to(device) for v,x in parts['fit']['features'].items()}
            before_functions=capture_functions(bank,task,probe)
            phases=[]
            if arm=='fresh_only' or (arm=='old_then_new' and task==0):sequence=['new','new']
            elif arm=='old_only':sequence=['old'] if task else []
            else:sequence=['old','new']
            for phase_index,phase in enumerate(sequence):
                fit=fit_phase(model,parts['fit'],task_indices(task),device,phase,
                              seed=config['seed']+1009*task+100000*phase_index)
                fit['fit_metrics']=cache_metrics(model,parts['fit'],task_indices(task),device)
                fit['holdout_metrics']=cache_metrics(model,parts['holdout'],task_indices(task),device)
                phases.append(fit)
                if phase=='old':
                    if arm=='old_only':old_stage_coeff=bank.old_coefficients[task].detach().clone()
                    elif not torch.equal(old_stage_coeff,bank.old_coefficients[task]):raise RuntimeError('Old stage differs between paired arms')
            after_functions=capture_functions(bank,task,probe)
            if any(not torch.equal(before_functions[k],after_functions[k]) for k in before_functions):raise RuntimeError('Frozen function changed')
            model.set_parax_runtime_enabled(True);audits[arm].finish(model,task)
            records[arm].append(evaluate(model,val_loader,device,task,config['threshold'],config['amp'],output/arm/'val_scores'/f'task{task}.npz'))
            views=evaluate_view_diagnostics(model,val_loader,device,task,config['threshold'],config['amp'],output/arm/'view_val_scores'/f'task{task}.npz')
            write(output/arm/f'task{task}_views.json',views)
            write(output/arm/f'task{task}_audit.json',dict(phases=phases,**identity,**recovery,
                  complete_old_functions_unchanged=True,source_checkpoint_sha256=source_hashes[str(task)],
                  fit_metrics=cache_metrics(model,parts['fit'],task_indices(task),device),
                  holdout_metrics=cache_metrics(model,parts['holdout'],task_indices(task),device),
                  source_base_tensor_sha256=restored,policy=bank.policy_summary(task),
                  donor_provenance='fresh_only birth-task checkpoint from this batch and fit partition',
                  new_initialization_sha256=initial[0]))
            target=output/arm/'compact_checkpoints'/f'task{task}.pth';target.parent.mkdir(exist_ok=True)
            torch.save(checkpoint_payload(model,task,git,arm),target)
        # Install only AFTER every current-task result was sealed/saved. No
        # existing task reads this new donor; all future tasks share it exactly.
        for bank in banks.values():
            bank.install_donor(banks['fresh_only'],task)
        donor_hashes={parameter_hash((n,v) for n,v in b.state_dict().items() if n.startswith('donor_')) for b in banks.values()}
        if len(donor_hashes)!=1:raise RuntimeError('Common donor pools diverged')
        write(output/f'task{task}_donor_install.json',dict(birth_task=task,source_checkpoint_sha256=_sha256(output/'fresh_only'/'compact_checkpoints'/f'task{task}.pth'),
              common_donor_sha256=next(iter(donor_hashes)),visible_from_task=task+1))
        previous_base={k:v for k,v in audits['fresh_only']._protected_state(model,task+1).items() if not k.startswith('parax.')}
        write(output/'summary.json',dict(status='fitting',completed_tasks=task+1,
              arms={k:dict(task_metrics=[asdict(x) for x in rows],**summarize_tasks(rows)) for k,rows in records.items()}))
        print(f'staged fit/holdout/validation completed task={task}',flush=True)
    write(output / "fit_complete.json", {"tasks": 8, "test_accessed": False, "model_selection": False})
    if validation_only:
        # Check every saved task module through the same restore used by future
        # evaluation, without constructing a Test dataset or doing optimization.
        restored_checkpoints = []
        for policy in policies:
            restored_model = build_expanding_model(config, load_openai_clip_visual(clip), policy).float().to(device)
            for task in range(8):
                path = output / policy / 'compact_checkpoints' / f'task{task}.pth'
                before_sha = _sha256(path)
                payload = _torch_load(path)
                original = _torch_load(source / 'compact_checkpoints' / f'task{task}.pth')
                restore_fitted(restored_model, payload, original, task, git, policy)
                if _sha256(path) != before_sha:
                    raise RuntimeError('Validation restoration changed checkpoint')
                restored_checkpoints.append(dict(arm=policy, task=task, checkpoint_sha256=before_sha,
                                                base_and_access_restored=True, training_updates=0))
            del restored_model
        write(output / 'validation_checkpoint_restore_audit.json', dict(checkpoints=restored_checkpoints,
              test_accessed=False, optimization_updates=0))
        finish_validation_only(output, records, source, cached, source_hashes, cache_hashes, args.smoke)
        return
    # No test dataset exists until every training operation has finished.
    test_dataset = source_dataset(config, args.data_root, args.face_manifest_root, "test")
    validate_classes(test_dataset)
    test_rows = {p: [] for p in policies}
    for policy in policies:
        model = build_expanding_model(config, load_openai_clip_visual(clip), policy).float().to(device)
        for task in range(8):
            path = output / policy / "compact_checkpoints" / f"task{task}.pth"
            saved_sha = _sha256(path)
            payload = _torch_load(path)
            original = _torch_load(source / "compact_checkpoints" / f"task{task}.pth")
            restore_fitted(model, payload, original, task, git, policy)
            before = model_digest(model)
            test_loader = loader(test_dataset, seen_indices(task), config, args.smoke)
            test_rows[policy].append(evaluate(model, test_loader, device, task, config["threshold"], config["amp"],
                                             output / policy / "test_scores" / f"task{task}.npz"))
            views = evaluate_view_diagnostics(model, test_loader, device, task, config["threshold"], config["amp"],
                                             output / policy / "view_test_scores" / f"task{task}.npz")
            if model_digest(model) != before or _sha256(path) != saved_sha:
                raise RuntimeError("Read-only test changed model/checkpoint")
            write(output / policy / f"task{task}_test_audit.json", {"state_before": before, "state_after": before,
                  "checkpoint_sha256": saved_sha, "training_updates": 0, "views": views})
            print(f"read-only test completed policy={policy} task={task}", flush=True)
    if source_hashes != {str(t): _sha256(source / "compact_checkpoints" / f"task{t}.pth") for t in range(8)}:
        raise RuntimeError("Source checkpoints changed")
    if cache_hashes != {str(t): _sha256(cached / "train_cache" / f"task{t}.pth") for t in range(8)}:
        raise RuntimeError("Source TRAIN caches changed")
    write(output / "summary.json", {"status": "smoke_complete" if args.smoke else "complete", "completed_tasks": 8,
          "source_checkpoints_unchanged": True, "source_caches_unchanged": True,
          "validation": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in records.items()},
          "test": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in test_rows.items()},
          "baseline_test_from_source": source_summary["test_metrics"],
          "test_model_selection": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache-source", "output", "data-root", "clip-checkpoint", "face-manifest-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--validation-only", action="store_true", help="Never construct or evaluate a Test dataset")
    parser.add_argument("--old-only-recheck", action="store_true", help="Only baseline/old-only candidates plus fit-only donor control; requires validation-only")
    parser.add_argument("--smoke", action="store_true", help="All eight tasks, only first64 samples per split; not performance")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
