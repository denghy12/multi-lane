"""Full validation with immutable birth-task checkpoints; no optimization."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .evaluate_joint26_checkpoint import build_model, make_source
from .export_compact_test_scores import _read_json, _sha256, _torch_load, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import (CLASS_ORDER, TASK_SIZES, LabelView, compact_model_state_dict, compute_metrics,
                     dataset_view, git_metadata, ids_digest, seen_indices, set_seed,
                     tensor_state_digest, validate_classes)


def load_scores(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def audit_source(source):
    config, summary = _read_json(source/'config.json'), _read_json(source/'seed_summary.json')
    audit, history = _read_json(source/'paired_protocol_audit.json'), _read_json(source/'training_history.json')
    drift = _read_json(source/'image_stream_transfer_audit.json')
    require(summary['status'] == 'complete' and summary['config'] == config, 'Incomplete/config mismatch')
    require(config['training_protocol'] == 'incremental' and config['task_sizes'] == list(TASK_SIZES)
            and config['class_order'] == list(CLASS_ORDER) and config['max_tasks'] == 8
            and config['image_stream_transfer_audit'], 'Expected original eight-task transfer run')
    require(config['adapter_mode'] == 'disabled' and config['parax_mode'] in ('disabled','image')
            and config['selector_mode'] == config['prompt_mode'] == 'shared'
            and config['selector_conditioning'] == 'disabled' and config['view_fusion'] == 'fixed_three_view'
            and config['view_classifier_mode'] == 'shared_post_fusion', 'Unsupported architecture')
    require(config['git']['dirty'] is False and config['reporting_split'] == 'val'
            and config['save_compact_checkpoints'], 'Invalid provenance')
    require(summary['completed_epochs'] == 240 and config['epochs_per_task'] == 30, 'Wrong epoch budget')
    require(audit['frozen_visual_unchanged'] and audit['frozen_visual_sha256_before'] == audit['frozen_visual_sha256_after'], 'Source changed CLIP')
    updates = 0
    for task in range(8):
        rows = history[str(task)]
        require(len(rows) == 30 and all(r['skipped_optimizer_steps'] == 0 and r['amp_loss_scale_end'] == 1024
                and math.isfinite(r['current_loss']) for r in rows), 'Unstable/incomplete source')
        count = sum(r['optimizer_steps'] for r in rows)
        require(count == 30*math.ceil(audit['tasks'][str(task)]['train_instances']/config['train_batch_size']), 'Wrong source update budget')
        require(drift['tasks'][str(task)]['old_lane_parameters_unchanged'], 'Source changed old task parameters')
        if config['parax_mode'] == 'image':
            require(all(r['parax_grad_finite'] == 1 for r in rows), 'Nonfinite source gradients')
        updates += count
    require(updates == summary['completed_optimizer_updates'], 'Source update sum differs')
    if config['parax_mode'] == 'image':
        require(config['parax_layer_indices'] == [10] and config['parax_rank'] == 32
                and config['parax_num_experts'] == 3 and not config['parax_freeze_center_after_task0']
                and not config['parax_task_local_gate'] and not config['parax_level_conditioned']
                and not config.get('parax_freeze_all_after_task0', False), 'Different ParaX method')
    return config, summary, audit


def owned_task_state(state, task, sizes):
    start, stop = sum(sizes[:task]), sum(sizes[:task+1])
    result = dict(selectors=state['selectors'][task], head_weight=state['head.weight'][start:stop],
                  class_mask=state['task_class_mask'][task])
    if 'head.bias' in state:
        result['head_bias'] = state['head.bias'][start:stop]
    result.update({key: value[:,task] for key,value in state.items() if key.startswith('prompts.')})
    return result


def assert_owned_equal(birth, final, task, sizes):
    a,b = owned_task_state(birth,task,sizes), owned_task_state(final,task,sizes)
    require(a.keys() == b.keys() and all(torch.equal(v,b[k]) for k,v in a.items()), 'Task-owned state differs from final checkpoint')


def assemble_blocks(blocks, sizes):
    """Join by stable IDs and exact class ownership, never by best AP."""
    require(len(blocks) == len(sizes), 'Missing task block')
    ids = blocks[0]['sample_ids']
    require(len(set(ids.tolist())) == len(ids), 'Duplicate sample IDs')
    targets = blocks[0]['full_targets']
    require(targets.shape == (len(ids),sum(sizes)), 'Wrong full-label shape')
    result = dict(sample_ids=ids.copy(),targets=targets.copy(),class_indices=np.arange(sum(sizes)),
                  face_reliable=blocks[0]['face_reliable'].copy())
    keys = ['logits','probabilities'] + [f'{v}_{suffix}' for v in ('full','person','face') for suffix in ('logits','probabilities')]
    joined={k:[] for k in keys};start=0
    for task,(block,count) in enumerate(zip(blocks,sizes)):
        require(len(block['sample_ids']) == len(ids) and len(set(block['sample_ids'].tolist())) == len(ids), 'Invalid block IDs')
        positions={str(v):i for i,v in enumerate(block['sample_ids'])}
        require(all(str(v) in positions for v in ids), 'Sample set differs')
        rows=[positions[str(v)] for v in ids]
        require(np.array_equal(block['class_indices'],np.arange(start,start+count)), 'Incorrect class ownership')
        require(np.array_equal(block['full_targets'][rows],targets), 'Full targets differ')
        require(np.array_equal(block['face_reliable'][rows],result['face_reliable']), 'Face reliability differs')
        for key in keys:
            value=block[key][rows]
            require(value.shape == (len(ids),count) and np.isfinite(value).all(), 'Invalid prediction block')
            if key.endswith('probabilities'):
                require((value >= 0).all() and (value <= 1).all(), 'Invalid probabilities')
            joined[key].append(value)
        start+=count
    result.update({k:np.concatenate(v,axis=1) for k,v in joined.items()})
    return result


def infer(model, loader, device, amp, task, sizes, max_batches=None):
    """Evaluate all seen lanes, retaining only this task's owned class block."""
    start,stop=sum(sizes[:task]),sum(sizes[:task+1])
    rows,ids,lengths,reliable=[],[],[],[]
    with torch.no_grad():
        for i,(images,targets,samples) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            reliable.append(images['face_reliable'].cpu().numpy().astype(bool))
            with torch.cuda.amp.autocast(enabled=amp and device.type=='cuda'):
                logits,views=model.seen_logits_with_views(move_model_inputs(images,device))
            values=dict(logits=logits.float().cpu(),**{f'{v}_logits':x.float().cpu() for v,x in views.items()})
            values.update({k.replace('logits','probabilities'):torch.sigmoid(v) for k,v in list(values.items())})
            rows.append(dict(full_targets=targets.cpu().numpy(),**{k:v.numpy() for k,v in values.items()}))
            ids.extend(str(x) for x in samples);lengths.append(len(samples))
    require(bool(rows),'Empty validation loader')
    full={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    full.update(sample_ids=np.asarray(ids),face_reliable=np.concatenate(reliable),batch_lengths=np.asarray(lengths))
    block={k:(v if k=='full_targets' else v[:,start:stop]) for k,v in full.items() if k not in ('sample_ids','face_reliable','batch_lengths')}
    block.update(sample_ids=full['sample_ids'],face_reliable=full['face_reliable'],batch_lengths=full['batch_lengths'],class_indices=np.arange(start,stop))
    return block,full


def score_metrics(data,threshold):
    result={}
    for view in ('fused','full','person','face','face_reliable'):
        key='probabilities' if view=='fused' else f'{"face" if view=="face_reliable" else view}_probabilities'
        mask=data['face_reliable'] if view=='face_reliable' else np.ones(len(data['targets']),dtype=bool)
        require(bool(mask.any()),'Empty metric cohort')
        result[view]=asdict(compute_metrics(7,torch.from_numpy(data[key][mask]),torch.from_numpy(data['targets'][mask]),threshold))
    return result


def replay(model,validation,source,config,task,device):
    loader=DataLoader(dataset_view(validation,seen_indices(task),include_sample_id=True),
                      batch_size=config['eval_batch_size'],shuffle=False,num_workers=0)
    images,targets,ids=next(iter(loader))
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=device.type=='cuda' and config['amp']):
        logits=model.seen_logits(move_model_inputs(images,device)).float().cpu().numpy()
    expected=load_scores(source/f'val_scores/task{task}.npz')
    require(np.array_equal(expected['sample_ids'][:len(ids)],np.asarray(ids))
            and np.array_equal(expected['targets'][:len(ids)],targets.numpy()),'Replay labels/IDs differ')
    difference=float(np.abs(logits-expected['logits'][:len(ids)]).max())
    require(difference <= .01,f'Checkpoint replay differs: {difference}')
    return difference


def evaluate_snapshots(source,output,data_root,clip_checkpoint,face_manifest_root,preflight=False,device_name='cuda',split='val'):
    require(split in ('val','test'),'Unsupported evaluation split')
    begin=time.time()
    config,summary,audit=audit_source(source)
    require(not output.exists(),'Output already exists')
    metadata=git_metadata(Path(__file__).resolve().parents[2]);require(not metadata['dirty'],'Evaluator source is dirty')
    require(_sha256(clip_checkpoint) == config['clip_checkpoint_sha256'],'CLIP file changed')
    require((Path(config['data_root']).resolve() == data_root.resolve()),'Dataset root changed')
    manifest_paths=[face_manifest_root/'manifests/val.jsonl',face_manifest_root/'detector_config.json']
    if split=='test':manifest_paths.append(face_manifest_root/'manifests/test.jsonl')
    manifest_hashes={str(p):_sha256(p) for p in manifest_paths}
    require(manifest_hashes[str(manifest_paths[0])] == config['face_manifest']['artifact_sha256']['val_manifest'],'Validation Face manifest changed')
    require(manifest_hashes[str(manifest_paths[1])] == config['face_manifest']['artifact_sha256']['detector_config'],'Face detector config changed')
    if split=='test':
        require(manifest_hashes[str(manifest_paths[2])] == config['face_manifest']['artifact_sha256']['test_manifest'],'Test Face manifest changed')
    files=[source/name for name in ('config.json','seed_summary.json','paired_protocol_audit.json','image_stream_transfer_audit.json','training_history.json')]
    files += [source/f'{folder}/task{t}.{suffix}' for t in range(8) for folder,suffix in (('compact_checkpoints','pth'),('val_scores','npz'))]
    if split=='test':files.append(source/'test_scores/task7.npz')
    hashes={str(p):_sha256(p) for p in files}
    set_seed(config['seed'],config['tf32']);device=torch.device(device_name)
    require(device.type=='cpu' or torch.cuda.is_available(),'CUDA unavailable')
    model=build_model(config,load_openai_clip_visual(clip_checkpoint),task_sizes=config['task_sizes']).float().to(device)
    visual_hash=tensor_state_digest(model.visual_encoder.state_dict())
    require(visual_hash == audit['frozen_visual_sha256_after'],'Restored CLIP differs')
    validation=make_source(config,data_root,face_manifest_root,'val');validate_classes(validation)
    require(len(validation)==2397,'Expected complete2397-person validation')
    loader=None
    expected_people=2397 if split=='val' else 5368
    final_payload=_torch_load(source/'compact_checkpoints/task7.pth')
    output.mkdir(parents=True);blocks=[];task_rows={}
    for task in ([0,7] if preflight else range(8)):
        payload=_torch_load(source/f'compact_checkpoints/task{task}.pth')
        assert_owned_equal(payload['model'],final_payload['model'],task,config['task_sizes'])
        restore_compact_model_state(model,payload,task,config['git']);model.requires_grad_(False).eval()
        before=tensor_state_digest(compact_model_state_dict(model))
        difference=replay(model,validation,source,config,task,device)
        if loader is None:
            cohort=validation if split=='val' else make_source(config,data_root,face_manifest_root,'test')
            validate_classes(cohort);require(len(cohort)==expected_people,'Incomplete evaluation cohort')
            loader=DataLoader(LabelView(cohort,range(len(cohort)),range(26),include_sample_id=True),
                              batch_size=config['eval_batch_size'],shuffle=False,num_workers=config['workers'])
        block,full=infer(model,loader,device,config['amp'],task,config['task_sizes'],1 if preflight else None)
        require(tensor_state_digest(compact_model_state_dict(model))==before,'Inference changed weights')
        require(all(p.grad is None for p in model.parameters()),'Inference created parameter gradients')
        np.savez_compressed(output/f'task{task}_owned_scores.npz',**block)
        if not preflight: blocks.append(block)
        task_rows[str(task)]=dict(checkpoint_sha256=hashes[str(source/f'compact_checkpoints/task{task}.pth')],
                                  replay_max_logit_difference=difference,owned_parameters_equal_final=True,
                                  weights_unchanged=True,samples=len(block['sample_ids']))
        print(f'SNAPSHOT_{split.upper()} task={task} samples={len(block["sample_ids"])} replay={difference:.8g}',flush=True)
        if task==7:
            final=dict(full)
            final['targets']=final.pop('full_targets');final['class_indices']=np.arange(26)
            np.savez_compressed(output/f'final_model_{split}.npz',**final)
    result=dict(status='preflight_passed' if preflight else 'complete',source=str(source.resolve()),source_config=config,
                evaluator_git=metadata,seed=config['seed'],preflight=preflight,evaluation_split=split,
                test_images_or_scores_opened=split=='test',test_weight_search=False,training_performed=False,optimizer_updates=0,
                source_sha256=hashes,face_manifest_sha256=manifest_hashes,tasks=task_rows)
    if not preflight:
        assembled=assemble_blocks(blocks,config['task_sizes'])
        require(np.array_equal(assembled['sample_ids'],final['sample_ids']) and np.array_equal(assembled['targets'],final['targets']),'Final/assembled cohorts differ')
        np.savez_compressed(output/f'assembled_{split}.npz',**assembled)
        result.update(final_metrics=score_metrics(final,config['threshold']),snapshot_metrics=score_metrics(assembled,config['threshold']),
                      samples=len(assembled['sample_ids']),sample_ids_sha256=ids_digest(assembled['sample_ids'].tolist()))
        reference=load_scores(source/f'{split}_scores/task7.npz')
        require(np.array_equal(final['sample_ids'],reference['sample_ids']) and np.array_equal(final['targets'],reference['targets']),'Full final replay cohort differs')
        result['final_replay_max_logit_difference']=float(np.abs(final['logits']-reference['logits']).max())
        require(result['final_replay_max_logit_difference'] <= .01,'Full final checkpoint replay differs')
        require(abs(result['final_metrics']['fused']['mAP']-summary['metrics' if split=='val' else 'test_metrics']['final_mAP']) <= .02,'Final validation mAP differs')
    require(tensor_state_digest(model.visual_encoder.state_dict())==visual_hash,'Inference changed CLIP')
    require(all(_sha256(Path(p))==v for p,v in hashes.items()),'Source artifact changed')
    require(all(_sha256(Path(p))==v for p,v in manifest_hashes.items()),'Face artifacts changed')
    result.update(source_unchanged=True,clip_unchanged=True,elapsed_seconds=time.time()-begin)
    (output/'evaluation_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(f'SNAPSHOT_{split.upper()}_COMPLETE '+result['status'],flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('source','output','data-root','clip-checkpoint','face-manifest-root'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--split',choices=('val','test'),default='val')
    args=parser.parse_args()
    evaluate_snapshots(args.source,args.output,args.data_root,args.clip_checkpoint,args.face_manifest_root,args.preflight,args.device,args.split)
