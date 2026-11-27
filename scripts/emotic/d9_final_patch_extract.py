"""Capture current-task final CLS and every final Frozen Forward spatial patch."""
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.final_patch_fusion import VIEWS, WIDTH, PROJECTION_SEED, project_tokens
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import set_seed
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_fill_image_source import native_scores, assert_train_roles, source_mode
from scripts.emotic.d9_image_gate_extract import OriginalImages, pack_images, keep_images
from scripts.emotic.evaluate_d9_relation import write_json, stage_loader
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_FILES = ('multi_lane/track_a/model.py', 'multi_lane/track_a/r1_crossvit.py',
    'multi_lane/track_a/r1_frozen_cross_view.py', 'multi_lane/track_a/d9_relation_fusion.py',
    'multi_lane/track_a/final_patch_fusion.py', 'scripts/emotic/d9_final_patch_extract.py')
SCHEMA = 'current_task_final_cls_spatial_patches_v1'


def capture_identity():
    return {p:sha256_file(ROOT/p) for p in CAPTURE_FILES}


def validate_capture(capture, batch):
    if set(capture) != set(VIEWS):
        raise ValueError('Missing a branch capture')
    for view in VIEWS:
        c, p = capture[view]['cls'], capture[view]['patch']
        if (set(capture[view]) != {'cls','patch'} or c.ndim != 2 or p.ndim != 3
                or len(c) != batch or len(p) != batch or p.shape[1] != 196
                or c.shape[1] != 512 or p.shape[2] != 768
                or not torch.isfinite(c).all() or not torch.isfinite(p).all()
                or c.requires_grad or p.requires_grad):
            raise ValueError('Expected detached ViT-B/16 final CLS512 and 196 spatial patches768')


@torch.inference_mode()
def verify_capture(model, source, task, workers):
    path = source/'evaluation/val'/f'task{task}.npz'
    meta = read(path.with_suffix('.json')); require_digest(path, meta['scores_sha256'])
    validation = build_joint_source(input_paths(), 'val')
    loader = stage_loader(validation, task, False, meta['protocol']['batch_size'], workers)
    images, _, ids = next(iter(loader)); inputs = move_model_inputs(images, torch.device('cuda'))
    capture = {}; before = {k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('branches.full.visual_encoder.')}
    with torch.autocast('cuda'):
        plain = model(inputs, all_seen=True)
        hooked = model(inputs, all_seen=True, final_tokens=capture)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(ids), z['sample_ids'][:len(ids)])
        for i, view in enumerate(VIEWS):
            torch.testing.assert_close(plain['branches'][view], hooked['branches'][view], rtol=0, atol=0)
            np.testing.assert_array_equal(hooked['branches'][view].float().cpu().numpy(), z['branch_logits'][:len(ids), :, i])
    validate_capture(capture, len(ids))
    for k,value in before.items():torch.testing.assert_close(model.state_dict()[k], value, rtol=0, atol=0)
    return dict(people=len(ids), native_batch_size=meta['protocol']['batch_size'],
                capture_on_off_max_difference=0., native_logits_max_difference=0., state_unchanged=True)


@torch.inference_mode()
def calculate(model, loader, batch_size):
    names, cls_rows, patch_rows, counts, flags_rows = [], [], [], [], []
    start_time=time.monotonic()
    for groups in loader:
        ids, full, person, face, flags = zip(*groups)
        sizes=[len(f) for f in flags]; masks=torch.cat(flags)
        owner=torch.repeat_interleave(torch.arange(len(ids)),torch.tensor(sizes))
        full,person,face=torch.stack(full),torch.cat(person),torch.cat(face)
        for start in range(0,len(masks),batch_size):
            rows=slice(start,start+batch_size)
            pixels=dict(full=full[owner[rows]].cuda(),person=person[rows].cuda(),face=face[rows].cuda())
            inputs={**pixels,**{'evidence_'+k:v for k,v in pixels.items()},'evidence_face_reliable':masks[rows].cuda()}
            capture={}
            with torch.autocast('cuda'): model(inputs,all_seen=True,final_tokens=capture)
            validate_capture(capture,len(masks[rows]))
            cls=torch.stack([project_tokens(capture[v]['cls'],v,'cls') for v in VIEWS],1)
            patches=torch.stack([project_tokens(capture[v]['patch'],v,'patch') for v in VIEWS],1)
            cls[~inputs['evidence_face_reliable'],2]=0
            patches[~inputs['evidence_face_reliable'],2]=0
            cls_rows.append(cls.cpu().numpy());patch_rows.append(patches.half().cpu().numpy())
        names.extend(ids);counts.extend(sizes);flags_rows.append(masks.numpy())
        if len(names)==len(loader.dataset) or len(names)%100<len(ids):
            print(f'PATCHES {len(names)}/{len(loader.dataset)} images; {time.monotonic()-start_time:.1f}s',flush=True)
    if names!=loader.dataset.names:raise ValueError('Complete image roster order changed')
    counts=np.asarray(counts,dtype=np.int64);flags=np.concatenate(flags_rows)
    inverse=np.repeat(np.arange(len(names)),counts)
    return dict(image_ids=np.asarray(names),roster_ids=np.asarray(loader.dataset.sample_ids),
                counts=counts,roster_inverse=inverse,roster_reliable=flags,
                reliable_counts=np.bincount(inverse[flags],minlength=len(names)),
                cls=np.concatenate(cls_rows),patch=np.concatenate(patch_rows))


def extract(cfg,name,out,split,task,smoke):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2);set_seed(cfg['seed'],True)
    source=Path(cfg['sources'][name]['path']);prior=Path(cfg['cache'])/name/'features'
    native,_,digest,checkpoint=native_scores(source,prior,split,task)
    inputs=input_paths();batch=cfg['extraction']['batch_size'];workers=cfg['extraction']['workers']
    spec=dict(schema=SCHEMA,task=task,split=split,smoke=smoke,source_mode=source_mode(source),
              native_sha256=digest,checkpoint_sha256=checkpoint,config_sha256=sha256_file(source/'config.json'),
              roles_sha256=sha256_file(source/'train_image_roles.json'),batch_size=batch,
              projection_seed=PROJECTION_SEED,width=WIDTH,patch_dtype='float16',cls_dtype='float32',
              normalization='per-token layer_norm before fixed normalized Gaussian columns',
              capture_code=capture_identity(),
              mat_sha256=sha256_file(inputs.data_root/'CVPR17_Annotations.mat'),
              manifest_sha256=sha256_file(inputs.face_manifest_root/'manifests'/f'{split}.jsonl'))
    path=out/split/f'task{task}.npz';meta=path.with_suffix('.json')
    if meta.exists():
        old=read(meta)
        if old['protocol']!=spec:raise ValueError('Existing final-token feature protocol differs')
        require_digest(path,old['archive_sha256']);return
    if path.exists():raise ValueError('Unsealed patch cache exists; use a fresh output directory')
    with np.load(native,allow_pickle=False) as z:
        dataset=OriginalImages(inputs,split,z['sample_ids'],z['reliable'],smoke)
    if split=='train':assert_train_roles(source,dataset.names)
    loader=DataLoader(dataset,batch_sampler=pack_images(map(len,dataset.rows),batch),num_workers=workers,
                      collate_fn=keep_images,worker_init_fn=single_thread_worker,
                      **({'prefetch_factor':1} if workers else {}))
    model=load_evaluation_model(source,task).requires_grad_(False).eval()
    replay=verify_capture(model,source,task,workers);fields=calculate(model,loader,batch)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**fields)
    tmp.replace(path)
    write_json(meta,dict(protocol=spec,archive_sha256=sha256_file(path),replay=replay,
               images=len(dataset.names),roster_people=len(dataset.sample_ids),
               roster_sha256=hashlib.sha256('\n'.join(dataset.sample_ids).encode()).hexdigest(),
               raw_split_people=dataset.raw_people,scored_people=dataset.scored_people))
    del model;torch.cuda.empty_cache()
    print(f'CAPTURED {name}/{split}/task{task}: {len(dataset.names)} images, all 196 patches',flush=True)
