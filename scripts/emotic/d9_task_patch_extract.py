"""Read each seen Task Forward's own final CLS and Frozen9 image stream."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from multi_lane.track_a.final_patch_fusion import VIEWS, WIDTH, PROJECTION_SEED, project_tokens
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import set_seed
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_fill_image_source import native_scores
from scripts.emotic.d9_final_patch_extract import validate_capture, CAPTURE_FILES
from scripts.emotic.d9_final_patch_readout import PatchStage, subset
from scripts.emotic.d9_summary_readout import load_stage as load_base, subset as subset_base, availability
from scripts.emotic.d9_image_gate_extract import OriginalImages, pack_images, keep_images, group_key
from scripts.emotic.run_r1_frozen_cross_view import input_paths
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.evaluate_d9_relation import stage_loader
from scripts.emotic.d9_paired_a_evaluate import write_verified

SCHEMA = 'all_seen_task_final_tokens_v1'
ROOT = Path(__file__).resolve().parents[2]


def capture_code():
    return {p:sha256_file(ROOT/p) for p in (*CAPTURE_FILES, 'scripts/emotic/d9_task_patch_extract.py')}


def load_features(cfg, name, out, split, task, head=None, smoke=False):
    """Labels/logits always belong to stage task; feature head may be older."""
    head = task if head is None else head
    if not 0 <= head <= task or split == 'train' and head != task:
        raise ValueError('Cannot use future features or old labels during fitting')
    base = load_base(Path(cfg['sources'][name]['path']), Path(cfg['cache'])/name/'features',
                     split, task, False, current_only=split == 'train')
    parent = Path(cfg['parent'])
    path = (parent/name/'features'/split/f'task{task}.npz' if split == 'train'
            else out/'features'/split/f'task{task}'/f'head{head}.npz')
    meta = read(path.with_suffix('.json'))
    require_digest(path, meta['archive_sha256'])
    spec = meta['protocol']
    for key in ('native_sha256','checkpoint_sha256','config_sha256'):
        if spec[key] != base.provenance[key]:
            raise ValueError('Feature/native identity differs: '+key)
    if split == 'train':
        parent_code = read(parent/'protocol.json')['code_sha256']
        if any(parent_code[k] != v for k,v in spec['capture_code'].items()):
            raise ValueError('Unrecognized parent capture code')
        if spec['task'] != task or spec['split'] != split or spec['smoke']:
            raise ValueError('Expected full original TRAIN cache')
    elif (spec['schema'] != SCHEMA or spec['task'] != task or spec['head'] != head
          or spec['split'] != split or spec['smoke'] != smoke or spec['capture_code'] != capture_code()):
        raise ValueError('Wrong task path or capture code')
    with np.load(path, allow_pickle=False) as z:
        ids = z['image_ids']
        positions = {str(k):i for i,k in enumerate(base.ids)}
        if len(set(ids.tolist())) != len(ids) or any(str(k) not in positions for k in ids):
            raise ValueError('Unknown or duplicate original image')
        base = subset_base(base, np.array([positions[str(k)] for k in ids]))
        for k,value in [('counts',base.counts),('reliable_counts',base.reliable_counts)]:
            np.testing.assert_array_equal(z[k],value)
        owner = np.repeat(np.arange(len(ids)),base.counts)
        np.testing.assert_array_equal(z['roster_inverse'],owner)
        roster, flags = z['roster_ids'],z['roster_reliable']
        if len(set(roster.tolist())) != len(roster) or flags.dtype != np.bool_:
            raise ValueError('Invalid person roster')
        np.testing.assert_array_equal([group_key(p,split) for p in roster],ids[owner])
        mapping = {str(p):i for i,p in enumerate(roster)}
        take = [mapping[str(p)] for p in base.person_ids]
        np.testing.assert_array_equal(flags[take],base.reliable.numpy())
        np.testing.assert_array_equal(owner[take],base.inverse.numpy())
        np.testing.assert_array_equal(np.bincount(owner[flags],minlength=len(ids)),base.reliable_counts)
        if hashlib.sha256('\n'.join(roster.tolist()).encode()).hexdigest() != meta['roster_sha256']:
            raise ValueError('Roster digest differs')
        cls,patch = z['cls'],z['patch']
        if (cls.shape != (len(roster),3,32) or patch.shape != (len(roster),3,196,32)
                or cls.dtype != np.float32 or patch.dtype != np.float16
                or not np.isfinite(cls).all() or not np.isfinite(patch).all()
                or np.any(cls[~flags,2]) or np.any(patch[~flags,2])):
            raise ValueError('Expected complete, finite, masked 196-patch inputs')
        stage = PatchStage(base,torch.from_numpy(cls),torch.from_numpy(patch),roster,owner,
                           torch.from_numpy(flags),meta['archive_sha256'])
    if split == 'train' and smoke:
        kinds = availability(base)
        keep = np.sort(np.concatenate([np.flatnonzero(kinds == k)[:12] for k in (0,1,2)]))
        stage = subset(stage,keep)
    return stage


@torch.inference_mode()
def verify_capture(model, source, task, workers):
    meta = read(source/'evaluation/val'/f'task{task}.json')
    path = source/'evaluation/val'/f'task{task}.npz'
    require_digest(path,meta['scores_sha256'])
    data = build_joint_source(input_paths(),'val')
    loader = stage_loader(data,task,False,meta['protocol']['batch_size'],workers)
    images,_,ids = next(iter(loader))
    inputs = move_model_inputs(images,torch.device('cuda'))
    before = {k:v.detach().clone() for k,v in model.state_dict().items()
              if not k.startswith('branches.full.visual_encoder.')}
    all_tokens, current = {},{}
    with torch.autocast('cuda'):
        plain = model(inputs,all_seen=True)
        actual = model(inputs,all_seen=True,task_final_tokens=all_tokens,final_tokens=current)
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(ids,z['sample_ids'][:len(ids)])
        for i,v in enumerate(VIEWS):
            torch.testing.assert_close(plain['branches'][v],actual['branches'][v],rtol=0,atol=0)
            np.testing.assert_array_equal(actual['branches'][v].float().cpu().numpy(),z['branch_logits'][:len(ids),:,i])
            if set(all_tokens[v]) != set(range(task+1)):
                raise ValueError('Missing task path')
            for key in ('cls','patch'):
                torch.testing.assert_close(all_tokens[v][task][key],current[v][key],rtol=0,atol=0)
    for head in range(task+1):
        validate_capture({v:all_tokens[v][head] for v in VIEWS},len(ids))
    for k,value in before.items():
        torch.testing.assert_close(model.state_dict()[k],value,rtol=0,atol=0)
    return dict(native_logits_exact=True,current_capture_exact=True,state_unchanged=True,
                heads=list(range(task+1)),people=len(ids))


@torch.inference_mode()
def extract(cfg,name,out,split,task,smoke):
    if split not in ('val','test'):
        raise ValueError('Reuse sealed TRAIN features')
    set_seed(cfg['seed'],True);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    source = Path(cfg['sources'][name]['path'])
    native,_,digest,checkpoint = native_scores(source,Path(cfg['cache'])/name/'features',split,task)
    paths = input_paths()
    spec = dict(schema=SCHEMA,task=task,split=split,smoke=smoke,
                native_sha256=digest,checkpoint_sha256=checkpoint,
                config_sha256=sha256_file(source/'config.json'),capture_code=capture_code(),
                width=WIDTH,projection_seed=PROJECTION_SEED,batch_size=64,
                mat_sha256=sha256_file(paths.data_root/'CVPR17_Annotations.mat'),
                manifest_sha256=sha256_file(paths.face_manifest_root/'manifests'/f'{split}.jsonl'))
    dest = out/'features'/split/f'task{task}'
    existing = []
    for head in range(task+1):
        p = dest/f'head{head}.npz'
        if p.with_suffix('.json').exists():
            metadata = read(p.with_suffix('.json'))
            if metadata['protocol'] != dict(spec,head=head):raise ValueError('Task feature protocol changed')
            require_digest(p,metadata['archive_sha256']);existing.append(head)
        elif p.exists():raise ValueError('Unsealed task feature file')
    if len(existing) == task+1:return
    with np.load(native,allow_pickle=False) as z:
        dataset = OriginalImages(paths,split,z['sample_ids'],z['reliable'],smoke)
    loader = DataLoader(dataset,batch_sampler=pack_images(map(len,dataset.rows),64),
                        num_workers=cfg['extraction']['workers'],collate_fn=keep_images,
                        worker_init_fn=single_thread_worker,
                        **({'prefetch_factor':1} if cfg['extraction']['workers'] else {}))
    model = load_evaluation_model(source,task).requires_grad_(False).eval()
    replay = verify_capture(model,source,task,cfg['extraction']['workers'])
    names,counts,flags_rows=[],[],[]
    cls_rows = {h:[] for h in range(task+1)}
    patch_rows = {h:[] for h in range(task+1)}
    for groups in loader:
        ids,full,person,face,flags=zip(*groups)
        sizes=[len(x) for x in flags];mask=torch.cat(flags)
        owner=torch.repeat_interleave(torch.arange(len(ids)),torch.tensor(sizes))
        full,person,face=torch.stack(full),torch.cat(person),torch.cat(face)
        for start in range(0,len(mask),64):
            rows=slice(start,start+64)
            pixels=dict(full=full[owner[rows]].cuda(),person=person[rows].cuda(),face=face[rows].cuda())
            inputs={**pixels,**{'evidence_'+k:v for k,v in pixels.items()},'evidence_face_reliable':mask[rows].cuda()}
            capture={}
            with torch.autocast('cuda'):model(inputs,all_seen=True,task_final_tokens=capture)
            for h in range(task+1):
                values={v:capture[v][h] for v in VIEWS};validate_capture(values,len(mask[rows]))
                cls=torch.stack([project_tokens(values[v]['cls'],v,'cls') for v in VIEWS],1)
                patches=torch.stack([project_tokens(values[v]['patch'],v,'patch') for v in VIEWS],1)
                cls[~inputs['evidence_face_reliable'],2]=0;patches[~inputs['evidence_face_reliable'],2]=0
                cls_rows[h].append(cls.cpu().numpy());patch_rows[h].append(patches.half().cpu().numpy())
        names.extend(ids);counts.extend(sizes);flags_rows.append(mask.numpy())
    if names != dataset.names:raise ValueError('Image order differs')
    counts=np.array(counts);flags=np.concatenate(flags_rows);owner=np.repeat(np.arange(len(names)),counts)
    common=dict(image_ids=np.array(names),roster_ids=np.array(dataset.sample_ids),counts=counts,
                roster_inverse=owner,roster_reliable=flags,
                reliable_counts=np.bincount(owner[flags],minlength=len(names)))
    dest.mkdir(parents=True,exist_ok=True)
    for h in range(task+1):
        cls=np.concatenate(cls_rows.pop(h));patch=np.concatenate(patch_rows.pop(h))
        if h == task and not smoke:
            oldpath=Path(cfg['parent'])/name/'features'/split/f'task{task}.npz'
            oldmeta=read(oldpath.with_suffix('.json'));require_digest(oldpath,oldmeta['archive_sha256'])
            with np.load(oldpath,allow_pickle=False) as z:
                for k,value in {**common,'cls':cls,'patch':patch}.items():
                    np.testing.assert_array_equal(value,z[k])
        path=dest/f'head{h}.npz'
        if h not in existing:
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,**common,cls=cls,patch=patch)
            temp.replace(path)
            write_verified(path.with_suffix('.json'),dict(protocol=dict(spec,head=h),
                archive_sha256=sha256_file(path),replay=replay,
                roster_sha256=hashlib.sha256('\n'.join(dataset.sample_ids).encode()).hexdigest()))
        del cls,patch
    del model;torch.cuda.empty_cache()
    print(f'CAPTURED {name}/{split}/task{task}: {len(names)} images, {task+1} own task paths',flush=True)

