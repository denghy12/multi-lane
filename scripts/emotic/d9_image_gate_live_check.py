"""Check a few complete original images against a fresh task7 all-heads forward."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.r1_crossvit import VIEWS
from multi_lane.track_a.runner import TASK_SIZES, set_seed
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_image_gate_extract import OriginalImages, calculate, keep_images, pack_images
from scripts.emotic.d9_image_gate_readout import METHODS, fused, image_rows, probabilities
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def sample_rosters(names, counts, reliable_counts):
    # Only immutable, label-independent roster metadata determines selection.
    order = np.argsort(names, kind='stable')
    selected = []
    for mask in (counts == reliable_counts, reliable_counts == 0,
                 (reliable_counts > 0) & (reliable_counts < counts)):
        subset = order[mask[order]]
        multi = subset[counts[subset] > 1]
        single = subset[counts[subset] == 1]
        selected.extend(multi[:2])
        selected.extend(single[:1])
    if not selected or not np.any(counts[selected] > 1):
        raise ValueError('Expected multi-person examples')
    return np.array(sorted(selected), dtype=np.int64)


def errors(live, cached):
    a = np.asarray(live, dtype=np.float64)
    b = np.asarray(cached, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid shape/values in live check')
    d = np.abs(a-b)
    return {'max_absolute': float(d.max()), 'mean_absolute': float(d.mean()), 'n': int(d.size)}


@torch.inference_mode()
def branches_forward(model, dataset, batch_size, rows):
    result, positions, images = [], [], []
    for i, name in enumerate(dataset.names):
        _, full, person, face, reliable = dataset[i]
        for start in range(0, len(person), batch_size):
            j = slice(start, start+batch_size)
            p, f, ok = person[j].cuda(), face[j].cuda(), reliable[j].cuda()
            u = full.unsqueeze(0).expand(len(p), -1, -1, -1).contiguous().cuda()
            inputs = dict(full=u, evidence_full=u, person=p, evidence_person=p,
                          face=f, evidence_face=f, evidence_face_reliable=ok)
            with torch.autocast('cuda'):
                output = model(inputs, all_seen=True)
            actual = torch.stack([output['branches'][v].float().cpu() for v in VIEWS], dim=-1)
            # A box without a scored label is still included in gate features.
            source_ids = [dataset.sample_ids[k] for k in dataset.rows[i][j]]
            for k, sid in enumerate(source_ids):
                if sid in rows:
                    positions.append(rows[sid])
                    images.append(i)
                    result.append(actual[k])
    if len(set(positions)) != len(rows) or len(result) != len(rows):
        raise ValueError('Fresh branch forward omitted or duplicated an official scored person')
    return torch.stack(result), np.asarray(positions), torch.tensor(images)


def check(source, root, model, split, batch_size, fit):
    task = len(TASK_SIZES)-1
    readout = root/'readout'
    record = read(readout/f'{split}_results.json')
    if record['smoke'] or len(record['stages']) != len(TASK_SIZES):
        raise ValueError('Expected closed eight-task evaluation')
    require_digest(readout/'fit.json', record['fit_sha256'])
    stage = record['stages'][task]
    score_path = source/'evaluation'/split/f'task{task}.npz'
    feat_path = root/'features'/split/f'task{task}.npz'
    weight_path = readout/'weights'/split/f'task{task}.npz'
    for p,key in ((score_path,'archive_sha256'),(feat_path,'features_sha256'),(weight_path,'weights_sha256')):
        require_digest(p, stage[key])
    with np.load(feat_path) as cache, np.load(weight_path) as w, np.load(score_path) as scores:
        names = cache['image_ids']
        np.testing.assert_array_equal(names,w['image_ids'])
        ix = sample_rosters(names, cache['counts'], cache['reliable_counts'])
        scored_ids = scores['sample_ids']
        take = np.isin(image_rows(scored_ids, split), names[ix])
        dataset = OriginalImages(input_paths(), split, scored_ids[take], scores['reliable'][take], False)
        loader = DataLoader(dataset, batch_sampler=pack_images(map(len,dataset.rows),batch_size),
                            num_workers=0, collate_fn=keep_images)
        features = calculate(model, task, loader, batch_size, False)
        for key in ('image_ids','counts','reliable_counts','all_reliable'):
            np.testing.assert_array_equal(features[key],cache[key][ix])
        feature_errors = {k:errors(features[k],cache[k][ix]) for k in ('full','relation')}
        rows = {sid:j for j,sid in zip(np.flatnonzero(take),scored_ids[take])}
        live, positions, inverse = branches_forward(model, dataset, batch_size, rows)
        cached = torch.from_numpy(scores['branch_logits'][positions]).float()
        if live.shape[1:] != (sum(TASK_SIZES),3):
            raise ValueError('Live output failed to cover all views/heads')
        if list(scores['view_order']) != list(VIEWS):
            raise ValueError('Cache view order differs')
        branch_errors = {v:errors(live[:,:,j],cached[:,:,j]) for j,v in enumerate(VIEWS)}
        raw = torch.from_numpy(np.concatenate((features['full'],features['relation']),axis=1))
        norm = fit['normalization']
        x = (raw - torch.tensor(norm['mean'])) / torch.tensor(norm['scale'])
        xs = {'constant':x[:,:0], 'full':x[:,:8], 'full_relation':x}
        allowed = torch.from_numpy(features['all_reliable'])
        gates = fit['states'][task]['gates']
        compared = {}
        for m in (*METHODS,'image_fixed'):
            if m == 'image_fixed':
                actual = torch.tensor([[.8,.2,0.],[.64,.16,.2]])[allowed.long()]
            else:
                matrix = torch.tensor(gates[m]['matrix']).reshape(xs[m].shape[1],3)
                actual = probabilities(torch.tensor(gates[m]['bias']),matrix,xs[m],allowed)
            previous = torch.from_numpy(w[m][ix])
            estimated, original = fused(live,actual,inverse), fused(cached,previous,inverse)
            compared[m] = dict(weights=errors(actual,previous), fused_logits=errors(estimated,original),
                               fused_probabilities=errors(estimated.sigmoid(),original.sigmoid()))
        return dict(split=split,task=task, archive_sha256=stage['archive_sha256'],
                    fit_sha256=record['fit_sha256'], features_sha256=stage['features_sha256'],
                    weights_sha256=stage['weights_sha256'], images=dataset.names,
                    roster_people=features['counts'].tolist(), reliable_people=features['reliable_counts'].tolist(),
                    scored_people=len(rows),classes=sum(TASK_SIZES),branch_logits=branch_errors,
                    features=feature_errors,readouts=compared)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True, help='formal experiment directory')
    parser.add_argument('--batch-size',type=int,default=12)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Expected a positive batch size')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1,True)
    fit = read(args.root/'readout/fit.json')
    require_digest(args.source/'compact_checkpoints/task7.pth',fit['states'][-1]['checkpoint_sha256'])
    model = load_evaluation_model(args.source,7).requires_grad_(False).eval()
    results = {s:check(args.source,args.root,model,s,args.batch_size,fit) for s in ('val','test')}
    write_verified(args.root/'live_check.json', dict(batch_size=args.batch_size,splits=results))
    for s,r in results.items():
        print(s,r['scored_people'],'people, errors in full_relation',r['readouts']['full_relation'],flush=True)
