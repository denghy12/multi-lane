"""Train a single fusion rule per stage and broadcast each image's vector to all its people and heads."""
import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, average_precision, compute_metrics, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.r1_asymmetric_forward import fuse_logits

METHODS = ('constant', 'full', 'full_relation')
PERMUTATIONS = 100


@dataclass
class Stage:
    image_ids: np.ndarray
    full: torch.Tensor
    relation: torch.Tensor
    counts: np.ndarray
    reliable_counts: np.ndarray
    usable: torch.Tensor
    inverse: torch.Tensor
    observed: torch.Tensor
    reliable: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    historical_logits: torch.Tensor
    archive_sha256: str
    features_sha256: str
    checkpoint_sha256: str


def image_rows(sample_ids, split):
    keys = []
    for item in sample_ids.tolist():
        image, sep, person = item.rpartition('#person=')
        if not sep or not image.startswith(split + ':') or not person.isdecimal():
            raise ValueError(f'Invalid original image in {item}')
        keys.append(image)
    if len(set(sample_ids.tolist())) != len(keys):
        raise ValueError('Duplicate labeled person')
    return np.asarray(keys)


def load_stage(source, features, split, task, smoke):
    training = split == 'train'
    if training:
        path = source/'paired_A'/f'task{task}_held.npz'
        meta = read(source/'paired_A'/f'task{task}_center.json')
        digest = meta['archive_sha256']
        checkpoint = meta['checkpoint_sha256']
        if meta['protocol']['classes'] != list(task_indices(task)):
            raise ValueError('Mismatched current TRAIN classes')
        size = TASK_SIZES[task]
    else:
        path = source/'evaluation'/split/f'task{task}.npz'
        meta = read(path.with_suffix('.json'))
        digest = meta['scores_sha256']
        checkpoint = meta['protocol']['checkpoint_sha256']
        size = sum(TASK_SIZES[:task+1])
    protocol = meta['protocol']
    if protocol['task'] != task or protocol['split'] != split or protocol['smoke']:
        raise ValueError('Expected archived formal score protocol')
    require_digest(path, digest)
    feat = features/split/f'task{task}.npz'
    with np.load(feat, allow_pickle=False) as f, np.load(path, allow_pickle=False) as z:
        image_ids = f['image_ids']
        keys = image_rows(z['sample_ids'], split)
        positions = {k: i for i,k in enumerate(image_ids.tolist())}
        if len(positions) != len(image_ids) or not len(positions):
            raise ValueError('Expected distinct original images')
        take = np.isin(keys, image_ids)
        if set(keys[take].tolist()) != set(positions) or not (smoke or take.all()):
            raise ValueError('Feature images and archived scoring population differ')
        inverse = np.array([positions[k] for k in keys[take]], dtype=np.int64)
        observed = np.bincount(inverse, minlength=len(image_ids))
        counts = f['counts']
        rc = f['reliable_counts']
        usable = f['all_reliable']
        reliable = z['reliable'][take]
        if any(arr.shape != (len(image_ids),) for arr in (counts,rc,usable)):
            raise ValueError('Inconsistent image roster lengths')
        if (usable.dtype != np.bool_ or reliable.dtype != np.bool_ or not np.array_equal(usable, rc == counts)
                or np.any(counts < observed) or np.any(rc < np.bincount(inverse[reliable], minlength=len(counts)))
                or np.any(counts-rc < np.bincount(inverse[~reliable], minlength=len(counts)))):
            raise ValueError('Roster and scored person availability disagree')
        branches = torch.from_numpy(z['branch_logits'][take]).float()
        labels = torch.from_numpy(z['targets'][take]).float()
        full = torch.from_numpy(f['full']).float()
        relation = torch.from_numpy(f['relation']).float()
        historical = torch.empty(0) if training else torch.from_numpy(z['fixed_logits'][take]).float()
        if (branches.shape != (len(inverse),size,3) or labels.shape != branches.shape[:2]
                or full.shape != (len(image_ids),8) or relation.ndim != 2 or relation.shape[0] != len(full)
                or not relation.shape[1] or not all(torch.isfinite(x).all() for x in (branches,labels,full,relation))
                or not ((labels == 0) | (labels == 1)).all()):
            raise ValueError('Invalid score or feature shapes/values')
        if not training:
            if list(z['view_order']) != ['full','person','face'] or list(z['task_sizes']) != list(TASK_SIZES[:task+1]):
                raise ValueError('Unexpected view or head order')
            replay = fuse_logits(*branches.unbind(-1), torch.from_numpy(reliable))
            torch.testing.assert_close(replay,historical,atol=5e-6,rtol=2e-6)
    return Stage(image_ids,full,relation,counts,rc,torch.from_numpy(usable),torch.from_numpy(inverse),
                 torch.from_numpy(observed),torch.from_numpy(reliable),branches,labels,historical,
                 digest,sha256_file(feat),checkpoint)


def standardized_features(stage, normalization):
    raw = torch.cat((stage.full,stage.relation),dim=1)
    if normalization is None:
        std = raw.std(dim=0,unbiased=False)
        normalization = dict(mean=raw.mean(dim=0).tolist(), scale=torch.where(std>1e-6,std,1.).tolist())
    x = (raw-torch.tensor(normalization['mean']))/torch.tensor(normalization['scale'])
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite standardized features')
    return dict(constant=x[:,:0],full=x[:,:8],full_relation=x), normalization


def probabilities(bias, matrix, x, usable):
    scores = bias[usable.long()] + x @ matrix
    masked = scores.clone()
    masked[~usable,2] = -torch.inf
    return masked.softmax(dim=1)


def fused(branch, weights, inverse):
    row = weights[inverse]
    return (branch[:,:,0]*row[:,0:1] + branch[:,:,1]*row[:,1:2] + branch[:,:,2]*row[:,2:3])


def image_loss(scores, labels, inverse, counts):
    losses = F.binary_cross_entropy_with_logits(scores, labels, reduction='none').mean(dim=1)
    return (losses/counts[inverse]).sum()/len(counts)


def fit_gate(stage, x, previous, steps):
    b = torch.zeros(2,3)
    m = torch.zeros(x.shape[1],3)
    if previous is not None:
        b.copy_(torch.tensor(previous['bias']))
        m.copy_(torch.tensor(previous['matrix']).reshape_as(m))
    b.requires_grad_()
    m.requires_grad_()
    opt = torch.optim.Adam([b,m], lr=.01)
    for _ in range(steps):
        opt.zero_grad()
        weights = probabilities(b,m,x,stage.usable)
        logits = fused(stage.logits,weights,stage.inverse)
        loss = image_loss(logits,stage.targets,stage.inverse,stage.observed)
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(v.grad).all() for v in (b,m)):
            raise FloatingPointError('Nonfinite TRAIN loss/gradient')
        opt.step()
    return dict(bias=b.detach().tolist(),matrix=m.detach().tolist(),steps=steps)


def train(source, features, out, smoke):
    states, normalization, previous = [],None,{}
    for task in range(1 if smoke else len(TASK_SIZES)):
        stage = load_stage(source,features,'train',task,smoke)
        xs, normalization = standardized_features(stage,normalization)
        gates = {m:fit_gate(stage,xs[m],previous.get(m),300) for m in METHODS}
        states.append(dict(task=task,gates=gates,people=len(stage.inverse),images=len(stage.image_ids),
                           archive_sha256=stage.archive_sha256,features_sha256=stage.features_sha256,
                           checkpoint_sha256=stage.checkpoint_sha256))
        previous = gates
        print(f'fit task{task} {len(stage.image_ids)} images {len(stage.inverse)} people',flush=True)
    write_verified(out/'fit.json',dict(smoke=smoke,states=states,normalization=normalization,
                   protocol='natural images; one common gate across all seen heads; image-balanced TRAIN BCE; '
                            'task0 TRAIN feature normalization; inherit parameters and reset Adam per task'))


def mask_means(weights, usable):
    mean = weights.clone()
    for flag in (True,False):
        mask = usable == flag
        if mask.any():
            mean[mask] = weights[mask].mean(dim=0)
    return mean


def permute_images(ids, usable, rng):
    donors = np.arange(len(ids))
    ordered = np.argsort(ids,kind='stable')
    for flag in (True,False):
        rows = ordered[usable[ordered] == flag]
        if len(rows) > 1:
            cycle = rng.permutation(rows)
            donors[cycle] = np.roll(cycle,1)
    return torch.from_numpy(donors)


def groups(stage):
    mask = stage.usable[stage.inverse]
    mixed = (stage.reliable_counts>0) & (stage.reliable_counts<stage.counts)
    masks = dict(all=torch.ones_like(mask),reliable=stage.reliable,unreliable=~stage.reliable,
                 image_all_faces=mask,image_face_off=~mask,image_mixed=torch.from_numpy(mixed)[stage.inverse])
    return {g:v for g,v in masks.items() if v.any()}


def permutation_ap(stage,weights,masks,rng,repetitions):
    # Compute official AP separately for every exchange before averaging.
    labels = stage.targets.numpy()
    result = {m:{g:[] for g in masks} for m in weights}
    for repeat in range(repetitions):
        donor = permute_images(stage.image_ids,stage.usable.numpy(),rng)
        for m,w in weights.items():
            logits = fused(stage.logits,w[donor],stage.inverse)
            for g,mask in masks.items():
                prob = logits[mask].sigmoid().numpy()
                y = labels[mask.numpy()]
                result[m][g].append([100*average_precision(s,a) for s,a in zip(prob.T,y.T)])
    report = {}
    for m,by_group in result.items():
        report[m] = {}
        for g,samples in by_group.items():
            ap = np.asarray(samples)
            report[m][g] = dict(mAP=float(ap.mean()),per_class_ap=ap.mean(axis=0).tolist(),
                                shuffle_std_mAP=float(ap.mean(axis=1).std()))
    return report


def save_weights(path, stage, weights):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(image_ids=stage.image_ids,counts=stage.counts,reliable_counts=stage.reliable_counts,
                all_reliable=stage.usable.numpy(),**{m:w.numpy() for m,w in weights.items()})
    if path.exists():
        with np.load(path,allow_pickle=False) as prior:
            if set(prior.files) != set(data):
                raise ValueError('Weights archive has different schema')
            for key,array in data.items():
                np.testing.assert_array_equal(prior[key],array)
    else:
        tmp = path.with_suffix(f'.tmp.{os.getpid()}.npz')
        np.savez_compressed(tmp,**data)
        tmp.replace(path)


def evaluate(source,features,out,split,smoke):
    fit = read(out/'fit.json')
    digest = sha256_file(out/'fit.json')
    if fit['smoke'] != smoke or len(fit['states']) != (1 if smoke else len(TASK_SIZES)):
        raise ValueError('Inconsistent TRAIN lock')
    seal = out/'validation_seal.json'
    if split == 'test':
        locked = read(seal)
        require_digest(out/'fit.json',locked['fit_sha256'])
        require_digest(out/'val_results.json',locked['val_sha256'])
    stages = []
    for row in fit['states']:
        task = row['task']
        stage = load_stage(source,features,split,task,smoke)
        if row['checkpoint_sha256'] != stage.checkpoint_sha256:
            raise ValueError('Unpaired branch checkpoints')
        xs,_ = standardized_features(stage,fit['normalization'])
        gates = row['gates']
        weights = {m:probabilities(torch.tensor(gates[m]['bias']),torch.tensor(gates[m]['matrix']).reshape(xs[m].shape[1],3),
                                   xs[m],stage.usable) for m in METHODS}
        weights['image_fixed'] = torch.tensor([[.80,.20,0.],[.64,.16,.20]])[stage.usable.long()]
        for w in weights.values():
            if (not torch.isfinite(w).all() or not torch.allclose(w.sum(1),torch.ones(len(w)),atol=1e-6)
                    or torch.any(w[~stage.usable,2] != 0)):
                raise ValueError('Invalid per-image weights')
        wpath = out/'weights'/split/f'task{task}.npz'
        save_weights(wpath,stage,weights)
        for m in ('full','full_relation'):
            weights[m+'_mean'] = mask_means(weights[m],stage.usable)
        predictions = {m:fused(stage.logits,w,stage.inverse) for m,w in weights.items()}
        predictions['historical_fixed'] = stage.historical_logits
        masks = groups(stage)
        metrics = {m:{g:asdict(compute_metrics(task,logits[mask].sigmoid(),stage.targets[mask],.5))
                      for g,mask in masks.items()} for m,logits in predictions.items()}
        if not smoke:
            historical = read(source/'evaluation'/split/f'task{task}.json')['metrics']['fixed']
            if any(metrics['historical_fixed'][g] != value for g,value in historical.items()):
                raise ValueError('Historical fixed metrics did not replay')
        permutation = permutation_ap(stage,{m:weights[m] for m in ('full','full_relation')},masks,
                                     np.random.default_rng(314159+task),PERMUTATIONS)
        counts_per_group = {g:int(mask.sum()) for g,mask in masks.items()}
        stages.append(dict(task=task,images=len(stage.image_ids),people=len(stage.inverse),groups=counts_per_group,
                           features_sha256=stage.features_sha256,archive_sha256=stage.archive_sha256,
                           weights_sha256=sha256_file(wpath),metrics=metrics,permutation=permutation))
        print(split,task,{m:round(metrics[m]['all']['mAP'],4) for m in metrics},flush=True)
    summary = {}
    for row in stages:
        for m,gs in row['metrics'].items():
            for g,v in gs.items():
                summary.setdefault(m,{}).setdefault(g,[]).append(v['mAP'])
        for m,gs in row['permutation'].items():
            for g,v in gs.items():
                summary.setdefault(m+'_shuffled',{}).setdefault(g,[]).append(v['mAP'])
    # If a stratum was absent in any stage, do not call a partial average an eight-stage score.
    summary = {m:{g:dict(final_mAP=vals[-1],average_mAP=float(np.mean(vals))) for g,vals in gs.items()
                  if len(vals)==len(stages)} for m,gs in summary.items()}
    dest = out/f'{split}_results.json'
    report = dict(split=split,smoke=smoke,fit_sha256=digest,stages=stages,summary=summary,permutations=PERMUTATIONS)
    if split == 'test':
        report['val_seal_sha256'] = sha256_file(seal)
    write_verified(dest,report)
    if split == 'val':
        write_verified(seal,dict(fit_sha256=digest,val_sha256=sha256_file(dest)))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('source','features','out'):
        cli.add_argument('--'+key,type=Path,required=True)
    cli.add_argument('--phase',choices=('fit','val','test'),required=True)
    cli.add_argument('--smoke',action='store_true')
    args = cli.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    if args.phase == 'fit':
        train(args.source,args.features,args.out,args.smoke)
    else:
        evaluate(args.source,args.features,args.out,args.phase,args.smoke)
