"""Decompose changes to Sadness AP on the same task6 people and frozen experts."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, average_precision, task_indices
from scripts.emotic.d9_sadness_pairs import attribute
from scripts.emotic.d9_sadness_replay import replay
from scripts.emotic.evaluate_d9_relation import write_json

TASK = 6


def ranked(prob, y):
    ranks = np.empty(len(prob), dtype=int)
    ranks[prob.argsort()[::-1]] = np.arange(1, len(prob) + 1)
    return dict(ap=100 * average_precision(prob, y),
                positives_in_top50=int(y[ranks <= 50].sum()),
                median_rank_positive=float(np.median(ranks[y.astype(bool)]))), ranks


def run(prior, tcp, out, split, mode):
    out.mkdir(exist_ok=True, parents=True)
    lock = out / 'val_lock.json'
    if split == 'test':
        record = json.loads(lock.read_text())
        if record['mode'] != mode or record['val_sha256'] != sha256_file(out/'val.json'):
            raise ValueError('Validation report does not match this protocol')
    d = replay(prior, tcp, TASK, split, source_mode=mode)
    col = task_indices(TASK).index(CLASS_ORDER.index('Sadness'))
    y = d['y'][:, col].astype(bool)
    valid = d['reliable']
    wc = d['w_c']
    z = d['z'][:, col]
    wf = np.where(valid[:,None], [[.64,.16,.20]], [[.80,.20,0.]])
    probabilities = {name: torch.from_numpy(d['logits_' + name]).float().sigmoid().numpy()[:,col]
                     for name in ('fixed', 'a', 'c')}
    pc = probabilities['c']
    samples = dict(ids=d['ids'], sadness=y, reliable=valid, branch_logits=z,
                   weights_a=d['w_a'], weights_c=wc, logits_a=d['logits_a'][:,col],
                   logits_c=d['logits_c'][:,col], logits_fixed=d['logits_fixed'][:,col])
    comparisons = {}
    for base, wb in [('a', d['w_a']), ('fixed', wf)]:
        pb = probabilities[base]
        scores = dict(base=pb, positive_only=np.where(y,pc,pb),
                      negative_only=np.where(y,pb,pc), all_c=pc)
        rows, ranks = {}, {}
        for name, p in scores.items():
            rows[name], ranks[name] = ranked(p,y)
        for row, key in [('base',base), ('all_c','c')]:
            np.testing.assert_allclose(rows[row]['ap'], d['ap_'+key][col], rtol=0, atol=1e-7)
        pos, neg, both, start = [rows[k]['ap'] for k in ('positive_only','negative_only','all_c','base')]
        symmetric = dict(positive=((pos-start)+(both-neg))/2, negative=((neg-start)+(both-pos))/2)
        groups = attribute(d['ids'],y,z,wb,wc,valid,ranks['base'],ranks['all_c'])
        comparisons[base] = dict(rankings=rows, order_averaged_ap_change=symmetric, **groups)
        for name, ranks_ in ranks.items():
            samples[base+'_rank_'+name] = ranks_
    archive = out/(split+'_people.npz')
    np.savez_compressed(archive,**samples)
    report = dict(task=TASK, emotion='Sadness', split=split, mode=mode, people=len(y), positives=int(y.sum()),
                  group_mean_ap=float(d['ap_c_mean'][col]), comparisons=comparisons,
                  source=d['metadata'], people_sha256=sha256_file(archive))
    write_json(out/(split+'.json'), report)
    if split == 'val':
        write_json(lock,dict(mode=mode, val_sha256=sha256_file(out/'val.json')))
    print(mode, split, {k:{n:round(row['ap'],4) for n,row in c['rankings'].items()}
                        for k,c in comparisons.items()}, flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('prior','tcp','out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--mode',choices=('sample_full','sample_holdout'),required=True)
    parser.add_argument('--split',choices=('val','test'),required=True)
    run(**vars(parser.parse_args()))
