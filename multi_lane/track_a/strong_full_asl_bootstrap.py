"""Paired image-cluster uncertainty, validation only; never selects test settings."""
import json
import numpy as np
from .strong_full_asl import ARMS


def score(p,y):
    order=p.argsort(axis=0)[::-1];pos=np.take_along_axis(y,order,axis=0)==1
    c=pos.cumsum(axis=0);n=c[-1].copy();c[~pos]=0
    return float((100*(c/np.arange(1,len(pos)+1)[:,None]).sum(axis=0)/(n+1e-8)).mean())


def summarize_bootstrap(root, repeats=1000):
    cells={}
    for arm in ARMS:
        with np.load(root/arm/'full_val/scores/task7.npz') as z:
            if not cells:ids=z['sample_ids'].copy();y=z['targets'].copy()
            assert np.array_equal(ids,z['sample_ids']) and np.array_equal(y,z['targets'])
            cells[arm]=z['probabilities'].copy()
    groups={}
    for i,name in enumerate(ids):groups.setdefault(str(name).rsplit('#person=',1)[0],[]).append(i)
    groups=list(groups.values());rng=np.random.default_rng(20261023)
    pairs=[('S_ASL','S_BCE'),('T_ASL','T_BCE'),('T_ASL','S_ASL'),('T_ASL_ATTN','T_ASL'),
        ('T_ASL_POS','T_ASL'),('T_ASL_BOTH','T_ASL'),('T_ASL_BOTH','T_ASL_ATTN'),('T_ASL_BOTH','T_ASL_POS')]
    points={a:score(p,y) for a,p in cells.items()};draws={a+'-'+b:[] for a,b in pairs}
    for _ in range(repeats):
        ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]);m={a:score(p[ix],y[ix]) for a,p in cells.items()}
        for a,b in pairs:draws[a+'-'+b].append(m[a]-m[b])
    result=dict(repeats=repeats,unit='original_image',people=len(ids),images=len(groups),training_seeds=1,
        contrasts={a+'-'+b:dict(point_pp=points[a]-points[b],ci95_pp=np.percentile(draws[a+'-'+b],[2.5,97.5]).tolist()) for a,b in pairs})
    (root/'bootstrap_validation_final.json').write_text(json.dumps(result,indent=2)+'\n')
