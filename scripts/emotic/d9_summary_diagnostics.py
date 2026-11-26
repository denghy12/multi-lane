"""Paired original-image bootstrap for the current-head pilot, using official AP."""
import numpy as np
import torch

from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_summary_readout import metrics


def bootstrap(out, split, repetitions, seed):
    report = read(out/f'{split}_results.json')
    data = []
    for row in report['stages']:
        path = out/split/f"task{row['task']}.npz"
        require_digest(path, row['predictions_sha256'])
        with np.load(path, allow_pickle=False) as z:
            data.append({k: z[k] for k in z.files})
    universe = sorted(set().union(*(set(d['image_ids'].tolist()) for d in data)))
    positions = {k: i for i, k in enumerate(universe)}
    # Repeated original images have the same bootstrap multiplicity across tasks.
    mappings = [np.array([positions[k] for k in d['image_ids']])[d['inverse']] for d in data]
    methods = ('summary_mean', 'summary_correspondence')
    differences = {m: {k: [] for k in ('shared', 'permutation')} for m in methods}
    invalid = 0
    rng = np.random.default_rng(seed)
    for repeat in range(repetitions):
        multiplicity = np.bincount(rng.integers(len(universe), size=len(universe)), minlength=len(universe))
        selections = [np.repeat(np.arange(len(d['targets'])), multiplicity[owner]) for d, owner in zip(data, mappings)]
        if any(not len(sel) or np.any(d['targets'][sel].sum(0) == 0)
               or np.any((1-d['targets'][sel]).sum(0) == 0) for d, sel in zip(data, selections)):
            invalid += 1
            continue
        delta = {m: {k: [] for k in ('shared', 'permutation')} for m in methods}
        for d, sel in zip(data, selections):
            labels = torch.from_numpy(d['targets'][sel])
            def ap(values):
                return metrics(torch.from_numpy(values[sel]), labels)['mAP']
            shared = ap(d['shared_logits'])
            for m in methods:
                own = ap(d[m+'_logits'])
                exchanged = np.mean([ap(x) for x in d[m+'_permuted_logits']])
                delta[m]['shared'].append(own-shared)
                delta[m]['permutation'].append(own-exchanged)
        for m in methods:
            for k in delta[m]:
                differences[m][k].append(float(np.mean(delta[m][k])))
        if (repeat+1) % 100 == 0:
            print(f'{split} original-image bootstrap {repeat+1}/{repetitions}', flush=True)
    result = dict(repetitions=repetitions, valid=repetitions-invalid, invalid_missing_class_support=invalid,
                  seed=seed, shared_image_multiplicity_across_tasks=True,
                  comparison='own current-head AP minus shared / mean AP of all archived exchanges', methods={})
    for m in methods:
        result['methods'][m] = {}
        for k, values in differences[m].items():
            result['methods'][m][k] = dict(mean=float(np.mean(values)) if values else None,
                ci95=np.quantile(values, [.025, .975]).tolist() if values else None)
    write_verified(out/f'{split}_bootstrap.json', result)
    return result


def decision(out):
    report, intervals = read(out/'val_results.json'), read(out/'val_bootstrap.json')
    results = {}
    for m in ('summary_mean', 'summary_correspondence'):
        deltas = {k: [] for k in ('fixed', 'shared', 'permutation', 'mean_scores')}
        if m == 'summary_correspondence':
            deltas['summary_mean'] = []
        for row in report['stages']:
            current = row['methods']
            own = current[m]['groups']['all']['mAP']
            for k in deltas:
                other = (current[m]['permutation']['mAP'] if k == 'permutation' else
                         current[m]['mean_scores']['all']['mAP'] if k == 'mean_scores' else
                         current[k]['groups']['all']['mAP'])
                deltas[k].append(own-other)
        conditions = {k: dict(mean_delta=float(np.mean(v)), improved_tasks=int(np.sum(np.array(v)>0)),
                             passed=bool(np.mean(v)>0 and np.sum(np.array(v)>0)>=2)) for k,v in deltas.items()}
        ci = intervals['methods'][m]
        supported = all(v['ci95'] is not None and v['ci95'][0] > 0 for v in ci.values())
        passed = all(v['passed'] for v in conditions.values()) and supported and not report['smoke']
        results[m] = dict(conditions=conditions, bootstrap_positive=supported,
                          evidence='current-head pilot supported' if passed else 'not confirmed',
                          eligible_for_separate_eight_task_plan=passed)
    write_verified(out/'validation_decision.json', results)
    return results
