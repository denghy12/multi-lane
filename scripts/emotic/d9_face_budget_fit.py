"""Fit paired A on current-task TRAIN holdouts with matched Full/Person and original 90% Face."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_paired_a_evaluate import load_training_pair, write_verified
from scripts.emotic.d9_quality_fit import LR, STEPS


def aligned_mix(original, matched):
    """Take each view from its designated budget within the same person's same variant."""
    old, old_variants = original
    new, new_variants = matched
    for field in ('sample_ids', 'reliable', 'targets'):
        np.testing.assert_array_equal(old[field], new[field], err_msg=f'Unmatched TRAIN {field}')

    def mix(old_array, new_array):
        if old_array.shape != new_array.shape or old_array.shape != (*old['targets'].shape, 3):
            raise ValueError('Expected matching [people, current classes, 3 views] logits')
        if not np.isfinite(new_array).all():
            raise ValueError('Non-finite matched TRAIN logits')
        return np.concatenate((new_array[..., :2], old_array[..., 2:]), axis=-1)

    natural = dict(new, branch_logits=mix(old['branch_logits'], new['branch_logits']))
    variants = {v: {'branch_logits': mix(old_variants[v]['branch_logits'], new_variants[v]['branch_logits'])}
                for v in ('person', 'face')}
    return natural, variants


def replay(pair, center, weights, name):
    reference = fit_average(*pair)
    np.testing.assert_array_equal(reference['params']['bias'], center, err_msg=f'{name} center mismatch')
    np.testing.assert_array_equal(reference['weights'], weights, err_msg=f'{name} weights mismatch')


def fit(matched, old, old_pairs, out):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    config = read(old/'config.json')
    new_config = read(matched/'config.json')
    if (config['mode'] != 'sample_holdout' or config['smoke'] or config['seed'] != 1 or config['tasks'] != 8
            or config['image_split'] != 'global_10pct_hash_rank' or not config['protect_branches']
            or {k: v for k, v in new_config.items() if k != 'equal_updates'} != config
            or 'equal_updates' not in new_config):
        raise ValueError('Unmatched source recipes')
    old_lock_path = old_pairs/'readout'/'centers_lock.json'
    old_lock = read(old_lock_path)
    old_hashes = {name+'_sha256': sha256_file(old/file) for name,file in
                  [('config','config.json'), ('roles','train_image_roles.json')]}
    if any(old_lock[k] != v for k, v in old_hashes.items()):
        raise ValueError('Original 90% source configuration or TRAIN roles changed')
    require_digest(matched/'train_image_roles.json', old_hashes['roles_sha256'])
    roles = read(old/'train_image_roles.json')
    new_hash = sha256_file(matched/'config.json')
    new_pairs = matched/'paired_A'
    new_seal = read(new_pairs/'centers_lock.json')
    keys = {str(t) for t in range(len(TASK_SIZES))}
    if set(new_seal) != keys or set(old_lock['tasks']) != keys:
        raise ValueError('Incomplete source centers')

    tasks = {}
    for t in range(len(TASK_SIZES)):
        k = str(t)
        classes = list(task_indices(t))
        prior = old_lock['tasks'][k]
        require_digest(old/'cohorts'/f'task{t}.json', prior['cohort_sha256'])
        require_digest(matched/'cohorts'/f'task{t}.json', prior['cohort_sha256'])
        require_digest(old/'compact_checkpoints'/f'task{t}.pth', prior['checkpoint_sha256'])
        cohort = read(old/'cohorts'/f'task{t}.json')['fusion']
        old_data, old_variants, meta = load_training_pair(
            old_pairs/'pairs', roles, t, dict(prior, fusion_cohort=cohort), old_hashes)
        if (prior['fit_classes'] != classes or prior['pairs_sha256'] != meta['archive_sha256']
                or (meta['input_size'], meta['reduced_size']) != (224, 112)):
            raise ValueError(f'Original TRAIN pair mismatch at task{t}')
        old_pair = old_data, old_variants
        replay(old_pair, prior['paired']['center'], prior['paired']['weights'], f'original task{t}')

        center_path = new_pairs/f'task{t}_center.json'
        require_digest(center_path, new_seal[k])
        new = read(center_path)
        checkpoint_hash = new['checkpoint_sha256']
        require_digest(matched/'compact_checkpoints'/f'task{t}.pth', checkpoint_hash)
        expected = dict(split='train', task=t, classes=classes, smoke=False, init='equal_available_views',
                        previous_center_sha256=new_seal.get(str(t-1)), steps=STEPS, lr=LR, optimizer='adam',
                        variants=['natural', 'person_112', 'reliable_face_112'], config_sha256=new_hash,
                        cohort_sha256=prior['cohort_sha256'], roles_sha256=old_hashes['roles_sha256'],
                        checkpoint_sha256=checkpoint_hash)
        if expected != new['protocol'] or new['fusion_cohort'] != cohort:
            raise ValueError(f'Matched TRAIN pair mismatch at task{t}')
        path = new_pairs/f'task{t}_held.npz'
        require_digest(path, new['archive_sha256'])
        with np.load(path, allow_pickle=False) as z:
            new_data = {field: z[field] for field in old_data}
            new_variants = {v: {'branch_logits': z[v+'_branch_logits']} for v in ('person','face')}
        new_pair = new_data, new_variants
        hybrid = aligned_mix(old_pair, new_pair)
        replay(new_pair, new['center'], new['group_weights'], f'matched task{t}')
        unreliable = ~new_data['reliable']
        for table, reference in ((hybrid[0], new_data), (hybrid[1]['person'], new_variants['person']),
                                 (hybrid[1]['face'], new_variants['face'])):
            np.testing.assert_array_equal(table['branch_logits'][unreliable, :, :2],
                                          reference['branch_logits'][unreliable, :, :2])
        candidate = fit_average(*hybrid)
        unreliable_delta = np.max(np.abs(np.subtract(candidate['weights'][0], new['group_weights'][0])))
        tasks[k] = dict(center=candidate['params']['bias'], group_weights=candidate['weights'], fit=candidate['fit'],
                        fit_classes=classes, people=cohort['people'], reliable_face=cohort['reliable_face'],
                        ids_sha256=cohort['sample_id_sha256'], cohort_sha256=prior['cohort_sha256'],
                        old_checkpoint_sha256=prior['checkpoint_sha256'], matched_checkpoint_sha256=checkpoint_hash,
                        old_pairs_sha256=meta['archive_sha256'], matched_pairs_sha256=new['archive_sha256'],
                        full_person_source='matched', face_source='old', source_centers_replayed=True,
                        unreliable_full_person_inputs_replayed=True,
                        unreliable_weights_max_abs_delta_vs_matched=float(unreliable_delta))
        print(f'task{t}: {cohort["people"]} TRAIN people, paired A {candidate["weights"]}', flush=True)
    protocol = dict(split='train', seed=1, roles='permanent 10% original-image holdout',
                    init='equal_available_views', lr=LR, steps=STEPS, optimizer='adam', loss='balanced_paired_bce',
                    variants=['natural', 'person_112', 'reliable_face_112'], tasks=list(range(len(TASK_SIZES))))
    write_verified(out/'centers_lock.json', dict(tasks=tasks, protocol=protocol, old_config_sha256=old_hashes['config_sha256'],
                   matched_config_sha256=new_hash, roles_sha256=old_hashes['roles_sha256'],
                   old_centers_lock_sha256=sha256_file(old_lock_path),
                   matched_centers_lock_sha256=sha256_file(new_pairs/'centers_lock.json')))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for opt in ('matched', 'old', 'old-pairs', 'out'):
        cli.add_argument('--'+opt, type=Path, required=True)
    fit(**vars(cli.parse_args()))
