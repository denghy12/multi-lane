"""Lock the historical 90% split and the full-TRAIN update/epoch clock."""
import math


def epoch_budget(history, task, people, epochs, batch_size):
    """Verify actual historical updates, including the final short batch."""
    count = math.ceil(people / batch_size)
    rows = history['epochs']
    if history['coverage']['count'] != people or len(rows) != epochs:
        raise ValueError(f'Incomplete full-TRAIN reference at task {task}')
    for epoch, row in enumerate(rows, 1):
        expected = dict(task=task, epoch=epoch, examples=people, updates=count, skipped=0)
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError(f'Invalid full-TRAIN update history at task {task} epoch {epoch}')
    return dict(examples=people, updates=count, last_batch_size=1 + (people - 1) % batch_size)


def load_reference_budget(full, holdout, config):
    from multi_lane.track_a.r1_crossvit_experiment import read, sha
    from multi_lane.track_a.runner import TASK_SIZES

    old_full = read(full/'config.json')
    old_holdout = read(holdout/'config.json')
    different = {k for k in old_full.keys() | old_holdout.keys() if old_full.get(k) != old_holdout.get(k)}
    if (old_full['mode'] != 'sample_full' or old_holdout['mode'] != 'sample_holdout'
            or different - {'mode', 'image_split', 'fusion_asl'}
            or old_full['smoke'] or old_full['tasks'] != len(TASK_SIZES) or old_full['epochs'] != 30
            or old_full['batch_size'] != 64):
        raise ValueError('Expected paired complete sample_full and sample_holdout references')
    permitted = {'smoke', 'epochs', 'tasks', 'batch_size', 'micro_batch_size', 'gate_updates'} if config['smoke'] else set()
    if any(config.get(k) != v for k, v in old_holdout.items() if k not in permitted):
        raise ValueError('Branch, augmentation, precision or optimizer recipe differs from sample_holdout')
    targets = {}
    for t in range(config['tasks']):
        hist = full/'history'/f'task{t}.json'
        cohort = full/'cohorts'/f'task{t}.json'
        full_cohort = read(cohort)['branch']
        history = read(hist)
        if history['coverage']['sample_id_sha256'] != full_cohort['sample_id_sha256']:
            raise ValueError(f'Full reference cohort and history differ at task {t}')
        budget = epoch_budget(history, t, full_cohort['people'], old_full['epochs'], old_full['batch_size'])
        targets[str(t)] = dict(**budget, full_history_sha256=sha(hist), full_cohort_sha256=sha(cohort),
                               holdout_cohort_sha256=sha(holdout/'cohorts'/f'task{t}.json'))
    return dict(mode='cycle_90pct_to_full_epoch_size', tasks=targets,
                full_config_sha256=sha(full/'config.json'), holdout_config_sha256=sha(holdout/'config.json'),
                roles_sha256=sha(holdout/'train_image_roles.json'),
                smoke_extra_examples=3 if config['smoke'] else None)


def check_epoch_budget(task, epoch, config, optimizer, scheduler, row):
    """Match CosineAnnealingLR's epoch ticks and each parameter group's actual LR."""
    ref = config['equal_updates']['tasks'][str(task)]
    sample = row['sampling']
    n = sample['unique_people'] + config['equal_updates']['smoke_extra_examples'] if config['smoke'] else ref['examples']
    expected = math.ceil(n / config['batch_size'])
    expected_batches = [config['batch_size']] * (expected-1) + [1 + (n-1) % config['batch_size']]
    if (row['updates'] != expected or row['examples'] != n or row['skipped']
            or sample['examples'] != n or sample['batch_sizes'] != expected_batches
            or sample['duplicate_within_batches'] or scheduler.last_epoch != epoch-1):
        raise RuntimeError(f'Task {task} epoch {epoch} failed update-budget audit')
    factor = (1 + math.cos(math.pi*(epoch-1) / config['epochs'])) / 2
    rates = [g['lr'] for g in optimizer.param_groups]
    if any(not math.isclose(lr, base*factor, abs_tol=1e-15, rel_tol=1e-11)
           for lr, base in zip(rates, scheduler.base_lrs)):
        raise RuntimeError(f'Task {task} epoch {epoch} cosine LR mismatch')
    row['branch_lrs'] = rates
