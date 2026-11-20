"""Compare the two 90% branches with their own TRAIN-fitted paired fusion weights."""
import argparse
import os
from pathlib import Path

from scripts.emotic.d9_a_eight_eval import read, require_digest


SPLITS = ('val', 'test')
METRICS = ('final_mAP', 'average_mAP')


def load_run(root, meta):
    lock_path = root / 'centers_lock.json'
    lock = read(lock_path)
    seal_path = root / 'validation_seal.json'
    seal = read(seal_path)
    require_digest(lock_path, seal['centers_sha256'])
    require_digest(root / 'val_results.json', seal['val_results_sha256'])
    if set(lock) != {str(t) for t in range(8)}:
        raise ValueError(f'Expected eight sealed TRAIN centers in {root}')
    roles = set()
    for t in range(8):
        center_path = root / f'task{t}_center.json'
        require_digest(center_path, lock[str(t)])
        protocol = read(center_path)['protocol']
        if (protocol['task'] != t or protocol['split'] != 'train' or protocol['smoke']
                or protocol['init'] != 'equal_available_views'
                or (protocol.get('fit_mode') == 'replay_meta_task_checkpoint') != meta):
            raise ValueError(f'Unexpected TRAIN center at {center_path}')
        roles.add(protocol['roles_sha256'])
    if len(roles) != 1:
        raise ValueError(f'TRAIN roles differ across stages in {root}')
    reports = {s: read(root / f'{s}_results.json') for s in SPLITS}
    for s, r in reports.items():
        if (r['split'] != s or r['smoke'] or r['centers_sha256'] != seal['centers_sha256']
                or [row['stage'] for row in r['stages']] != list(range(8))):
            raise ValueError(f'Incomplete or different {s} run in {root}')
    require_digest(seal_path, reports['test']['validation_seal_sha256'])
    return reports, roles.pop()


def report(out, reference, full_comparison):
    meta, meta_roles = load_run(out, meta=True)
    original, original_roles = load_run(reference, meta=False)
    if meta_roles == original_roles:
        raise ValueError('Expected independently assigned TRAIN image rosters')
    full = []
    for s in SPLITS:
        comp = read(full_comparison / f'{s}_comparison.json')
        if comp['split'] != s or comp['official_result_sha256']['equal_updates'] != original[s]['native_result_sha256']:
            raise ValueError(f'Historical {s} control refers to another trajectory')
        full.extend(comp['branch_summary']['fixed']['all']['values']['full'][k] for k in METRICS)

    def values(run, method):
        return [run[s]['summary'][method]['all'][k] for s in SPLITS for k in METRICS]

    def row(label, scores, signed=False):
        fmt = '+.4f' if signed else '.4f'
        return '| ' + label + ' | ' + ' | '.join(format(x, fmt) for x in scores) + ' |'

    original_a, meta_a = values(original, 'A'), values(meta, 'A')
    meta_fixed = values(meta, 'fixed')
    columns = ['| 方法 | Val 最终 | Val 阶段平均 | Test 最终 | Test 阶段平均 |', '|---|---:|---:|---:|---:|']
    lines = ['# 各自用留出 TRAIN 学融合比例，两套 90% 分支的八阶段对照', '',
             '两套 Frozen9 分支均为 seed1，按原图留出约 10%。每个任务从等权学习当前任务的比例，'
             '仅使用本组 checkpoint 在本组留出 TRAIN 上得到的自然图与降清晰度预测。'
             '可用 Face 与不可用 Face 的人物分别共用一套比例。评价均为自然清晰度。', '', *columns,
             row('原随机，固定比例', values(original, 'fixed')), row('原随机，自己学比例', original_a),
             row('人数／人脸分层，固定比例', meta_fixed), row('人数／人脸分层，自己学比例', meta_a),
             row('全 TRAIN，固定比例', full), '', '## 分层分支学比例后的差值', '', *columns]
    for label, control in [('相对同组固定', meta_fixed), ('相对原随机自己学比例', original_a), ('相对全 TRAIN 固定', full)]:
        lines.append(row(label, [a-b for a, b in zip(meta_a, control)], signed=True))
    lines.extend(['', '## 分层分支逐阶段的融合收益', '',
                  '| 阶段 | Val 固定 | Val 学习 | Val 差值 | Test 固定 | Test 学习 | Test 差值 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for t in range(8):
        cells = [str(t)]
        for s in SPLITS:
            metrics = meta[s]['stages'][t]['metrics']
            fixed, learned = (metrics[m]['all']['mAP'] for m in ('fixed', 'A'))
            cells.extend([f'{fixed:.4f}', f'{learned:.4f}', f'{learned-fixed:+.4f}'])
        lines.append('| ' + ' | '.join(cells) + ' |')
    result = '\n'.join(lines) + '\n'
    path = out / 'results.md'
    if path.exists() and path.read_text() != result:
        raise ValueError(f'Existing report differs at {path}')
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(result)
    tmp.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'reference', 'full-comparison'):
        parser.add_argument('--' + name, type=Path, required=True)
    report(**vars(parser.parse_args()))
