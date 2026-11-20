"""Write the natural-input ranking comparison from sealed eight-stage evaluations."""
import argparse
from pathlib import Path

from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS


def report(out):
    splits = [read(out / f'{s}_results.json') for s in ('val', 'test')]
    seal = read(out / 'validation_seal.json')
    require_digest(out / 'val_results.json', seal['val_results_sha256'])
    require_digest(out / 'validation_seal.json', splits[1]['validation_seal_sha256'])
    if any(len(s['stages']) != 8 or s['decision'] != seal['decision'] for s in splits):
        raise ValueError('Incomplete or unsealed result')
    names = {'fixed':'手设比例', 'paired':'双模糊 BCE', 'rank':'双模糊 BCE 加自然图排序',
             'rank_reliable_only':'仅在可靠 Face 人物使用新比例', 'rank_unreliable_only':'仅在无可靠 Face 人物使用新比例'}
    lines = ['# 双模糊分类训练加入自然图排序', '',
             '八阶段评估均使用自然图。两组均只用留出 TRAIN 更新融合比例，两个单人群替换仅供归因。']
    for group, title in [('all', '全体人物'), ('reliable', '有可靠 Face'), ('unreliable','无可靠 Face')]:
        lines.extend(['', '## '+title, '', '| 读出 | Val 最终 | Val 阶段平均 | Test 最终 | Test 阶段平均 |',
                      '|---|---:|---:|---:|---:|'])
        methods = names if group == 'all' else ('fixed','paired','rank')
        for key in methods:
            values = [r['summary'][key][group][metric] for r in splits for metric in MAP_KEYS]
            lines.append('| '+' | '.join([names[key]]+[f'{v:.4f}' for v in values])+' |')
    lines.extend(['','## 各阶段新组相对原组的全体 mAP 变化','',
                  '| 阶段 | Val 差值 | Test 差值 |','|---|---:|---:|'])
    for stage in range(8):
        metrics = [s['stages'][stage]['metrics'] for s in splits]
        changes = [m['rank']['all']['mAP']-m['paired']['all']['mAP'] for m in metrics]
        lines.append(f'| {stage} | {changes[0]:+.4f} | {changes[1]:+.4f} |')
    lines.extend(['', 'Validation 封存方案为 **'+names[seal['decision']['winner']]+'**。'])
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', required=True, type=Path)
    report(cli.parse_args().out)
