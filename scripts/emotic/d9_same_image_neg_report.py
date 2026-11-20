"""Summarize the preregistered 2x2 cohort and objective comparison."""
import argparse
from pathlib import Path

from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS


NAMES = {'fixed': '同模型手设比例', 'paired': 'A 原人群 BCE', 'rank': 'B 原人群 BCE＋排序',
         'expanded_bce': 'C 补人 BCE', 'expanded_rank': 'D 补人 BCE＋排序'}


def report(out):
    runs = [read(out / f'{split}_results.json') for split in ('val', 'test')]
    seal = read(out / 'validation_seal.json')
    require_digest(out / 'val_results.json', seal['val_results_sha256'])
    require_digest(out / 'validation_seal.json', runs[1]['validation_seal_sha256'])
    if any(len(run['stages']) != 8 or run['decision'] != seal['decision'] for run in runs):
        raise ValueError('Incomplete eight stages or inconsistent decision')

    def four(method, group):
        return [run['summary'][method][group][k] for run in runs for k in MAP_KEYS]

    header = ['| 方案 | Val 最终 | Val 八阶段平均 | Test 最终 | Test 八阶段平均 |','|---|---:|---:|---:|---:|']
    lines = ['# 当期留出原图内补充全阴性人物', '', 'seed1，同一套八阶段 Frozen9，融合比例只在 TRAIN 学习。']
    for group, title in [('all', '全体人物'),('reliable','有可靠 Face'),('unreliable','无可靠 Face')]:
        lines.extend(['','## '+title,'',*header])
        for k, label in NAMES.items():
            vals = four(k,group)
            lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines.extend(['','## 全体人物的四项差值','',*header])
    for left,right in [('expanded_bce','paired'),('expanded_rank','rank'),('expanded_rank','expanded_bce'),
                       ('rank','paired'),('expanded_rank','paired')]:
        values = [x-y for x,y in zip(four(left,'all'),four(right,'all'))]
        lines.append('| '+NAMES[left]+' 减 '+NAMES[right]+' | '+' | '.join(f'{v:+.4f}' for v in values)+' |')
    lines.extend(['','## 各阶段全体人物变化','',
                  '| 阶段 | Val C−A | Val D−C | Test C−A | Test D−C |','|---|---:|---:|---:|---:|'])
    for i in range(8):
        vals = []
        for run in runs:
            m = run['stages'][i]['metrics']
            vals.extend(m[x]['all']['mAP']-m[y]['all']['mAP'] for x,y in
                        [('expanded_bce','paired'),('expanded_rank','expanded_bce')])
        lines.append(f'| {i} | '+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    lines.extend(['','Validation 封存方案为 **'+NAMES[seal['decision']['winner']]+'**。'])
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    report(cli.parse_args().out)
