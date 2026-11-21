"""Report a preregistered reliable-Face blur-strength by natural-mass comparison."""
import argparse
from pathlib import Path

from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS

NAMES = {'fixed': '同模型手设比例', 'paired_112': 'A 112尺寸，三版等权',
         'gentle_168': '168尺寸，三版等权', 'clean_112': '112尺寸，自然图占一半',
         'clean_168': '168尺寸，自然图占一半'}


def report(out):
    runs = [read(out / f'{split}_results.json') for split in ('val', 'test')]
    seal = read(out/'validation_seal.json')
    require_digest(out/'val_results.json', seal['val_results_sha256'])
    require_digest(out/'validation_seal.json', runs[1]['validation_seal_sha256'])
    if any(len(run['stages']) != 8 or run['decision'] != seal['decision'] for run in runs):
        raise ValueError('Incomplete or inconsistent eight-stage result')
    def four(name, group):
        return [run['summary'][name][group][key] for run in runs for key in MAP_KEYS]
    header = ['| 方法 | Val 最终 | Val 阶段平均 | Test 最终 | Test 阶段平均 |','|---|---:|---:|---:|---:|']
    lines = ['# 双模糊 A 的强度和自然图训练份额', '',
             'seed1，原人群，双模糊 BCE。可靠组损失每人归一，无可靠 Face 的输出完全沿用 A。']
    for group,title in [('all','全体人物'),('reliable','可靠 Face 人物'),('unreliable','无可靠 Face 人物')]:
        lines.extend(['','## '+title,'',*header])
        for method,label in NAMES.items():
            lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in four(method,group))+' |')
    lines.extend(['','## 同底座差值','',*header])
    for a,b in [('gentle_168','paired_112'),('clean_112','paired_112'),('clean_168','paired_112'),
                ('clean_168','gentle_168'),('clean_168','clean_112')]:
        vals = [x-y for x,y in zip(four(a,'all'),four(b,'all'))]
        lines.append('| '+NAMES[a]+' 减 '+NAMES[b]+' | '+' | '.join(f'{v:+.4f}' for v in vals)+' |')
    lines.extend(['','## 各阶段相对 A 的变化','',
                  '| 阶段 | Val 168 | Val 半份自然 | Val 两者 | Test 168 | Test 半份自然 | Test 两者 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for t in range(8):
        values = []
        for run in runs:
            m = run['stages'][t]['metrics']
            values.extend(m[x]['all']['mAP']-m['paired_112']['all']['mAP'] for x in
                          ('gentle_168','clean_112','clean_168'))
        lines.append(f'| {t} | '+' | '.join(f'{v:+.4f}' for v in values)+' |')
    lines.extend(['','Validation 封存选择为 **'+NAMES[seal['decision']['winner']]+'**。'])
    (out/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    report(p.parse_args().out)
