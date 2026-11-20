"""Write paired and single-perturbation results for the eight frozen stages."""
import argparse
import json
from pathlib import Path

from scripts.emotic.d9_a_eight_eval import require_digest

METHODS = {
    'fixed': '全体沿用原固定比例',
    'natural': '可靠组仅自然图',
    'person': '可靠组仅模糊身体',
    'face': '可靠组仅模糊人脸',
    'paired': '可靠组分别模糊身体和人脸',
}
METRICS = ('final_mAP', 'average_mAP')


def report(out):
    seal = json.loads((out/'validation_seal.json').read_text())
    runs = [json.loads((out/f'{split}_results.json').read_text()) for split in ('val', 'test')]
    require_digest(out/'validation_seal.json', runs[1]['validation_seal_sha256'])
    if any(len(r['stages']) != 8 for r in runs):
        raise ValueError('The final report requires all eight stages')

    def four(name, group):
        return [r['summary'][name][group][m] for r in runs for m in METRICS]

    def table_row(name, vals, signed=False):
        fmt = '+.4f' if signed else '.4f'
        return '| ' + ' | '.join([name] + [format(x, fmt) for x in vals]) + ' |'

    headers = ['| 可靠 Face 人群的融合训练输入 | Val 最终 | Val 阶段平均 | Test 最终 | Test 阶段平均 |',
               '|---|---:|---:|---:|---:|']
    lines = ['# 可靠 Face 人群清晰度对照', '',
             '各方案用于相同的自然图评价。四个学习组对无可靠 Face 的人物都采用已封存的原配对比例。'
             '仅自然图一行指可靠组的训练条件。三个训练位置均保留相同权重，未使用的扰动由自然图填充。']
    for group, title in [('all','全体人物'),('reliable','仅可靠 Face 人物'),('unreliable','仅无可靠 Face 人物')]:
        lines.extend(['', f'## {title}', '', *headers])
        for key, label in METHODS.items():
            lines.append(table_row(label, four(key, group)))
    lines.extend(['', '## 同人配对差值', '',
                  '在全体人物上直接重新排名。每行是前一方案减后一方案，单位为 mAP 点。','', *headers])
    for k, b in [('person','paired'),('face','paired'),('natural','paired'),
                 ('person','natural'),('face','natural'),('paired','face'),('paired','person')]:
        vals = [a-x for a,x in zip(four(k,'all'),four(b,'all'))]
        lines.append(table_row(METHODS[k]+' 减 '+METHODS[b], vals, signed=True))
    lines.extend(['', '## 相对双扰动的逐阶段变化', '',
                  '| 阶段 | Val 仅自然 | Val 仅身体 | Val 仅人脸 | Test 仅自然 | Test 仅身体 | Test 仅人脸 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for t in range(8):
        differences = []
        for r in runs:
            metrics = r['stages'][t]['metrics']
            differences.extend(metrics[k]['all']['mAP']-metrics['paired']['all']['mAP']
                               for k in ('natural','person','face'))
        lines.append(table_row(str(t), differences, signed=True))
    selection = seal['decision']['winner']
    lines.extend(['', 'Validation 封存的方案为 **'+METHODS[selection]+'**。全部四组均在封存后同批报告 Test。'])
    (out/'results.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    report(parser.parse_args().out)
