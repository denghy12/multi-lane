"""Summarize locked TRAIN amplitudes and paired held-out mAP differences."""
import argparse
import json
from pathlib import Path

from multi_lane.track_a.openai_clip_loader import sha256_file


NAMES = {
    'historical_fixed': '历史固定，逐人处理缺脸',
    'image_fixed': '图级固定',
    'constant': 'TRAIN 学习常量（0% 调整）',
    'full': 'Full，自由逐图调整（100%）',
    'shrink_full': 'Full，TRAIN 学幅度',
    'full_relation': 'Full＋跨视图关联，自由调整（100%）',
    'shrink_full_relation': 'Full＋跨视图关联，TRAIN 学幅度',
}
DELTA_CONTROLS = ('constant', 'image_fixed', 'unshrunk', 'shuffled', 'mean')


def load(path):
    return json.loads(path.read_text())


def report(out):
    fit = load(out/'fit.json')
    summaries = {}
    for split in ('val', 'test'):
        result = load(out/f'{split}_results.json')
        if (len(result['stages']) != len(fit['states'])
                or [s['task'] for s in result['stages']] != list(range(len(fit['states'])))):
            raise ValueError(f'Incomplete {split} stages')
        summaries[split] = result['summary']

    def four(method):
        return [summaries[split][method]['all'][key] for split in ('val','test')
                for key in ('final_mAP', 'average_mAP')]

    lines = ['# 由 TRAIN 确定整图调整幅度' + ('（小跑）' if fit['smoke'] else ''), '',
             '以 TRAIN 学到的图级可用性常量为中心，将原图自身的比例沿直线缩放。'
             '每阶段的单一幅度适用于全部已见任务头、同图所有人和两种 Face 可用性。'
             '0% 表示常量，100% 表示使用模型给该图的原始比例。', '',
             '| 任务阶段 | 只看 Full | Full＋跨视图关联 |',
             '|---|---:|---:|']
    for s in fit['states']:
        amps = s['lambda']
        for a in amps.values():
            if not (0 <= a <= 1 and abs(a*20-round(a*20)) < 1e-8):
                raise ValueError('Unexpected TRAIN grid')
        lines.append(f"| {s['task']} | {amps['full']:.0%} | {amps['full_relation']:.0%} |")
    lines.extend(['', '| 同一套视觉分支的读出 | Val 最终 | Val 阶段平均 | Test 最终 | Test 阶段平均 |',
                  '|---|---:|---:|---:|---:|'])
    for method, name in NAMES.items():
        lines.append(f'| {name} | '+' | '.join(f'{x:.4f}' for x in four(method))+' |')
    lines.extend(['', '## 只改变整图调整幅度后的配对差值', '',
                  '单位为 mAP 点。正值表示本图收缩比例的 mAP 更高。交换使用 100 次同可用性整图交换的平均 mAP。'
                  '组均值使用该评价集无标签的原图权重，仅用于检查图间差异是否有价值。', '',
                  '| 图级输入 | 对照 | Val 最终差 | Val 阶段平均差 | Test 最终差 | Test 阶段平均差 |',
                  '|---|---|---:|---:|---:|---:|'])
    for gate in ('full','full_relation'):
        m='shrink_'+gate
        diffs={}
        for name in DELTA_CONTROLS:
            baseline = {'constant':'constant','image_fixed':'image_fixed','unshrunk':gate,
                        'shuffled':m+'_shuffled','mean':m+'_mean'}[name]
            label = {'constant':'TRAIN 学习常量','image_fixed':'图级固定','unshrunk':'100% 调整',
                     'shuffled':'同幅度跨图交换','mean':'同幅度组均值'}[name]
            differences = [x-y for x,y in zip(four(m), four(baseline))]
            diffs[name] = differences
            lines.append('| '+('Full' if gate=='full' else 'Full＋关联')+f' | {label} | '+
                         ' | '.join(f'{x:+.4f}' for x in differences)+' |')
        decision = all(diffs[name][i] > 1e-9 for name in ('constant','shuffled') for i in (0,1))
        if gate=='full_relation':
            val_decision = '通过' if decision else '未通过'
    lines.extend(['', f'主候选 Full＋关联{val_decision}预先设定的 Validation 条件'
                  '（最终和阶段平均均严格超过学习常量及同掩码跨图交换）。'
                  '超过图级固定是独立的性能比较。', '',
                  '若某阶段幅度为零，该阶段使用学习常量，不产生图与图之间的内容差异。'
                  '本试验只评价这两种已训练读出、这条插值路径及 seed1。', '',
                  f'封存训练文件 SHA-256 为 `{sha256_file(out/"fit.json")}`。'])
    (out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    report(parser.parse_args().out)
