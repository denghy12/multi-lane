"""Verify per-image weights against source predictions and report image-gate ablations."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.runner import TASK_SIZES, average_precision

NAMES = {
    'historical_fixed': '历史固定，按人物处理缺脸',
    'image_fixed': '固定，按原图处理缺脸',
    'constant': '按图可用性学习常量',
    'full': '整图 Full 特征',
    'full_relation': '整图 Full 与跨视图关联',
    'full_mean': 'Full 图权重换成组均值',
    'full_shuffled': 'Full 图权重跨图交换',
    'full_relation_mean': '关联组图权重换成组均值',
    'full_relation_shuffled': '关联组图权重跨图交换',
}
SUBGROUPS = {'single': '单目标图', 'multi': '多目标图', 'mixed': '人脸可用性混合的原图',
             'sacrificed': '图级规则关闭其可靠 Face 的人物'}


def read(path):
    return json.loads(path.read_text())


def check_and_score(root, source, split, results, smoke):
    diagnostics = []
    for stage_row in results['stages']:
        t = stage_row['task']
        with ExitStack() as stack:
            features = stack.enter_context(np.load(root/'features'/split/f'task{t}.npz'))
            archive = stack.enter_context(np.load(root/'readout'/'weights'/split/f'task{t}.npz'))
            native = stack.enter_context(np.load(source/'evaluation'/split/f'task{t}.npz'))
            names = archive['image_ids'].astype(str)
            lookup = {name: i for i, name in enumerate(names)}
            fidx = {str(name): i for i, name in enumerate(features['image_ids'])}
            if len(lookup) != len(names):
                raise ValueError('Duplicate original image in weight archive')
            order = [fidx[k] for k in names]
            counts, reliable_counts = features['counts'][order], features['reliable_counts'][order]
            image_mask = counts == reliable_counts
            np.testing.assert_array_equal(image_mask, archive['all_reliable'])
            np.testing.assert_array_equal(counts, archive['counts'])
            row_images = [str(x).rsplit('#person=', 1)[0] for x in native['sample_ids']]
            selected = np.array([k in lookup for k in row_images])
            if not smoke and not selected.all():
                raise ValueError(f'{split} stage {t}: image weights missing for native evaluation people')
            ids = np.array([lookup[k] for k in row_images if k in lookup])
            reliable = native['reliable'][selected]
            score = torch.as_tensor(native['branch_logits'][selected], dtype=torch.float32)
            labels = native['targets'][selected, :score.shape[1]]
            groups = dict(single=counts[ids] == 1, multi=counts[ids] > 1,
                          mixed=(reliable_counts[ids] > 0) & ~image_mask[ids],
                          sacrificed=reliable & ~image_mask[ids])
            if (not len(ids) or np.any(image_mask[ids] & ~reliable)
                    or score.shape[1] != sum(TASK_SIZES[:t+1])):
                raise ValueError(f'{split} stage {t}: invalid image roster or task dimensions')
            rows = {}
            for method in ('historical_fixed','image_fixed','constant','full','full_relation'):
                if method == 'historical_fixed':
                    logits = torch.from_numpy(native['fixed_logits'][selected].copy()).float()
                else:
                    weights = archive[method]
                    if (weights.shape != (len(names), 3) or not np.isfinite(weights).all()
                            or np.any(weights < 0) or np.any(weights[~image_mask, 2] != 0)):
                        raise ValueError(f'{method}: invalid shared image weights')
                    np.testing.assert_allclose(weights.sum(1), 1, rtol=0, atol=2e-6)
                    w = torch.from_numpy(weights[ids]).float()
                    logits = score[:,:,0]*w[:,0:1] + score[:,:,1]*w[:,1:2] + score[:,:,2]*w[:,2:3]
                probs = logits.sigmoid().numpy()
                def mean_ap(chosen):
                    return float(np.mean([100*average_precision(probs[chosen,c], labels[chosen,c])
                                          for c in range(score.shape[1])]))
                overall = mean_ap(np.ones(len(ids), dtype=bool))
                np.testing.assert_allclose(overall, stage_row['metrics'][method]['all']['mAP'],
                                           rtol=0, atol=2e-4, err_msg=f'{method} {split} stage {t} replay')
                rows[method] = {key: mean_ap(mask) for key, mask in groups.items() if mask.any()}
            diagnostics.append(dict(task=t, counts={k:int(mask.sum()) for k, mask in groups.items()},
                                    original_images=len(names), scores=rows))
    return diagnostics


def generate(root, source, smoke):
    torch.set_num_threads(2)
    reports, diagnostics = {}, {}
    for split in ('val','test'):
        result = read(root/'readout'/f'{split}_results.json')
        if len(result['stages']) != (1 if smoke else 8):
            raise ValueError('Incomplete stages')
        diagnostics[split] = check_and_score(root, source, split, result, smoke)
        reports[split] = result['summary']
    def four(method):
        return [reports[s][method]['all'][key] for s in ('val','test') for key in ('final_mAP','average_mAP')]
    def numbers(values):
        return ' | '.join(f'{x:.4f}' for x in values)
    def diff_row(method, other):
        return [a-b for a,b in zip(four(method), four(other))]
    lines = ['# 原图级融合实验' + ('（预检）' if smoke else ''), '',
             '每张原图在同一阶段只生成一组三路比例，同图所有目标人物、所有已见类别共享。'
             '每人分别使用原分类头的分数。在图内有任一目标缺可靠人脸时，图级方法全部关闭 Face 末端分数。'
             '固定三路 checkpoint，使用对应阶段留出 TRAIN 的自然图预测学习融合器。', '',
             '| 读出 | Val 最终 | Val 阶段平均 | Test 最终 | Test 阶段平均 |',
             '|---|---:|---:|---:|---:|']
    for name,label in NAMES.items():
        lines.append(f'| {label} | {numbers(four(name))} |')
    lines.extend(['', '## 每一步对全体人物 mAP 的影响', '',
                  '| 对照 | Val 最终差 | Val 阶段平均差 | Test 最终差 | Test 阶段平均差 |',
                  '|---|---:|---:|---:|---:|'])
    for method,other in [('image_fixed','historical_fixed'),('constant','image_fixed'),('full','constant'),
                         ('full_relation','full'),('full_relation','constant'),('full_relation','full_relation_mean'),
                         ('full_relation','full_relation_shuffled')]:
        delta=diff_row(method,other)
        lines.append(f'| {NAMES[method]} 减 {NAMES[other]} | '+ ' | '.join(f'{x:+.4f}' for x in delta)+' |')
    lines.extend(['', '## 最终阶段按图中目标情况拆分', '',
                  '各人群分别计算所属样本的排序。子群 mAP 不能按人数相加得到全体 mAP。以下为最终阶段，逐阶段详见配套 JSON。','',
                  '| 数据与人群 | 人数 | 原固定 | 图级固定 | 学习常量 | 整图 | 整图加关联 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for split, data in diagnostics.items():
        end = data[-1]
        for g,label in SUBGROUPS.items():
            if end['counts'][g]:
                values = [end['scores'][m][g] for m in ('historical_fixed','image_fixed','constant','full','full_relation')]
                lines.append(f"| {split} · {label} | {end['counts'][g]} | {numbers(values)} |")
    lines.extend(['', '归因时，新旧固定之差同时量化整图缺脸策略的代价或收益。'
                  '同规则下的常量、整图及整图加关联依次衡量平均配比、图像内容和跨视图关系的贡献。'
                  '同组均值和整图随机置换衡量图像对应性。结论限 seed1。'])
    (root/'image_subgroups.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
    (root/'results.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True,type=Path)
    p.add_argument('--source', required=True,type=Path)
    p.add_argument('--smoke', action='store_true')
    generate(**vars(p.parse_args()))
