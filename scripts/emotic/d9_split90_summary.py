"""Seal all Validation comparisons, then report the four TRAIN splits on Validation and Test."""
import argparse
import os
from pathlib import Path

STRATEGIES = ('hash2', 'meta', 'online')
COLUMNS = (('full', 'all'), ('person', 'all'), ('face', 'reliable'), ('fixed', 'all'))
METRICS = ('final_mAP', 'average_mAP')
RANDOM = ('historical', 'hash2')
STRATIFIED = ('meta', 'online')
NAMES = {'historical': '原随机 90%', 'hash2': '第二份随机 90%', 'meta': '按图中人数和人脸情况分层',
         'online': '按当前任务情绪分层', 'full_train': '全 TRAIN 参照'}
VIEWS = {'full': '整图 Full', 'person': '身体 Person', 'face': '可靠 Face', 'fixed': '三路固定融合'}
REFERENCE_COMPARISON = Path('../multi-lane-main-d9-holdout-equal-updates/output/'
                            'd9_equal_updates/seed1_20261119/formal/comparison')


def above_both(table, strategy, method):
    """Both scores must be strictly higher than both random allocations."""
    return all(table[strategy][method][metric] > table[control][method][metric]
               for metric in METRICS for control in RANDOM)


def report_markdown(tables):
    def values(name, method):
        return [tables[split][name][method][key] for split in ('val', 'test') for key in METRICS]

    def row(label, scores, signed=False):
        numbers = [format(value, '+.4f' if signed else '.4f') for value in scores]
        return '| ' + ' | '.join([label, *numbers]) + ' |'

    def difference(name, control, method):
        return [x - y for x, y in zip(values(name, method), values(control, method))]

    def points(scores):
        return '／'.join(f'{value:+.4f}' for value in scores)

    heading = '| 原图划分 | 验证最终 | 验证阶段平均 | 测试最终 | 测试阶段平均 |'
    separator = '|---|---:|---:|---:|---:|'
    lines = ['# 同样保留 90% 训练图片，哪种划分让三路学得更好？', '',
             '四组均用 EMOTIC 的 11,623 张原图训练分支，留出 1,292 张。模型 seed1、每任务 30 轮、'
             '共 13,950 次分支更新。同一原图的人物留在同一侧。正式评价使用自然清晰度的验证集和测试集。', '',
             '所有组沿用相同的 Frozen9 跨视图交互和固定融合比例。人脸可靠时，Full、Person、Face 各占 '
             '64%、16%、20%。其余人物使用 Full 80%、Person 20%。最终值为第八阶段 mAP，阶段平均为八个阶段 mAP 的平均值。', '',
             '## 固定融合与各分支成绩', '']
    for method in ('fixed', 'full', 'person', 'face'):
        title = VIEWS[method] + ('（仅可靠人脸人物）' if method == 'face' else '（全体人物）')
        lines.extend(['### ' + title, '', heading, separator])
        for name in (*RANDOM, *STRATIFIED, 'full_train'):
            lines.append(row(NAMES[name], values(name, method)))
        lines.append('')

    lines.extend(['## 与两份随机名单的配对差值', '',
                  '正数表示高于该随机参照。每一行都在相同评价人物和相同读出上相减，单位为 mAP 点。', '',
                  '| 分层方法 | 参照 | 读出 | 验证最终差 | 验证平均差 | 测试最终差 | 测试平均差 |',
                  '|---|---|---|---:|---:|---:|---:|'])
    for name in STRATIFIED:
        for control in RANDOM:
            for method in ('fixed', 'full', 'person', 'face'):
                lines.append(row(' | '.join((NAMES[name], NAMES[control], VIEWS[method])),
                                 difference(name, control, method), signed=True))
    lines.extend(['', '## 怎样解释这组结果', '',
                  '先按已封存的 Validation 判断。只有固定融合的两项指标均高于两份随机名单，且至少同一条单路的'
                  '两项验证指标也都高于两份随机名单，才把该划分列为候选。下列每对差值依次是最终和阶段平均。', ''])
    lines.append(f'- 只换一份随机名单，固定融合的验证变化为 {points(difference("hash2", "historical", "fixed")[:2])}，'
                 f'测试变化为 {points(difference("hash2", "historical", "fixed")[2:])}。这是 seed1 下普通换图已产生的变化。')
    for name in STRATIFIED:
        val = tables['val']
        fusion_pass = above_both(val, name, 'fixed')
        supporting = [VIEWS[method] for method, _ in COLUMNS if method != 'fixed' and above_both(val, name, method)]
        verdict = '通过候选条件' if fusion_pass and supporting else '未通过候选条件'
        fusion = '验证融合两项均超过两份随机名单' if fusion_pass else '验证融合未能在两项上同时超过两份随机名单'
        branches = '同向单路有' + '、'.join(supporting) if supporting else '没有单路在两项验证指标上同时超过两份随机名单'
        delta_full = difference(name, 'full_train', 'fixed')[2:]
        test = ('测试融合的最终和阶段平均均高于两份随机名单' if above_both(tables['test'], name, 'fixed')
                else '测试融合未在两项指标上同时高于两份随机名单')
        lines.append(f'- **{NAMES[name]}{verdict}。** {fusion}，{branches}。{test}。相对全 TRAIN 固定融合，'
                     f'测试差值为 {points(delta_full)}。')
    lines.extend(['', '训练更新次数、模型和固定读出已配平。比较反映 seed1 下换用这些具体原图名单造成的整体变化，'
                  '涉及训练人物、情绪覆盖及组批变化。单路分数有助于定位变化，同一划分规则的可重复性仍需独立配对复验。', ''])
    return '\n'.join(lines)


def with_absolute_reference(summary, source_hashes, reference):
    if source_hashes != {reference['official_result_sha256']['equal_updates']} or summary['split'] != reference['split']:
        raise ValueError('Full-TRAIN report is paired to a different historical equal-updates evaluation')
    scores = {}
    for method, group in COLUMNS:
        historical = reference['branch_summary'][method][group]['values']
        if historical['equal_updates'] != summary['table']['historical'][method]:
            raise ValueError('Historical equal-updates scores differ from the full-TRAIN comparison')
        scores[method] = historical['full']
    return dict(summary['table'], full_train=scores)


def summarize(root, split, reference_comparison=REFERENCE_COMPARISON):
    from multi_lane.track_a.openai_clip_loader import sha256_file
    from scripts.emotic.d9_a_eight_eval import read, require_digest
    from scripts.emotic.d9_paired_a_evaluate import write_verified

    seal = root / 'validation_seal.json'
    if split == 'test':
        locked = read(seal)
        require_digest(root/'val_summary.json', locked['summary_sha256'])
        for name in STRATEGIES:
            require_digest(root/name/'comparison/val_comparison.json', locked['comparison_sha256'][name])
    reports, digests, table = {}, {}, {}
    for name in STRATEGIES:
        path = root/name/'comparison'/f'{split}_comparison.json'
        reports[name] = read(path)
        if reports[name]['strategy'] != name or reports[name]['split'] != split:
            raise ValueError(f'Unexpected report at {path}')
        digests[name] = sha256_file(path)
        data = reports[name]['summary']
        reference = {m: data[m][g]['reference'] for m, g in COLUMNS}
        if 'historical' in table and table['historical'] != reference:
            raise ValueError('Splits used different historical controls')
        table['historical'] = reference
        table[name] = {m: data[m][g]['candidate'] for m, g in COLUMNS}
    summary_file = root/f'{split}_summary.json'
    write_verified(summary_file, dict(split=split, table=table, comparison_sha256=digests))
    if split == 'val':
        write_verified(seal, dict(summary_sha256=sha256_file(summary_file), comparison_sha256=digests))
    else:
        tables = {}
        for part in ('val', 'test'):
            summary = read(root/f'{part}_summary.json')
            official = read(reference_comparison/f'{part}_comparison.json')
            source_reports = (reports.values() if part == 'test' else
                              (read(root/name/'comparison/val_comparison.json') for name in STRATEGIES))
            source_hashes = {report['native_sha256']['reference'] for report in source_reports}
            tables[part] = with_absolute_reference(summary, source_hashes, official)
        report = report_markdown(tables)
        path = root/'results.md'
        if path.exists() and path.read_text(encoding='utf-8') != report:
            raise ValueError(f'Result differs from existing artifact: {path}')
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(report, encoding='utf-8')
        tmp.replace(path)
    print(split, table, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--reference-comparison', type=Path, default=REFERENCE_COMPARISON)
    args = parser.parse_args()
    summarize(args.root, args.split, args.reference_comparison)
