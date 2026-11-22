"""Publish completed missing-table cells with audited historical references; no model imports."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path


CATEGORIES = {
    'baseline': ('共同参照：保留三路自己的识别训练',
                 '三路分类分数按可用视图加权：$z= w_Fz_F+w_Pz_P+w_Rz_R$。'),
    'global': ('1．中间相似度与匹配情况：全部patch直接平均',
               '$r=[平均读取量,平均集中程度,平均Q/K余弦,平均Value余弦]$；'
               '$w(x)=\\mathrm{MaskedSoftmax}(b+f_\\theta([r_P,r_R]))$。'
               '有界版本限制输出相对分数的调整；本人／错配组只改变TRAIN监督。'),
    'weighted': ('1．中间相似度与匹配情况：按当前任务关注的位置汇总',
                 '$r_{t,v}=\\sum_j s_{t,v,j}r_{v,j}$；'
                 '$w_t(x)=\\mathrm{MaskedSoftmax}(b+f_\\theta([r_{t,P},r_{t,R}]))$。'),
    'summary': ('2．直接用第10层任务摘要的内容预测比例',
                '$\\bar G_v=M^{-1}\\sum_jG_{v,j}$；'
                '$w(x)=\\mathrm{MaskedSoftmax}(b+f_\\theta([\\bar G_F,\\bar G_P,\\bar G_R]))$。'
                '这里没有显式计算摘要余弦。'),
    'task10': ('3．直接用第10层任务CLS的内容预测比例——只评task6当前三个类别',
               '$x=[P_Fc_F^{10},P_Pc_P^{10},P_Rc_R^{10}]$；'
               '$w(x)=\\mathrm{MaskedSoftmax}(b+Wx)$。各路固定投影4维，'
               '可靠性监督教每路评分接近分支分给真实标签的概率。'),
    'final': ('4．最终任务CLS：直接用内容，或者先算相似度',
              '内容版：$x=[P_Fh_F,P_Ph_P,P_Rh_R]$。相似度版：'
              '$x=[\\cos(h_F,h_P),\\cos(h_F,h_R),\\cos(h_P,h_R)]$。'
              '两者均通过 $w=\\mathrm{MaskedSoftmax}(b+Wx)$ 输出比例；余弦使用原始等维CLS。'),
    'combined': ('5．第10层CLS内容与中间匹配信息一起使用——只评task0／3／6当前新类',
                 '$x=[P_Fc_F^{10},P_Pc_P^{10},P_Rc_R^{10},r_P,r_R]$。'
                 '直接预测：$w=\\mathrm{MaskedSoftmax}(b+Wx)$；保持Full时：'
                 '$[w_P,w_R]=(1-w_F)\\mathrm{Softmax}(f_\\theta(x))$；'
                 '冲突筛选：$w=w_{固定}+\\mathbf1[采纳]\\Delta w$。'),
    'image': ('6．整张原图共用一个比例',
              '$u(I)=P(N^{-1}\\sum_jg^9_{F,j})$；'
              '$w(I)=\\mathrm{MaskedSoftmax}(b+f_\\theta([u(I),\\bar r_P,\\bar r_R]))$。'
              '只看整图时去掉关联项；限制幅度时：'
              '$w_{最终}=(1-\\lambda)w_{共用}+\\lambda w(I)$，$\\lambda$仅由TRAIN确定。'),
}
EXPECTED = [(name, 'equal90') for name in
            ('global_relation', 'bounded_bce', 'own_benefit', 'shuffled_benefit', 'task_summary',
             'tcp_fixed', 'tcp_A', 'tcp_C_tcp', 'combined_fixed', 'quality_B', 'quality_C',
             'budget_person', 'disagreement_middle')]
EXPECTED += [('final_cosine', mode) for mode in ('full', 'equal90')]
EXPECTED += [(name, 'full') for name in
             ('image_fixed', 'image_constant', 'image_full', 'image_full_relation',
              'shrink_full', 'shrink_full_relation')]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Artifact hash differs: {path}')


def metric_pair(summary, method):
    row = summary[method]['all']
    values = [float(row[k]) for k in ('final_mAP', 'average_mAP')]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'Nonfinite metric: {method}')
    return values


def read_eight_pair(folder, style):
    paths = ([folder / 'evaluation' / split / 'result.json' for split in ('val', 'test')]
             if style == 'replay' else [folder / f'{split}_results.json' for split in ('val', 'test')])
    if not all(p.exists() for p in paths):
        return None
    reports = [read(p) for p in paths]
    for split, report in zip(('val', 'test'), reports):
        if report.get('smoke') is not False or report['split'] != split or len(report['stages']) != 8:
            raise ValueError(f'Expected complete formal eight-stage {split} result at {folder}')
    artifacts = {str(p): digest(p) for p in paths}
    if style == 'replay':
        for stage in reports[1]['stages']:
            if stage['protocol']['validation_result_sha256'] != digest(paths[0]):
                raise ValueError('Replay Test refers to different Validation output')
        for split in ('val', 'test'):
            proof = folder / 'branch_replay' / f'{split}_predictions.json'
            if not proof.exists():
                return None
            audit = read(proof)
            if audit['status'] != 'exact' or len(audit['stages']) != 8 or any(
                    row['status'] != 'exact' for row in audit['stages']):
                raise ValueError('Prediction replay was not exact')
            artifacts[str(proof)] = digest(proof)
    else:
        seal = folder / 'validation_seal.json'
        if not seal.exists():
            return None
        lock = read(seal)
        seal_key = 'val_seal_sha256' if style == 'image' else 'validation_seal_sha256'
        checked(seal, reports[1][seal_key])
        checked(paths[0], lock['val_sha256' if style == 'image' else 'val_results_sha256'])
        checked(folder / 'fit.json', lock['fit_sha256'])
        artifacts[str(seal)] = digest(seal)
        artifacts[str(folder / 'fit.json')] = digest(folder / 'fit.json')
    return reports, artifacts


def collect(root):
    reference_path = Path(__file__).with_name('d9_fill_historical_references.json')
    reference = read(reference_path)
    rows = deepcopy(reference['rows'])
    for row in rows.values():
        row['results'] = {key: dict(value, source='historical_verified_report')
                          for key, value in row.pop('historical').items()}
    artifacts = {str(reference_path): digest(reference_path)}

    def put(key, mode, val, test, origin):
        if any(not math.isfinite(float(v)) for v in [*val, *test]):
            raise ValueError(f'Nonfinite completed metric: {key}')
        rows[key]['results'][mode] = dict(val=list(val), test=list(test),
                                        source='new_computed_result', origin=str(origin))

    for mode in ('full', 'equal90'):
        folder = root / f'replay_{mode}'
        pair = read_eight_pair(folder, 'replay')
        if pair is None:
            continue
        reports, files = pair
        artifacts.update(files)
        methods = ['final_cosine']
        if mode == 'equal90':
            methods += ['global_relation', 'bounded_bce', 'own_benefit', 'shuffled_benefit', 'task_summary']
        for method in methods:
            put(method, mode, *(metric_pair(r['summary'], method) for r in reports), folder)
        # Pair each newly learned result with the fixed output from the very same replay.
        put('baseline', mode, *(metric_pair(r['summary'], 'fixed') for r in reports), folder)

    for style, sub in [('image', 'readout'), ('shrink', 'shrink')]:
        folder = root / 'image_full' / sub
        pair = read_eight_pair(folder, style)
        if pair is None:
            continue
        reports, files = pair
        artifacts.update(files)
        names = {'image_fixed': 'image_fixed', 'image_constant': 'constant',
                 'image_full': 'full', 'image_full_relation': 'full_relation'} if style == 'image' else {
                 'shrink_full': 'shrink_full', 'shrink_full_relation': 'shrink_full_relation'}
        for row, method in names.items():
            put(row, 'full', *(metric_pair(r['summary'], method) for r in reports), folder)

    folder = root / 'probes_equal90'
    combined = folder / 'results.json'
    if combined.exists():
        data = read(combined)
        artifacts[str(combined)] = digest(combined)
        for family, ids in data['sources'].items():
            val_path = folder / family / ('tcp_validation_lock.json' if family == 'tcp' else 'validation_seal.json')
            test_path = folder / family / ('tcp_test_results.json' if family == 'tcp' else 'test_results.json')
            checked(val_path, ids['validation_sha256'])
            checked(test_path, ids['test_sha256'])
            locked, test = read(val_path), read(test_path)
            checked(val_path, test['validation_lock_sha256' if family == 'tcp' else 'validation_seal_sha256'])
            for filename, expected in locked['artifact_hashes' if family == 'tcp' else 'artifacts'].items():
                checked(folder / family / filename, expected)
            artifacts.update({str(val_path): digest(val_path), str(test_path): digest(test_path)})
        mappings = [('tcp_fixed', 'tcp', 'fixed'), ('tcp_A', 'tcp', 'A'), ('tcp_C_tcp', 'tcp', 'C_tcp'),
                    ('combined_fixed', 'quality', 'fixed'), ('quality_B', 'quality', 'B'),
                    ('quality_C', 'quality', 'C'), ('budget_person', 'budget', 'person'),
                    ('disagreement_middle', 'disagreement', 'middle')]
        for key, family, method in mappings:
            report = data['rows'][family]
            if report['tasks'] != ([6] if family == 'tcp' else [0, 3, 6]):
                raise ValueError('Wrong probe evaluation task scope')
            put(key, 'equal90', [report['validation'][method]], [report['test'][method]], combined)
        for split in ('validation', 'test'):
            baseline = data['rows']['quality'][split]['fixed']
            if any(abs(data['rows'][family][split]['fixed'] - baseline) > 1e-8
                   for family in ('budget', 'disagreement')):
                raise ValueError('Current-head fixed controls differ between probe families')
    missing = [dict(method=key, coverage=mode, name=rows[key]['name']) for key, mode in EXPECTED
               if rows[key]['results'].get(mode, {}).get('source') != 'new_computed_result']
    return dict(schema=1, root=str(root), complete=not missing, expected_missing_cells=len(EXPECTED),
                newly_completed_cells=len(EXPECTED) - len(missing), pending=missing,
                historical_documents=reference['source_documents'],
                historical_result_artifacts=reference.get('source_result_artifacts', {}),
                artifacts=artifacts, rows=rows)


def cell(row, mode, split):
    result = row['results'].get(mode)
    return '待完成' if result is None else '／'.join(f'{value:.4f}' for value in result[split])


def markdown(report):
    text = ['# 融合比例实验补表', '',
            f'本轮待补项目共{report["expected_missing_cells"]}项，已完成{report["newly_completed_cells"]}项。'
            + ('全部补齐。' if report['complete'] else '未完成处保留“待完成”，不填估算值。'), '',
            '实验共同条件：EMOTIC历史划分、seed1；保留Frozen9的Person和Face跨视图交互。'
            '全TRAIN使用全部当前任务合法TRAIN人物；90%一侧只使用原随机划分、已补足13,950次分支更新的模型。'
            '融合监督全部来自TRAIN，Validation／Test不更新融合参数。', '',
            '全TRAIN的融合人物已按原分支配方参加过识别训练；90%一侧用永久留出的10%原图人物学习融合。'
            '完整八阶段与task6／三个当前头诊断分别列出。历史结果保留原报告四位小数，新结果由封存产物读取。', '']
    for category, (title, formula) in CATEGORIES.items():
        members = [r for r in report['rows'].values() if r['category'] == category]
        scoped = members[0]['scope'] != 'eight_stage'
        text.extend(['## ' + title, '',
                     ('评价范围：task6当前三个类别的mAP。' if category == 'task10' else
                      '评价范围：先计算task0／3／6各自当前新类mAP，再将三个任务等权平均。') if scoped else
                     '评价范围：完整八阶段；单元格依次为最终mAP／阶段平均mAP。', '',
                     '| 方法 | 比例读取什么信息 | 全TRAIN Validation | 全TRAIN Test | 补足更新90% Validation | 补足更新90% Test |',
                     '| --- | --- | ---: | ---: | ---: | ---: |'])
        for row in members:
            text.append('| ' + ' | '.join([row['name'], row['input'], cell(row, 'full', 'val'),
                         cell(row, 'full', 'test'), cell(row, 'equal90', 'val'), cell(row, 'equal90', 'test')]) + ' |')
        text.extend(['', '计算简式：' + formula, ''])
    text.extend(['## 数字来源与比较限制', '',
                 '表中的“待完成”仅表示尚无本批核验完成的结果。八阶段补跑要求原分支及固定融合预测逐值重放；'
                 'task6和三个任务的诊断直接复用既有checkpoint提取特征，不作为重新训练八任务的成绩。', '',
                 '图级方法使同图所有人物及已见类别共用比例，Face可用性也按整图确定，须优先与“同一张图用固定比例”比较。'
                 '全TRAIN图级三折只隔离融合层拟合，分支此前仍见过这些TRAIN图片。', '',
                 '旧全TRAIN早期门控与新增90%门控的融合训练人群不同；新增90%侧按原任务分支更新时钟读取留出人物。'
                 '不要把两列差值只解释成特征接口的变化。', '',
                 '机器可读results.json记录每项来源、结果文件SHA-256和历史报告快照SHA-256。'])
    return '\n'.join(text) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Batch root containing formal/, or formal/ itself')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-complete', action='store_true', help='Write partial report but fail if any of the 21 cells is unfinished')
    args = parser.parse_args()
    root = args.root / 'formal' if (args.root / 'formal').is_dir() else args.root
    out = args.output or root / 'report'
    report = collect(root)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out / 'completed_tables.md').write_text(markdown(report))
    print(f'Completed {report["newly_completed_cells"]}/{report["expected_missing_cells"]}: {out}')
    if args.require_complete and not report['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
