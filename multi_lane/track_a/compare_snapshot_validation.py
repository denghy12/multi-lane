"""Summarize paired full-validation historical checkpoint compositions."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from .compare_joint26 import require
from .snapshot_validation import load_scores

METHODS=('THREE_VIEW','THREE_VIEW_PARAX')


def summarize(root):
    reports={}
    cohort=None
    for seed in (0,1,2):
        reports[seed]={}
        initial=[]
        for method in METHODS:
            path=root/f'seed{seed}'/method
            report=json.loads((path/'evaluation_summary.json').read_text())
            require(report['status']=='complete' and not report['training_performed'] and report['optimizer_updates']==0
                    and not report['test_images_or_scores_opened'] and report['source_unchanged'] and report['clip_unchanged']
                    and report['samples']==2397 and report['seed']==seed,'Evaluation audit failed')
            require(report['source_config']['parax_mode']==('image' if method.endswith('PARAX') else 'disabled'),'Wrong method')
            initial.append(report['source_config']['initial_selector_prompt_classifier_sha256'])
            data=load_scores(path/'assembled_validation.npz')
            if cohort is None:cohort=data
            else:require(all(np.array_equal(data[k],cohort[k]) for k in ('sample_ids','targets','face_reliable','class_indices')),'Paired cohorts differ')
            if method=='THREE_VIEW':
                require(abs(report['snapshot_metrics']['fused']['mAP']-report['final_metrics']['fused']['mAP'])<=.03,
                        'No-ParaX history composition unexpectedly changed AP')
            reports[seed][method]=report
        require(len(set(initial))==1,'Source base initialization differs')
    metrics={}
    for method in METHODS:
        metrics[method]={}
        for kind in ('final_metrics','snapshot_metrics'):
            metrics[method][kind]={}
            for view in ('fused','full','person','face','face_reliable'):
                values=[reports[s][method][kind][view]['mAP'] for s in (0,1,2)]
                metrics[method][kind][view]=dict(mean=statistics.mean(values),std=statistics.stdev(values),seeds=values)
    differences={}
    for label,first,a,second,b in (
        ('protected_minus_baseline','THREE_VIEW_PARAX','snapshot_metrics','THREE_VIEW','snapshot_metrics'),
        ('protected_minus_shared_live','THREE_VIEW_PARAX','snapshot_metrics','THREE_VIEW_PARAX','final_metrics')):
        values=[reports[s][first][a]['fused']['mAP']-reports[s][second][b]['fused']['mAP'] for s in (0,1,2)]
        differences[label]=dict(mean=statistics.mean(values),std=statistics.stdev(values),seeds=values)
    result=dict(status='complete',evaluation_split='val',samples=2397,training_performed=False,test_images_or_scores_opened=False,
                paired_audit_passed=True,metrics=metrics,paired_differences=differences,
                decision_note='Historical composition isolates post-introduction changes but does not undo how later tasks were trained; not an upper bound for all stability methods.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=['# 完整2397人validation的历史任务变换保护对照','','全部26类；三个seed均值±标准差；没有新训练或test访问。','',
           '| 三路模型与预测方式 | Validation mAP |','| --- | ---: |']
    for method,kind,title in [('THREE_VIEW','final_metrics','无ParaX，最终模型'),('THREE_VIEW','snapshot_metrics','无ParaX，各类别取出生任务checkpoint'),
                              ('THREE_VIEW_PARAX','final_metrics','内部共享ParaX，最终模型'),('THREE_VIEW_PARAX','snapshot_metrics','内部ParaX，各类别取出生任务checkpoint')]:
        value=metrics[method][kind]['fused'];lines.append(f'| {title} | {value["mean"]:.4f} ± {value["std"]:.4f} |')
    for label,row in differences.items():lines.extend(['',f'{label}: mean={row["mean"]:+.4f}; seed0/1/2={row["seeds"]}'])
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print('SNAPSHOT_VALIDATION_PAIRED_SUMMARY_COMPLETE',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root)
