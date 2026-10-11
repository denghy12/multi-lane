"""Summarize paired full-validation historical checkpoint compositions."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from .compare_joint26 import require
from .snapshot_validation import load_scores

METHODS=('THREE_VIEW','THREE_VIEW_PARAX')


def summarize(root, seeds=(0,1,2), split='val'):
    expected_people=2397 if split=='val' else 5368
    def stats(values):
        return dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None,seeds=values)
    reports={}
    cohort=None
    for seed in seeds:
        reports[seed]={}
        initial=[]
        for method in METHODS:
            path=root/f'seed{seed}'/method
            report=json.loads((path/'evaluation_summary.json').read_text())
            require(report['status']=='complete' and not report['training_performed'] and report['optimizer_updates']==0
                    and report['test_images_or_scores_opened']==(split=='test') and report['evaluation_split']==split and report['source_unchanged'] and report['clip_unchanged']
                    and report['samples']==expected_people and report['seed']==seed,'Evaluation audit failed')
            require(report['source_config']['parax_mode']==('image' if method.endswith('PARAX') else 'disabled'),'Wrong method')
            initial.append(report['source_config']['initial_selector_prompt_classifier_sha256'])
            data=load_scores(path/f'assembled_{split}.npz')
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
                values=[reports[s][method][kind][view]['mAP'] for s in seeds]
                metrics[method][kind][view]=stats(values)
    differences={}
    for label,first,a,second,b in (
        ('protected_minus_baseline','THREE_VIEW_PARAX','snapshot_metrics','THREE_VIEW','snapshot_metrics'),
        ('protected_minus_shared_live','THREE_VIEW_PARAX','snapshot_metrics','THREE_VIEW_PARAX','final_metrics')):
        values=[reports[s][first][a]['fused']['mAP']-reports[s][second][b]['fused']['mAP'] for s in seeds]
        differences[label]=stats(values)
    result=dict(status='complete',evaluation_split=split,samples=expected_people,training_performed=False,test_images_or_scores_opened=split=='test',test_weight_search=False,seed_ids=list(seeds),
                paired_audit_passed=True,metrics=metrics,paired_differences=differences,
                decision_note='Historical composition isolates post-introduction changes but does not undo how later tasks were trained; not an upper bound for all stability methods.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines=[f'# 完整{expected_people}人{split}的历史任务变换保护对照','',f'全部26类；seed={list(seeds)}；没有新训练或权重搜索。单seed不计算标准差。','',
           f'| 三路模型与预测方式 | {split} mAP |','| --- | ---: |']
    for method,kind,title in [('THREE_VIEW','final_metrics','无ParaX，最终模型'),('THREE_VIEW','snapshot_metrics','无ParaX，各类别取出生任务checkpoint'),
                              ('THREE_VIEW_PARAX','final_metrics','内部共享ParaX，最终模型'),('THREE_VIEW_PARAX','snapshot_metrics','内部ParaX，各类别取出生任务checkpoint')]:
        value=metrics[method][kind]['fused'];formatted=f'{value["mean"]:.4f}'
        if value['std'] is not None:formatted+=f' ± {value["std"]:.4f}'
        lines.append(f'| {title} | {formatted} |')
    for label,row in differences.items():lines.extend(['',f'{label}: mean={row["mean"]:+.4f}; seed0/1/2={row["seeds"]}'])
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    print('SNAPSHOT_VALIDATION_PAIRED_SUMMARY_COMPLETE',flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--split',choices=('val','test'),default='val')
    args=parser.parse_args();summarize(args.root,tuple(args.seeds),args.split)
