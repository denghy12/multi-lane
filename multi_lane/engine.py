# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for MULTI-LANE
# Thomas De Min thomas.demin@unitn.it
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import json
from typing import Iterable, List
import wandb
import datetime
from html import escape

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from timm.utils import accuracy

import multi_lane.utils as utils
import multi_lane.datasets as datasets


DETAIL_METRICS = [
    'ap', 'f1', 'precision', 'recall', 'support',
    'predicted_positive', 'tp', 'fp', 'fn', 'tn',
]


def _safe_divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def _average_precision(scores: torch.Tensor, targets: torch.Tensor):
    positives = int(targets.sum().item())
    if positives == 0:
        return 0.0

    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order].float()
    true_positive_cumsum = torch.cumsum(sorted_targets, dim=0)
    rank = torch.arange(1, sorted_targets.numel() + 1, device=scores.device).float()
    precision_at_rank = true_positive_cumsum / rank
    return float((precision_at_rank * sorted_targets).sum().item() / positives)


def _multilabel_class_details(logits: torch.Tensor, targets: torch.Tensor, class_ids: List[int]):
    scores = torch.sigmoid(logits.detach().cpu())
    predictions = scores.gt(0.8)
    targets = targets.detach().cpu().bool()
    details = {}

    for class_id in class_ids:
        pred = predictions[:, class_id]
        target = targets[:, class_id]

        tp = int((pred & target).sum().item())
        fp = int((pred & ~target).sum().item())
        fn = int((~pred & target).sum().item())
        tn = int((~pred & ~target).sum().item())
        support = int(target.sum().item())
        predicted_positive = int(pred.sum().item())
        precision = _safe_divide(tp, predicted_positive)
        recall = _safe_divide(tp, support)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        details[class_id] = {
            'ap': _average_precision(scores[:, class_id], target.float()),
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'support': support,
            'predicted_positive': predicted_positive,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }

    return details


def _support_counts(targets, num_classes):
    counts = [0 for _ in range(num_classes)]
    for target in targets:
        for class_id in target:
            counts[int(class_id)] += 1
    return counts


def _class_task_map(class_mask):
    mapping = {}
    for task_id, classes in enumerate(class_mask):
        for class_id in classes:
            mapping[int(class_id)] = task_id
    return mapping


def _class_names(dataset, num_classes):
    classes = getattr(dataset, 'classes', None)
    if classes is None:
        return [str(i) for i in range(num_classes)]
    return [str(classes[i]) for i in range(num_classes)]


def _detail_report_path(args):
    run_name = args.name or args.dataset.replace('Split-', '').lower()
    file_name = f'{run_name}_per_class_task_table.html'
    return os.path.join(args.output_dir, 'detail', file_name)


def _write_detail_report(path, title, table_columns, table_rows, overall_columns, overall_rows):
    parent = os.path.abspath(os.path.dirname(path))
    if not os.path.exists(parent):
        os.makedirs(parent)

    json_table_columns = json.dumps(table_columns, ensure_ascii=False)
    json_table_rows = json.dumps(table_rows, ensure_ascii=False, allow_nan=False)
    json_overall_columns = json.dumps(overall_columns, ensure_ascii=False)
    json_overall_rows = json.dumps(overall_rows, ensure_ascii=False, allow_nan=False)
    escaped_title = escape(title)

    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escaped_title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #1f2937;
    }}
    h1 {{
      margin: 0 0 14px;
      font-size: 28px;
      line-height: 1.2;
    }}
    h2 {{
      margin: 18px 0 10px;
      font-size: 18px;
      line-height: 1.2;
    }}
    .toolbar {{
      align-items: center;
      display: flex;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .segmented-control {{
      display: inline-flex;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      overflow: hidden;
      background: #ffffff;
    }}
    .segmented-control button {{
      border: 0;
      border-right: 1px solid #cbd5e1;
      background: #ffffff;
      color: #334155;
      cursor: pointer;
      font-size: 13px;
      padding: 7px 12px;
    }}
    .segmented-control button:last-child {{
      border-right: 0;
    }}
    .segmented-control button.active {{
      background: #0f172a;
      color: #ffffff;
    }}
    .layout-hint {{
      color: #64748b;
      font-size: 13px;
    }}
    .table-wrap {{
      max-height: calc(100vh - 130px);
      overflow: auto;
      border: 1px solid #d1d5db;
    }}
    .overall-wrap {{
      max-height: 240px;
      overflow: auto;
      border: 1px solid #d1d5db;
      margin-bottom: 16px;
    }}
    table.metric-table {{
      border-collapse: collapse;
      font-size: 12px;
      white-space: nowrap;
    }}
    .metric-table th,
    .metric-table td {{
      border: 1px solid #e5e7eb;
      padding: 6px 8px;
      text-align: right;
    }}
    .metric-table th {{
      position: sticky;
      top: 0;
      background: #f3f4f6;
      z-index: 2;
    }}
    .metric-table th:nth-child(1),
    .metric-table td:nth-child(1) {{
      position: sticky;
      left: 0;
      min-width: 64px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(2),
    .metric-table td:nth-child(2) {{
      position: sticky;
      left: 64px;
      min-width: 150px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(3),
    .metric-table td:nth-child(3) {{
      position: sticky;
      left: 214px;
      min-width: 80px;
      text-align: left;
      background: #ffffff;
      z-index: 1;
    }}
    .metric-table th:nth-child(4),
    .metric-table td:nth-child(4) {{
      position: sticky;
      left: 294px;
      min-width: 80px;
      text-align: right;
      background: #ffffff;
      z-index: 1;
      box-shadow: 1px 0 0 #e5e7eb;
    }}
    .metric-table th:nth-child(-n+4) {{
      background: #f3f4f6;
      z-index: 3;
    }}
  </style>
</head>
<body>
  <h1>{escaped_title}</h1>
  <h2>总体指标</h2>
  <div class="overall-wrap">
    <table class="metric-table" id="overallTable"></table>
  </div>
  <div class="toolbar">
    <div class="segmented-control" role="group" aria-label="column layout">
      <button type="button" class="active" data-layout="task">按 task 分组</button>
      <button type="button" data-layout="metric">按指标分组</button>
    </div>
    <span class="layout-hint" id="layoutHint"></span>
  </div>
  <div class="table-wrap">
    <table class="metric-table" id="metricTable"></table>
  </div>
  <script>
    const tableColumns = {json_table_columns};
    const tableRows = {json_table_rows};
    const overallColumns = {json_overall_columns};
    const overallRows = {json_overall_rows};
    const metricPreference = ["ap", "f1", "precision", "recall", "support", "predicted_positive", "tp", "fp", "fn", "tn"];
    const frozenColumnCount = 4;
    const indexColumns = tableColumns.slice(0, frozenColumnCount);
    const metricColumns = tableColumns.slice(frozenColumnCount);

    function parseMetricColumn(column) {{
      const match = /^task(\\d+)_(.+)$/.exec(column);
      if (!match) {{
        return null;
      }}
      return {{
        column,
        task: Number(match[1]),
        metric: match[2],
      }};
    }}

    const parsedColumns = metricColumns
      .map(parseMetricColumn)
      .filter(Boolean);
    const tasks = [...new Set(parsedColumns.map((item) => item.task))]
      .sort((a, b) => a - b);
    const discoveredMetrics = [...new Set(parsedColumns.map((item) => item.metric))];
    const metrics = [
      ...metricPreference.filter((metric) => discoveredMetrics.includes(metric)),
      ...discoveredMetrics.filter((metric) => !metricPreference.includes(metric)),
    ];
    const knownMetricColumns = new Set(
      parsedColumns.map((item) => item.column)
    );
    const extraColumns = metricColumns.filter(
      (column) => !knownMetricColumns.has(column)
    );

    function taskFirstColumns() {{
      return [
        ...indexColumns,
        ...tasks.flatMap((task) =>
          metrics
            .map((metric) => `task${{task}}_${{metric}}`)
            .filter((column) => tableColumns.includes(column))
        ),
        ...extraColumns,
      ];
    }}

    function metricFirstColumns() {{
      return [
        ...indexColumns,
        ...metrics.flatMap((metric) =>
          tasks
            .map((task) => `task${{task}}_${{metric}}`)
            .filter((column) => tableColumns.includes(column))
        ),
        ...extraColumns,
      ];
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return "NaN";
      }}
      if (typeof value === "number") {{
        return Number.isInteger(value) ? String(value) : value.toFixed(6);
      }}
      return String(value);
    }}

    function renderStaticTable(tableId, columns, rows) {{
      const table = document.getElementById(tableId);
      const thead = document.createElement("thead");
      const headerRow = document.createElement("tr");
      columns.forEach((column) => {{
        const th = document.createElement("th");
        th.textContent = column;
        headerRow.appendChild(th);
      }});
      thead.appendChild(headerRow);

      const tbody = document.createElement("tbody");
      rows.forEach((row) => {{
        const tr = document.createElement("tr");
        columns.forEach((column) => {{
          const td = document.createElement("td");
          td.textContent = formatValue(row[column]);
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      table.replaceChildren(thead, tbody);
    }}

    function renderTable(columns) {{
      renderStaticTable("metricTable", columns, tableRows);
    }}

    function setLayout(layout) {{
      const isMetricLayout = layout === "metric";
      renderTable(isMetricLayout ? metricFirstColumns() : taskFirstColumns());
      document.querySelectorAll("[data-layout]").forEach((button) => {{
        button.classList.toggle("active", button.dataset.layout === layout);
      }});
      document.getElementById("layoutHint").textContent = isMetricLayout
        ? "当前列顺序：task0-7 的 ap，然后 task0-7 的 f1，依次类推。"
        : "当前列顺序：task0 的所有指标，然后 task1 的所有指标，依次类推。";
    }}

    renderStaticTable("overallTable", overallColumns, overallRows);
    document.querySelectorAll("[data-layout]").forEach((button) => {{
      button.addEventListener("click", () => setLayout(button.dataset.layout));
    }});
    setLayout("task");
  </script>
</body>
</html>
'''

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(html)

    json_path = os.path.splitext(path)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump({
            'table_columns': table_columns,
            'table_rows': table_rows,
            'overall_columns': overall_columns,
            'overall_rows': overall_rows,
        }, fp, ensure_ascii=False, indent=2, allow_nan=False)


def _update_multilabel_detail_report(args, class_mask, val_dataset, task_id, predictions,
                                     targets, loss, mAP, of1, cf1):
    if not utils.is_main_process():
        return

    if not hasattr(args, 'detail_report_state'):
        class_names = _class_names(val_dataset, args.num_classes)
        class_tasks = _class_task_map(class_mask)
        counts = _support_counts(val_dataset.targets, args.num_classes)
        args.detail_report_state = {
            'rows': [
                {
                    'class_id': class_id,
                    'class_name': class_names[class_id],
                    'class_task': class_tasks.get(class_id),
                    'count': counts[class_id],
                }
                for class_id in range(args.num_classes)
            ],
            'overall_rows': [],
        }

    seen_classes = sorted([c for m in class_mask[:task_id + 1] for c in m])
    details = _multilabel_class_details(predictions, targets, seen_classes)
    for row in args.detail_report_state['rows']:
        class_id = row['class_id']
        class_task = row['class_task']
        for metric in DETAIL_METRICS:
            column = f'task{task_id}_{metric}'
            row[column] = details[class_id][metric] if class_task <= task_id and class_id in details else None

    args.detail_report_state['overall_rows'].append({
        'task': task_id,
        'seen_classes': len(seen_classes),
        'samples': int(targets.shape[0]),
        'mAP': float(mAP.item()),
        'amAP': float(np.mean([row['mAP'] for row in args.detail_report_state['overall_rows']] + [mAP.item()])),
        'oF1': float(of1.item()),
        'cF1': float(cf1.item()),
        'loss': float(loss),
    })

    table_columns = ['class_id', 'class_name', 'class_task', 'count']
    for current_task in range(args.num_tasks):
        for metric in DETAIL_METRICS:
            table_columns.append(f'task{current_task}_{metric}')

    for row in args.detail_report_state['rows']:
        for column in table_columns:
            if column not in row:
                row[column] = None

    overall_columns = ['task', 'seen_classes', 'samples', 'mAP', 'amAP', 'oF1', 'cF1', 'loss']
    title = f'{args.name or args.dataset.replace("Split-", "").lower()}_per_class_metrics'
    path = _detail_report_path(args)
    _write_detail_report(
        path=path,
        title=title,
        table_columns=table_columns,
        table_rows=args.detail_report_state['rows'],
        overall_columns=overall_columns,
        overall_rows=args.detail_report_state['overall_rows'],
    )
    print(f"Saved detail report to {path}")


def train_one_epoch(model: nn.Module, criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, set_training_mode=True, task_id=-1, 
                    class_mask=None, run=None, args=None,):

    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for i, (input, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits, feats, frozen_feats, sim, tasks = model(input)

        # here is the trick to mask out classes of non-current tasks
        fill_value = float('-inf') if type(criterion) == torch.nn.CrossEntropyLoss else 0
        logits = utils.mask_logits(logits, class_mask, args.num_classes, [task_id], fill_value=fill_value)

        if type(criterion) == torch.nn.BCEWithLogitsLoss:
            target = utils.mask_logits(target, class_mask, args.num_classes, [task_id], fill_value=0)

        if type(criterion) == torch.nn.CrossEntropyLoss:
            loss = criterion(logits / args.temperature, target)
        else:
            loss = criterion(logits / args.temperature, target.float())
        
        if args.head_mode  == 'task':
            loss -= sim

        if type(criterion) == torch.nn.CrossEntropyLoss:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            mAP = utils.mean_average_precision(logits, target)
            clean_logits = utils.remove_logits(logits, class_mask, [task_id])
            clean_target = utils.remove_logits(target, class_mask, [task_id])
            of1 = utils.f1_score_overall(clean_logits, clean_target)
            cf1, _ = utils.f1_score_per_class(clean_logits, clean_target)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        
        loss.backward()
        if i % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        # metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Loss'].update(loss.item(), n=input.shape[0])
        if type(criterion) == torch.nn.CrossEntropyLoss:
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        else:
            metric_logger.meters['mAP'].update(mAP.item(), n=input.shape[0])
            metric_logger.meters['oF1'].update(of1.item(), n=input.shape[0])
            metric_logger.meters['cF1'].update(cf1.item(), n=input.shape[0])

        if args.head_mode  == 'task':
            metric_logger.meters['Sim'].update(sim.item(), n=input.shape[0])

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    wandb_dict = {}
    if run is not None:
        if 'Acc@1' in metric_logger.meters:
            wandb_dict['train/acc1'] = metric_logger.meters['Acc@1'].global_avg
        if 'Acc@5' in metric_logger.meters:
            wandb_dict['train/acc5'] = metric_logger.meters['Acc@5'].global_avg
        if 'mAP' in metric_logger.meters:
            wandb_dict['train/mAP'] = metric_logger.meters['mAP'].global_avg
        if 'oF1' in metric_logger.meters:
            wandb_dict['train/oF1'] = metric_logger.meters['oF1'].global_avg
        if 'cF1' in metric_logger.meters:
            wandb_dict['train/cF1'] = metric_logger.meters['cF1'].global_avg
        if 'Loss' in metric_logger.meters:
            wandb_dict['train/loss'] = metric_logger.meters['Loss'].global_avg
        if 'Lr' in metric_logger.meters:
            wandb_dict['train/lr'] = metric_logger.meters['Lr'].global_avg
        if 'Sim' in metric_logger.meters:
            wandb_dict['train/sim'] = metric_logger.meters['Sim'].global_avg
    return wandb_dict


@torch.no_grad()
def evaluate(model: nn.Module, criterion, data_loader: Iterable, device, task_id=-1, tasks_so_far=-1,
             class_mask=None, args=None,):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()

    predictions = []
    targets = []
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits, feats, frozen_feats, sim, tasks = model(input, eval=True)

        # here is the trick to mask out classes of non-current tasks
        logits = utils.mask_logits(logits, class_mask, args.num_classes, list(range(tasks_so_far+1)))
        loss = criterion(logits / args.temperature, target)
        
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        acc_task = (tasks == task_id).float().mean()

        metric_logger.meters['Loss'].update(loss.item())
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
        if args.head_mode == 'task':
            metric_logger.meters['Sim'].update(sim.item(), n=input.shape[0])
            metric_logger.meters['Acc_Task'].update(acc_task.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    stats = '* '
    stats += f"Acc@1 {metric_logger.meters['Acc@1'].global_avg:.3f} Acc@5 {metric_logger.meters['Acc@5'].global_avg:.3f} "
    stats += f"Loss {metric_logger.meters['Loss'].global_avg:.3f}"
    if args.head_mode  == 'task':
        stats += f" Acc_Task {metric_logger.meters['Acc_Task'].global_avg:.3f}"
    
    print(stats)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: nn.Module, criterion, data_loader, device, task_id=-1, acc_matrix=None,
                      class_mask=None, run=None, args=None,):
    stat_matrix = np.zeros((4, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, criterion=criterion, data_loader=data_loader[i]['val'], 
                              device=device, task_id=i, tasks_so_far=task_id, class_mask=class_mask, 
                              args=args)

        # save stats
        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc_Task'] if args.head_mode  == 'task' else 0

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)
    diagonal = np.diag(acc_matrix)

    result_str = f"[Average accuracy till task{task_id+1}]"
    result_str += f"\tAcc@1: {avg_stat[0]:.4f}\tAcc@5: {avg_stat[1]:.4f}"
    
    if args.head_mode  == 'task':
        result_str += f"\tAcc Task: {avg_stat[3]:.4f}"
    
    result_str += f"\tLoss: {avg_stat[2]:.4f}"

    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    wandb_dict = {}
    if run is not None:
        for task in range(task_id+1):
            wandb_dict[f'test/task{task}_acc'] = stat_matrix[0][task]
            if args.head_mode  == 'task':
                wandb_dict[f'test/task{task}_acc_task'] = stat_matrix[3][task]
        wandb_dict.update({
            f'test/avg_acc': avg_stat[0],
        })
        if task_id > 0:
            wandb_dict.update({
                'test/forgetting': forgetting,
            })

    return wandb_dict


@torch.no_grad()
def evaluate_till_now_multi(model: nn.Module, criterion, data_loader, device: torch.device, 
                            task_id=-1, mAP_vector=None, class_mask=None, run=None, args=None,
                            val_dataset=None):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    #! ------- Extract predictions ------
    predictions = []
    targets = []

    header = f'Till task {task_id+1}'
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits, feats, frozen_feats, sim, tasks = model(input, eval=True)

        # here is the trick to mask out classes of non-current tasks
        logits = utils.mask_logits(logits, class_mask, args.num_classes, task_id=list(range(task_id+1)), 
                                fill_value=0)
        target = utils.mask_logits(target, class_mask, args.num_classes, task_id=list(range(task_id+1)),
                                fill_value=0)
        
        loss = criterion(logits / args.temperature, target.float())
        metric_logger.meters['Loss'].update(loss.item(), n=input.shape[0])

        predictions.append(logits)
        targets.append(target)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)
    #! ----------------------------------
    
    #$ ------- Calculate metrics ----------
    mAP = utils.mean_average_precision(predictions, targets)
    mAP_vector[task_id] = mAP.item()
    clean_predictions = utils.remove_logits(predictions, class_mask, list(range(task_id+1)))
    clean_targets = utils.remove_logits(targets, class_mask, list(range(task_id+1)))
    of1 = utils.f1_score_overall(clean_predictions, clean_targets)
    cf1, class_wise_cf1 = utils.f1_score_per_class(clean_predictions, clean_targets)
    #$ ------------------------------------

    result_str = f"[Average performances till task {task_id+1}]"
    result_str += f"\tmAP: {mAP.item():.4f}\tamAP: {mAP_vector[:task_id+1].mean():.4f}\toF1: {of1.item():.4f}\tcF1: {cf1.item():.4f}"
    result_str += f"\tLoss: {metric_logger.meters['Loss'].global_avg:.4f}"
    print(result_str)

    if val_dataset is not None:
        _update_multilabel_detail_report(
            args=args,
            class_mask=class_mask,
            val_dataset=val_dataset,
            task_id=task_id,
            predictions=predictions,
            targets=targets,
            loss=metric_logger.meters['Loss'].global_avg,
            mAP=mAP,
            of1=of1,
            cf1=cf1,
        )

    wandb_dict = {}
    if run is not None:
        wandb_dict['test/mAP'] = mAP.item()
        wandb_dict['test/oF1'] = of1.item()
        wandb_dict['test/cF1'] = cf1.item()
        wandb_dict['test/loss'] = metric_logger.meters['Loss'].global_avg

    return wandb_dict


def train_and_evaluate(model: nn.Module, model_without_ddp: nn.Module, 
                       criterion, data_loader: Iterable, device: torch.device, class_mask=None, 
                       args=None):
    
    if args.wandb and args.rank == 0:
        wandb_name = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}-{args.name}'
        run = wandb.init(
            project=args.project,
            entity='user',
            name=wandb_name,
            notes=args.notes,
            config=args,
        )
    else:
        run = None

    # create matrix to save end-of-task accuracies
    if type(criterion) == torch.nn.CrossEntropyLoss:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    else:
        mAP_vector = np.zeros((args.num_tasks, 1))

    for task_id in range(args.num_tasks):
        # transfer parameters to new task
        if task_id > 0:
            if args.distributed:
                model.module.next_task()
            else:
                model.next_task()

        # Create new optimizer for each task to clear optimizer status
        optimizer = utils.get_optimizer(args, model_without_ddp)
        lr_scheduler = None
        if args.sched == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        train_dataloader = data_loader[task_id]['train']
        
        wandb_dict = {}
        for epoch in range(args.epochs):
            stats = train_one_epoch(model=model,criterion=criterion, data_loader=train_dataloader,
                                    optimizer=optimizer, device=device, epoch=epoch,
                                    set_training_mode=True, task_id=task_id, class_mask=class_mask, 
                                    run=run, args=args,)                
            if lr_scheduler:
                lr_scheduler.step()
        wandb_dict.update(stats)

        if type(criterion) == torch.nn.CrossEntropyLoss:
            stats = evaluate_till_now(model=model, criterion=criterion,data_loader=data_loader,
                                    device=device, task_id=task_id, acc_matrix=acc_matrix, 
                                    class_mask=class_mask, run=run, args=args)

        else:
            _, val_dataset = datasets.get_dataset(args.dataset.replace('Split-', ''), 
                                                  datasets.build_transform(is_train=True, args=args),
                                                  datasets.build_transform(is_train=False, args=args),
                                                  args=args)
            seen_classes = [c for m in class_mask[:task_id+1] for c in m]
            val_seen_indices = []
            for k in range(len(val_dataset.targets)):
                if set(val_dataset.targets[k]).intersection(set(seen_classes)) != set():
                    val_seen_indices.append(k)
                    
            val_seen_dataset = torch.utils.data.Subset(val_dataset, val_seen_indices)
            val_seen_dataloader = torch.utils.data.DataLoader(val_seen_dataset, batch_size=24, 
                                                              shuffle=False, num_workers=args.num_workers, 
                                                              pin_memory=True)
            stats = evaluate_till_now_multi(model=model, criterion=criterion, data_loader=val_seen_dataloader,
                                            device=device, task_id=task_id, mAP_vector=mAP_vector, 
                                            class_mask=class_mask, run=run, args=args, val_dataset=val_dataset)
            
    
        wandb_dict.update(stats)
        if run is not None:
            run.log(wandb_dict)
    
