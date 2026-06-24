# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for MULTI_LANE
# Author: Thomas De Min thomas.demin@unitn.it
# ------------------------------------------

import sys
import argparse
import datetime
import random
import os
import json
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model

from multi_lane.configs import CONFIGS
from multi_lane.datasets import build_continual_dataloader
from multi_lane.engine import *
import multi_lane.models # used to register custom vit architecture
from multi_lane import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower()
        if value in ('true', '1', 'yes', 'y'):
            return True
        if value in ('false', '0', 'no', 'n'):
            return False
    raise ValueError(f'Invalid boolean value: {value}')

def _dataset_class_names(dataset):
    seen = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        classes = getattr(current, 'classes', None)
        category_names = getattr(current, 'category_names', None)
        category2name = getattr(current, 'category2name', None)

        if isinstance(category_names, dict) and len(category_names) > 0:
            values = [category_names[i] for i in sorted(category_names.keys())]
            if all(isinstance(v, str) for v in values):
                return [str(v) for v in values]

        if isinstance(category2name, dict) and len(category2name) > 0:
            values = [category2name[i] for i in sorted(category2name.keys())]
            if all(isinstance(v, str) for v in values):
                return [str(v) for v in values]

        if classes is not None:
            return [str(c) for c in classes]

        current = getattr(current, 'dataset', None)

    return None

def _class_names_from_loaders(data_loader):
    for task_loaders in data_loader:
        for split in ('val', 'train'):
            class_names = _dataset_class_names(task_loaders[split].dataset)
            if class_names is not None:
                return class_names
    return None

def _log_class_order(args, class_mask, class_names):
    if class_mask is None or class_names is None:
        return

    order = []
    for task_id, class_ids in enumerate(class_mask):
        task_classes = [
            {
                'class_id': int(class_id),
                'class_name': str(class_names[int(class_id)]),
            }
            for class_id in class_ids
        ]
        order.append({'task': int(task_id), 'classes': task_classes})
        joined = ', '.join(f"{item['class_id']}:{item['class_name']}" for item in task_classes)
        print(f"[Class order] task {task_id}: {joined}")

    if not utils.is_main_process() or not getattr(args, 'output_dir', None):
        return
    run_name = args.name or args.dataset.replace('Split-', '').lower()
    detail_dir = os.path.join(args.output_dir, 'detail')
    os.makedirs(detail_dir, exist_ok=True)
    path = os.path.join(detail_dir, f'{run_name}_class_order.json')
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(order, fp, ensure_ascii=False, indent=2)
    print(f"Saved class order to {path}")

def _resolve_eval_checkpoint(args):
    candidates = []
    for path in (getattr(args, 'eval_checkpoint', None), getattr(args, 'eval_dir', None)):
        if not path:
            continue
        if os.path.isdir(path):
            candidates.append(os.path.join(path, 'checkpoints.pth'))
        else:
            candidates.append(path)

    if getattr(args, 'output_dir', None):
        candidates.append(os.path.join(args.output_dir, 'checkpoints.pth'))

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        'No evaluation checkpoint found. Pass --eval_checkpoint or point --eval_dir/output_dir '
        'to a directory containing checkpoints.pth.'
    )

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # train_dataloader, val_dataloader = tmp_dl(args)
    data_loader, class_mask = build_continual_dataloader(args)

    model_name = args.backbone if getattr(args, 'backbone', None) else args.model
    model = create_model(
        model_name,
        pretrained=as_bool(args.pretrained),
        num_classes=args.num_classes,
        args=args,
    )
    model.init(args)
    model.to(device)
    model.class_mask = class_mask #! TMP
    class_names = _class_names_from_loaders(data_loader)
    if class_names is not None and hasattr(model, 'set_class_names'):
        model.set_class_names(class_names)
    _log_class_order(args, class_mask, class_names)
        
    # freeze everything except head and layernorm
    learnable_params = []
    for n, p in model.named_parameters():
        if utils.is_trainable(args, n):
            p.requires_grad = True
            learnable_params.append((n, p))
        else:
            p.requires_grad = False
        
    n_parameters = sum(p.numel() for _, p in learnable_params)
    print(f'Name: {args.name}')
    print(f'Description: {args.notes}')
    print(f"Learnable Parameters {[n for n, _ in learnable_params]}")
    print('Number of params:', n_parameters)
    print(args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0 * args.accumulate_grad_batches
    if args.head_mode in ('clip_ddp', 'ddp') \
    and getattr(args, 'ddp_optimizer_lr', None) is not None:
        args.lr = float(args.ddp_optimizer_lr)
    print(f'Effective optimizer lr: {args.lr}')

    if args.opt == 'sgd':
        args.opt_betas = None

    if 'COCO' in args.dataset or 'VOC' in args.dataset or 'EMOTIC' in args.dataset:
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    if args.eval:
        checkpoint_path = _resolve_eval_checkpoint(args)
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(state_dict, strict=True)
        evaluate_checkpoint(model=model, criterion=criterion, data_loader=data_loader,
                            device=device, class_mask=class_mask, args=args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    train_and_evaluate(model=model, 
                       model_without_ddp=model_without_ddp, 
                       criterion=criterion, 
                       data_loader=data_loader, 
                       device=device,
                       class_mask=class_mask, 
                       args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    if args.store_model:
        output_path = os.path.join(args.output_dir, 'checkpoints.pth')
        print(f"Saving trained model to {output_path}")
        state_dict = {
            'args': args,
            'model': model_without_ddp.state_dict(),
        }

        torch.save(state_dict, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    
    subparser = parser.add_subparsers(dest='subparser_name')
    config_parser = subparser.add_parser(config)
    
    get_args_parser = CONFIGS[config]
    get_args_parser(config_parser)

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    
    sys.exit(0)
