import argparse

def restricted_float(x, range=(0.0, 1.0)):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < range[0] or x > range[1]:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]" % (x,))
    return x

def non_negative_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    
    if x < 0:
        raise argparse.ArgumentTypeError("%r is negative" % (x,))
    return x

def str2bool(x):
    if isinstance(x, bool):
        return x
    value = str(x).lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError("%r is not a valid boolean value" % (x,))

def add_clip_ddp_args(subparsers: argparse.ArgumentParser):
    subparsers.add_argument('--ddp_prompt_length', type=int, default=16,
                            help='DDP class-specific prompt token length')
    subparsers.add_argument('--ddp_prompt_layers', type=int, default=5,
                            help='Number of final CLIP visual layers using DDP interlayer prompts')
    subparsers.add_argument('--ddp_pcd', type=str2bool, default=True,
                            help='Use DDP Progressive Confidence Decoupling at evaluation time')
    subparsers.add_argument('--ddp_tau_max', type=float, default=3.0,
                            help='Maximum PCD temperature for the final task')
    subparsers.add_argument('--ddp_gamma', type=float, default=0.7,
                            help='PCD curriculum exponent')
    subparsers.add_argument('--ddp_similarity_aggregation', type=str,
                            choices=[
                                'mean', 'max', 'cls', 'pooled_cls',
                                'patch_mean', 'patch_max', 'cls_plus_patch_max', 'topk_mean',
                                'paper_attention',
                            ],
                            default='mean',
                            help='How to turn DDP visual evidence into one class score')
    subparsers.add_argument('--ddp_similarity_topk', type=int, default=5,
                            help='Number of patch tokens averaged when ddp_similarity_aggregation=topk_mean')
    subparsers.add_argument('--ddp_paper_attention_scale', type=float, default=20.0,
                            help='Official DDP positive-token attention scale before token softmax')
    subparsers.add_argument('--ddp_prompt_norm_mode', type=str,
                            choices=['legacy', 'prompted'], default='legacy',
                            help='legacy normalizes only image tokens before prompt attention; prompted normalizes [prompt; image] jointly')
    subparsers.add_argument('--ddp_logit_scale_mode', type=str,
                            choices=['clip', 'none', 'paper'], default='clip',
                            help='Legacy shared train/eval scale; paper uses the official fixed DDP scale')
    subparsers.add_argument('--ddp_train_logit_scale_mode', type=str,
                            choices=['inherit', 'clip', 'none', 'paper'], default='inherit',
                            help='Training-only DDP margin scale; inherit uses ddp_logit_scale_mode')
    subparsers.add_argument('--ddp_eval_logit_scale_mode', type=str,
                            choices=['inherit', 'clip', 'none', 'paper'], default='inherit',
                            help='Evaluation-only DDP margin scale; inherit uses ddp_logit_scale_mode')
    subparsers.add_argument('--ddp_paper_logit_scale', type=float, default=100.0,
                            help='Official DDP final pair-logit scale (20 token scale times 5)')
    subparsers.add_argument('--ddp_loss_mode', type=str,
                            choices=['bce_mean', 'paper_sum'], default='bce_mean',
                            help='Current mean BCE or official DDP summed binary-softmax BCE')
    subparsers.add_argument('--ddp_loss_weight', type=float, default=0.03,
                            help='Weight applied to paper_sum DDP loss')
    subparsers.add_argument('--ddp_text_init', type=str,
                            choices=['random', 'same', 'semantic'], default='random',
                            help='DDP text prompt initialization: random keeps the old behavior; same ties pos/neg starts; semantic seeds pos/neg prompts from template words')
    subparsers.add_argument('--ddp_positive_text_template', type=str,
                            default='a photo containing a {}.',
                            help='Positive DDP text template used to seed semantic text prompts')
    subparsers.add_argument('--ddp_negative_text_template', type=str,
                            default='a photo without a {}.',
                            help='Negative DDP text template used to seed semantic text prompts')
    subparsers.add_argument('--ddp_train_text_prompts', type=str2bool, default=True,
                            help='Allow gradients/optimizer updates for DDP text prompts')
    subparsers.add_argument('--ddp_train_visual_prompts', type=str2bool, default=True,
                            help='Allow gradients/optimizer updates for DDP visual prompts')
    subparsers.add_argument('--ddp_prompt_polarity', type=str,
                            choices=['both', 'positive', 'negative'], default='both',
                            help='Which positive/negative prompt branch is trainable; both branches are still used in forward')
    subparsers.add_argument('--ddp_class_chunk_size', type=int, default=4,
                            help='Number of classes processed per DDP visual prompt chunk')
    subparsers.add_argument('--ddp_optimizer_scope', type=str,
                            choices=['per_task', 'continual'], default='per_task',
                            help='Recreate prompt optimizer each task or keep one optimizer across tasks')
    subparsers.add_argument('--ddp_clear_frozen_optimizer_state', type=str2bool, default=True,
                            help='Clear Adam state for non-current prompts so frozen knowledge anchors cannot drift')
    subparsers.add_argument('--ddp_optimizer_lr', type=float, default=None,
                            help='Unscaled DDP optimizer LR override; official supplemental code uses 5.9e-3')
    subparsers.add_argument('--ddp_scheduler_mode', type=str,
                            choices=['legacy_cosine', 'paper_multistep', 'none'],
                            default='legacy_cosine',
                            help='DDP scheduler implementation; paper_multistep follows the supplemental code')
    subparsers.add_argument('--ddp_scheduler_milestones', type=int, nargs='+',
                            default=[0, 20],
                            help='Global epoch milestones for DDP paper_multistep scheduler')
    subparsers.add_argument('--ddp_scheduler_gamma', type=float, default=0.1,
                            help='Decay factor for DDP paper_multistep scheduler')
    subparsers.add_argument('--ddp_train_transform', type=str,
                            choices=['legacy', 'paper'], default='legacy',
                            help='Legacy MULTI-LANE crop/flip or official DDP resize/cutout/RandAugment')
    subparsers.add_argument('--ddp_diagnostics', type=str2bool, default=True,
                            help='Print and save DDP evaluation score diagnostics')
    subparsers.add_argument('--ddp_diagnostic_thresholds', type=float, nargs='+',
                            default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            help='Thresholds used for DDP diagnostic precision/recall/F1 curves')
    subparsers.add_argument('--ddp_eval_score_mode', type=str,
                            choices=['logits', 'probability'], default='logits',
                            help='DDP evaluation score semantics: logits uses sigmoid(scaled logits), probability uses strict DDP p_pos')
    subparsers.add_argument('--ddp_eval_threshold', type=float, default=0.8,
                            help='Main DDP F1 threshold for the selected ddp_eval_score_mode')
    subparsers.add_argument('--ddp_score_dump', type=str2bool, default=False,
                            help='Dump per-sample DDP s_pos/s_neg/margin/probability/scaled-logit arrays during evaluation')
    subparsers.add_argument('--clip_normalize_input', type=str2bool, default=False,
                            help='Use CLIP mean/std image normalization instead of no normalization or ImageNet normalization')
    subparsers.add_argument('--eval_checkpoint', type=str, default=None,
                            help='Checkpoint path for --eval; falls back to --eval_dir or output_dir/checkpoints.pth')
