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
                            choices=['mean', 'max', 'cls'], default='mean',
                            help='How to aggregate token-wise DDP similarities')
    subparsers.add_argument('--ddp_class_chunk_size', type=int, default=4,
                            help='Number of classes processed per DDP visual prompt chunk')
