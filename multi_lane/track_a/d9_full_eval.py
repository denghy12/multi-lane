"""True- or swapped-box readout of the same trained Full path."""
import argparse
from pathlib import Path

from .r1_crossvit_experiment import read, validation


def main():
    parser = argparse.ArgumentParser(__doc__)
    for flag in ('training-root', 'data-root', 'clip-checkpoint', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--d9-box-mode', choices=('true', 'swapped'), required=True)
    parser.add_argument('--score-dir', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = read(args.training_root/'config.json')
    if not config.get('d9_full_box') or (config['r1_kind'], config['r1_receiver']) != ('native', 'full'):
        parser.error('Expected a target-box native Full checkpoint')
    args.face_manifest_root = None
    validation(args)


if __name__ == '__main__':
    main()
