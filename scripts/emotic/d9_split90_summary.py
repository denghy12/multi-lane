"""Freeze all four-way Validation comparisons before opening the three Test reports."""
import argparse
from pathlib import Path

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified

STRATEGIES = ('hash2', 'meta', 'online')
COLUMNS = (('full', 'all'), ('person', 'all'), ('face', 'reliable'), ('fixed', 'all'))


def summarize(root, split):
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
        reference = {m: data[m][g]['reference'] for m,g in COLUMNS}
        if 'historical' in table and table['historical'] != reference:
            raise ValueError('Splits used different historical controls')
        table['historical'] = reference
        table[name] = {m: data[m][g]['candidate'] for m,g in COLUMNS}
    summary_file = root/f'{split}_summary.json'
    write_verified(summary_file, dict(split=split, table=table, comparison_sha256=digests))
    if split == 'val':
        write_verified(seal, dict(summary_sha256=sha256_file(summary_file), comparison_sha256=digests))
    print(split, table, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', choices=('val','test'), required=True)
    args = parser.parse_args()
    summarize(args.root, args.split)
