"""One Frozen9 descriptor and availability decision per original EMOTIC image."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker, source_runs, read
from multi_lane.track_a.runner import build_transforms, set_seed
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths
from scripts.emotic.d9_fill_image_source import native_scores, assert_train_roles, source_mode


PROJECTION_SEED = 20261120
LAYER = 8


def group_key(person_id, split):
    key, sep, person = str(person_id).rpartition('#person=')
    if not sep or not key.startswith(split + ':') or not person.isdecimal():
        raise ValueError(f'Invalid {split} person ID: {person_id}')
    return key


def select_images(rosters, required, smoke):
    """Keep whole images; smoke covers the face availability patterns present."""
    names = sorted(required)
    if not names or not set(names).issubset(rosters):
        raise ValueError('Missing annotation roster for a requested original image')
    if smoke:
        by_kind = defaultdict(list)
        for name in names:
            masks = [row[3] for row in rosters[name]]
            by_kind[(all(masks), any(masks))].append(name)
        names = sorted({n for group in by_kind.values() for n in group[:12]})
    return names


class OriginalImages(EMOTIC):
    """Read annotations as a roster of boxes, without consulting emotion labels."""
    def __init__(self, inputs, split, scored_ids, scored_reliability, smoke):
        self.split = split
        self.image_root = inputs.data_root / 'cvpr_emotic'
        mat = loadmat(inputs.data_root / 'CVPR17_Annotations.mat', squeeze_me=True,
                      struct_as_record=False, variable_names=[split])
        records = {}
        manifest = inputs.face_manifest_root / 'manifests' / f'{split}.jsonl'
        for line in manifest.read_text().splitlines():
            item = json.loads(line)
            person = item['sample_id']
            if person in records:
                raise ValueError('Duplicate face manifest person')
            records[person] = item
        rosters = {}
        raw_mask = {}
        for item in self._as_list(mat[split]):
            people = []
            for index, annotation in enumerate(self._as_list(item.person)):
                person = self._sample_id(split, item.folder, item.filename, index)
                image = group_key(person, split)
                record = records.get(person, {})
                valid = bool(record.get('valid_face') and not record.get('ambiguous_match'))
                reliable = (valid and float(record.get('face_short_side', 0.)) >= 24.
                            and float(record.get('face_detection_score', 0.)) >= .6)
                if person in raw_mask:
                    raise ValueError(f'Duplicate annotated target {person}')
                raw_mask[person] = reliable
                people.append((person, annotation.body_bbox, valid, reliable))
            if people:
                rosters[image] = people
        if not records.keys() <= raw_mask.keys():
            raise ValueError('Face manifest contains unknown target boxes')
        self.missing_manifest = sorted(raw_mask.keys() - records.keys())
        self.raw_people = len(raw_mask)
        scored_mask = dict(zip(scored_ids.tolist(), scored_reliability.tolist()))
        if (len(scored_mask) != len(scored_ids) or not scored_mask.keys() <= records.keys()
                or any(raw_mask[p] != r for p, r in scored_mask.items())):
            raise ValueError('Historical scored-person availability or identity differs from raw roster')
        self.names = select_images(rosters, {group_key(p, split) for p in scored_mask}, smoke)
        selected = set(self.names)
        self.scored_people = sum(group_key(p, split) in selected for p in scored_mask)
        self.rows, self.body_bboxes, self.face_records = [], [], []
        self.sample_ids, self.face_training_valid, self.face_reliable = [], [], []
        for image in self.names:
            start = len(self.sample_ids)
            for person, body, valid, reliable in rosters[image]:
                self.sample_ids.append(person)
                self.body_bboxes.append(body)
                self.face_records.append(records.get(person, {}))
                self.face_training_valid.append(valid)
                self.face_reliable.append(reliable)
            self.rows.append(range(start, len(self.sample_ids)))
        self.transforms = {}
        for view, path in source_runs(inputs.reference_root).items():
            cfg = read(path / 'config.json')
            if view == 'person':
                self.person_crop_margin = cfg['person_crop_margin']
            _, self.transforms[view] = build_transforms(
                cfg['input_normalization'], cfg['train_crop_scale'], input_mode=cfg['input_mode'],
                person_transform_mode=cfg['person_transform_mode'],
                person_color_jitter_strength=cfg['person_color_jitter_strength'],
                person_color_jitter_probability=cfg['person_color_jitter_probability'])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        rows = self.rows[idx]
        relative = self.names[idx].partition(':')[2]
        with Image.open(self.image_root / relative) as original:
            image = original.convert('RGB')
        return (self.names[idx], self.transforms['full'](image),
                torch.stack([self.transforms['person'](self._crop_person(image, self.body_bboxes[i])) for i in rows]),
                torch.stack([self.transforms['face'](self._crop_face(image, i)) for i in rows]),
                torch.tensor([self.face_reliable[i] for i in rows], dtype=torch.bool))


def pack_images(sizes, budget):
    """Never separate an original roster. Oversized groups are chunked on the GPU."""
    batches, current, count = [], [], 0
    for i, size in enumerate(sizes):
        if current and count + size > budget:
            batches.append(current)
            current, count = [], 0
        current.append(i)
        count += size
    if current:
        batches.append(current)
    return batches


def keep_images(items):
    return items


def prefix(branch, images):
    tokens = branch._visual_tokens(images)
    for block in branch.visual_encoder.transformer.resblocks[:LAYER+1]:
        tokens = block(tokens.permute(1, 0, 2)).permute(1, 0, 2)
    return tokens


def read_relations(model, task, full, person, face, reliable):
    active = torch.ones_like(reliable)
    person_agent = model.branches['person'].cross_agent(task, LAYER)
    face_agent = model.branches['face'].cross_agent(task, LAYER)
    full_patches = full[:, 1:]
    _, p_rel = person_agent(person, [full_patches, face[:, 1:]], [active, reliable],
                            return_full_relation=True, detach_relation=True)
    _, f_rel = face_agent(face, [full_patches, person[:, 1:]], [active, active],
                          return_full_relation=True, detach_relation=True)
    return torch.cat((p_rel, f_rel), dim=-1)


def verify_relations(model, task, full, person, face, reliable, actual):
    for index, (view, pixels) in enumerate((('person', person), ('face', face))):
        capture = {}
        branch = model.branches[view]
        inputs = dict(full=pixels, evidence_full=full, evidence_person=person, evidence_face=face,
                      evidence_face_reliable=reliable)
        branch.encode_lanes(inputs, False, full_relations=capture, detach_relation=True)
        torch.testing.assert_close(actual[:, 4*index:4*index+4], capture[task], rtol=.005, atol=.002)


def summarize_people(relation, flags):
    p = relation[:, :4].mean(0)
    f = relation[flags, 4:].mean(0) if flags.any() else np.zeros(4, dtype=relation.dtype)
    return np.concatenate((p, f))


@torch.inference_mode()
def calculate(model, task, loader, batch_size, smoke):
    donor = model.branches['person']
    if (model.branches['face'].visual_encoder is not donor.visual_encoder
            or model.branches['full'].visual_encoder is not donor.visual_encoder):
        raise ValueError('Source model does not use identical frozen CLIP backbones')
    for view in ('face','person'):
        branch = model.branches[view]
        if branch.frozen_cross_active_layers != (LAYER,) or branch.frozen_cross_evidence != 'all':
            raise ValueError('Expected Frozen9 two-sender receivers')
    # Fixed, data-independent projection, identical across tasks and splits.
    matrix = torch.randn(donor.width, 8, generator=torch.Generator().manual_seed(PROJECTION_SEED))
    matrix = torch.nn.functional.normalize(matrix, dim=0).cuda()
    names, full_rows, rel_rows, counts, r_counts = [], [], [], [], []
    for batch_index, groups in enumerate(loader):
        ids, full_images, person_images, face_images, flags = zip(*groups)
        masks = torch.cat(flags)
        sizes = [len(f) for f in flags]
        owners = torch.repeat_interleave(torch.arange(len(ids)), torch.tensor(sizes))
        full_pixels = torch.stack(full_images).cuda()
        with torch.autocast('cuda'):
            full = prefix(donor, full_pixels)
        full_rows.append((full[:, 1:].float().mean(dim=1) @ matrix).cpu().numpy())
        persons, faces = torch.cat(person_images), torch.cat(face_images)
        parts = []
        for start in range(0, len(masks), batch_size):
            rows = slice(start, start + batch_size)
            p = persons[rows].cuda()
            f = faces[rows].cuda()
            mask = masks[rows].cuda()
            sel = owners[rows].cuda()
            with torch.autocast('cuda'):
                actual = read_relations(model, task, full[sel], prefix(donor, p), prefix(donor, f), mask)
                if smoke and batch_index == 0 and start == 0:
                    verify_relations(model, task, full_pixels[sel], p, f, mask, actual)
            parts.append(actual.cpu().numpy())
        relations = np.concatenate(parts, axis=0)
        for group, mask in zip(np.split(relations, np.cumsum(sizes)[:-1]), flags):
            rel_rows.append(summarize_people(group, mask.numpy()))
        names.extend(ids)
        counts.extend(sizes)
        r_counts.extend(int(m.sum()) for m in flags)
    counts, r_counts = np.asarray(counts), np.asarray(r_counts)
    fields = dict(image_ids=np.asarray(names), full=np.concatenate(full_rows), relation=np.stack(rel_rows),
                  counts=counts, reliable_counts=r_counts, all_reliable=(r_counts == counts))
    if (names != loader.dataset.names or np.any(counts < 1) or not np.isfinite(fields['full']).all()
            or not np.isfinite(fields['relation']).all()):
        raise ValueError('Missing or invalid image-level features')
    return fields


def extract(source, out, split, task, workers, batch_size, smoke):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    set_seed(1, True)
    torch.set_num_threads(2)
    inputs = input_paths()
    mode = source_mode(source)
    native, meta, score_digest, checkpoint_digest = native_scores(source, out, split, task)
    manifest = inputs.face_manifest_root/'manifests'/f'{split}.jsonl'
    spec = dict(schema=1, task=task, split=split, smoke=smoke, projection_seed=PROJECTION_SEED,
                native_sha256=score_digest, checkpoint_sha256=checkpoint_digest,
                batch_size=batch_size, mat_sha256=sha256_file(inputs.data_root/'CVPR17_Annotations.mat'),
                manifest_sha256=sha256_file(manifest), config_sha256=sha256_file(source/'config.json'))
    dest = out/split/f'task{task}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = dest.with_suffix('.json')
    if meta_path.exists():
        previous = read(meta_path)
        if previous['protocol'] != spec or sha256_file(dest) != previous['archive_sha256']:
            raise ValueError(f'Existing feature provenance mismatch: {dest}')
        print(f'{dest} verified', flush=True)
        return
    with np.load(native, allow_pickle=False) as z:
        dataset = OriginalImages(inputs, split, z['sample_ids'], z['reliable'], smoke)
    if split == 'train':
        assert_train_roles(source, dataset.names)
    loader = DataLoader(dataset, batch_sampler=pack_images(map(len, dataset.rows), batch_size), num_workers=workers,
                        collate_fn=keep_images, worker_init_fn=single_thread_worker,
                        **({'prefetch_factor': 1} if workers else {}))
    model = load_evaluation_model(source, task).requires_grad_(False).eval()
    fields = calculate(model, task, loader, batch_size, smoke)
    tmp = dest.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **fields)
    tmp.replace(dest)
    write_json(meta_path, dict(protocol=spec, archive_sha256=sha256_file(dest),
               images=len(dataset.names), roster_people=len(dataset.sample_ids),
               raw_split_people=dataset.raw_people, missing_face_records_in_split=dataset.missing_manifest,
               scored_people_in_selected_images=dataset.scored_people,
               roster_sha256=hashlib.sha256('\n'.join(dataset.sample_ids).encode()).hexdigest(),
               all_reliable_images=int(fields['all_reliable'].sum())))
    print(f'{dest}: {len(dataset.names)} original images, {len(dataset.sample_ids)} complete roster people', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('source','out'):
        parser.add_argument('--'+k, type=Path, required=True)
    parser.add_argument('--split', choices=('train','val','test'), required=True)
    parser.add_argument('--task', type=int, choices=range(8), required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size < 1:
        parser.error('Require nonnegative workers and positive batch size')
    extract(**vars(args))
