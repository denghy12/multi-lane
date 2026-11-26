"""Extract Task10 Summarized Patches statistics from frozen, verified D9 branches."""
import hashlib
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import VIEWS
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import set_seed
from multi_lane.track_a.summary_logit_fusion import image_descriptors, summary_statistics
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_fill_image_source import native_scores, assert_train_roles, source_mode
from scripts.emotic.d9_image_gate_extract import OriginalImages, pack_images, keep_images
from scripts.emotic.evaluate_d9_relation import write_json, stage_loader
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


SCHEMA = 'task10_summary_pairs_v1'


@torch.inference_mode()
def verify_capture(model, source, task, workers):
    """Keep archived batch boundaries: AMP changes with batching in this project."""
    path = source/'evaluation'/'val'/f'task{task}.npz'
    meta = read(path.with_suffix('.json'))
    require_digest(path, meta['scores_sha256'])
    validation = build_joint_source(input_paths(), 'val')
    loader = stage_loader(validation, task, False, meta['protocol']['batch_size'], workers)
    images, _, ids = next(iter(loader))
    inputs = move_model_inputs(images, torch.device('cuda'))
    capture = {}
    with torch.autocast('cuda'):
        plain = model(inputs, all_seen=True)
        hooked = model(inputs, all_seen=True, summary_tokens=capture)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(np.asarray(ids), archive['sample_ids'][:len(ids)])
        for index, view in enumerate(VIEWS):
            torch.testing.assert_close(plain['branches'][view], hooked['branches'][view], rtol=0, atol=0)
            np.testing.assert_array_equal(hooked['branches'][view].float().cpu().numpy(),
                                          archive['branch_logits'][:len(ids), :, index])
    summary_statistics(capture, inputs['evidence_face_reliable'].reshape(-1).bool(), task)
    return dict(people=len(ids), native_batch_size=meta['protocol']['batch_size'],
                capture_on_off_max_difference=0., native_logits_max_difference=0.)


@torch.inference_mode()
def calculate(model, task, loader, batch_size):
    started = time.monotonic()
    names, mean_rows, relation_rows, counts, reliable_counts = [], [], [], [], []
    for groups in loader:
        ids, full, person, face, flags = zip(*groups)
        sizes = [len(m) for m in flags]
        masks = torch.cat(flags)
        owner = torch.repeat_interleave(torch.arange(len(ids)), torch.tensor(sizes))
        full, person, face = torch.stack(full), torch.cat(person), torch.cat(face)
        parts = []
        for start in range(0, len(masks), batch_size):
            rows = slice(start, start+batch_size)
            pixels = dict(full=full[owner[rows]].cuda(), person=person[rows].cuda(), face=face[rows].cuda())
            inputs = {**pixels, **{'evidence_'+k: v for k, v in pixels.items()},
                      'evidence_face_reliable': masks[rows].cuda()}
            capture = {}
            with torch.autocast('cuda'):
                model(inputs, all_seen=True, summary_tokens=capture)
            parts.append(summary_statistics(capture, inputs['evidence_face_reliable'], task).cpu())
        descriptors = image_descriptors(torch.cat(parts), masks, sizes)
        names.extend(ids)
        mean_rows.append(descriptors['summary_mean'].numpy())
        relation_rows.append(descriptors['summary_correspondence'].numpy())
        counts.extend(sizes)
        reliable_counts.extend(int(m.sum()) for m in flags)
        if len(names) == len(loader.dataset) or len(counts) % 100 < len(ids):
            elapsed = time.monotonic()-started
            print(f'extract task{task}: {len(names)}/{len(loader.dataset)} images, '
                  f'{sum(counts)} people, {elapsed:.1f}s', flush=True)
    if names != loader.dataset.names:
        raise ValueError('Image roster order changed')
    return dict(image_ids=np.asarray(names), summary_mean=np.concatenate(mean_rows),
                summary_correspondence=np.concatenate(relation_rows), counts=np.asarray(counts),
                reliable_counts=np.asarray(reliable_counts))


def extract(source, out, split, task, workers=2, batch_size=64, smoke=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    inputs = input_paths()
    native, _, digest, checkpoint = native_scores(source, out, split, task)
    spec = dict(schema=SCHEMA, layer_index=9, temperature=.1, task=task, split=split, smoke=smoke,
                source_mode=source_mode(source), native_sha256=digest, checkpoint_sha256=checkpoint,
                batch_size=batch_size, config_sha256=sha256_file(source/'config.json'),
                roles_sha256=sha256_file(source/'train_image_roles.json'),
                mat_sha256=sha256_file(inputs.data_root/'CVPR17_Annotations.mat'),
                manifest_sha256=sha256_file(inputs.face_manifest_root/'manifests'/f'{split}.jsonl'))
    dest = out/split/f'task{task}.npz'
    meta_path = dest.with_suffix('.json')
    if meta_path.exists():
        previous = read(meta_path)
        if previous['protocol'] != spec:
            raise ValueError(f'Existing feature protocol differs: {dest}')
        require_digest(dest, previous['archive_sha256'])
        return
    if dest.exists():
        raise ValueError(f'Unsealed feature archive exists; use a fresh output directory: {dest}')
    with np.load(native, allow_pickle=False) as z:
        dataset = OriginalImages(inputs, split, z['sample_ids'], z['reliable'], smoke)
    if split == 'train':
        assert_train_roles(source, dataset.names)
    loader = DataLoader(dataset, batch_sampler=pack_images(map(len, dataset.rows), batch_size),
                        num_workers=workers, collate_fn=keep_images, worker_init_fn=single_thread_worker,
                        **({'prefetch_factor': 1} if workers else {}))
    model = load_evaluation_model(source, task).requires_grad_(False).eval()
    replay = verify_capture(model, source, task, workers)
    fields = calculate(model, task, loader, batch_size)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **fields)
    tmp.replace(dest)
    write_json(meta_path, dict(protocol=spec, archive_sha256=sha256_file(dest), replay=replay,
               images=len(dataset.names), roster_people=len(dataset.sample_ids),
               scored_people_in_selected_images=dataset.scored_people,
               raw_split_people=dataset.raw_people, missing_face_records_in_split=dataset.missing_manifest,
               roster_sha256=hashlib.sha256('\n'.join(dataset.sample_ids).encode()).hexdigest()))
    del model
    torch.cuda.empty_cache()
    print(f'{dest}: {len(dataset.names)} images; exact native/capture replay passed', flush=True)
