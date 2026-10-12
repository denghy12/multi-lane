"""Locked full-cohort validation, then test, of the whole-frozen image ParaX."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .compare_whole_parax_freeze import audit_control
from .evaluate_joint26_checkpoint import build_model, make_source
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .runner import CLASS_ORDER, TASK_SIZES, LabelView, compact_model_state_dict, dataset_view, git_metadata, seen_indices, set_seed, tensor_state_digest, validate_classes
from .snapshot_validation import audit_source, infer, load_scores, replay, score_metrics


def read(path):
    return json.loads(path.read_text())


def seen_payload(full, task):
    stop = sum(TASK_SIZES[:task+1])
    require(full['full_targets'].shape == (len(full['sample_ids']), 26), 'Incomplete ground truth')
    require(full['logits'].shape[1] == stop, 'Wrong seen predictions')
    result = {k: v for k, v in full.items() if k not in ('full_targets', 'batch_lengths')}
    result.update(targets=full['full_targets'][:, :stop], class_indices=np.arange(stop))
    return result


def metrics_at_task(data, threshold, task):
    result = score_metrics(data, threshold)
    for value in result.values():
        value['task_id'] = task
    return result


def filter_seen_people(data):
    mask = data['targets'].any(axis=1)
    return {k: (v if k == 'class_indices' else v[mask]) for k, v in data.items()}


def replay_saved_cohort(model, cohort, saved_path, config, task, device):
    """Strictly replay saved predictions with their original people and batch boundaries.

    Filtering a full-cohort forward afterwards does not restore its batch layout.
    AMP kernels may round differently after unseen-only people change the batches.
    """
    expected = load_scores(saved_path)
    loader = DataLoader(dataset_view(cohort, seen_indices(task), include_sample_id=True),
                        batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['workers'])
    _, actual = infer(model, loader, device, config['amp'], task, config['task_sizes'])
    require(np.array_equal(expected['sample_ids'], actual['sample_ids'])
            and np.array_equal(expected['targets'], actual['full_targets'])
            and np.array_equal(expected['class_indices'], np.arange(sum(config['task_sizes'][:task+1]))),
            'Saved-score replay cohort/classes mismatch')
    require(np.array_equal(expected['batch_lengths'], actual['batch_lengths']),
            'Saved-score replay batch boundaries differ')
    require(expected['logits'].shape == actual['logits'].shape
            and np.isfinite(expected['logits']).all() and np.isfinite(actual['logits']).all(),
            'Invalid saved-score replay logits')
    delta = float(np.abs(expected['logits']-actual['logits']).max())
    require(delta <= .01, f'Complete saved-score replay differs with original batches: {delta}')
    return delta


def summarize_curve(tasks):
    """Only use a fixed full cohort; do not confuse added negative people with forgetting."""
    ids = sorted(map(int, tasks))
    final = tasks[str(ids[-1])]['metrics']['fused']['per_class_ap']
    old_count = sum(TASK_SIZES[:ids[-1]])
    forgetting = []
    for col in range(old_count):
        previous = [tasks[str(t)]['metrics']['fused']['per_class_ap'][col] for t in ids[:-1]
                    if col < tasks[str(t)]['metrics']['fused']['seen_classes']]
        forgetting.append(max(0., max(previous)-final[col]))
    return dict(final_mAP=tasks[str(ids[-1])]['metrics']['fused']['mAP'],
                average_mAP=float(np.mean([tasks[str(t)]['metrics']['fused']['mAP'] for t in ids])),
                forgetting=float(np.mean(forgetting)) if forgetting else 0.,
                forgetting_definition='mean max(0, best previous class AP minus final AP) over old classes, fixed full cohort')


def evaluate(source, reference_root, pilot_reference, output, clip_checkpoint, data_root,
             face_manifest_root, split='val', preflight=False, smoke_updates=0, validation_result=None):
    require(split in ('val', 'test') and not output.exists(), 'Wrong split/existing output')
    require(not (split == 'test' and (preflight or smoke_updates)), 'Test requires the complete locked run')
    config, audit = audit_control(source, reference_root/'THREE_VIEW_PARAX', smoke_updates, 8,
                                  pilot_reference if not smoke_updates else None)
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    require(not metadata['dirty'], 'Dirty evaluator')
    previous = None
    if split == 'test':
        require(validation_result is not None, 'Complete validation must precede test')
        previous = read(validation_result)
        require(previous['status'] == 'complete' and previous['split'] == 'val'
                and previous['seed'] == config['seed'] and previous['evaluator_git']['commit'] == metadata['commit'],
                'Invalid preceding validation')
    require(_sha256(clip_checkpoint) == config['clip_checkpoint_sha256'], 'CLIP file changed')
    sources = {'baseline': reference_root/'THREE_VIEW', 'shared_live': reference_root/'THREE_VIEW_PARAX', 'whole_frozen': source}
    output.mkdir(parents=True)
    results = {}
    cohort_ids = cohort_targets = cohort_face = None
    for name, root in sources.items():
        c = read(root/'config.json')
        pa = read(root/'paired_protocol_audit.json')
        require(c['seed'] == config['seed'] and c['initial_selector_prompt_classifier_sha256'] == config['initial_selector_prompt_classifier_sha256'], 'Initialization/seed mismatch')
        require(Path(c['data_root']).resolve() == data_root.resolve(), 'Dataset root changed')
        if name != 'whole_frozen':
            audit_source(root)
        files = [root/f for f in ('config.json', 'seed_summary.json', 'training_history.json', 'paired_protocol_audit.json', 'image_stream_transfer_audit.json')]
        files += [root/f'{folder}/task{t}.{suffix}' for t in range(8)
                  for folder, suffix in (('compact_checkpoints', 'pth'), ('val_scores', 'npz'))]
        hashes = {str(p): _sha256(p) for p in files}
        if previous is not None:
            require(previous['results'][name]['source_sha256'] == hashes, 'Source changed since validation')
        manifests = {str(face_manifest_root/'detector_config.json'): c['face_manifest']['artifact_sha256']['detector_config']}
        for selected_split in ('val', split):
            manifests[str(face_manifest_root/f'manifests/{selected_split}.jsonl')] = c['face_manifest']['artifact_sha256'][selected_split+'_manifest']
        require(all(_sha256(Path(p)) == value for p, value in manifests.items()), 'Face artifacts changed')
        set_seed(c['seed'], c['tf32'])
        device = torch.device('cuda')
        model = build_model(c, load_openai_clip_visual(clip_checkpoint), task_sizes=c['task_sizes']).float().to(device)
        clip_hash = tensor_state_digest(model.visual_encoder.state_dict())
        require(clip_hash == pa['frozen_visual_sha256_after'], 'Restored CLIP differs')
        validation = make_source(c, data_root, face_manifest_root, 'val')
        validate_classes(validation)
        require(len(validation) == 2397, 'Incomplete validation')
        loader = None
        rows = {}
        for task in ([0, 7] if preflight else range(8)):
            restore_compact_model_state(model, _torch_load(root/f'compact_checkpoints/task{task}.pth'), task, c['git'])
            model.requires_grad_(False).eval()
            difference = replay(model, validation, root, c, task, device)
            # Never create a test dataset before checking a restored checkpoint against validation.
            if loader is None:
                cohort = validation if split == 'val' else make_source(c, data_root, face_manifest_root, 'test')
                validate_classes(cohort)
                require(len(cohort) == (2397 if split == 'val' else 5368), 'Incomplete evaluation cohort')
                loader = DataLoader(LabelView(cohort, range(len(cohort)), range(26), include_sample_id=True),
                                    batch_size=64, num_workers=2, shuffle=False)
            before = tensor_state_digest(compact_model_state_dict(model))
            _, full = infer(model, loader, device, c['amp'], task, c['task_sizes'], 1 if preflight else None)
            require(tensor_state_digest(compact_model_state_dict(model)) == before and all(p.grad is None for p in model.parameters()), 'Inference changed weights/gradients')
            if cohort_ids is None:
                cohort_ids, cohort_targets, cohort_face = full['sample_ids'], full['full_targets'], full['face_reliable']
            require(np.array_equal(cohort_ids, full['sample_ids']) and np.array_equal(cohort_targets, full['full_targets'])
                    and np.array_equal(cohort_face, full['face_reliable']), 'Evaluation cohorts differ')
            data = seen_payload(full, task)
            destination = output/name
            destination.mkdir(exist_ok=True)
            np.savez_compressed(destination/f'task{task}.npz', **data)
            row = dict(samples=len(data['sample_ids']), validation_replay_max_logit_difference=difference)
            if preflight:
                # Cover the original failure mode, not just a first-batch smoke.
                row['saved_filtered_replay_max_logit_difference'] = replay_saved_cohort(
                    model, validation, root/f'val_scores/task{task}.npz', c, task, device)
                row['saved_score_replay_batch_policy'] = 'original_filtered_cohort_and_batch_boundaries'
            if not preflight:
                metrics = metrics_at_task(data, c['threshold'], task)
                filtered = filter_seen_people(data)
                start = sum(TASK_SIZES[:task])
                row.update(metrics=metrics, filtered_seen_people_metrics=metrics_at_task(filtered, c['threshold'], task),
                           new_class_mAP=float(np.mean(metrics['fused']['per_class_ap'][start:])),
                           old_class_mAP=float(np.mean(metrics['fused']['per_class_ap'][:start])) if start else None)
                saved_path = root/f'{split}_scores/task{task}.npz'
                if saved_path.exists():
                    saved = load_scores(saved_path)
                    require(np.array_equal(saved['sample_ids'], filtered['sample_ids']) and np.array_equal(saved['targets'], filtered['targets']), 'Saved-score cohort mismatch')
                    # Compare the saved function in exactly its original batch layout.
                    # Full-cohort scores remain independently computed, without replacement.
                    row['full_cohort_vs_saved_max_logit_difference'] = float(np.abs(saved['logits']-filtered['logits']).max())
                    row['saved_filtered_replay_max_logit_difference'] = replay_saved_cohort(
                        model, cohort, saved_path, c, task, device)
                    row['saved_score_replay_batch_policy'] = 'original_filtered_cohort_and_batch_boundaries'
            rows[str(task)] = row
            print(f'WHOLE_FROZEN_{split.upper()} method={name} task={task} samples={row["samples"]}', flush=True)
        require(tensor_state_digest(model.visual_encoder.state_dict()) == clip_hash, 'CLIP changed during inference')
        require(all(_sha256(Path(p)) == v for p, v in hashes.items()), 'Source changed during inference')
        require(all(_sha256(Path(p)) == v for p, v in manifests.items()), 'Face artifacts changed during inference')
        results[name] = dict(tasks=rows, source=str(root.resolve()), source_sha256=hashes,
                             face_manifest_sha256=manifests, source_unchanged=True, clip_unchanged=True)
        if not preflight:
            results[name]['curve_summary'] = summarize_curve(rows)
        del model
        torch.cuda.empty_cache()
    result = dict(status='preflight_passed' if preflight else 'complete', split=split, seed=config['seed'],
                  samples=len(cohort_ids), evaluator_git=metadata, training_performed=False, optimizer_updates=0,
                  test_loaded=split == 'test', test_weight_search=False, validation_preceded_test=previous is not None,
                  freeze_audit=audit, results=results)
    (output/'comparison.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    if not preflight:
        lines = [f'# Entire shared ParaX frozen after task0: seed0 full {split}', '',
                 '| Completed task | Seen classes | No ParaX | Shared-live ParaX | Whole-frozen ParaX | Frozen minus baseline |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for task in range(8):
            values = [results[n]['tasks'][str(task)]['metrics']['fused']['mAP'] for n in sources]
            lines.append(f'| {task} | {sum(TASK_SIZES[:task+1])} | '+ ' | '.join(f'{x:.4f}' for x in values+[values[2]-values[0]])+' |')
        (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('WHOLE_FROZEN_EVALUATION_COMPLETE '+result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('source', 'reference-root', 'output', 'clip-checkpoint', 'data-root', 'face-manifest-root'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('pilot-reference', 'validation-result'):
        parser.add_argument('--'+key, type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--smoke-updates', type=int, default=0)
    args = parser.parse_args()
    evaluate(**vars(args))
