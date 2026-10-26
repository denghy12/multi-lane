"""Real feature-mean forward of two complete frozen MULTI-LANE sources."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .head_anchored_eight import runtime
from .historical_b32_backbone_attribution import method_state
from .paired_transforms import move_model_inputs
from .protected_view_forward import (VIEWS, ProtectedViewFusion, install, models,
    references, verify_batch, write_scores)
from .run_historical_b32_test_attribution import read, write
from .runner import (TaskMetrics, dataset_view, seen_indices,
    set_seed, summarize_tasks, tensor_state_digest)

CONDITIONS = ('A', 'B', 'R5', 'LOGIT_MEAN')
MAX_AMP_LOGIT_DIFFERENCE = .03125


class FixedModelMeanFusion(ProtectedViewFusion):
    """Average AFTER each source's fixed view fusion, BEFORE the common head.

    No new normalization, gate, head fitting, or parameter averaging is used.
    A and B keep their own Full summaries and cross-view attention dependencies.
    """
    def __init__(self, a, b=None):
        super().__init__(a, b)
        for model in (a, b):
            if model is None:
                continue
            if (model.view_fusion != 'fixed_three_view'
                    or model.view_classifier_mode != 'shared_post_fusion'
                    or model.view_fusion_module.logit_residual):
                raise ValueError('Fixed mean requires fixed views and a shared linear head')

    def compose_mean(self, images, features):
        if self.b is None:
            raise ValueError('The fixed mean requires both complete sources')
        lanes = self.a._lane_ids(True)
        fa, wa = self.a.view_fusion_module(features['A'], lanes, images['face_reliable'])
        fb, wb = self.b.view_fusion_module(features['B'], lanes, images['face_reliable'])
        if not torch.equal(wa, wb):
            raise ValueError('Source view weights differ')
        fused = .5 * fa + .5 * fb
        views = {v: .5 * features['A'][v] + .5 * features['B'][v] for v in VIEWS}
        z, zv = self.a._lane_logits_with_views(fused, views, images, lanes)
        return {'fused': self.combine(z), **{v: self.combine(zv[v]) for v in VIEWS}}, wa

    def forward(self, images, arm='R5'):
        features, sources = self.sources(images, need_b=arm != 'A')
        if arm in ('A', 'B'):
            return sources[arm]
        if arm != 'R5':
            raise ValueError(arm)
        return self.compose_mean(images, features)[0]


def evaluate(args):
    git = runtime(); set_seed(1, True); start = time.time()
    a, b = models(args); model = None
    visual_before = tensor_state_digest(a.visual_encoder.state_dict())
    data = make_source(read(args.source_run/'config.json'), args.data_root,
                       args.face_manifest_root, args.split)
    tasks = [0, 2, 7] if args.action == 'preflight' else list(range(8))
    results = {arm: [] for arm in CONDITIONS}; audits = []
    for task in tasks:
        source = install(args, a, b, task)
        if model is None:
            model = FixedModelMeanFusion(a, b)
        protected = {k: tensor_state_digest(method_state(m)) for k, m in (('A', a), ('B', b))}
        ref = references(args, args.split, task)
        loader = DataLoader(dataset_view(data, seen_indices(task), True), batch_size=64,
            shuffle=False, num_workers=2, pin_memory=False,
            generator=torch.Generator().manual_seed(87001+task))
        rows = {arm: {v: [] for v in ('fused', *VIEWS)} for arm in CONDITIONS}
        ids=[]; labels=[]; reliability=[]; weights=[]; batches=[]
        differences = {v: 0. for v in ('fused', *VIEWS)}
        with torch.no_grad():
            for images, y, batch_ids in loader:
                reliable = images['face_reliable'].bool().cpu()
                x = move_model_inputs(images, torch.device('cuda'))
                with torch.cuda.amp.autocast():
                    features, sources = model.sources(x)
                    mean, w = model.compose_mean(x, features)
                # Replay every source output against historical complete-model predictions.
                verify_batch(ref, len(ids), batch_ids, y, reliable, sources, {})
                offline = {v: .5*sources['A'][v].float() + .5*sources['B'][v].float()
                           for v in ('fused', *VIEWS)}
                outputs = dict(sources, R5=mean, LOGIT_MEAN=offline)
                for v in differences:
                    error = (mean[v].float()-offline[v]).abs().max().item()
                    assert error <= MAX_AMP_LOGIT_DIFFERENCE, (task, v, error)
                    differences[v] = max(differences[v], error)
                for arm, values in outputs.items():
                    for v, z in values.items():
                        rows[arm][v].append(z.float().cpu())
                ids.extend(batch_ids); labels.append(y.float()); reliability.append(reliable)
                weights.append(w.float().cpu()); batches.append(len(y))
                if args.action == 'preflight':
                    break
        assert len(ids) == (64 if args.action == 'preflight' else len(ref['A']['sample_ids']))
        assert all(protected[k] == tensor_state_digest(method_state(m)) for k, m in (('A', a), ('B', b)))
        y = torch.cat(labels); reliable = torch.cat(reliability)
        for arm in CONDITIONS:
            provenance = {v: arm if arm in ('A', 'B') else '.5A+.5B' for v in VIEWS}
            value = write_scores(args.output/arm, task, ids, y, reliable,
                {v: torch.cat(z) for v, z in rows[arm].items()}, torch.cat(weights), batches,
                view_sources=provenance)
            results[arm].append(value)
            # Also verify the serialized source probabilities, including historical batch sigmoid.
            if arm in ('A', 'B'):
                with np.load(args.output/arm/'view_scores'/f'task{task}.npz') as saved:
                    for v in ('fused', *VIEWS):
                        for suffix in ('_logits', '_probabilities'):
                            assert np.array_equal(saved[v+suffix], ref[arm][v+suffix][:len(ids)])
            print('EVALUATE', args.split, 'task', task, arm, value['mAP'], flush=True)
        audit = dict(task=task, people=len(ids), sources=source, source_predictions_exact=True,
            serialized_sources_exact=True, parameters_unchanged=True,
            mean_before_head=True, no_new_normalization=True,
            maximum_absolute_logit_difference_from_logit_mean=differences,
            amp_absolute_tolerance=MAX_AMP_LOGIT_DIFFERENCE)
        audits.append(audit); write(args.output/f'task{task}_audit.json', audit)
    assert tensor_state_digest(a.visual_encoder.state_dict()) == visual_before
    write(args.output/'result.json', dict(status='complete', action=args.action, git=git,
        split=args.split, seed=1, tasks=tasks, training_updates=0,
        evaluation_scope='samples_intersect_seen_classes', test_used_for_parameter_search=True,
        source_predictions_exact=True, source_parameters_unchanged=True,
        visual_weights_shared=True, frozen_tokens_cached=False, fixed_mean_coefficient=.5,
        mean_before_head=True, metrics=results,
        summaries={arm: summarize_tasks([TaskMetrics(**r) for r in values])
                   for arm, values in results.items()} if args.action == 'eval' else {},
        audits=audits, elapsed_seconds=time.time()-start,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def benchmark(args):
    git=runtime(); set_seed(1, True); a,b=models(args, need_b=args.arm != 'A')
    source=install(args, a, b, 7); model=FixedModelMeanFusion(a,b)
    data=make_source(read(args.source_run/'config.json'), args.data_root, args.face_manifest_root, 'val')
    images,_,_=next(iter(DataLoader(dataset_view(data, seen_indices(7), True), batch_size=64,
        shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(87008))))
    x=move_model_inputs(images, torch.device('cuda'))
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(3): model(x,args.arm)
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter(); start.record()
        for _ in range(10): model(x,args.arm)
        end.record(); torch.cuda.synchronize()
        gpu_ms=start.elapsed_time(end)/10; wall_seconds=(time.perf_counter()-wall)/10
        peak=torch.cuda.max_memory_allocated()/1024**2; reserved=torch.cuda.max_memory_reserved()/1024**2
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    with_flops=True) as profile: model(x,args.arm)
    visual=sum(p.numel() for p in a.visual_encoder.parameters()); total=sum(p.numel() for p in model.parameters())
    write(args.output/'result.json', dict(status='complete', git=git, arm=args.arm, task=7,
        batch_size=64, warmup=3, repeats=10, sources=source, training_updates=0,
        visual_parameters=visual, total_stored_parameters=total, stored_nonvisual_parameters=total-visual,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        visual_weights_shared=True, frozen_tokens_cached=False, source_passes=1 if args.arm=='A' else 2,
        frozen_view_forwards=3 if args.arm=='A' else 6, gpu_milliseconds_per_batch=gpu_ms,
        wall_seconds_per_batch=wall_seconds, people_per_second=64/(gpu_ms/1000),
        peak_allocated_mib=peak, peak_reserved_mib=reserved,
        profile_reported_flops_per_batch=sum(e.flops for e in profile.key_averages()),
        profiler_scope='covered matrix multiplication and convolution; not all operators',
        baseline_memory_measured_in_separate_process=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'eval', 'benchmark'))
    for name in ('source-run','person-run','dual-run','reference-root','data-root','clip-checkpoint','face-manifest-root','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', choices=('val','test'), default='val')
    parser.add_argument('--arm', choices=('A','R5'), default='R5')
    args=parser.parse_args()
    (benchmark if args.action=='benchmark' else evaluate)(args)
