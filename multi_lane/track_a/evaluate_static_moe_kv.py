"""Validation-only Selector key/value interventions on immutable checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .evaluate_static_moe import audit_source, build_static_model, make_source, read
from .evaluate_static_moe_v2 import source_hashes
from .evaluate_whole_parax_eight_task import seen_payload, metrics_at_task, summarize_curve
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .runner import LabelView, TASK_SIZES, git_metadata, set_seed, tensor_state_digest, compact_model_state_dict, validate_classes
from .selector_kv_diagnostic import SELECTOR_EVIDENCE_MODES
from .snapshot_validation import infer, replay


def collect_diagnostics(records, sample_ids, task):
    """Each view's record stream follows the same deterministic loader order."""
    result={"sample_ids":sample_ids}
    for view in ('full','person','face'):
        selected=[r for r in records if r['view']==view]
        require(bool(selected) and all(r['task']==task and r['layer']==1 for r in selected),
                "Missing/wrong diagnostic lane or layer")
        for metric in selected[0]['values']:
            values=torch.cat([r['values'][metric] for r in selected]).numpy()
            require(values.shape==(len(sample_ids),) and np.isfinite(values).all(),
                    "Invalid sample-level diagnostic shape/value")
            result[view+'__'+metric]=values
    return result


def evaluate(source_batch: Path, output: Path, version: str, method: str,
             data_root: Path, clip_checkpoint: Path, face_manifest_root: Path,
             smoke: bool=False):
    metadata=git_metadata(Path(__file__).resolve().parents[2])
    require(not metadata['dirty'] and not output.exists(), "Dirty evaluator or existing output")
    require(method in ('SHARED','MOE_UNIFORM','MOE_OVERLAP') and version in ('v1','v2'),
            "Unsupported source")
    require(not (version=='v2' and method=='MOE_UNIFORM'), "v2 uniform source does not exist")
    source=source_batch/method
    config, audit=audit_source(source)
    require(config['adapter_mode']=='image_token' and config['adapter_layer_indices']==[1],
            "Diagnostic requires the original layer1 image evidence adapter")
    require(config.get('adapter_residual_bound_ratio',0)==(.1 if version=='v2' else 0),
            "Source version/bound mismatch")
    require(Path(config['data_root']).resolve()==data_root.resolve(), "Dataset root differs")
    require(_sha256(clip_checkpoint)==config['clip_checkpoint_sha256'], "CLIP checkpoint changed")
    manifests={face_manifest_root/'detector_config.json':config['face_manifest']['artifact_sha256']['detector_config'],
               face_manifest_root/'manifests/val.jsonl':config['face_manifest']['artifact_sha256']['val_manifest']}
    require(all(_sha256(p)==sha for p,sha in manifests.items()), "Validation Face provenance differs")
    hashes=source_hashes(source_batch,method)
    references=[source_batch/'full_val'/method/f'task{t}.npz' for t in range(8)]
    reference_hashes={str(p):_sha256(p) for p in references}
    set_seed(config['seed'],config['tf32'])
    device=torch.device('cuda')
    model=build_static_model(config,load_openai_clip_visual(clip_checkpoint)).float().to(device)
    clip_hash=tensor_state_digest(model.visual_encoder.state_dict())
    require(clip_hash==audit['frozen_visual_sha256_after'], "Restored CLIP differs")
    cohort=make_source(config,data_root,face_manifest_root,'val')
    validate_classes(cohort)
    require(len(cohort)==2397, "Incomplete validation cohort")
    count=64 if smoke else 2397
    loader=DataLoader(LabelView(cohort,range(count),range(26),include_sample_id=True),
                      batch_size=64,num_workers=2,shuffle=False)
    output.mkdir(parents=True)
    result=dict(status='running',version=version,method=method,split='val',samples=count,
                smoke=smoke,seed=config['seed'],modes=list(SELECTOR_EVIDENCE_MODES),
                training_performed=False,optimizer_updates=0,test_accessed=False,
                diagnostic_lane='current_task_only',diagnostic_layer=1,
                residual_statistics='candidate residual; sample token/channel energy sum denominators',
                evaluator_git=metadata,source_sha256=hashes,reference_sha256=reference_hashes,
                tasks={mode:{} for mode in SELECTOR_EVIDENCE_MODES})
    for task in range(8):
        payload=_torch_load(source/'compact_checkpoints'/f'task{task}.pth')
        restore_compact_model_state(model,payload,task,config['git'])
        model.set_selector_evidence_mode('both');model.enable_selector_sample_diagnostics(None)
        model.requires_grad_(False).eval()
        replay_difference=replay(model,cohort,source,config,task,device)
        state_before=tensor_state_digest(compact_model_state_dict(model))
        with np.load(references[task],allow_pickle=False) as z:
            expected={k:z[k] for k in ('sample_ids','targets','logits','face_reliable')}
        for mode in SELECTOR_EVIDENCE_MODES:
            model.set_selector_evidence_mode(mode)
            model.enable_selector_sample_diagnostics(task)
            _,full=infer(model,loader,device,config['amp'],task,TASK_SIZES)
            data=seen_payload(full,task)
            require(all(np.array_equal(data[k],expected[k][:count]) for k in ('sample_ids','targets','face_reliable')),
                    "Cohort IDs/labels/Face mask differ from source")
            diagnostics=collect_diagnostics(model.pop_selector_sample_diagnostics(),data['sample_ids'],task)
            if mode in ('value_only','none'):
                require(all(float(diagnostics[v+'__attention_kl_current_to_frozen'].max())==0
                            for v in ('full','person','face')), "Frozen key attention changed")
            row=dict(samples=count,metrics=metrics_at_task(data,config['threshold'],task),
                     validation_replay_max_logit_difference=replay_difference,
                     diagnostics={k:{'mean':float(v.mean()),'p50':float(np.quantile(v,.5)),
                                     'p95':float(np.quantile(v,.95)),'max':float(v.max())}
                                  for k,v in diagnostics.items() if k!='sample_ids'})
            if mode=='both':
                difference=float(np.abs(data['logits']-expected['logits'][:count]).max())
                require(difference<=.01, f"Original forward replay differs: {difference}")
                row['full_cohort_original_max_logit_difference']=difference
            require(tensor_state_digest(compact_model_state_dict(model))==state_before
                    and all(p.grad is None for p in model.parameters()), "Inference changed state/gradients")
            destination=output/mode;destination.mkdir(exist_ok=True)
            np.savez_compressed(destination/f'task{task}.npz',**data)
            np.savez_compressed(destination/f'task{task}_diagnostics.npz',**diagnostics)
            result['tasks'][mode][str(task)]=row
            print(f'STATIC_MOE_KV version={version} method={method} task={task} mode={mode} mAP={row["metrics"]["fused"]["mAP"]:.6f}',flush=True)
        (output/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
    require(tensor_state_digest(model.visual_encoder.state_dict())==clip_hash, "Inference changed CLIP")
    require(source_hashes(source_batch,method)==hashes
            and all(_sha256(Path(p))==sha for p,sha in reference_hashes.items()), "Source changed")
    require(all(_sha256(p)==sha for p,sha in manifests.items()), "Face artifacts changed")
    result.update(status='smoke_passed' if smoke else 'complete',source_unchanged=True,
                  curve_summary={mode:summarize_curve(tasks) for mode,tasks in result['tasks'].items()})
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source-batch','output','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--version',choices=('v1','v2'),required=True)
    p.add_argument('--method',choices=('SHARED','MOE_UNIFORM','MOE_OVERLAP'),required=True)
    p.add_argument('--smoke',action='store_true')
    evaluate(**vars(p.parse_args()))
