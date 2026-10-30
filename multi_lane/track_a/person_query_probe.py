"""Fixed TRAIN-person interventions for own short-run query checkpoints; no update."""
import argparse
from pathlib import Path

import torch

from .person_query_conditioning import trace_person_query
from .person_grounded_breakpoints import compare_fields,pair_geometry,spatial_response
from .private_summary_evaluation import load_checkpoint,grouped_donors,paired_batches
from .protected_joint_experiment import build,runtime
from .evaluate_joint26_checkpoint import make_source
from .runner import set_seed,dataset_view,seen_indices,tensor_state_digest
from .historical_b32_backbone_attribution import method_state
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read,write
from .export_compact_test_scores import _sha256


def run(args):
    git=runtime();set_seed(1,True);cfg=read(args.training_root/'config.json')
    assert cfg['kind']=='protected_person_query_single_factor' and cfg['git']==git
    if args.output.exists():raise ValueError('Probe output exists; keep previous evidence')
    model=build(cfg['model_template'],Path(cfg['clip_checkpoint']),cfg['arm'])
    visual=tensor_state_digest(model.visual_encoder.state_dict())
    source=make_source(cfg['model_template'],Path(cfg['data_root']),Path(cfg['face_manifest_root']),'train')
    source.multi_view_transform.emit_spatial_geometry=True;records=[]
    for task in read(args.training_root/'training_complete.json')['tasks']:
        path,sha=load_checkpoint(model,args.training_root,cfg,task);before=tensor_state_digest(method_state(model))
        view=dataset_view(source,seen_indices(task),True);ids=[source.sample_ids[i] for i in view.indices]
        batches=[b for b in paired_batches(ids) if len(b)>=32][:1]
        if not batches:raise AssertionError('No complete TRAIN paired batch')
        images,y,names=next(iter(torch.utils.data.DataLoader(view,batch_sampler=batches,num_workers=0,
            generator=torch.Generator().manual_seed(931650+task))))
        images=move_model_inputs(images,torch.device('cuda'));donor,eligible=grouped_donors(names,images['spatial_visibility']>0)
        if not bool(eligible.any()):raise AssertionError('No visible TRAIN person donor')
        same=(images['full']==images['full'][donor]).flatten(1).all(-1)
        if bool((eligible & ~same).any()):raise AssertionError('Person boxes are not in the same Full crop coordinates')
        with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
            model.seen_logits_with_views(images);native=model._native_delta.clone();cache=model._interaction_cache;lanes={}
            geometry=pair_geometry(images['spatial_bbox'],images['spatial_visibility'],donor,model._private_sources['full'][0].shape[1]-1)
            for i,lane in enumerate(cache['lane_ids']):
                module=model.interaction_bank[lane];person=cache['features']['person'][:,i]
                tokens={v:model._private_sources[v][0] for v in ('full','person','face')}
                anchors={v:model._private_sources[v][1][i] for v in tokens}
                arguments=(module,tokens,anchors,cache['base'][:,i],person,images['spatial_bbox'],images['spatial_visibility'],cache['reliable'])
                a,_=trace_person_query(*arguments);b,_=trace_person_query(*arguments,condition_person=person[donor])
                if not torch.equal(a['delta'],native[:,i]):raise AssertionError('Own probe changes native correction')
                if not torch.equal(a['readout_query'],b['readout_query']):raise AssertionError('Late query changed')
                az,_=trace_person_query(*arguments,zero_condition=True)
                bz,_=trace_person_query(*arguments,condition_person=person[donor],zero_condition=True)
                if not torch.equal(az['delta'],bz['delta']):raise AssertionError('Zero-condition negative control failed')
                stats=compare_fields(a,b);stats.update(spatial_response(a,b,geometry))
                lanes[str(lane)]={k:float(v[eligible].mean()) for k,v in stats.items()}
        assert before==tensor_state_digest(method_state(model)) and sha==_sha256(path)
        records.append(dict(task=task,checkpoint_sha256=sha,samples=len(y),eligible=int(eligible.sum()),
            sample_ids=list(names),lanes=lanes,parameters_unchanged=True))
    assert visual==tensor_state_digest(model.visual_encoder.state_dict())
    write(args.output/'result.json',dict(status='complete',arm=cfg['arm'],git=git,split='train',
        evaluation_data_used_for_probe=False,training_updates=0,records=records,
        purpose='implementation_and_signal_check_only_not_mAP_success'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
