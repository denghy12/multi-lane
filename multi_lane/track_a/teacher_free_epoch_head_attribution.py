"""Read-only epoch trajectories and representation-by-head factorial attribution."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .historical_b32_backbone_attribution import build, method_state, component
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read, write
from .runner import compute_metrics, dataset_view, seen_indices, set_seed, tensor_state_digest
from .teacher_free_shared_learning import runtime, evaluation_loader

ARMS = ("U0", "U2", "U4")
EPOCHS = (1, 5, 10, 20, 30)
CONDITIONS = (("U0","U0"), ("U0","U2"), ("U2","U0"), ("U2","U2"),
              ("U0","U4"), ("U4","U0"), ("U4","U4"))
VIEWS = ("fused", "full", "person", "face")
CORE = ("model.py", "adapter.py", "view_fusion.py", "paired_transforms.py",
        "runner.py", "historical_b32_backbone_attribution.py",
        "evaluate_joint26_checkpoint.py", "teacher_free_shared_learning.py")


def label(representation, head):
    if (representation, head) not in CONDITIONS:
        raise ValueError("Unregistered factorial condition")
    return representation + "_H" + head[1:]


def checkpoint_path(root, arm, epoch):
    if arm not in ARMS or epoch not in EPOCHS:
        raise ValueError("Only predeclared task0 checkpoints may be evaluated")
    return root / arm / "train/epoch_recovery/task0" / ("epoch%02d.pth" % epoch)


def code_audit(commit):
    root = Path(__file__).resolve().parents[2]
    paths = ["multi_lane/track_a/"+name for name in CORE]
    paths += ["multi_lane/continual_datasets/continual_datasets.py"]
    for path in paths:
        old = subprocess.check_output(["git","show",commit+":"+path],cwd=root)
        if old != (root/path).read_bytes():
            raise ValueError("Existing inference dependency changed: "+path)
    return dict(status="verified",existing_inference_files_byte_identical=paths)


def validate_payload(saved, config, epoch):
    if saved["config"] != config or saved["task_id"] != 0:
        raise ValueError("Epoch checkpoint provenance does not match its own run")
    if saved["training_state"]["next_epoch"] != epoch:
        raise ValueError("Checkpoint epoch is incorrect")
    if saved["provenance"]["next_epoch"] != epoch:
        raise ValueError("Saved epoch boundary is inconsistent")
    if saved["source_git"] != config["git"]:
        raise ValueError("Source code provenance changed")
    if epoch == 1 and config["arm"] != "U0":
        if saved["training_state"]["history"][0]["complementary_ramp"] != 0:
            raise ValueError("First epoch is not an identity control")
    return saved["model"]


def load_sources(args):
    configs = {a:read(args.source_root/a/"train/config.json") for a in ARMS}
    base = configs["U0"]
    if base["seed"] != 1 or base["epochs_per_task"] != 30 or base["smoke"]:
        raise ValueError("Wrong training recipe")
    for a, cfg in configs.items():
        if (cfg["model_template"] != base["model_template"] or cfg["git"] != base["git"]
                or cfg["arm"] != a or cfg["seed"] != 1 or cfg["smoke"]):
            raise ValueError("Source models do not form the matched experiment")
        completed = read(args.source_root/a/"train/training_complete.json")
        if completed["tasks"] != [0] or completed["status"] != "complete":
            raise ValueError("This diagnostic requires the complete task0 screen")
    audit = code_audit(base["git"]["commit"])
    if _sha256(args.clip_checkpoint) != base["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint differs from the training source")
    states, hashes = {}, {}
    for epoch in EPOCHS:
        for a in ARMS:
            path = checkpoint_path(args.source_root,a,epoch)
            hashes[str(path)] = _sha256(path)
            saved = torch.load(path,map_location="cpu",weights_only=False)
            states[a,epoch] = validate_payload(saved,configs[a],epoch)
    for a in ARMS[1:]:
        if set(states[a,1]) != set(states["U0",1]) or not all(
                torch.equal(v,states[a,1][k]) for k,v in states["U0",1].items()):
            raise ValueError("Epoch1 model identity control failed")
    for a in ARMS:
        path = args.source_root/a/"train/compact_checkpoints/task0.pth"
        hashes[str(path)] = _sha256(path)
        if hashes[str(path)] != read(args.source_root/a/"train/training_complete.json")["checkpoints"]["0"]["sha256"]:
            raise ValueError("Final compact source SHA differs from completed training")
        saved = _torch_load(path)
        if not all(torch.equal(v,saved["model"][k]) for k,v in states[a,30].items()):
            raise ValueError("Epoch30 does not reproduce the final checkpoint")
    return configs, states, hashes, audit


def install_state(model, state):
    missing = model.load_state_dict(state,strict=False)
    if missing.unexpected_keys or any(not k.startswith("visual_encoder.") for k in missing.missing_keys):
        raise ValueError("Epoch checkpoint omitted task state")
    model.restore_task(0)
    model.requires_grad_(False).eval()
    model.cross_view_runtime_enabled = False


def readout(features, weight, bias, mask, seen_classes=5):
    """Use the original pre-head-normalized features and task ownership mask."""
    result = {}
    for name in VIEWS:
        values = F.linear(features[name],weight,bias)
        result[name] = (values*mask.unsqueeze(0)).sum(1)[:, :seen_classes]
    return result


def probability_rows(values):
    # Match the historical CPU sigmoid separately for each original batch.
    logits = {name:value.float().cpu() for name,value in values.items()}
    return logits, {name:torch.sigmoid(value) for name,value in logits.items()}


def verify_replay(actual, reference, ids, targets, offset=0):
    n = len(ids)
    if (not np.array_equal(reference["sample_ids"][offset:offset+n],np.asarray(ids))
            or not np.array_equal(reference["targets"][offset:offset+n],targets.numpy())):
        raise ValueError("Reference replay population or labels changed")
    differences = {}
    for name in VIEWS:
        for suffix in ("_logits","_probabilities"):
            key = name+suffix
            wanted = reference[key][offset:offset+n]
            if not np.array_equal(actual[key],wanted):
                raise AssertionError(("native_prediction_replay_failed",key,
                    float(np.max(np.abs(actual[key]-wanted)))))
            differences[key] = 0.
    return dict(people=n,all_logits_and_probabilities_exact=True,differences=differences)


def bce_summary(logits, targets):
    logits = logits.double()
    values = (F.softplus(logits)-targets.double()*logits).mean(0)
    return dict(mean=float(values.mean()),per_class=values.tolist(),
                scope="current_task_five_classes_no_hidden_class_constant")


def summarize_case(logits, probabilities, targets, reliable):
    result = {}
    for view in VIEWS:
        result[view] = dict(metrics=asdict(compute_metrics(0,probabilities[view],targets,.5)),
            bce=bce_summary(logits[view],targets))
    if bool(reliable.any()):
        result["face_reliable"] = dict(metrics=asdict(compute_metrics(
            0,probabilities["face"][reliable],targets[reliable],.5)),
            bce=bce_summary(logits["face"][reliable],targets[reliable]))
    result["subgroups"] = {}
    for name,mask in (("reliable",reliable),("unreliable",~reliable)):
        if bool(mask.any()):
            result["subgroups"][name] = {v:dict(metrics=asdict(compute_metrics(
                0,probabilities[v][mask],targets[mask],.5)),
                bce=bce_summary(logits[v][mask],targets[mask])) for v in ("full","fused")}
    return result


def factorial(values):
    rows = []
    for arm in ("U2","U4"):
        for view in (*VIEWS,"face_reliable"):
            base = values[label("U0","U0")][view]
            representation = values[label(arm,"U0")][view]
            head = values[label("U0",arm)][view]
            own = values[label(arm,arm)][view]
            for metric in ("mAP","bce"):
                def pick(x):
                    return x["metrics"]["mAP"] if metric == "mAP" else x["bce"]["mean"]
                a,b,c,d = map(pick,(base,representation,head,own))
                rows.append(dict(arm=arm,view=view,metric=metric,baseline=a,
                    representation_at_baseline_head=b-a,head_at_baseline_representation=c-a,
                    interaction=d-b-c+a,total=d-a,
                    interpretation="conditional_readout_contrasts_not_unique_training_causal_effects"))
    return rows


def state_drift(state, baseline):
    rows = {}
    for group in ("A","S","P","H"):
        def coordinates(s):
            values=[]
            for k,v in s.items():
                if component(k) != group:
                    continue
                if group == "A":
                    if int(k.split(".")[2]) != 0:continue
                elif group == "S":v=v[:1]
                elif group == "P":v=v[:,:1]
                elif group == "H":v=v[:5]
                values.append(v.double().reshape(-1))
            return torch.cat(values)
        a,b=coordinates(state),coordinates(baseline)
        rows[group]=dict(exact=torch.equal(a,b),relative_l2=float((a-b).norm()/b.norm()),
            cosine=float(a.dot(b)/(a.norm()*b.norm())))
    return rows


def evaluate(args):
    started=time.time()
    git=runtime()
    set_seed(1,True)
    configs,states,hashes,code=load_sources(args)
    base=configs["U0"]
    cfg=dict(kind="readonly_epoch_representation_head_attribution",git=git,
        source_git=base["git"],source_root=str(args.source_root),seed=1,task=0,classes=list(range(5)),
        epochs=list(EPOCHS),conditions=[label(*c) for c in CONDITIONS],split=args.split,smoke=args.smoke,
        batch_size=64,workers=args.workers,worker_threads=1,threads=72,amp=True,tf32=True,
        evaluation_scope="samples_intersect_seen_classes",training_updates=0,
        source_parameter_assembly_for_deployment=False,optimizer_created=False,
        test_used_for_development=True,source_sha256=dict(hashes),inference_code_audit=code,
        inputs=dict(data_root=str(args.data_root),clip=str(args.clip_checkpoint),face=str(args.face_manifest_root)))
    config_path=args.output/"config.json"
    if config_path.exists():
        if read(config_path) != cfg:raise ValueError("Restart changed diagnostic protocol")
    else:write(config_path,cfg)
    model=build(base["model_template"],args.clip_checkpoint)
    model.requires_grad_(False).eval()
    visual=tensor_state_digest(model.visual_encoder.state_dict())
    source=make_source(base["model_template"],args.data_root,args.face_manifest_root,args.split)
    view=dataset_view(source,seen_indices(0),True)
    if args.smoke:view=Subset(view,range(min(64,len(view))))
    loader=evaluation_loader(view,0,args.workers)
    references={}
    for a in ARMS:
        p=args.source_root/a/args.split/"view_scores/task0.npz"
        hashes[str(p)] = _sha256(p)
        with np.load(p,allow_pickle=False) as z:references[a]={k:z[k].copy() for k in z.files}
    completed=[]
    cohorts=None
    for epoch in EPOCHS:
        folder=args.output/("epoch%02d"%epoch)
        marker=folder/"completion.json"
        if marker.exists():
            row=read(marker)
            if row["status"]!="complete" or row["epoch"]!=epoch:raise ValueError("Invalid epoch completion")
            if not all(_sha256(Path(p))==s for p,s in row["output_sha256"].items()):
                raise ValueError("Existing diagnostic output was modified")
            completed.append(row);continue
        tick=time.time()
        head={a:(states[a,epoch]["head.weight"].to("cuda"),
                 states[a,epoch]["head.bias"].to("cuda")) for a in ARMS}
        buffers={label(*c):{k:[] for k in (*VIEWS,)} for c in CONDITIONS}
        probabilities={key:{v:[] for v in VIEWS} for key in buffers}
        ids_all=[];ys=[];reliable_all=[];lengths=[];offset=0
        for batch_index,(images,targets,ids) in enumerate(loader):
            reliable=images["face_reliable"].bool().cpu()
            x=move_model_inputs(images,torch.device("cuda"))
            lengths.append(len(ids));ids_all.extend(str(i) for i in ids);ys.append(targets.float());reliable_all.append(reliable)
            for representation in ARMS:
                install_state(model,states[representation,epoch])
                before=tensor_state_digest(method_state(model)) if batch_index==0 else None
                with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
                    fused,features=model.encode_lanes_with_views(x,all_seen_lanes=True)
                    evidence=dict(fused=fused,**features)
                    mask=model.task_class_mask[[0]].to(dtype=fused.dtype)
                    for r,h in CONDITIONS:
                        if r!=representation:continue
                        values=readout(evidence,*head[h],mask)
                        z,p=probability_rows(values)
                        key=label(r,h)
                        for name in VIEWS:
                            buffers[key][name].append(z[name]);probabilities[key][name].append(p[name])
                    if batch_index==0:
                        native,branches=model.seen_logits_with_views(x)
                        own=label(representation,representation)
                        for name,value in dict(fused=native,**branches).items():
                            if not torch.equal(value.float().cpu(),buffers[own][name][-1]):
                                raise AssertionError(("native_feature_readout_changed",epoch,representation,name))
                if before is not None and before!=tensor_state_digest(method_state(model)):
                    raise AssertionError("Readout changed model parameters")
                if epoch==30:
                    own=label(representation,representation)
                    replay={name+suffix:(buffers[own][name][-1] if suffix=="_logits" else probabilities[own][name][-1]).numpy()
                            for name in VIEWS for suffix in ("_logits","_probabilities")}
                    verify_replay(replay,references[representation],ids,targets,offset)
            if epoch==1:
                ref=label("U0","U0")
                for name in buffers:
                    for v in VIEWS:
                        if not torch.equal(buffers[name][v][-1],buffers[ref][v][-1]):
                            raise AssertionError("Epoch1 cross-readout identity failed")
            offset+=len(ids)
        y=torch.cat(ys);reliable=torch.cat(reliable_all)
        cohort=(tuple(ids_all),tensor_state_digest({"targets":y,"face_reliable":reliable}))
        if cohorts is not None and cohorts!=cohort:raise AssertionError("Epoch cohort changed")
        cohorts=cohort
        results={};outputs=[]
        for r,h in CONDITIONS:
            key=label(r,h)
            z={v:torch.cat(buffers[key][v]) for v in VIEWS}
            p={v:torch.cat(probabilities[key][v]) for v in VIEWS}
            arrays=dict(schema_version=np.asarray(1),task_id=np.asarray(0),epoch=np.asarray(epoch),
                representation=np.asarray(r),head_source=np.asarray(h),sample_ids=np.asarray(ids_all,dtype=np.str_),
                class_indices=np.arange(5,dtype=np.int64),targets=y.numpy(),
                face_reliable=reliable.numpy(),batch_lengths=np.asarray(lengths),
                probability_device=np.asarray("cpu"),probability_dtype=np.asarray("float32"),
                probability_operation=np.asarray("torch.sigmoid"),torch_version=np.asarray(str(torch.__version__)))
            for v in VIEWS:
                arrays[v+"_logits"]=z[v].numpy();arrays[v+"_probabilities"]=p[v].numpy()
            folder.mkdir(parents=True,exist_ok=True)
            path=folder/(key+".npz");np.savez_compressed(path,**arrays);outputs.append(path)
            results[key]=summarize_case(z,p,y,reliable)
        summary=folder/"metrics.json"
        write(summary,dict(epoch=epoch,split=args.split,conditions=results,
            factorial=factorial(results),parameter_drift={a:state_drift(states[a,epoch],states["U0",epoch]) for a in ARMS}))
        outputs.append(summary)
        if visual!=tensor_state_digest(model.visual_encoder.state_dict()) or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Frozen source or gradients changed")
        row=dict(status="complete",epoch=epoch,people=len(ids_all),elapsed_seconds=time.time()-tick,
            output_sha256={str(p):_sha256(p) for p in outputs},readout_did_not_update_parameters=True,
            epoch1_identity_checked=epoch==1,epoch30_native_all_predictions_exact=epoch==30)
        write(marker,row);completed.append(row)
        write(args.output/"progress.json",dict(status="running",completed=completed))
        print("EPOCH_HEAD_COMPLETE",args.split,epoch,len(ids_all),"seconds",round(time.time()-tick,2),flush=True)
        del head,buffers,probabilities
    if any(_sha256(Path(p))!=sha for p,sha in hashes.items()):raise AssertionError("Source file SHA changed")
    write(args.output/"result.json",dict(status="complete",config=cfg,epochs=list(EPOCHS),
        completed=completed,source_sha256_unchanged=True,source_files_sha256=hashes,training_updates=0,
        source_weights_written=False,elapsed_seconds=time.time()-started,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))
    print("READONLY_ATTRIBUTION_COMPLETE",args.split,flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","data-root","clip-checkpoint","face-manifest-root","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--split",choices=("val","test"),default="val")
    p.add_argument("--workers",type=int,choices=(0,2),default=2)
    p.add_argument("--smoke",action="store_true")
    return p


if __name__=="__main__":
    evaluate(parser().parse_args())
