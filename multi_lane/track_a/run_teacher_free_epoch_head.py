"""GPU0-only smoke then parallel Validation/Test read-only attribution."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_historical_b32_test_attribution import read,write
from .teacher_free_shared_learning import runtime


def command(args, split, output, smoke=False):
    result=[sys.executable,"-u","-m","multi_lane.track_a.teacher_free_epoch_head_attribution",
        "--source-root",str(args.source_root),"--data-root",str(args.data_root),
        "--clip-checkpoint",str(args.clip_checkpoint),"--face-manifest-root",str(args.face_manifest_root),
        "--split",split,"--workers","2","--output",str(output)]
    return result+["--smoke"] if smoke else result


def run(args):
    git=runtime()
    args.output.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    with (args.output/"queue.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:raise RuntimeError("Diagnostic queue already running") from e
        queue=args.logs/"queue.json"
        if queue.exists():
            previous=read(queue)
            for row in previous.get("records",[]):
                if row.get("exit") is None:
                    p=Path("/proc")/str(row["pid"])/"cmdline"
                    if p.exists() and b"multi_lane.track_a.teacher_free_epoch_head_attribution" in p.read_bytes():
                        raise RuntimeError("Previous diagnostic child remains active")
        cfg=dict(git=git,gpu=0,epochs=[1,5,10,20,30],arms=["U0","U2","U4"],
            source_root=str(args.source_root),data_root=str(args.data_root),
            clip_checkpoint=str(args.clip_checkpoint),face_manifest_root=str(args.face_manifest_root),
            training_updates=0,batch_size=64,workers=2,threads=72,amp=True,tf32=True,
            task=0,current_classes=5,conditions_per_epoch=7,
            evaluation_scope="samples_intersect_seen_classes",test_used_for_development=True,
            no_test_best_epoch_selection=True)
        if (args.output/"protocol.json").exists():
            if read(args.output/"protocol.json")!=cfg:raise ValueError("Protocol changed on restart")
        else:write(args.output/"protocol.json",cfg)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="72",MKL_NUM_THREADS="72")
        records=[];started=time.time()
        def start(name,cmd):
            with (args.logs/(name+".log")).open("a") as f:
                process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            row=dict(name=name,pid=process.pid,command=cmd,started_unix=time.time())
            records.append(row);write(queue,dict(status="running",records=records))
            print("START",name,process.pid,flush=True)
            return process,row
        try:
            smoke=args.output/"smoke_val"
            if not (smoke/"result.json").exists():
                p,row=start("smoke_val",command(args,"val",smoke,True))
                row["exit"]=p.wait();row["finished_unix"]=time.time()
                if row["exit"]:raise RuntimeError("Smoke failed; do not run full attribution")
            sr=read(smoke/"result.json")
            if sr["status"]!="complete" or not sr["config"]["smoke"]:
                raise ValueError("Smoke completion is invalid")
            jobs=[]
            for split in ("val","test"):
                output=args.output/split
                if (output/"result.json").exists():
                    if read(output/"result.json")["status"]=="complete":continue
                while True:
                    free=int(subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.free",
                        "--format=csv,noheader,nounits"],text=True).strip())
                    shm=os.statvfs("/dev/shm")
                    if free>=4500 and shm.f_bavail*shm.f_frsize>=768*1024**2:break
                    write(queue,dict(status="waiting_for_gpu0_memory",records=records))
                    time.sleep(3)
                jobs.append(start(split,command(args,split,output)))
                time.sleep(8)
            failures=[]
            for p,row in jobs:
                row["exit"]=p.wait();row["finished_unix"]=time.time()
                if row["exit"]:failures.append(row)
                write(queue,dict(status="running",records=records))
            if failures:raise RuntimeError(("Attribution child failed",failures))
            for split in ("val","test"):
                result=read(args.output/split/"result.json")
                if result["status"]!="complete" or not result["source_sha256_unchanged"]:
                    raise ValueError("Full split integrity check failed")
            write(args.output/"completion.json",dict(status="complete",git=git,protocol=cfg,
                records=records,training_updates=0,source_weights_written=False,
                elapsed_seconds=time.time()-started))
            write(queue,dict(status="complete",records=records))
            print("EPOCH_HEAD_ALL_COMPLETE",flush=True)
        except BaseException as error:
            write(args.output/"failure.json",dict(error=repr(error),records=records,
                source_weights_written=False))
            raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source-root","data-root","clip-checkpoint","face-manifest-root","output","logs"):
        p.add_argument("--"+name,type=Path,required=True)
    run(p.parse_args())
