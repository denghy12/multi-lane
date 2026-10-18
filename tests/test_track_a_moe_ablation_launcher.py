"""Six distinct arms, shared control, paired evaluation and failure draining."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class AblationLauncherTest(unittest.TestCase):
    def launch(self, fail_stage=''):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);repo=root/'repo';scripts=repo/'scripts/emotic';scripts.mkdir(parents=True)
            for name in ('run_moe_layer_routing_ablation.sh','run_moe_layer_routing_arm.sh'):
                shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/emotic'/name,scripts/name)
            (repo/'.gitignore').write_text('/output/\n/logs/\n')
            for args in (('init','-q'),('add','.'),('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')):
                subprocess.run(['git',*args],cwd=repo,check=True)
            data=root/'data';data.mkdir();(data/'CVPR17_Annotations.mat').touch()
            clip=root/'clip.pt';clip.touch()
            manifests=root/'face/manifests';manifests.mkdir(parents=True)
            for split in ('train','val','test'):(manifests/(split+'.jsonl')).touch()
            fake=root/'fake_python'
            fake.write_text('#!'+sys.executable+'\n'+'''
import json,os,sys,time
if sys.argv[1]=='-':
    # Skip only the real GPU preflight; execute manifest/status code normally.
    if len(sys.argv)>2:
        sys.argv.pop(1)
        exec(sys.stdin.read())
    sys.exit(0)
module=sys.argv[sys.argv.index('-m')+1]
arm=os.environ['ARM']
stage='train' if module.endswith('.runner') else ('verify' if module.endswith('.verify_moe_activate_freeze') else sys.argv[sys.argv.index('--split')+1])
def event(kind):
    with open(os.environ['EVENT_PATH'],'a') as handle:
        handle.write(json.dumps(dict(arm=arm,stage=stage,event=kind,time=time.time_ns(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),args=sys.argv))+'\\n')
event('start')
time.sleep(.5 if stage in ('val','test') else .1)
event('end')
fail=os.environ.get('FAIL_STAGE')
sys.exit(7 if (fail=='train' and stage=='train' and arm=='L1_K2_F3') or (fail=='test' and stage=='test' and arm=='L1_K3_F2') else 0)
''')
            fake.chmod(0o755);events=root/'events.jsonl'
            env=dict(os.environ,PYTHON=str(fake),RUN_ID='fixture',SEED='0',DATA_ROOT=str(data),
                     CLIP_CHECKPOINT=str(clip),FACE_MANIFEST_ROOT=str(manifests.parent),
                     EVENT_PATH=str(events),FAIL_STAGE=fail_stage)
            env.pop('SMOKE_UPDATES',None)
            process=subprocess.run(['bash',str(scripts/'run_moe_layer_routing_ablation.sh')],env=env,
                                   capture_output=True,text=True,timeout=25)
            manifest=json.loads((repo/'output/moe_layer_routing_ablation/fixture/batch_manifest.json').read_text())
            rows=[json.loads(line) for line in events.read_text().splitlines()]
            statuses={p.parent.name:p.read_text().strip() for p in (repo/'logs/moe_layer_routing_ablation/fixture').glob('*/evaluation_status.txt')}
            return process.returncode,manifest,rows,statuses

    def test_six_unique_runs_correct_gpus_and_two_shared_control_references(self):
        code,manifest,rows,statuses=self.launch()
        self.assertEqual(code,0)
        train=[r for r in rows if r['stage']=='train' and r['event']=='start']
        self.assertEqual(len(train),6)
        self.assertEqual(len({r['arm'] for r in train}),6)
        self.assertEqual(set(manifest['layer_group'])&set(manifest['routing_freezing_group']),{'L1_K2_F2'})
        expected={r['arm']:r for r in manifest['arms']}
        for r in train:
            spec=expected[r['arm']];args=r['args']
            self.assertEqual(r['gpu'],str(spec['gpu']))
            for arg,value in (('--adapter-layer-indices',spec['layer']),('--moe-top-k',spec['router_top_k']),('--moe-freeze-count',spec['freeze_count'])):
                self.assertEqual(args[args.index(arg)+1],str(value))
        for arm in expected:
            times={(r['stage'],r['event']):r['time'] for r in rows if r['arm']==arm}
            self.assertLess(max(times[s,'start'] for s in ('val','test')),min(times[s,'end'] for s in ('val','test')))
            self.assertLess(times['train','end'],times['verify','start'])
            self.assertLess(times['verify','end'],min(times[s,'start'] for s in ('val','test')))
        self.assertEqual(set(statuses.values()),{'val_exit=0 test_exit=0'})
        self.assertEqual(manifest['status'],'complete')

    def test_eval_failure_drains_all_arms_and_both_splits(self):
        code,manifest,rows,statuses=self.launch('test')
        self.assertEqual(code,1)
        self.assertEqual(len(manifest['arm_status']),6)
        self.assertEqual(manifest['arm_status']['L1_K3_F2']['exit_code'],1)
        self.assertEqual(statuses['L1_K3_F2'],'val_exit=0 test_exit=7')
        self.assertEqual(len([r for r in rows if r['stage'] in ('val','test') and r['event']=='end']),12)
        self.assertEqual(manifest['status'],'failed')

    def test_training_failure_prevents_its_evaluation_and_drains_other_arms(self):
        code,manifest,rows,statuses=self.launch('train')
        self.assertEqual(code,1)
        self.assertEqual(manifest['arm_status']['L1_K2_F3']['exit_code'],7)
        self.assertEqual({r['stage'] for r in rows if r['arm']=='L1_K2_F3'},{'train'})
        self.assertEqual(len(statuses),5)
        self.assertEqual(len(manifest['arm_status']),6)


if __name__=='__main__':unittest.main()
