import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.final_patch_fusion import METHODS, LEARNED, WIDTH, PROJECTION_SEED
from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.d9_final_patch_extract import SCHEMA, capture_identity
from scripts.emotic.d9_final_patch_readout import train,evaluate,load_stage
from scripts.emotic.d9_final_patch_run import CONFIG,FIT,check,lock_protocol
from scripts.emotic.evaluate_d9_relation import metrics_for_stage
from scripts.emotic.r1_asymmetric_forward import fuse_logits
from test_d9_final_patch import population


class FinalPatchPipelineTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_locked_two_sources_eight_methods_and_eight_stages(self):
        cfg=read(CONFIG);check(cfg)
        self.assertEqual(len(cfg['sources'])*len(cfg['methods']),16)
        self.assertEqual(len(LEARNED),6)
        cfg['tasks']=[0,3,6]
        with self.assertRaises(ValueError):check(cfg)
        cfg=read(CONFIG);cfg['sources']['full']['lr']=.01
        with self.assertRaises(ValueError):check(cfg)

    def test_documentation_commit_keeps_original_phase_identity_but_code_change_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            record=dict(digest='original',code_sha256={'model.py':'content'},config={'seed':1},
                        head='execution-commit',branch='experiment')
            lock_protocol(base,record)
            same=dict(record,head='documentation-commit')
            self.assertEqual(lock_protocol(base,same)['head'],'execution-commit')
            with self.assertRaises(ValueError):lock_protocol(base,dict(same,digest='changed-code'))
    def test_cache_retains_every_patch_and_matches_complete_and_supervised_rosters(self):
        stage=population();cfg=read(CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);path=root/'train/task0.npz';path.parent.mkdir()
            patches=np.zeros((len(stage.roster_ids),3,196,WIDTH),dtype=np.float16)
            def save(patches):
                np.savez_compressed(path,image_ids=stage.base.ids,roster_ids=stage.roster_ids,counts=stage.base.counts,
                                    reliable_counts=stage.base.reliable_counts,roster_inverse=stage.roster_inverse,
                                    roster_reliable=stage.roster_reliable.numpy(),cls=stage.cls.numpy(),patch=patches)
                spec=dict(schema=SCHEMA,split='train',task=0,smoke=False,width=WIDTH,projection_seed=PROJECTION_SEED,
                          capture_code=capture_identity(),patch_dtype='float16',cls_dtype='float32',
                          native_sha256=stage.base.provenance['native_sha256'],checkpoint_sha256='frozen',config_sha256='config')
                meta=dict(protocol=spec,archive_sha256=sha256_file(path),roster_sha256=hashlib.sha256('\n'.join(stage.roster_ids).encode()).hexdigest())
                path.with_suffix('.json').write_text(json.dumps(meta))
            save(patches)
            with patch('scripts.emotic.d9_final_patch_readout.load_base',return_value=stage.base):
                actual=load_stage(cfg,'equal90',root,'train',0,False)
                self.assertEqual(len(actual.roster_ids),int(stage.base.counts.sum()))
                self.assertEqual(len(actual.base.person_ids),24)
                save(patches[:,:,:10])
                with self.assertRaises(ValueError):load_stage(cfg,'equal90',root,'train',0,False)
                save(patches);meta=read(path.with_suffix('.json'));meta['protocol']['checkpoint_sha256']='changed';path.with_suffix('.json').write_text(json.dumps(meta))
                with self.assertRaises(ValueError):load_stage(cfg,'equal90',root,'train',0,False)

    def test_eight_stage_full_pipeline_native_masks_controls_and_val_seal(self):
        cfg=read(CONFIG);fitting=dict(FIT,steps=2,permutations=2,patch_swaps=1)
        def load(cfg,name,features,split,task,smoke):return population(split,task)
        def fixed(source,features,split,task,stage):return fuse_logits(*stage.logits.unbind(-1),stage.reliable)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'out';source=root/'source';cfg['sources']['equal90']['path']=str(source)
            for split in ('val','test'):
                rows=[]
                for task in range(8):
                    base=population(split,task).base
                    rows.append(dict(task=task,scores_sha256=split,metrics=metrics_for_stage(task,base.targets,base.reliable,{'fixed':fixed(None,None,split,task,base)})))
                p=source/'evaluation'/split/'result.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(stages=rows)))
            with patch('scripts.emotic.d9_final_patch_readout.load_stage',side_effect=load),patch('scripts.emotic.d9_final_patch_readout.canonical_fixed',side_effect=fixed):
                train(cfg,'equal90',out,list(range(8)),fitting,False,'cpu')
                with self.assertRaises(FileNotFoundError):evaluate(cfg,'equal90',out,'test',list(range(8)),fitting,False,'cpu')
                evaluate(cfg,'equal90',out,'val',list(range(8)),fitting,False,'cpu')
                result=evaluate(cfg,'equal90',out,'test',list(range(8)),fitting,False,'cpu')
                self.assertTrue(result['target_comparable']);self.assertEqual(set(result['summary']),set(METHODS))
                with np.load(out/'test/task7.npz',allow_pickle=False) as z:
                    self.assertEqual(z['targets'].shape[1],26)
                    native=population('test',7).base.logits.numpy()
                    for method in LEARNED:
                        np.testing.assert_array_equal(z[method+'_logits'],(native*z[method+'_weights'][:,None,:]).sum(2))
                        self.assertTrue((z[method+'_weights'][~z['reliable'],2]==0).all())
                    for method in ('cls_mean','cls_regions','cls_patch','cls_patch_full'):
                        self.assertIn(method+'_patch_swapped_logits',z.files)
                    self.assertIn('cls_regions_spatial_shuffled_logits',z.files)
                (out/'selection.json').write_text((out/'selection.json').read_text()+'\n')
                with self.assertRaises(ValueError):evaluate(cfg,'equal90',out,'test',list(range(8)),fitting,False,'cpu')

if __name__=='__main__':unittest.main()
