"""Keep one task-head population resident; cache only unlearned descriptors."""
from dataclasses import replace
import hashlib
import torch
from scripts.emotic.d9_final_patch_readout import inputs,predict
from multi_lane.track_a.summary_logit_fusion import fused_logits


class StagePredictor:
    def __init__(self,stage,device):
        self.device=device
        self.stage=replace(stage,cls=stage.cls.to(device),patch=stage.patch.to(device),
                           roster_reliable=stage.roster_reliable.to(device))
        self.descriptors={}

    @torch.no_grad()
    def __call__(self,gate,*,patch_donors=None,spatial_order=None,diagnose=False):
        if len(gate.readers):
            return predict(self.stage,gate,self.device,patch_donors=patch_donors,
                           spatial_order=spatial_order,diagnose=diagnose)
        def identity(array):
            return None if array is None else hashlib.sha256(array.tobytes()).hexdigest()
        key=(gate.method,identity(patch_donors),identity(spatial_order))
        gate=gate.to(self.device).eval()
        if key not in self.descriptors:
            cls,patch,counts,flags=inputs(self.stage,self.device)
            if patch_donors is not None:patch=patch[torch.as_tensor(patch_donors,device=self.device)]
            if spatial_order is not None:patch=torch.stack([patch[:,v,order] for v,order in enumerate(spatial_order)],1)
            self.descriptors[key]=gate.describe(cls,patch,counts,flags)[0]
        scores=gate.scores(self.descriptors[key]).cpu()
        base=self.stage.base
        return scores,fused_logits(base.logits,scores,base.inverse,base.reliable),{}
