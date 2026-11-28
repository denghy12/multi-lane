"""Cache frozen inputs without changing reader math or deterministic reductions."""
import torch
from multi_lane.track_a.final_patch_fusion import roster_mean, VIEWS


class PreparedDescriptor:
    """One population, one gate. Never cache a trainable reader output."""
    def __init__(self, gate, cls, patch, counts, flags):
        self.gate = gate
        if not len(gate.readers):
            with torch.no_grad():
                self.fixed, _ = gate.describe(cls, patch, counts, flags)
            self.parts = None
            return
        self.fixed = None
        # Validate the original path once, including shapes and finite patches.
        with torch.no_grad():
            gate.describe(cls, patch, counts, flags)
        self.counts = counts.cpu()
        self.flags = flags.cpu()
        self.cls = cls.detach().float().clone()
        self.cls[:, 2] = self.cls[:, 2].masked_fill(~flags[:, None], 0)
        self.parts = []
        for i in range(3):
            g = patch[:, i].detach().float()
            if i == 2:
                g = g.masked_fill(~flags[:, None, None], 0)
            self.parts.append(g)
        self.fraction = roster_mean(flags[:, None].float(), counts)
        self.size = (counts.float()/(counts+1))[:, None]

    def __call__(self):
        if self.fixed is not None:
            return self.fixed
        parts = []
        for i, v in enumerate(VIEWS):
            c = self.cls[:, i]
            result, _ = self.gate.readers[v](c, self.parts[i], self.cls[:, 0])
            # Retain the original CPU prefix-sum order and its exact gradient.
            parts.append(roster_mean(torch.cat((c, result), 1), self.counts,
                                     self.flags if i == 2 else None))
        return torch.cat((*parts, self.fraction, self.size), 1)
