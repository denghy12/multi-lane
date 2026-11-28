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
        self.ends = self.counts.cumsum(0)
        prefix = torch.cat((torch.zeros(1), self.flags.float().cumsum(0)))
        valid_counts = (prefix[self.ends]-prefix[self.ends-self.counts]).clamp_min(1)
        width = 2*gate.width
        self.denominator = torch.cat((self.counts[:, None].expand(-1, 2*width),
                                      valid_counts[:, None].expand(-1, width)), 1)

    def __call__(self):
        if self.fixed is not None:
            return self.fixed
        parts = []
        for i, v in enumerate(VIEWS):
            c = self.cls[:, i]
            result, _ = self.gate.readers[v](c, self.parts[i], self.cls[:, 0])
            features = torch.cat((c, result), 1)
            if i == 2:
                features = features.masked_fill(~self.flags.to(features.device)[:, None], 0)
            parts.append(features)
        # Same independent prefix sum per column, one transfer for all three views.
        value = torch.cat(parts, 1).cpu()
        prefix = torch.cat((value.new_zeros((1, value.shape[1])), value.cumsum(0)), 0)
        pooled = ((prefix[self.ends]-prefix[self.ends-self.counts])/self.denominator).to(self.cls.device)
        return torch.cat((pooled, self.fraction, self.size), 1)
