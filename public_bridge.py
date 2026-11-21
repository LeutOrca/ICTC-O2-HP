"""Source-preserving public O2-HP adapter for MACE's first scalar TP only.

The angular block is embedded in canonical integer O(2) blocks. All unused
couplings have zero external weights. This is a padded scalar-times-angular
identity, not a general O(3)->O(2) MACE convolution converter.
"""
import torch
from ictc_o2_hp import UvuTensorProduct


class PublicScalarTP(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        if (str(reference.irreps_in1) != "128x0e" or reference.weight_numel != 512
                or reference.irreps_in2.dim != 16 or reference.irreps_out.dim != 2048
                or len(reference.instructions) != 4):
            raise ValueError("Public bridge supports only frozen first scalar TP")
        self.products = torch.nn.ModuleList([
            UvuTensorProduct(128, 1, degree, parity="canonical",
                            shared_weights=False, internal_weights=False)
            for degree in range(4)])
        self.weight_numel = reference.weight_numel
        self.irreps_in1, self.irreps_in2 = reference.irreps_in1, reference.irreps_in2
        self.irreps_out = reference.irreps_out
        self.selected = []
        for tp in self.products:
            selected = []
            for p, (i, j, k) in enumerate(tp.paths):
                if i == 0 and j == k:
                    coeff = tp._coefficients[p, :tp.entries_out[k].dim, 0,
                                             :tp.entries_in2[j].dim]
                    if not torch.equal(coeff, torch.eye(coeff.shape[0], dtype=coeff.dtype)):
                        raise ValueError("Selected scalar CG is not the identity")
                    selected.append(p)
            self.selected.append(tuple(selected))

    def forward(self, x, angular, weight):
        edges, channels = x.shape
        outputs = []
        for degree, (tp, selected) in enumerate(zip(self.products, self.selected)):
            width = 2 * degree + 1
            left = torch.cat((x, x.new_zeros(edges, channels * (width - 1))), dim=1)
            start = degree * degree
            h = angular[:, start:start + width]
            # Real e3nn order -l..l -> scalar, then each real two-dimensional pair.
            permutation = [degree] + [j for m in range(1, degree + 1)
                                      for j in (degree - m, degree + m)]
            right = torch.cat([h[:, j:j + 1] for j in permutation], dim=1)
            radial = weight[:, degree * channels:(degree + 1) * channels]
            zero = torch.zeros_like(radial)
            expanded = torch.cat([radial if p in selected else zero
                                  for p in range(len(tp.paths))], dim=1)
            raw = tp(left, right, expanded)
            blocks, offset = [], 0
            for entry in tp.entries_out:
                block = raw[:, offset:offset + channels * entry.dim]
                blocks.extend(block.reshape(edges, channels, entry.dim).unbind(-1))
                offset += channels * entry.dim
            inverse = [permutation.index(j) for j in range(width)]
            outputs.append(torch.stack([blocks[j] for j in inverse], dim=-1).flatten(1))
        return torch.cat(outputs, dim=1)


def install(model):
    model.interactions[0].conv_tp = PublicScalarTP(model.interactions[0].conv_tp).to(
        device=next(model.parameters()).device, dtype=torch.float32)
    return model
