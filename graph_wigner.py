"""Keep official Wigner values and masked derivatives without dynamic row shapes."""
import importlib
import math
from types import MethodType

from e3nn import o3
import torch


def install(rotation, device):
    module = importlib.import_module(rotation.__class__.__module__.rsplit(".", 1)[0] + ".wigner")
    # The official path converts these immutable constants on every invocation.
    module._Jd = [value.to(device=device, dtype=torch.float32) for value in module._Jd]
    axis = torch.tensor([0., 1., 0.], device=device)
    threshold = .999999

    def to_wigner(layer, edge_rotation, start_lmax, end_lmax):
        x = edge_rotation[:, :, 1]
        alpha, beta = o3.xyz_to_angles(x)
        matrix = o3.angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2)
        matrix = torch.bmm(matrix, edge_rotation)
        gamma = torch.atan2(matrix[..., 0, 2], matrix[..., 0, 0])
        if layer.use_rotation_mask:
            yprod = (x @ axis).detach()
            live = (yprod > -threshold) & (yprod < threshold)
            alpha = torch.where(live, alpha, alpha.detach())
            gamma = torch.where(live, gamma, gamma.detach())
            detached_beta = torch.where(yprod > threshold, 0., beta.detach())
            detached_beta = torch.where(yprod < -threshold, math.pi, detached_beta)
            beta = torch.where(live, beta, detached_beta)
        size = (end_lmax + 1) ** 2 - start_lmax ** 2
        result = torch.zeros(len(alpha), size, size, device=edge_rotation.device)
        offset = 0
        for degree in range(start_lmax, end_lmax + 1):
            block = module.wigner_D(degree, alpha, beta, gamma)
            end = offset + block.shape[1]
            result[:, offset:end, offset:end] = block
            offset = end
        return result if layer.use_rotation_mask else result.detach()

    rotation._rotation_to_wigner_matrix = MethodType(to_wigner, rotation)
