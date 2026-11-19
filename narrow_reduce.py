"""Deterministic receiver sum with one index per row of the message matrix."""
from contextlib import contextmanager

import torch
from mace.modules import blocks
from mace.tools.scatter import scatter_sum as original_sum


def index_sum(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if (dim == 0 and out is None and dim_size is not None and reduce == "sum"
            and src.ndim == 2 and index.ndim == 1 and src.size(0) == index.numel()):
        return src.new_zeros((dim_size, src.size(1))).index_add_(0, index, src)
    return original_sum(src, index, dim, out, dim_size, reduce)


@contextmanager
def replace_block_scatter():
    saved = blocks.scatter_sum
    blocks.scatter_sum = index_sum
    try:
        yield
    finally:
        blocks.scatter_sum = saved


def install(bench):
    base_model, evaluate, configure = bench.model_for, bench.evaluate, bench.configure

    def deterministic():
        configure()
        torch.use_deterministic_algorithms(True)

    def model_for(backend, device):
        if backend == "native-indexadd":
            model = base_model("native", device)
            model._deterministic_narrow_index = True
            return model
        return base_model(backend, device)

    def evaluate_with_index(model, data):
        if not getattr(model, "_deterministic_narrow_index", False):
            return evaluate(model, data)
        with replace_block_scatter():
            return evaluate(model, data)

    bench.configure = deterministic
    bench.model_for = model_for
    bench.evaluate = evaluate_with_index
