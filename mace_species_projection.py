"""Replace only constant one-hot species projections in native TP TorchScript."""
import itertools

import torch
import triton
import triton.language as tl

_LIBRARY = None
_IDENTITY = itertools.count()


@triton.jit
def _species(Attrs, Species, Scale, WIDTH: tl.constexpr, ROW: tl.constexpr, COL: tl.constexpr,
             BLOCK: tl.constexpr):
    node = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    value = tl.load(Attrs + node * ROW + column * COL, column < WIDTH, other=0.)
    tl.device_assert(tl.sum((value != 0.).to(tl.int32), 0) <= 1,
                     "Species projection requires one-hot attributes")
    which = tl.argmax(value, 0)
    tl.store(Species + node, which)
    tl.store(Scale + node, tl.max(value, 0))


@triton.jit
def _project(Weight, Species, Scale, Output, ELEMENTS: tl.constexpr,
             IN: tl.constexpr, Z: tl.constexpr, OUT: tl.constexpr,
             S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, TILE: tl.constexpr):
    item = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    node, offset = item // (IN * OUT), item % (IN * OUT)
    left, right = offset // OUT, offset % OUT
    valid = item < ELEMENTS
    species = tl.load(Species + node, valid, other=0)
    scale = tl.load(Scale + node, valid, other=0.)
    weight = tl.load(Weight + left * S0 + species * S1 + right * S2, valid, other=0.)
    tl.store(Output + item, weight * scale, valid)


def project(attrs, weight):
    if (attrs.requires_grad or weight.requires_grad or attrs.dtype != torch.float32
            or weight.dtype != torch.float32 or not attrs.is_cuda or not weight.is_cuda
            or attrs.ndim != 2 or weight.ndim != 3 or attrs.shape[1] != weight.shape[1]):
        raise ValueError("One-hot projection requires constant CUDA FP32 attributes and weights")
    nodes, width = attrs.shape
    result = attrs.new_empty((nodes, weight.shape[0], weight.shape[2]))
    if nodes:
        species = torch.empty(nodes, device=attrs.device, dtype=torch.int32)
        scale = attrs.new_empty(nodes)
        _species[(nodes,)](attrs, species, scale, width, *attrs.stride(),
                           triton.next_power_of_2(width), num_warps=4, debug=True)
        _project[(triton.cdiv(result.numel(), 256),)](
            weight, species, scale, result, result.numel(), *weight.shape, *weight.stride(),
            256, num_warps=4, enable_fp_fusion=False,
        )
    return result


def register():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = torch.library.Library("hpx_mace", "DEF")
        _LIBRARY.define("onehot_projection(Tensor attrs, Tensor weight) -> Tensor")
        _LIBRARY.impl("onehot_projection", project, "CUDA")


def constant(value):
    direct = value.toIValue()
    if direct is not None:
        return direct
    node = value.node()
    if node.kind() in ("prim::ListConstruct", "prim::TupleConstruct"):
        return [constant(item) for item in node.inputs()]
    return None


class ProjectedGraph(torch.nn.Module):
    def __init__(self, original, function, keep_self):
        super().__init__()
        self.original, self.function, self.keep_self = original, function, keep_self

    def forward(self, x1, x2, weight):
        if self.keep_self:
            return self.function(self.original, x1, x2, weight)
        return self.function(x1, x2, weight)


def install(model):
    register()
    changed = []
    for index, block in enumerate(model.interactions):
        product = block.skip_tp
        original = product._compiled_main_left_right
        graph = original.forward.graph.copy()
        count = 0
        for node in list(graph.nodes()):
            if (node.kind() != "prim::CallFunction" or "tensordot" not in str(node.inputsAt(0).type())
                    or constant(node.inputsAt(3)) != [[1], [1]]):
                continue
            replacement = graph.create("hpx_mace::onehot_projection",
                                       [node.inputsAt(1), node.inputsAt(2)], 1)
            replacement.output().setType(node.output().type())
            replacement.insertBefore(node)
            node.output().replaceAllUsesWith(replacement.output())
            node.destroy()
            count += 1
        if count:
            keep_self = bool(list(next(graph.inputs()).uses()))
            if not keep_self:
                graph.eraseInput(0)
            torch._C._jit_pass_lint(graph)
            function = torch._C._create_function_from_graph(
                f"hpx_species_projected_{index}_{next(_IDENTITY)}", graph)
            product._compiled_main_left_right = ProjectedGraph(original, function, keep_self)
            changed.append(dict(interaction=index, projections=count))
    if not changed:
        raise RuntimeError("No native one-hot projection nodes matched the frozen TP graph")
    return changed
