"""Coalesce full-width slice adjoints for the native e3nn edge tensor products."""
from collections import defaultdict
import operator

from e3nn.o3._tensor_product._codegen import codegen_tensor_product_left_right
from e3nn.util import jit
import torch
from torch.fx import Node


def partition_slices(graph):
    slices = defaultdict(dict)
    for node in graph.nodes:
        if node.op != "call_function" or node.target is not operator.getitem:
            continue
        source, index = node.args
        if (not isinstance(source, Node) or not isinstance(index, tuple) or len(index) != 2
                or index[0] != slice(None) or not isinstance(index[1], slice)):
            continue
        section = index[1]
        if section.step is not None or not isinstance(section.stop, int):
            continue
        slices[source].setdefault((section.start or 0, section.stop), []).append(node)

    count = 0
    for source, groups in slices.items():
        sections = sorted(groups)
        if source.op != "call_method" or source.target != "reshape":
            continue
        width = source.args[-1]
        if (len(sections) < 2 or sections[0][0] != 0 or sections[-1][1] != width
                or any(a[1] != b[0] for a, b in zip(sections, sections[1:]))):
            continue
        if any(len(nodes) != 1 for nodes in groups.values()):
            continue
        with graph.inserting_after(source):
            split = graph.call_method("split", (source, [hi - lo for lo, hi in sections]), {"dim": 1})
        for index, section in enumerate(sections):
            original, = groups[section]
            with graph.inserting_before(original):
                replacement = graph.call_function(operator.getitem, (split, index))
            original.replace_all_uses_with(replacement)
            graph.erase_node(original)
        count += 1
    graph.lint()
    return count


def partition_edge_products(model):
    for block in model.interactions:
        product = block.conv_tp
        original = product._compiled_main_left_right
        compiled = codegen_tensor_product_left_right(
            product.irreps_in1, product.irreps_in2, product.irreps_out, product.instructions,
            product.shared_weights, product._specialized_code, product._optimize_einsums,
        )
        groups = partition_slices(compiled.graph)
        if groups == 0:
            raise RuntimeError("Generated tensor product has no disjoint full-width partitions")
        compiled.recompile()
        old_buffers = dict(original.named_buffers())
        if set(old_buffers) != set(dict(compiled.named_buffers())):
            raise RuntimeError("Regenerated e3nn tensor product has a different coefficient layout")
        for name, new in compiled.named_buffers():
            new.copy_(old_buffers[name])
        # The original checkpoint runs the generated e3nn operations as TorchScript.
        block.conv_tp._compiled_main_left_right = jit.compile(compiled, recurse=False).cuda()
        print(f"Partitioned {groups} differentiable slice inputs", flush=True)
