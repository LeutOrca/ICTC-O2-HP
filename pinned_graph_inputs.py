"""Install rebuilt CPU graph inputs directly into pinned CUDA Graph buffers."""
import torch

from capture_both_variants import Replay


class PinnedInputReplay(Replay):
    def __init__(self, evaluate, model, example):
        if any(not isinstance(value, torch.Tensor) for value in example.values()):
            raise ValueError("Pinned graph inputs require tensor-only inputs")
        self.inputs = {}
        self.host_views = {}
        self.packs = []
        self.copy_finished = torch.cuda.Event()
        self.host_copy_in_flight = False
        for dtype in dict.fromkeys(value.dtype for value in example.values()):
            fields = [(name, value) for name, value in example.items() if value.dtype == dtype]
            alignment = 256 // fields[0][1].element_size()
            sizes = [((value.numel() + alignment - 1) // alignment) * alignment
                     for _, value in fields]
            pinned = torch.zeros(sum(sizes), dtype=dtype, pin_memory=True)
            device = torch.empty(sum(sizes), dtype=dtype, device="cuda")
            for (name, value), host_view, device_view in zip(fields, pinned.split(sizes), device.split(sizes)):
                self.host_views[name] = host_view[:value.numel()].view_as(value)
                self.inputs[name] = device_view[:value.numel()].view_as(value)
                self.inputs[name].copy_(value)
            self.packs.append((pinned, device))
        self.capture(evaluate, model)

    def call(self, data):
        if data["positions"].is_cuda:
            return super().call(data)
        if data.keys() != self.host_views.keys():
            raise ValueError("Input fields differ from the captured graph")
        if self.host_copy_in_flight:
            self.copy_finished.synchronize()
        with torch.no_grad(), torch.profiler.record_function("mace_pinned_graph_input_transfer"):
            for name, value in data.items():
                view = self.host_views[name]
                if value.device.type != "cpu" or value.shape != view.shape or value.dtype != view.dtype:
                    raise ValueError(f"Graph input {name} requires another shape capture")
                view.copy_(value)
            for host, device in self.packs:
                device.copy_(host, non_blocking=True)
            self.copy_finished.record()
            self.host_copy_in_flight = True
        self.graph.replay()
        return {name: tensor.clone() for name, tensor in self.outputs.items()}
