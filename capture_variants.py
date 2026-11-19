"""CUDA Graph deployment experiment for shape-stable, strict-FP32 E/F."""
import torch


class Replay:
    def __init__(self, evaluate, model, example):
        self.inputs = {
            name: value.detach().clone() for name, value in example.items() if name != "_selector_groups"
        }
        if "_selector_groups" in example:
            self.inputs["_selector_groups"] = tuple(
                (z, indices.clone() if indices is not None else None)
                for z, indices in example["_selector_groups"]
            )
        self.graph = torch.cuda.CUDAGraph()
        worker = torch.cuda.Stream()
        current = torch.cuda.current_stream()
        worker.wait_stream(current)
        with torch.cuda.stream(worker):
            for _ in range(3):
                evaluate(model, self.inputs)
        current.wait_stream(worker)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, stream=worker):
            self.outputs = evaluate(model, self.inputs)

    def call(self, data):
        with torch.no_grad():
            for name, value in data.items():
                buf = self.inputs[name]
                if name == "_selector_groups":
                    self.copy_groups(buf, value)
                else:
                    if value.shape != buf.shape or value.dtype != buf.dtype:
                        raise ValueError(f"Graph input {name} requires another shape capture")
                    buf.detach().copy_(value)
            self.graph.replay()
            return {name: tensor.clone() for name, tensor in self.outputs.items()}

    @staticmethod
    def copy_groups(buffers, inputs):
        if len(buffers) != len(inputs):
            raise ValueError("This species composition requires another capture")
        for (old_z, old_index), (new_z, new_index) in zip(buffers, inputs):
            if (old_z != new_z or (old_index is None) != (new_index is None)
                    or (old_index is not None and old_index.shape != new_index.shape)):
                raise ValueError("This species composition requires another capture")
            if old_index is not None:
                old_index.copy_(new_index)


def install(bench, validate_live_input=False):
    eager_model = bench.model_for
    eager_evaluate = bench.evaluate

    def model_for(backend, device):
        if backend in ("native-graph", "hp-uvu-graph", "hp-uvu-species-graph", "hp-uvu-compact-graph"):
            model = eager_model(backend.removesuffix("-graph"), device)
            model._experimental_cuda_graph = True
            return model
        return eager_model(backend, device)

    def evaluate(model, inputs):
        if not getattr(model, "_experimental_cuda_graph", False):
            return eager_evaluate(model, inputs)
        if not hasattr(model, "_graph_execution"):
            capture = Replay(eager_evaluate, model, inputs)
            model._graph_execution = capture
            if validate_live_input:
                moved = inputs["positions"].detach().clone()
                moved[0, 0] += 0.005
                perturbed = dict(inputs, positions=moved)
                baseline = eager_evaluate(model, perturbed)
                errors = bench.errors(capture.call(perturbed), baseline)
                print("captured live input check", errors, flush=True)
                if not all(metric["passed"] for metric in errors.values()):
                    raise ValueError("Graph replay failed strict FP32 on an updated position buffer")
        return model._graph_execution.call(inputs)

    bench.model_for = model_for
    bench.evaluate = evaluate
