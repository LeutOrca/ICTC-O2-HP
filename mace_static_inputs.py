"""Reuse composition fields while rebuilding every native periodic neighbor list."""
import numpy as np
import torch


class CompositionInputs:
    def __init__(self, geometry, model, example, native_neighbors):
        self.atoms = tuple(np.array(row[1], copy=True) for row in geometry)
        self.cutoff = model.r_max_cpu
        self.neighbors = native_neighbors
        self.constant = {name: example[name] for name in ("node_attrs", "batch", "ptr", "head")}
        self.device = example["positions"].device
        self.calls = 0

    def __call__(self, geometry):
        if len(geometry) != len(self.atoms) or any(
                not np.array_equal(atoms, fixed) for (_, atoms, _), fixed in zip(geometry, self.atoms)):
            raise ValueError("Composition changed; rebuild the composition cache and CUDA Graph")
        edge_indexes, shifts, unit_shifts, cells, positions = ([] for _ in range(5))
        offset = 0
        for position, atoms, cell in geometry:
            edge, shift, unit, cell = self.neighbors(
                position, cutoff=self.cutoff, pbc=(True, True, True), cell=cell.copy())
            edge_indexes.append(edge + offset)
            shifts.append(shift)
            unit_shifts.append(unit)
            positions.append(position)
            cells.append(cell)
            offset += len(atoms)
        dynamic = {
            "edge_index": np.concatenate(edge_indexes, axis=1).astype(np.int64, copy=False),
            "shifts": np.concatenate(shifts).astype(np.float32),
            "unit_shifts": np.concatenate(unit_shifts).astype(np.float32),
            "positions": np.concatenate(positions).astype(np.float32),
            "cell": np.concatenate(cells, axis=0).astype(np.float32),
        }
        self.calls += 1
        return dict(self.constant, **{name: torch.from_numpy(value).to(self.device)
                                     for name, value in dynamic.items()})

    def receipt(self):
        return dict(calls=self.calls, cached_fields=list(self.constant),
                    constant_gpu_bytes=sum(value.numel() * value.element_size()
                                           for value in self.constant.values()),
                    composition_host_bytes=sum(value.nbytes for value in self.atoms),
                    neighbor_cache=False)
