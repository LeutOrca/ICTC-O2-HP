"""Run independent native matscipy graph builds concurrently while keeping batch order."""
import atexit
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from mace.data.neighborhood import get_neighborhood


_pool = None


def one_graph(spec):
    positions, cell, cutoff = spec
    return get_neighborhood(positions, cutoff=cutoff, pbc=(True, True, True), cell=cell.copy())


def neighborhoods(geometry, cutoff):
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
        atexit.register(_pool.shutdown)
    return list(_pool.map(one_graph, ((x, cell, cutoff) for x, _, cell in geometry)))
