"""Exercise cached CUDA launches on worker threads with no current context."""
import ctypes
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from benchmark import configure, error
from models import ROOT
from cuda_matrix import prepare, compiled_matrix, _checked


def main():
    configure()
    torch.manual_seed(20261121)
    x = torch.randn(64,64,device="cuda") * .1
    weight = torch.randn(64,64,device="cuda") * .1
    expected = x @ weight
    driver, _, _, _, _ = prepare()
    driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    output = torch.empty_like(expected)
    torch.cuda.synchronize()
    def worker(_):
        _checked(driver.cuCtxSetCurrent(None), "probe clear context")
        current = ctypes.c_void_p()
        _checked(driver.cuCtxGetCurrent(ctypes.byref(current)), "probe get context")
        assert not current.value
        compiled_matrix(x,weight,None,1.,0.,output=output)
        return True
    with ThreadPoolExecutor(max_workers=1) as pool:
        checks = list(pool.map(worker,range(20)))
    torch.cuda.synchronize()
    numerical = error(output,expected)
    if not numerical["passed"] or not all(checks):
        raise RuntimeError(numerical)
    (ROOT/"results/CUDA-CONTEXT-PROBE.json").write_text(json.dumps(dict(
        passed=True,threads=1,launches=20,forced_null_context=True,checks=numerical),indent=2))
    print("Cached CUDA worker-context launches: pass",flush=True)


if __name__ == "__main__":
    main()
