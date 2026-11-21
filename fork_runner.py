"""Independent CUDA-clean forkserver children, avoiding repeated package I/O."""
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import traceback

mp.set_forkserver_preload(["fork_prelude"])
CONTEXT=mp.get_context("forkserver")


def execute(command,log):
    os.setsid()
    with Path(log).open("w",buffering=1) as stream:
        os.dup2(stream.fileno(),1)
        os.dup2(stream.fileno(),2)
        sys.stdout=stream
        sys.stderr=stream
        try:
            import torch
            if torch.cuda._is_in_bad_fork() or torch.cuda.is_initialized():
                raise RuntimeError("Measurement child did not inherit a CUDA-clean process")
            import benchmark
            sys.argv=command[command.index(str(Path(benchmark.__file__).resolve())):]
            benchmark.main()
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            raise SystemExit(2)


def run(command,log,timeout):
    proc=CONTEXT.Process(target=execute,args=(command,str(log)))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        try:
            os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.join()
        return 124
    return proc.exitcode
