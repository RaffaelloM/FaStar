#!/usr/bin/env python3
"""Compile a single MLA kvc kernel with N=64 (N_div_n=1) for the GEMM probe.

N_div_n=1 means the A fill TAP's N_div_n dim has size 1 — the stride-0 is on
a size-1 dim (harmless: one iteration, stride irrelevant).  If this kernel
passes the probe (A=ones, B=identity -> C all 1.0) while the N=512 kvc
(N_div_n=8, stride-0 on size=8) fails, the stride-0-on-size>1 is confirmed
as the root cause.
"""
import os, sys, shutil, time
import numpy as np
from ml_dtypes import bfloat16
os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source", "IRON-devel"))
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device, config
from pathlib import Path
set_current_device(NPU2())
PROJ_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

TILE_M, TILE_K, TILE_N = 8, 64, 64

@iron.jit
def kvc64_op(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
             N: CompileTime[int], el: CompileTime[type]):
    m, k, n = TILE_M, TILE_K, TILE_N
    M_div_m, K_div_k, N_div_n = M // m, K // k, N // n
    mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el, output_dtype=el, vectorized=False)
    z = mm.zero
    fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="A", depth=2)
    fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="B", depth=2)
    fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="C", depth=2)
    def core(of_a, of_b, of_c, zero_k, mm_k):
        for _ in range_(M_div_m * N_div_n) if (M_div_m * N_div_n) > 1 else range(1):
            ec = of_c.acquire(1); zero_k(ec)
            for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                ea = of_a.acquire(1); eb = of_b.acquire(1); mm_k(ea, eb, ec)
                of_a.release(1); of_b.release(1)
            of_c.release(1)
    w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm], stack_size=0xD00)
    at = TensorAccessPattern((M, K), 0, [N_div_n, K_div_k, m, k], [0, k, K, 1])
    bt = TensorAccessPattern((K, N), 0, [N_div_n, K_div_k, k, n], [n, k*N, N, 1])
    ct = TensorAccessPattern((M, N), 0, [1, N_div_n, m, n], [m*N, n, N, 1])
    rt = Runtime()
    with rt.sequence(np.ndarray[(M,K), np.dtype[el]], np.ndarray[(K,N), np.dtype[el]], np.ndarray[(M,N), np.dtype[el]]) as (a,b,c):
        rt.start(w); tg = rt.task_group()
        rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
        rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
        rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(), rt).resolve_program()

if __name__ == "__main__":
    spec = kvc64_op.specialize(M=8, K=4096, N=64, el=bfloat16)
    xclbin, insts = spec.compile()
    shutil.copy2(xclbin, str(PROJ_ROOT / "fst_mla_kvc64.xclbin"))
    shutil.copy2(insts, str(PROJ_ROOT / "fst_mla_kvc64_insts.bin"))
    print(f"kvc64: 8x4096x64 (N_div_n=1) -> {os.path.getsize(str(PROJ_ROOT/'fst_mla_kvc64.xclbin'))}B xclbin + {os.path.getsize(str(PROJ_ROOT/'fst_mla_kvc64_insts.bin'))}B insts")