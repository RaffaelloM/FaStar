#!/usr/bin/env python3
"""Compile scalar shim-DMA GEMM xclbins for MLA operations.
Uses the same design as fst_expert_gemm.py (scalar MMUL, shim-DMA only, no mem-tile).
"""
import os, sys, time
import numpy as np
from ml_dtypes import bfloat16

os.environ.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
os.environ.setdefault("PEANO_INSTALL_DIR", os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
set_current_device(NPU2())

TILE_M = 8   # smaller M for MLA
TILE_K = 64
TILE_N = 64

def make_scalar_gemm(M_fix, K_fix, N_fix):
    @iron.jit
    def gemm(A: In, B: In, C: Out, *, M: CompileTime[int], K: CompileTime[int],
             N: CompileTime[int], el: CompileTime[type]):
        m, k, n = TILE_M, TILE_K, TILE_N
        mm = kernels.mm(dim_m=m, dim_k=k, dim_n=n, input_dtype=el, output_dtype=el, vectorized=False)
        z = mm.zero

        fifo_A = ObjectFifo(np.ndarray[(m*k,), np.dtype[el]], name="Ain", depth=2)
        fifo_B = ObjectFifo(np.ndarray[(k*n,), np.dtype[el]], name="Bin", depth=2)
        fifo_C = ObjectFifo(np.ndarray[(m*n,), np.dtype[el]], name="Cout", depth=2)

        def core(of_a, of_b, of_c, zero_k, mm_k):
            for _ in range_(M//m * N//n):
                ec = of_c.acquire(1); zero_k(ec)
                for _ in range_(K//k):
                    ea = of_a.acquire(1); eb = of_b.acquire(1)
                    mm_k(ea, eb, ec)
                    of_a.release(1); of_b.release(1)
                of_c.release(1)

        w = Worker(core, [fifo_A.cons(), fifo_B.cons(), fifo_C.prod(), z, mm])
        at = TensorAccessPattern((M,K),0,[N//n,K//k,m,k],[0,k,K,1])
        bt = TensorAccessPattern((K,N),0,[N//n,K//k,k,n],[n,k*N,N,1])
        ct = TensorAccessPattern((M,N),0,[1,N//n,m,n],[m*N,n,N,1])
        rt = Runtime()
        with rt.sequence(np.ndarray[(M,K), np.dtype[el]], np.ndarray[(K,N), np.dtype[el]],
                         np.ndarray[(M,N), np.dtype[el]]) as (a,b,c):
            rt.start(w); tg = rt.task_group()
            rt.fill(fifo_A.prod(), a, tap=at, task_group=tg)
            rt.fill(fifo_B.prod(), b, tap=bt, task_group=tg)
            rt.drain(fifo_C.cons(), c, tap=ct, task_group=tg, wait=True)
            rt.finish_task_group(tg)
        return Program(NPU2(), rt).resolve_program()
    return gemm.specialize(M=M_fix, K=K_fix, N=N_fix, el=bfloat16)

# Compile all MLA dimensions + expert
dims = [
    ("qc_shimdma",    8, 4096, 1024),
    ("qe_shimdma",    8, 1024, 4096),
    ("kvc_shimdma",   8, 4096, 512),
    ("qk_shimdma",    512, 512, 128),
    ("sv_shimdma",    512, 128, 512),
    ("oa_shimdma",    8, 4096, 4096),
    ("ob_shimdma",    8, 4096, 4096),
    ("gate_shimdma",  32, 4096, 2048),
    ("down_shimdma",  32, 2048, 4096),
]

for name, m, k, n in dims:
    t0 = time.perf_counter()
    spec = make_scalar_gemm(m, k, n)
    xclbin, insts = spec.compile()
    import shutil, os as _os
    target_x = f"/home/raffaele/Progetti/FaStar/{name}.xclbin"
    target_i = f"/home/raffaele/Progetti/FaStar/{name}_insts.bin"
    shutil.copy2(xclbin, target_x)
    shutil.copy2(insts, target_i)
    dt = time.perf_counter() - t0
    print(f"  {name}: {m}x{k}x{n} → {_os.path.getsize(target_x)}B xclbin + {_os.path.getsize(target_i)}B insts ({dt:.1f}s)", flush=True)

print("All scalar shim-DMA GEMM xclbins compiled.")
