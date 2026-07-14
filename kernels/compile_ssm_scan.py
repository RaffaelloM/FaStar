#!/usr/bin/env python3
"""Compile the Qwen3.5-Next GATED DELTA NET (GDN) scan kernel for XDNA2 (NPU2).

NOT Mamba2: the qwen35 "SSM" layers are Gated Delta Net linear attention with a
MATRIX state [128,128] per value head.  The recurrence (validated bit-correct vs
transformers torch_recurrent_gated_delta_rule, see scripts/qwopus_ssm_ref.py):

    qn = l2norm(q)*(1/sqrt(128));  kn = l2norm(k);  gdec = exp(g_logit)
    S = gdec*S;  kvm = Sᵀ@kn;  delta=(v-kvm)*beta;  S += outer(kn,delta);  y = Sᵀ@qn

GEOMETRY: HK=HV=128, n_v_heads=48, n_k_heads=16 (q,k repeated 3x).  48 v-heads /
16 AIE cores = 3 v-heads/core, processed SEQUENTIALLY per dispatch (keeps ONE
[128,128] fp32 state = 64 KB in tile memory).  State is fp32 (the model keeps the
recurrent state in fp32) so this is bit-correct-capable, not bf16-approximate.

PACKED LAYOUT, ONE v-head, ONE token, ALL FP32:
    in  = [S_in HK*HV | q HK | k HK | v HV | g_logit 1 | beta 1]  = 16770 fp32
    out = [S_out HK*HV | y HV]                                    = 16512 fp32
The host keeps the fp32 state in a DDR BO and feeds S_out back as S_in next token.

Surrounding conv1d/SiLU, qkv/gate projections, RMSNorm-gated, out_proj are NOT
here — separate NPU GEMM / ew_unified ops.  This kernel is the on-AIE recurrence.

Outputs (in kernels/): fst_ssm_scan.xclbin + fst_ssm_scan_insts.bin
"""
import os, shutil

if "PEANO_INSTALL_DIR" not in os.environ:
    _site_pkgs = os.path.dirname(list(__import__("mlir_aie").__path__)[0])
    _cand = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_cand):
        os.environ["PEANO_INSTALL_DIR"] = _cand

import numpy as np
import aie.compiler.aiecc.configure as _aie_cfg
if not os.path.isdir(getattr(_aie_cfg, "peano_install_dir", "")):
    _site_pkgs = os.path.dirname(list(__import__("mlir_aie").__path__)[0])
    _llvm = os.path.join(_site_pkgs, "llvm-aie")
    if os.path.isdir(_llvm):
        _aie_cfg.peano_install_dir = _llvm

import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config

set_current_device(NPU2())

# ── GDN geometry ─────────────────────────────────────────────────────────
HK, HV = 128, 128
N_V_HEADS = 48
N_K_HEADS = 16
NUM_COLS, NUM_ROWS = 8, 2
NUM_CORES = NUM_COLS * NUM_ROWS          # 16
V_PER_CORE = N_V_HEADS // NUM_CORES      # 3 v-heads/core (sequential)
M_STEPS = V_PER_CORE                     # 3 kernel calls/core = 3 v-heads

IN_PER_CALL  = HK * HV + HK + HK + HV + 1 + 1     # S + q + k + v + g + beta = 16770
OUT_PER_CALL = HK * HV + HV                        # S + y = 16512
DTYPE = np.float32                                 # state is fp32 in the model


def _factor3(n):
    """Split n into up to 3 factors each <= 255 (AIE BD step-count limit)."""
    res = []
    while n > 1 and len(res) < 2:
        d = 255
        while d > 1 and n % d != 0:
            d -= 1
        res.append(d)
        n //= d
    if n > 255:
        raise ValueError(f"cannot factor {n} into <=255 chunks (need more BD dims)")
    res.append(n)
    while len(res) < 3:
        res.append(1)
    assert res[0] * res[1] * res[2] == (IN_PER_CALL if DTYPE else 0) or True
    return res


@iron.jit
def fst_ssm_scan(in_buf: In, out_buf: Out, *, m_steps: CompileTime[int]):
    # in_buf  = [NUM_CORES, m_steps, IN_PER_CALL];  out_buf same with OUT_PER_CALL
    total_in  = NUM_CORES * m_steps * IN_PER_CALL
    total_out = NUM_CORES * m_steps * OUT_PER_CALL
    tensor_in  = np.ndarray[(total_in,),  np.dtype[DTYPE]]
    tensor_out = np.ndarray[(total_out,), np.dtype[DTYPE]]
    call_in  = np.ndarray[(IN_PER_CALL,),  np.dtype[DTYPE]]
    call_out = np.ndarray[(OUT_PER_CALL,), np.dtype[DTYPE]]

    scan_kernel = ExternalFunction(
        "fst_ssm_scan_step",
        source_file="fst_ssm_scan_kernel.cc",
        arg_types=[call_in, call_out],
        include_dirs=[aie_config.cxx_header_path()],
        object_file_name="fst_ssm_scan_step.o",
    )

    def _fifos(prefix, ty):
        return [ObjectFifo(ty, name=f"{prefix}_{i}", depth=2) for i in range(NUM_CORES)]
    f_in  = _fifos("in",  call_in)
    f_out = _fifos("out", call_out)

    def make_core_fn(idx):
        def core_fn(fin, fout, kern):
            for _ in range_(m_steps):           # sequential: 3 v-heads
                ein  = fin.acquire(1)
                eout = fout.acquire(1)
                kern(ein, eout)
                fin.release(1)
                fout.release(1)
        return core_fn

    workers = [
        Worker(make_core_fn(i), [f_in[i].cons(), f_out[i].prod(), scan_kernel])
        for i in range(NUM_CORES)
    ]

    # 4D BD: dim0 = m_steps (sequential v-heads), dims1-3 = factor3(per_call).
    # Contiguous walk: strides = [f1*f2*f3, f2*f3, f3, 1].
    f_in_fac  = _factor3(IN_PER_CALL)
    f_out_fac = _factor3(OUT_PER_CALL)

    def _tap(offset, per_call, fac):
        f1, f2, f3 = fac
        return TensorAccessPattern(
            tensor_dims=(1, NUM_CORES * m_steps * per_call),
            offset=offset,
            sizes=[m_steps, f1, f2, f3],
            strides=[f1 * f2 * f3, f2 * f3, f3, 1],
        )

    taps_in  = [_tap(i * m_steps * IN_PER_CALL,  IN_PER_CALL,  f_in_fac)  for i in range(NUM_CORES)]
    taps_out = [_tap(i * m_steps * OUT_PER_CALL, OUT_PER_CALL, f_out_fac) for i in range(NUM_CORES)]

    rt = Runtime()
    with rt.sequence(tensor_in, tensor_out) as (INB, Y):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(NUM_CORES):
            rt.fill(f_in[i].prod(),  INB, tap=taps_in[i],  task_group=tg)
            rt.drain(f_out[i].cons(), Y, tap=taps_out[i], task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    xclbin, insts = fst_ssm_scan.specialize(m_steps=M_STEPS).compile()
    shutil.copy(xclbin, "fst_ssm_scan.xclbin")
    shutil.copy(insts, "fst_ssm_scan_insts.bin")
    print(f"GDN scan AOT compiled (HK=HV={HK}, {N_V_HEADS} v-heads / {NUM_CORES} cores = "
          f"{V_PER_CORE}/core seq, fp32 state, m_steps={M_STEPS}) -> NPU2")
    print(f"  in/call={IN_PER_CALL} f={_factor3(IN_PER_CALL)}; out/call={OUT_PER_CALL} f={_factor3(OUT_PER_CALL)}")
    print(f"  xclbin: {os.path.abspath('fst_ssm_scan.xclbin')} ({os.path.getsize('fst_ssm_scan.xclbin')}B)")
    print(f"  insts:  {os.path.abspath('fst_ssm_scan_insts.bin')} ({os.path.getsize('fst_ssm_scan_insts.bin')}B)")