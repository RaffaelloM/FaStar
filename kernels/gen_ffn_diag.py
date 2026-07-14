#!/usr/bin/env python3
"""Diagnostic: is a pure-uint8 ObjectFifo packet correctly DMA'd + scalar-readable,
and does it stay fast+correct below vs above the 16 KB / 4096-word thresholds?
Emits fst_ffn_diag.xclbin with a 1-tile kernel that reads bytes at offsets
0, 4096, mid, last of a PKT_BYTES packet and writes them to output."""
import os,sys,shutil
from pathlib import Path
PROJ_ROOT=Path(os.path.dirname(os.path.abspath(__file__)))
IRON_PATH=str(PROJ_ROOT.parent/"Source"/"IRON-devel")
if IRON_PATH not in sys.path: sys.path.insert(0,IRON_PATH)
os.environ.setdefault("PATH",os.environ["PATH"]+":"+os.path.expanduser("~/.local/bin"))
os.environ.setdefault("PEANO_INSTALL_DIR",os.path.expanduser("~/.local/lib/python3.14/site-packages/llvm-aie"))
import numpy as np, aie.iron as iron
from aie.iron import In,Out,ObjectFifo,Program,Runtime,Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
from aie.utils import set_current_device
from aie.utils import config as aie_config
set_current_device(NPU2())
PKT_BYTES=int(os.environ.get("DIAG_PKT","8192"))   # 8192 (small) or 26240 (h-sized)
SRC=str(PROJ_ROOT/"fst_ffn_diag_kernel.cc"); INC=[aie_config.cxx_header_path()]
PKT_T=np.ndarray[(PKT_BYTES,),np.dtype[np.uint8]]
OUT_T=np.ndarray[(8,),np.dtype[np.float32]]
@iron.jit
def diag_op(inp:In,out:Out):
    k=ExternalFunction("diag_read",source_file=SRC,arg_types=[PKT_T,OUT_T,np.int32],include_dirs=INC,object_file_name="fst_ffn_diag.o")
    f_in=ObjectFifo(PKT_T,name="in",depth=2)
    f_out=ObjectFifo(OUT_T,name="out",depth=1)
    def core(fin,fo,k):
        o=fo.acquire(1)
        for _ in range_(16):
            p=fin.acquire(1); k(p,o,PKT_BYTES); fin.release(1)
        fo.release(1)
    w=Worker(core,[f_in.cons(),f_out.prod(),k])
    rt=Runtime()
    with rt.sequence(np.ndarray[(16*PKT_BYTES,),np.dtype[np.uint8]],np.ndarray[(8,),np.dtype[np.float32]]) as (inp,out):
        rt.start(w); tg=rt.task_group()
        rt.fill(f_in.prod(),inp,tap=TensorAccessPattern(tensor_dims=(1,16*PKT_BYTES),offset=0,sizes=[16,PKT_BYTES],strides=[PKT_BYTES,1]),task_group=tg)
        rt.drain(f_out.cons(),out,tap=TensorAccessPattern(tensor_dims=(8,),offset=0,sizes=[1,8],strides=[0,1]),task_group=tg,wait=True)
        rt.finish_task_group(tg)
    return Program(NPU2(),rt).resolve_program()
x,i=diag_op.compile(); shutil.copy(x,"fst_ffn_diag.xclbin"); shutil.copy(i,"fst_ffn_diag_insts.bin")
print(f"OK diag pkt={PKT_BYTES}B")
