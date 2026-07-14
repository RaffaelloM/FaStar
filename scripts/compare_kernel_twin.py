#!/usr/bin/env python3
"""Validate fst_ssm_scan_kernel.cc's recurrence math vs the bit-correct numpy/transformers
reference.  Builds the scalar twin (no aie_api), feeds identical random fp32 inputs to both,
diffs S_out and y.  Run from repo root."""
import subprocess, sys, numpy as np
sys.path.insert(0, "scripts")
from qwopus_ssm_ref import gdn_delta_rule, _l2norm_np, SCALE, HEAD_K, HEAD_V

TWIN = "scripts/qwopus_ssm_kernel_twin.cc"
BIN  = "/tmp/qwopus_ssm_kernel_twin"

subprocess.run(["g++", "-O2", "-std=c++17", TWIN, "-o", BIN], check=True)

rng = np.random.default_rng(7)
S0 = np.zeros((HEAD_K, HEAD_V), np.float32)
q = rng.standard_normal(HEAD_K).astype(np.float32)
k = rng.standard_normal(HEAD_K).astype(np.float32)
v = rng.standard_normal(HEAD_V).astype(np.float32)
g_logit, beta = -1.3, 0.37

# pre-norm/scale/exp UPSTREAM (as ew_unified would) -> the kernel's pure delta-rule inputs
qn = (_l2norm_np(q) * SCALE).astype(np.float32)
kn = _l2norm_np(k).astype(np.float32)
gdec = float(np.exp(g_logit))

# numpy reference = pure delta rule (same contract as the AIE kernel)
y_np, S_np = gdn_delta_rule(qn, kn, v, gdec, beta, S0.copy())

# twin: in = [S_in(HK*HV) | qn(HK) | kn(HK) | v(HV) | gdec(1) | beta(1)]
inp = np.concatenate([S0.reshape(-1), qn, kn, v, [gdec], [beta]]).astype(np.float32)
out = subprocess.run([BIN], input="\n".join(f"{x:.9g}" for x in inp),
                     capture_output=True, text=True, check=True).stdout.split()
S_t = np.array(out[:HEAD_K*HEAD_V], np.float32).reshape(HEAD_K, HEAD_V)
y_t = np.array(out[HEAD_K*HEAD_V:], np.float32)

# the kernel applies l2norm+scale+exp then the delta rule; gdn_scan_step_numpy does the same.
dy = float(np.max(np.abs(y_np - y_t)))
dS = float(np.max(np.abs(S_np - S_t)))
ok = dy < 1e-4 and dS < 1e-4
print(f"kernel-twin vs numpy-ref: max|Δy|={dy:.2e} max|ΔS|={dS:.2e} -> "
      f"{'OK (kernel math matches reference)' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)