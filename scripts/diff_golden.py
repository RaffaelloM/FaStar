#!/usr/bin/env python3
"""diff_golden.py — extract decode tokens from a run log and diff vs golden."""
import sys, re
def decode_ids(path):
    ids={}
    for line in open(path):
        # match: decode=ID (pos=N ...)
        for m in re.finditer(r'decode=(\d+)\s*\(pos=(\d+)', line):
            ids[int(m.group(2))]=int(m.group(1))
    return ids
run=decode_ids(sys.argv[1])
gold={}
for line in open(sys.argv[2]):
    m=re.match(r'(\d+)\s+(\d+)', line)
    if m: gold[int(m.group(1))]=int(m.group(2))
nok=0; first_bad=None
for p in sorted(gold):
    if p in run and run[p]==gold[p]: nok+=1
    elif first_bad is None: first_bad=(p, gold.get(p), run.get(p))
print(f"golden positions: {len(gold)}, run decode positions: {len(run)}")
print(f"match: {nok}/{len(gold)}")
if first_bad: print(f"FIRST MISMATCH: pos{first_bad[0]} golden={first_bad[1]} run={first_bad[2]}")
else: print("ALL MATCH — fused == 3-pass bit-identical")
