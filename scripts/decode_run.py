#!/usr/bin/env python3
"""decode_run.py — parse an engine run log for Q35 token ids and decode to text.

Reads a run log, extracts prefill-last + decode token ids, and decodes them with
the HF tokenizer (tokenizer.json).  Prints both per-token and the concatenated
visible text so coherence can be judged.
"""
import re, sys, os
from tokenizers import Tokenizer

TOK = sys.argv[2] if len(sys.argv) > 2 else None
if TOK is None:
    for c in ("tokenizer.json",
              "models/Qwopus3.6-27B-Coder-BF16/tokenizer.json"):
        if os.path.exists(c):
            TOK = c; break
if TOK is None or not os.path.exists(TOK):
    sys.exit("no tokenizer.json found")
t = Tokenizer.from_file(TOK)

log = sys.argv[1] if len(sys.argv) > 1 else "logs/run_bf16.txt"
txt = open(log).read()

ids = []
m = re.search(r"prefill[-_]last\s*=\s*(\d+)", txt)
if m:
    ids.append(("prefill-last", int(m.group(1))))
# Engine prints each decode token twice (with/without head=ms); dedup by pos=.
seen_pos = set()
for mm in re.finditer(r"decode\s*=\s*(\d+)\s*\(pos=(\d+)", txt):
    p = int(mm.group(2))
    if p in seen_pos:
        continue
    seen_pos.add(p)
    ids.append(("decode", int(mm.group(1))))

if not ids:
    for mm in re.finditer(r"decode\s*=\s*(\d+)", txt):
        ids.append(("decode", int(mm.group(1))))

tok_s = None
mts = re.search(r"Decode Tokens/sec[:\s]+([\d.]+)", txt)
if mts:
    tok_s = float(mts.group(1))

print(f"# {len(ids)} tokens from {log}  (tokenizer={TOK})" +
      (f"  tok/s={tok_s}" if tok_s is not None else ""))
order = []
for tag, tid in ids:
    s = t.decode([tid], skip_special_tokens=False)
    esc = s.encode('ascii', 'backslashreplace').decode('ascii')
    print(f"  {tag:12s} {tid:7d} -> {esc!r}")
    order.append(tid)

print("\n=== concatenated (skip_special_tokens=True) ===")
print(t.decode(order, skip_special_tokens=True))
print("\n=== concatenated (skip_special_tokens=False) ===")
print(t.decode(order, skip_special_tokens=False))