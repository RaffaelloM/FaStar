#!/usr/bin/env python3
"""fst_tokenize.py -- HuggingFace `tokenizers` bridge for the FaStar C++ engine.

The hand-rolled C++ BPETokenizer in fst_main.cpp cannot faithfully reproduce
DeepSeek's ByteLevel BPE pre-tokenizer (it needs the GPT-4-style Unicode regex
`\\p{L}/\\p{N}/...` which std::regex does not support, and it drops the space
that becomes the `Ġ` prefix the vocab keys on -- so words like `Ġtecnologia`
never match and the encoder falls back to id 0 = <BOS>, corrupting the prompt
and steering the model into Chinese).  Encoding is a once-per-prompt op, so we
shell out to the real `tokenizers` library here.

Usage:
    FST_PROMPT="<text>" python3 fst_tokenize.py <tokenizer.json>
    -> prints the token IDs space-separated on one line to stdout.

The prompt is passed via the FST_PROMPT env var (not argv) so apostrophes,
quotes, and other shell-meta characters survive verbatim.
"""
import os
import sys

from tokenizers import Tokenizer


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: fst_tokenize.py <tokenizer.json>", file=sys.stderr)
        return 2
    tok_path = sys.argv[1]
    prompt = os.environ.get("FST_PROMPT", "")
    tok = Tokenizer.from_file(tok_path)
    ids = tok.encode(prompt).ids
    print(" ".join(str(i) for i in ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())