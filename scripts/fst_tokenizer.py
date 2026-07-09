#!/usr/bin/env python3
"""fst_tokenizer.py — DeepSeek V4 Flash tokenizer wrapper.

Uses the upstream `tokenizers` library to load the HuggingFace
tokenizer.json shipped with the model.
"""
import os
from typing import List, Optional

from tokenizers import Tokenizer


BOS_ID = 0
EOS_ID = 1
PAD_ID = 2


class DeepSeekTokenizer:
    """Thin wrapper around the HF tokenizers library."""

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
    ):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(
                os.path.dirname(__file__),
                "Source",
                "models--deepseek-ai--DeepSeek-V4-Flash",
                "snapshots",
                "553034d7dd9e06c2eeaee68cf85a17d6d4754cf0",
                "tokenizer.json",
            )
        self._tok = Tokenizer.from_file(tokenizer_path)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        if not text:
            ids = []
        else:
            encoded = self._tok.encode(text)
            ids = encoded.ids

        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        if skip_special:
            ids = [i for i in ids if i >= 128]
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID


def main():
    tok = DeepSeekTokenizer()
    print(f"Vocab size: {tok.vocab_size}")

    for prompt in ["Hello", "How are you?", "DeepSeek V4 Flash"]:
        ids = tok.encode(prompt)
        decoded = tok.decode(ids)
        print(f"  '{prompt}' -> {ids} -> '{decoded}'")


if __name__ == "__main__":
    main()
