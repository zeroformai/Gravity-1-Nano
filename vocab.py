"""
vocab.py
--------
Compatibility wrapper for older imports.

New code should import from tokenizer.py.
"""

from tokenizer import (  # noqa: F401
    CharTokenizer,
    TiktokenTokenizer,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    create_tokenizer,
    load_tokenizer,
)
