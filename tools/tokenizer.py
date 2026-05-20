#!/usr/bin/env python3
"""Qwen2-0.5B tokenizer helper.

The C++/CUDA engine only handles integer token IDs. This script bridges text.

Usage:
  python tokenizer.py encode "Hello world"      # -> space-separated token IDs
  python tokenizer.py encode --file prompt.txt
  python tokenizer.py decode 9707 1879          # -> text
"""
import argparse
import sys

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2-0.5B"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("text", nargs="?", default=None)
    enc.add_argument("--file", default=None)

    dec = sub.add_parser("decode")
    dec.add_argument("ids", nargs="*", type=int)
    dec.add_argument("--file", default=None)

    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    if args.cmd == "encode":
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        elif args.text is not None:
            text = args.text
        else:
            text = sys.stdin.read()
        ids = tok.encode(text)
        print(" ".join(str(i) for i in ids))
    else:
        ids = args.ids
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                ids = [int(x) for x in f.read().split()]
        print(tok.decode(ids))


if __name__ == "__main__":
    main()
