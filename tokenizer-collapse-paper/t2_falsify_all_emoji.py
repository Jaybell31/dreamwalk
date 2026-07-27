#!/usr/bin/env python3
"""FALSIFICATION TEST for the headline phrase "ALL emoji -> [UNK]".

RAISED BY: a prior-art review flagged that bert-base-uncased's vocab.txt
contains some BMP symbol characters (heart, star, note, arrows). If any of
those are Emoji_Presentation codepoints, then "ALL emoji collapse" is FALSE
as written and must be narrowed to something like "all astral-plane
pictographs". Being factually corrected in public on the headline sentence
would cost more than the finding is worth, so we test it BEFORE publishing.

METHOD (no LLM judgement, pure set arithmetic):
  1. Pull the ACTUAL vocab of each 30.5k WordPiece model.
  2. Take the FULL official emoji set from the `emoji` package (not a
     hand-picked probe -- hand-picking is exactly how we would miss the
     exception).
  3. Intersect. Any emoji present as its own vocab token is a COUNTEREXAMPLE
     to the universal claim.
  4. For every counterexample, actually EMBED it against a carrier and check
     whether it separates -- because being in the vocab is necessary but not
     sufficient for the vector to differ.

OUTPUT: the exact wording the headline is allowed to use.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import emoji as EMOJI_PKG
from transformers import AutoTokenizer

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "intfloat/e5-base-v2",
          "thenlper/gte-base",
          "sentence-transformers/all-mpnet-base-v2"]

# FULL official emoji inventory, not a curated probe set.
ALL_EMOJI = sorted({e for e in EMOJI_PKG.EMOJI_DATA.keys()})
SINGLE_CP = [e for e in ALL_EMOJI if len(e) == 1]          # BMP + astral singles
BMP       = [e for e in SINGLE_CP if ord(e) <= 0xFFFF]
ASTRAL    = [e for e in SINGLE_CP if ord(e) > 0xFFFF]

print(f"official emoji sequences        {len(ALL_EMOJI)}")
print(f"  single-codepoint              {len(SINGLE_CP)}")
print(f"    BMP    (<= U+FFFF)          {len(BMP)}")
print(f"    astral (>  U+FFFF)          {len(ASTRAL)}")

report = []
for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    vocab = set(tok.get_vocab().keys())
    in_vocab = [e for e in SINGLE_CP if e in vocab]
    in_bmp    = [e for e in in_vocab if ord(e) <= 0xFFFF]
    in_astral = [e for e in in_vocab if ord(e) > 0xFFFF]
    unk = tok.unk_token_id

    # Does an in-vocab emoji actually survive tokenization as a distinct id?
    survivors = []
    for e in in_vocab:
        ids = tok(f"the market signal is {e}", add_special_tokens=False)["input_ids"]
        if unk not in ids:
            survivors.append(e)

    row = {"model": name,
           "vocab_size": len(vocab),
           "emoji_in_vocab": len(in_vocab),
           "bmp_in_vocab": in_bmp,
           "astral_in_vocab": in_astral,
           "tokenize_without_unk": survivors}
    report.append(row)
    print(f"\n=== {name}")
    print(f"  single-cp emoji present as vocab tokens : {len(in_vocab)}")
    print(f"    BMP     : {in_bmp}")
    print(f"    astral  : {in_astral}")
    print(f"  of those, tokenized with NO [UNK]       : {survivors}")

# If any survivor exists, check whether it also SEPARATES in vector space.
survivor_union = sorted({e for r in report for e in r["tokenize_without_unk"]})
print(f"\nunion of survivors across models: {survivor_union}")

if survivor_union:
    from sentence_transformers import SentenceTransformer
    probe = survivor_union[:8]
    while len(probe) < 2:
        probe = probe * 2
    print("\nvector separation test on survivors:")
    for name in MODELS[:3]:
        m = SentenceTransformer(name, device="cpu")
        sents = [f"the market signal is {e}" for e in probe]
        v = np.asarray(m.encode(sents, normalize_embeddings=True,
                                show_progress_bar=False), dtype=np.float32)
        d = len({x.tobytes() for x in v})
        print(f"  {name.split('/')[-1]:32} {d}/{len(probe)} distinct")
        del m

json.dump(report, open("/home/jason/dream_harness/t2_falsify.json", "w"), indent=1)

print("\n" + "=" * 66)
if not survivor_union:
    print("HEADLINE STANDS AS WRITTEN: no single-codepoint emoji survives")
    print("tokenization without [UNK] on any 30.5k-vocab model tested.")
else:
    print("HEADLINE MUST BE NARROWED. These emoji are exceptions:")
    print(f"  {survivor_union}")
    print("Use wording like 'all astral-plane pictographs' or state the")
    print("exception count explicitly. Do NOT say 'all emoji'.")
print("=" * 66)
