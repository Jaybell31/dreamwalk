#!/usr/bin/env python3
"""T13 -- WHICH words actually collide? The abstract currently says CJK/math/
currency "survive". T5 says otherwise on all four 30522 models:
  EMOJI 1/8, CHINESE 4/8, JAPANESE 6/8, CURRENCY 6/8, MATH 7/8.

Before we rewrite the claim we must know EXACTLY what collides with what, at
the token-ID level, so the paper states the harm precisely and cannot be
accused of inflating it.
"""
import json, collections, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

SETS = {
 "CHINESE":  ["医院","银行","学校","法院","机场","市场","农场","工厂"],
 "JAPANESE": ["病院","銀行","学校","裁判所","空港","市場","農場","工場"],
 "CURRENCY": ["\u20B9","\u20A9","\u20AA","\u20AB","\u20AC","\u00A3","\u00A5","\u20BD"],
 "MATH":     ["\u2211","\u222B","\u221A","\u2202","\u2207","\u221E","\u2248","\u2260"],
 "EMOJI":    ["\U0001F600","\U0001F602","\U0001F62D","\U0001F60D",
              "\U0001F914","\U0001F644","\U0001F621","\U0001F525"],
}
GLOSS = {
 "医院":"hospital","银行":"bank","学校":"school","法院":"court",
 "机场":"airport","市场":"market","农场":"farm","工厂":"factory",
 "病院":"hospital","銀行":"bank","裁判所":"court","空港":"airport",
 "市場":"market","農場":"farm","工場":"factory",
 "\u20B9":"rupee","\u20A9":"won","\u20AA":"shekel","\u20AB":"dong",
 "\u20AC":"euro","\u00A3":"pound","\u00A5":"yen","\u20BD":"ruble",
}
NAME = "sentence-transformers/all-MiniLM-L6-v2"
tok = AutoTokenizer.from_pretrained(NAME)
m = SentenceTransformer(NAME, device="cuda")
UNK = tok.unk_token_id
out = {}
for cat, words in SETS.items():
    E = m.encode(words, normalize_embeddings=True, show_progress_bar=False)
    groups = collections.defaultdict(list)
    for w, v in zip(words, E):
        groups[tuple(np.round(v, 6))].append(w)
    ids = {w: tok(w, add_special_tokens=False)["input_ids"] for w in words}
    coll = [g for g in groups.values() if len(g) > 1]
    print(f"\n=== {cat}  distinct={len(groups)}/8")
    for w in words:
        i = ids[w]
        nu = sum(1 for t in i if t == UNK)
        print(f"   {w:6} {GLOSS.get(w,''):9} ids={str(i):26} unk={nu}/{len(i)}")
    for g in coll:
        print(f"   COLLISION -> {' == '.join(f'{w}({GLOSS.get(w,w)})' for w in g)}")
    out[cat] = {"distinct": len(groups),
                "collisions": [[ [w, GLOSS.get(w, w)] for w in g] for g in coll],
                "ids": {w: ids[w] for w in words}}
json.dump(out, open("/home/jason/dream_harness/t13_collisions.json", "w"),
          indent=1, ensure_ascii=False)
print("\nwrote t13_collisions.json")
