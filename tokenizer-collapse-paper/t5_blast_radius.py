#!/usr/bin/env python3
"""T5 BLAST RADIUS -- emoji were never the point. What ELSE does the default
English embedding stack silently delete?

Same mechanism (30,522-token WordPiece, unknown glyph -> [UNK]), new probe sets.
If Chinese/Arabic/Devanagari collapse the same way, "emoji bug" is the wrong
headline and understates it by an order of magnitude.

Every probe set is 8 sentences differing ONLY in the varying element, in the
SAME carrier, so the comparison is apples to apples across scripts.

CONTROLS (mandatory, standing rule 1):
  WORDS_EN  positive control -- must be 8/8 on every model or the row is void.
  IDENTICAL negative control -- must be 1/8 or the harness is nondeterministic.

We report distinct RAW float32 vectors (not rounded) and retrieval top-1 with a
paraphrased query, because retrieval-at-chance is the number a practitioner
actually feels.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sentence_transformers import SentenceTransformer

CARRIER = "the document topic is {}"
QUERY   = "what does {} refer to in this document"

PROBES = {
    "WORDS_EN":   ["rising", "falling", "bullish", "bearish", "strong", "weak", "green", "red"],
    "EMOJI":      ["\U0001F680","\U0001F480","\U0001F402","\U0001F43B","\u2764\uFE0F","\U0001F602","\U0001F62D","\U0001F44D"],
    "CHINESE":    ["医院","银行","学校","法院","机场","市场","农场","工厂"],
    "JAPANESE":   ["病院","銀行","学校","裁判所","空港","市場","農場","工場"],
    "KOREAN":     ["병원","은행","학교","법원","공항","시장","농장","공장"],
    "ARABIC":     ["مستشفى","بنك","مدرسة","محكمة","مطار","سوق","مزرعة","مصنع"],
    "HINDI":      ["अस्पताल","बैंक","विद्यालय","न्यायालय","हवाई","बाज़ार","खेत","कारखाना"],
    "RUSSIAN":    ["больница","банк","школа","суд","аэропорт","рынок","ферма","завод"],
    "MATH":       ["\u2211","\u222B","\u221A","\u2202","\u2207","\u221E","\u2248","\u2260"],
    "CURRENCY":   ["\u20B9","\u20A9","\u20AA","\u20AB","\u20AC","\u00A3","\u00A5","\u20BD"],
}

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "intfloat/e5-base-v2",
          "thenlper/gte-base",
          "BAAI/bge-m3"]


def run(m, tok, items):
    sents = [CARRIER.format(x) for x in items]
    emb = np.asarray(m.encode(sents, normalize_embeddings=True,
                              show_progress_bar=False), dtype=np.float32)
    distinct = len({e.tobytes() for e in emb})
    ids = [tuple(tok(s, add_special_tokens=False)["input_ids"]) for s in sents]
    unk = tok.unk_token_id
    qs = [QUERY.format(x) for x in items]
    qe = np.asarray(m.encode(qs, normalize_embeddings=True,
                             show_progress_bar=False), dtype=np.float32)
    top1 = int(((qe @ emb.T).argmax(axis=1) == np.arange(len(items))).sum())
    return {"distinct": distinct, "n": len(items),
            "distinct_ids": len(set(ids)),
            "unk_inputs": sum(1 for i in ids if unk in i),
            "top1": top1}


out = []
for name in MODELS:
    m = SentenceTransformer(name, device="cpu")
    tok = m.tokenizer
    row = {"model": name, "vocab": int(tok.vocab_size)}
    for label, items in PROBES.items():
        row[label] = run(m, tok, items)
    row["IDENTICAL"] = run(m, tok, ["x"] * 8)
    row["control_ok"] = row["WORDS_EN"]["distinct"] == 8
    row["sanity_ok"] = row["IDENTICAL"]["distinct"] == 1
    out.append(row)
    print(f"\n=== {name}  vocab={row['vocab']}  "
          f"control_ok={row['control_ok']} sanity_ok={row['sanity_ok']}")
    for label in list(PROBES) :
        r = row[label]
        flag = "  <-- COLLAPSE" if r["distinct"] == 1 else ""
        print(f"    {label:10} distinct {r['distinct']}/8  ids {r['distinct_ids']}/8  "
              f"unk {r['unk_inputs']}/8  retrieval {r['top1']}/8{flag}")
    del m

json.dump(out, open("/home/jason/dream_harness/t5_blast_radius.json", "w"), indent=1)
print("\nwrote /home/jason/dream_harness/t5_blast_radius.json")
