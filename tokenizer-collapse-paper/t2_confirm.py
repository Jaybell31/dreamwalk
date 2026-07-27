#!/usr/bin/env python3
"""T2 CONFIRMATION -- is the collapse BIT-IDENTICAL, and does it survive
retrieval, and is it really the tokenizer?

The survey said 1/8 distinct vectors on 8 of 9 models. Before that becomes a
headline it has to survive the three ways it could still be an artifact:

  1. ROUNDING. "1/8 distinct" came from np.round(v,6). If the vectors differ in
     the 8th decimal the collapse is real-but-overstated. Here we compare RAW
     float32 bytes and report exact equality.
  2. ARTIFICIAL PROBE. Our carrier was a synthetic 5-word string. Real retrieval
     uses real documents. So we run an actual RETRIEVAL task: query "the market
     signal is <emoji>" against a corpus of the 8 emoji sentences, and measure
     top-1 accuracy. Chance = 12.5%.
  3. WRONG MECHANISM. We claim [UNK]. Prove it by showing the token ids are
     literally identical across all 8 inputs, and that the emoji position holds
     unk_token_id. If ids differ but vectors match, the mechanism is something
     else and the explanation must change.

CONTROL: the same three tests on DISTINCT_WORDS. If words also collapse, the
harness is broken and no emoji claim can be made.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sentence_transformers import SentenceTransformer

CARRIER = "the market signal is {}"
EMOJI = ["\U0001F680", "\U0001F480", "\U0001F402", "\U0001F43B",
         "\u2764\uFE0F", "\U0001F602", "\U0001F62D", "\U0001F44D"]
WORDS = ["rising", "falling", "bullish", "bearish",
         "strong", "weak", "green", "red"]

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "intfloat/e5-base-v2",
          "BAAI/bge-m3"]


def probe(m, tok, variants, tag):
    sents = [CARRIER.format(v) for v in variants]
    emb = m.encode(sents, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)

    # 1. bit-identical?
    raw_distinct = len({e.tobytes() for e in emb})

    # 3. identical token ids?
    idlists = [tuple(tok(s, add_special_tokens=False)["input_ids"]) for s in sents]
    id_distinct = len(set(idlists))
    unk = tok.unk_token_id
    unk_present = sum(1 for ids in idlists if unk in ids)

    # 2. retrieval: each sentence is its own query; can it find itself?
    #    (self-retrieval is trivially 100% -- so query with a PARAPHRASE carrier)
    qs = [f"what does {v} mean for the market" for v in variants]
    qe = np.asarray(m.encode(qs, normalize_embeddings=True,
                             show_progress_bar=False), dtype=np.float32)
    sim = qe @ emb.T
    top1 = int((sim.argmax(axis=1) == np.arange(len(variants))).sum())

    return {"tag": tag, "raw_distinct_vectors": raw_distinct, "n": len(variants),
            "distinct_token_id_seqs": id_distinct, "inputs_containing_UNK": unk_present,
            "retrieval_top1": top1, "retrieval_chance": round(100/len(variants), 1)}


out = []
for name in MODELS:
    m = SentenceTransformer(name, device="cpu")
    tok = m.tokenizer
    r = {"model": name, "vocab": int(tok.vocab_size)}
    r["emoji"] = probe(m, tok, EMOJI, "emoji")
    r["words"] = probe(m, tok, WORDS, "words")
    out.append(r)
    e, w = r["emoji"], r["words"]
    print(f"\n{name}  (vocab {r['vocab']})")
    print(f"  EMOJI  raw-distinct {e['raw_distinct_vectors']}/8  "
          f"distinct-id-seqs {e['distinct_token_id_seqs']}/8  "
          f"inputs-with-UNK {e['inputs_containing_UNK']}/8  "
          f"retrieval top1 {e['retrieval_top1']}/8 (chance 1/8)")
    print(f"  WORDS  raw-distinct {w['raw_distinct_vectors']}/8  "
          f"distinct-id-seqs {w['distinct_token_id_seqs']}/8  "
          f"inputs-with-UNK {w['inputs_containing_UNK']}/8  "
          f"retrieval top1 {w['retrieval_top1']}/8   <- CONTROL")
    del m

json.dump(out, open("/home/jason/dream_harness/t2_confirm.json", "w"), indent=1)
print("\nwrote /home/jason/dream_harness/t2_confirm.json")
