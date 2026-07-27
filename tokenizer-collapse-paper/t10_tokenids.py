#!/usr/bin/env python3
"""T10 -- THE EMBARRASSMENT CHECK. Mechanism at the TOKEN-ID level.

Two seats independently demanded this before submission:

  JULIUS: "record the exact token-ID sequence for all eight emoji variants. If
  the failing models produce DIFFERENT token ids, the tokenizer-collision
  mechanism is FALSE and the collapse happens downstream. If they produce one
  shared [UNK] id, call it MANY-TO-ONE UNKNOWN-TOKEN COLLAPSE, not deletion.
  If the emoji positions vanish entirely, deletion is confirmed."

  GEMINI: "verify the CURE is not itself collapsing. If demojize produces
  strings that also map to [UNK], you still get identical vectors and the cure
  is worse than the disease."

We have been saying "deleted at tokenization." That is a MECHANISM CLAIM and we
have not actually looked at the ids. If it is wrong, the abstract is wrong.

Three verdicts are possible and we commit to reporting whichever we get:
  DELETION            -> emoji contributes NO token at all
  MANY-TO-ONE [UNK]   -> all emoji share one [UNK] id (still identical seqs)
  DISTINCT IDS        -> mechanism claim is FALSE, collapse is downstream
"""
import json
from transformers import AutoTokenizer

MARKS = ["\U0001F436", "\U0001F431", "\U0001F42D", "\U0001F439",
         "\U0001F430", "\U0001F98A", "\U0001F43B", "\U0001F43C"]
MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "BAAI/bge-m3"]

import emoji as EMO
report = {}

for name in MODELS:
    tk = AutoTokenizer.from_pretrained(name)
    short = name.split("/")[-1]
    print(f"\n=== {short}  (vocab {tk.vocab_size})")

    # --- RAW EMOJI ---------------------------------------------------------
    seqs = []
    for m in MARKS:
        ids = tk(f"animal {m}", add_special_tokens=False)["input_ids"]
        seqs.append(tuple(ids))
    base = tk("animal", add_special_tokens=False)["input_ids"]
    uniq = set(seqs)
    unk = tk.unk_token_id

    print(f"  carrier 'animal' ids            {base}")
    for m, s in zip(MARKS[:3], seqs[:3]):
        print(f"  'animal {m}' -> {list(s)}  decoded={tk.decode(s)!r}")
    print(f"  DISTINCT id-sequences across 8 emoji: {len(uniq)}")

    extra = [list(s)[len(base):] for s in seqs]
    if len(uniq) == 1:
        tail = extra[0]
        if not tail:
            verdict = "DELETION (emoji contributes no token)"
        elif unk is not None and all(t == unk for t in tail):
            verdict = f"MANY-TO-ONE [UNK] (id {unk})"
        else:
            verdict = f"IDENTICAL non-UNK tail {tail}"
    else:
        verdict = "DISTINCT IDS -- collapse is NOT tokenizer-level"
    print(f"  RAW VERDICT: {verdict}")

    # --- CURED (demojize) --------------------------------------------------
    cseqs = [tuple(tk(EMO.demojize(f"animal {m}"),
                      add_special_tokens=False)["input_ids"]) for m in MARKS]
    cuniq = set(cseqs)
    cunk = sum(1 for s in cseqs for t in s if unk is not None and t == unk)
    print(f"  CURED distinct id-sequences: {len(cuniq)}/8   total [UNK] in cured: {cunk}")
    print(f"  cured example: {tk.decode(cseqs[0])!r}")
    cure_ok = len(cuniq) == 8 and cunk == 0
    print(f"  CURE VERDICT: {'SEPARATES CLEANLY' if cure_ok else 'CURE IS SUSPECT'}")

    report[short] = {"vocab": tk.vocab_size, "raw_distinct": len(uniq),
                     "raw_verdict": verdict, "cured_distinct": len(cuniq),
                     "cured_unk_tokens": cunk, "cure_ok": cure_ok}

json.dump(report, open("/home/jason/dream_harness/t10_tokenids.json", "w"), indent=1)
print("\n" + "=" * 70)
print("MECHANISM (what we may claim in the abstract):")
for k, v in report.items():
    print(f"  {k:22} raw={v['raw_verdict']:38} cure_ok={v['cure_ok']}")
print("=" * 70)
