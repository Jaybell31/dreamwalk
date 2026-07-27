#!/usr/bin/env python3
"""T9 -- REAL-CONTEXT BENCHMARK. Kills the "synthetic toy" attack.

THE ATTACK (independently named by all 3 council seats): "This is the known OOV
problem, disclosed 2021, demonstrated on a toy. You have not shown real-world
retrieval damage."

JULIUS'S DESIGN (devil's seat, adopted): use REAL, NATURALLY OCCURRING
emoji-bearing messages. For each message build an 8-document retrieval set with
IDENTICAL original wording and 8 plausible emoji alternatives; the true emoji is
the label. Nothing is synthetic except the distractor emoji -- the carrier text
is human-written.

WHY THIS IS THE RIGHT TEST: our earlier probes wrote the carrier text
ourselves, so a reviewer can say we engineered the collapse. Here the text comes
from tweet_eval, which we already audited (809 emoji-bearing texts, 1.35%
prevalence, top marks are the real ones people use).

PREREGISTERED INTERPRETATION (written before running):
  - collapsed models ~1/8 on REAL text  -> "toy" attack is dead, damage is real
  - collapsed models recover toward 8/8 -> real context rescues retrieval, and
    we must say so loudly and narrow the paper to short/low-context text
  - bge-m3 must beat the collapsed models or the harness is broken (neg control)
"""
import json, random
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import emoji as EMO

random.seed(0)

# --- real emoji-bearing messages -------------------------------------------
# DATA SOURCE NOTE (measured Jul27, cost one dead run): emoji in tweet_eval are
# NOT uniformly distributed across splits. train=45615 rows / 0 emoji,
# validation=2000 / 0 emoji, test=12284 / 794 emoji (6.46%). Using "train" here
# yields 0 trials and a ZeroDivisionError -- a 0-of-N that means UNTESTABLE
# (wrong input), NOT "no effect". Always print the pool size before scoring.
ds = load_dataset("tweet_eval", "sentiment", split="test")
EMOJI_SET = set(EMO.EMOJI_DATA.keys())


def emojis_in(s):
    return [ch for ch in s if ch in EMOJI_SET]


pool = []
for rec in ds:
    t = rec["text"]
    es = emojis_in(t)
    # exactly one distinct emoji, and enough carrier text to be a real message
    if len(set(es)) == 1 and len(t) > 60:
        pool.append((t, es[0]))
print(f"real single-emoji messages available: {len(pool)}")

# the 8 most common real emoji act as the distractor alphabet
from collections import Counter
top8 = [e for e, _ in Counter(e for _, e in pool).most_common(8)]
print("distractor alphabet (real usage):", " ".join(top8))

pool = [(t, e) for t, e in pool if e in top8]
random.shuffle(pool)
TRIALS = pool[:150]
print(f"trials: {len(TRIALS)}")
# FAIL LOUD on an empty pool. 0-of-N without a positive control is UNTESTABLE,
# not a negative result -- do not let it fall through into a division and get
# reported as an accuracy.
if len(TRIALS) < 30:
    raise SystemExit(
        f"ABORT: only {len(TRIALS)} real emoji-bearing trials found. "
        "This is an INPUT problem (wrong split / stripped corpus), not a "
        "measurement. Fix the data source before interpreting anything.")

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "BAAI/bge-m3"]

results = {}
for name in MODELS:
    # RUN 2 DIED AT THE 3000s TIMEOUT ON CPU WITH ZERO OUTPUT. Two fixes:
    #   (a) use the GPU -- 31GB is free, bge-m3 on CPU was the whole bottleneck
    #   (b) save after EVERY model so a timeout still leaves usable evidence
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(name, device=dev)
    short = name.split("/")[-1]
    for mode, tf in (("raw", lambda s: s), ("demojize", EMO.demojize)):
        hits = 0
        for text, true_e in TRIALS:
            # 8 candidates: same human text, swapped emoji. truth = original.
            cands = [text.replace(true_e, alt) for alt in top8]
            gold = top8.index(true_e)
            q = tf(text)
            E = m.encode([tf(c) for c in cands] + [q],
                         normalize_embeddings=True, show_progress_bar=False,
                         batch_size=16)
            sims = E[:-1] @ E[-1]
            if int(np.argmax(sims)) == gold:
                hits += 1
        acc = hits / len(TRIALS)
        results[f"{short}|{mode}"] = round(acc, 4)
        print(f"  {short:20} {mode:9} acc={acc:.3f}  ({hits}/{len(TRIALS)})",
              flush=True)
        # PARTIAL SAVE: a timeout must not destroy completed work.
        json.dump({"n": len(TRIALS), "alphabet": top8, "acc": results,
                   "complete": False},
                  open("/home/jason/dream_harness/t9_realcontext.json", "w"),
                  indent=1)
    del m

json.dump({"n": len(TRIALS), "alphabet": top8, "acc": results, "complete": True},
          open("/home/jason/dream_harness/t9_realcontext.json", "w"), indent=1)

print("\n" + "=" * 68)
print(f"REAL human-written carrier text. n={len(TRIALS)}. chance = 1/8 = 0.125")
for k, v in results.items():
    print(f"  {k:34} {v:.3f}")
print("=" * 68)
