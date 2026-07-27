#!/usr/bin/env python3
"""T12 EVAL -- powered semantic polarity test (n=96 pairs, 192 queries).

Pre-registered in t12_build.py BEFORE any result was seen:
  primary metric = pair-restricted 2-way accuracy, chance = 0.500
  H2: demojize > raw, two-sided exact binomial (McNemar), alpha = 0.05
      SIGNIFICANT -> keep "one-line fix restores retrieval"
      NOT         -> narrow paper to COLLAPSE + TIE ARTIFACT; demojize
                     reported as INSUFFICIENT. No fishing, no re-cutting.
  word-control must beat raw for every model, else corpus is void.
"""
import json, numpy as np, torch
from math import comb
from sentence_transformers import SentenceTransformer

PAIRS = json.load(open("/home/jason/dream_harness/t12_pairs.json"))
OUT = "/home/jason/dream_harness/t12_semantic.json"
WORD_BAD, WORD_GOOD = "a disaster", "a success"
MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "BAAI/bge-m3"]

DOCS, GB, GG = [], [], []
for p in PAIRS:
    GB.append(len(DOCS)); DOCS.append(p["bad_doc"])
    GG.append(len(DOCS)); DOCS.append(p["good_doc"])
N = len(PAIRS)
assert not any(ord(c) > 0x2500 for d in DOCS for c in d), "corpus not emoji-free"
print(f"{N} pairs, corpus {len(DOCS)} emoji-free docs, {2*N} queries")

import emoji as EMO

def build(mode):
    qs, gs, pid = [], [], []
    for i, p in enumerate(PAIRS):
        c = p["carrier"]
        if mode == "word":
            qb, qg = f"{c} {WORD_BAD}", f"{c} {WORD_GOOD}"
        else:
            qb, qg = f"{c} {p['bad_emoji']}", f"{c} {p['good_emoji']}"
            if mode == "demojize":
                qb, qg = EMO.demojize(qb), EMO.demojize(qg)
        qs += [qb, qg]; gs += [GB[i], GG[i]]; pid += [i, i]
    return qs, gs, pid

def mcnemar_p(b, c):
    """exact two-sided binomial on discordant pairs"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return min(1.0, p)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)
res, ties, correct = {}, {}, {}
for name in MODELS:
    m = SentenceTransformer(name, device=dev)
    short = name.split("/")[-1]
    D = m.encode(DOCS, normalize_embeddings=True, show_progress_bar=False,
                 batch_size=32)
    for mode in ("raw", "demojize", "word"):
        qs, gs, pid = build(mode)
        Q = m.encode(qs, normalize_embeddings=True, show_progress_bar=False,
                     batch_size=32)
        sims = Q @ D.T
        hits = []
        for j, (g, i) in enumerate(zip(gs, pid)):
            two = [GB[i], GG[i]]
            pick = two[int(np.argmax([sims[j][two[0]], sims[j][two[1]]]))]
            hits.append(int(pick == g))
        correct[f"{short}|{mode}"] = hits
        acc = float(np.mean(hits))
        corpus_top1 = float(np.mean([int(sims[j].argmax()) == g
                                     for j, g in enumerate(gs)]))
        res[f"{short}|{mode}"] = {"pair_2way": round(acc, 4),
                                  "corpus_top1": round(corpus_top1, 4),
                                  "hits": int(sum(hits)), "n": len(hits)}
        if mode == "raw":
            same = sum(int(np.allclose(Q[2*i], Q[2*i+1], atol=1e-6))
                       for i in range(N))
            ties[short] = {"identical": same, "of": N}
        print(f"  {short:20} {mode:9} pair2way={acc:.3f} "
              f"({sum(hits)}/{len(hits)})  corpus_top1={corpus_top1:.3f}",
              flush=True)
    del m
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("PRE-REGISTERED TEST: demojize vs raw, exact McNemar, alpha=0.05")
stats = {}
for name in MODELS:
    s = name.split("/")[-1]
    r, d = correct[f"{s}|raw"], correct[f"{s}|demojize"]
    b = sum(1 for x, y in zip(r, d) if x == 1 and y == 0)   # raw only
    c = sum(1 for x, y in zip(r, d) if x == 0 and y == 1)   # demojize only
    p = mcnemar_p(b, c)
    stats[s] = {"raw_only": b, "demojize_only": c, "p_mcnemar": round(p, 6),
                "significant": bool(p < 0.05)}
    print(f"  {s:20} raw_only={b:3} demojize_only={c:3}  p={p:.5f}  "
          f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")

json.dump({"n_pairs": N, "corpus": len(DOCS), "acc": res, "ties": ties,
           "mcnemar": stats, "complete": True}, open(OUT, "w"), indent=1)
print(f"\nidentical raw query pairs: {ties}")
print(f"wrote {OUT}")
