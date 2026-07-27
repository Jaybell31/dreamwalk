#!/usr/bin/env python3
"""T24 -- GPT'S DERANGEMENT KILLSHOT. Is the T19 rescue SEMANTICS or just ID?

THE CLAIM UNDER ATTACK (GPT, R11-E): "the correct same-entity spelling restores
semantic information." T19 self-retrieval cannot distinguish SEMANTIC RECOVERY
from ARBITRARY IDENTIFIER RECOVERY -- any unique in-vocabulary token can break
an [UNK] tie even if it means something completely different.

DESIGN (GPT's, implemented as specified):
  Three queries per entity, one fixed document index, nothing else varies:
    C = collided form            (e.g. 车  -> [UNK])
    S = correct distinct form    (e.g. 車  -> [1954])   the true rescue
    W = WRONG distinct form      (another entity's rescued token, deranged)
  W is assigned by DERANGEMENT so no entity ever receives its own token.
  100 independent derangements; we report the median W arm.

  D_sem   = R@1(S) - median_derangement R@1(W)
  RELABEL = [R@1(W) - R@1(C)] / [R@1(S) - R@1(C)]

PRE-REGISTERED GATES (GPT's numbers, not ours):
  ABORT the semantic-rescue claim if D_sem < 0.15, or if its permutation 95%
  lower bound <= 0.
  RELABEL GATE: if RELABEL >= 0.80, the mechanism is DE-COLLISION / arbitrary
  identity coding -- NOT recovered semantics -- and the paper's claim narrows
  to "collisions erase lexical identity; distinct token sequences restore
  separability."

WHY THIS MATTERS EVEN THOUGH T23 EXISTS. T23 showed identity WITHOUT semantics
(<U+533B> codepoints) recovers nothing -- identity is insufficient. T24 asks
the CONVERSE: is the gain T19 measured actually semantic, or would ANY
in-vocabulary CJK token have produced it? These are different questions and
both must be answered before the word "semantic" appears in the paper.
"""
import json
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import importlib.util

OUT = "/home/jason/dream_harness/t24_derangement.json"
spec = importlib.util.spec_from_file_location("t19", "/home/jason/dream_harness/t19_rescue.py")
t19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t19)
PAIRS = t19.PAIRS                      # (en, collided, rescued, is_ortho, doc)

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base"]
NDER = 100
rng = np.random.default_rng(20260727)


def derangement(n):
    """Random permutation with no fixed point."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def main():
    docs = [p[4] for p in PAIRS]
    coll = [p[1] for p in PAIRS]
    resc = [p[2] for p in PAIRS]
    n = len(PAIRS)
    idx = np.arange(n)
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # sanity: every rescued form must be tokenizer-distinct, every collided must collide
    assert all(100 in tk(c, add_special_tokens=False)["input_ids"] for c in coll)
    assert all(100 not in tk(r, add_special_tokens=False)["input_ids"] for r in resc)
    print(f"n={n} pairs, {NDER} derangements, index held fixed")

    res = {"n": n, "n_derangements": NDER,
           "gates": {"D_sem_abort_below": 0.15, "relabel_at_or_above": 0.80},
           "models": {}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)

        def r1(queries):
            Q = m.encode(queries, normalize_embeddings=True, show_progress_bar=False)
            return ((Q @ D.T).argmax(1) == idx)

        hit_C = r1(coll); hit_S = r1(resc)
        R_C, R_S = float(hit_C.mean()), float(hit_S.mean())

        # W arm: each entity gets ANOTHER entity's rescued token
        w_scores = []
        for _ in range(NDER):
            p = derangement(n)
            w_scores.append(float(r1([resc[p[k]] for k in range(n)]).mean()))
        w_scores = np.array(w_scores)
        R_W = float(np.median(w_scores))
        w_lo, w_hi = np.percentile(w_scores, [2.5, 97.5])

        D_sem = R_S - R_W
        # permutation lower bound: how often does a deranged arm match/beat S?
        p_perm = float(np.mean(w_scores >= R_S))
        D_sem_lo = R_S - float(np.percentile(w_scores, 97.5))   # conservative

        denom = R_S - R_C
        relabel = float((R_W - R_C) / denom) if denom > 0 else float("nan")

        semantic_ok = (D_sem >= 0.15) and (D_sem_lo > 0)
        relabelled = (not np.isnan(relabel)) and relabel >= 0.80

        print(f"  R@1  C(collided)={R_C:.3f}   S(correct)={R_S:.3f}   "
              f"W(deranged median)={R_W:.3f}  [95% {w_lo:.3f}-{w_hi:.3f}]")
        print(f"  D_sem = {D_sem:+.3f}   conservative lower bound {D_sem_lo:+.3f}   "
              f"perm p(W>=S) = {p_perm:.3f}")
        print(f"  RELABEL ratio = {relabel:.3f}  (gate 0.80)")
        print(f"  VERDICT: {'SEMANTIC RECOVERY SUPPORTED' if semantic_ok and not relabelled else 'RELABEL -- DE-COLLISION / ARBITRARY IDENTITY, NOT SEMANTICS' if relabelled else 'ABORT semantic claim (D_sem below gate)'}")

        res["models"][mname] = {
            "R_collided": R_C, "R_correct": R_S, "R_wrong_median": R_W,
            "W_ci95": [float(w_lo), float(w_hi)],
            "D_sem": float(D_sem), "D_sem_lower": float(D_sem_lo),
            "perm_p_W_ge_S": p_perm, "relabel_ratio": relabel,
            "semantic_supported": bool(semantic_ok and not relabelled),
            "relabelled_as_identity": bool(relabelled)}
        del m

    n_sem = sum(1 for v in res["models"].values() if v["semantic_supported"])
    res["verdict"] = (f"{n_sem}/3 models support SEMANTIC recovery; "
                      f"{sum(1 for v in res['models'].values() if v['relabelled_as_identity'])}/3 relabelled as identity-only")
    print(f"\n{res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
