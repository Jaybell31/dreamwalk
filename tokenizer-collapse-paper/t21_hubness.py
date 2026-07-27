#!/usr/bin/env python3
"""T21 -- HUBNESS. Settles the gte-base split PRE-REGISTERED in FOCUS.md.

THE SPLIT. T20 confirmed Gemini's semantic gate (xling cos < 0.75) in MiniLM
(0.507) and bge-base (0.685) but NOT in gte-base (0.846) -- yet gte-base STILL
fails 45.7% of DISTINCT-arm retrievals. Two readings were written down BEFORE
this test was run:
  (i)  the 0.75 threshold is model-relative and gte-base is simply scaled
       differently (its failed items, 0.819, still sit below its own mean 0.846
       -- same direction as the other two);
  (ii) a THIRD failure mode exists beyond collision and misalignment, visible
       only once alignment is adequate -- HUBNESS, where a few documents absorb
       most queries regardless of meaning.

PRE-REGISTERED GATE (copied verbatim from FOCUS.md, written before running):
  "If gte-base shows hub concentration >= 0.30 of queries on one document while
   its xling cos is >= 0.80, reading (ii) is supported and the paper needs a
   third mechanism section. If hubness is flat (<0.15), reading (i) holds and
   the gate was simply mis-scaled."

METRICS, DISTINCT arm only (the arm where collision is already excluded):
  * top_hub_share  -- fraction of queries whose argmax lands on the single
                      most-retrieved document. Chance = 1/n_distinct.
  * gini           -- inequality of the retrieved-document distribution.
  * n_never        -- documents that win ZERO queries (the starved tail).
A NEGATIVE CONTROL IS INCLUDED: the same metrics computed on the ENGLISH arm of
the same documents. English retrieval is near-ceiling, so if English also shows
high hub share the metric is measuring the CORPUS, not a failure mode, and the
test is void. This control is the difference between a result and an artifact.
"""
import json, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import importlib.util

OUT = "/home/jason/dream_harness/t21_hubness.json"
spec = importlib.util.spec_from_file_location("t17", "/home/jason/dream_harness/t17_within_model.py")
t17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t17)
ITEMS = t17.ITEMS

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base"]

def gini(counts):
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

def hub_stats(pred, n_docs):
    counts = np.bincount(pred, minlength=n_docs)
    return {"top_hub_share": float(counts.max() / len(pred)),
            "top_hub_doc": int(counts.argmax()),
            "gini": gini(counts),
            "n_never_retrieved": int((counts == 0).sum()),
            "chance_share": float(1.0 / n_docs)}

def main():
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    docs = [i[2] for i in ITEMS]; zh = [i[0] for i in ITEMS]; en = [i[1] for i in ITEMS]
    n = len(ITEMS)
    distinct = [k for k in range(n) if 100 not in tk(zh[k], add_special_tokens=False)["input_ids"]]
    nd = len(distinct)
    print(f"DISTINCT arm = {nd} items (collision already excluded)")

    res = {"n_distinct": nd, "gate": "hub>=0.30 & xling>=0.80 -> reading(ii); hub<0.15 -> reading(i)",
           "models": {}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode([docs[k] for k in distinct], normalize_embeddings=True, show_progress_bar=False)
        idx = np.arange(nd)

        QZ = m.encode([zh[k] for k in distinct], normalize_embeddings=True, show_progress_bar=False)
        QE = m.encode([en[k] for k in distinct], normalize_embeddings=True, show_progress_bar=False)
        pz = (QZ @ D.T).argmax(1); pe = (QE @ D.T).argmax(1)

        hz = hub_stats(pz, nd); he = hub_stats(pe, nd)
        r1z = float((pz == idx).mean()); r1e = float((pe == idx).mean())

        # NEGATIVE CONTROL: is the English arm also hubby? If so, metric is void.
        control_ok = he["top_hub_share"] < 0.15
        verdict = ("(ii) THIRD MECHANISM -- HUBNESS" if hz["top_hub_share"] >= 0.30
                   else "(i) gate mis-scaled, hubness flat" if hz["top_hub_share"] < 0.15
                   else "INDETERMINATE (0.15-0.30)")

        print(f"  Chinese arm R@1={r1z:.3f}  top_hub_share={hz['top_hub_share']:.3f} "
              f"(chance {hz['chance_share']:.3f})  gini={hz['gini']:.3f}  never-retrieved={hz['n_never_retrieved']}/{nd}")
        print(f"  ENGLISH CONTROL R@1={r1e:.3f}  top_hub_share={he['top_hub_share']:.3f}  gini={he['gini']:.3f}")
        print(f"  control {'VALID (english not hubby)' if control_ok else 'VOID -- english is ALSO hubby, metric measures corpus'}")
        print(f"  VERDICT: {verdict}")

        res["models"][mname] = {"chinese_r1": r1z, "english_r1": r1e,
                                "chinese_hub": hz, "english_hub": he,
                                "control_valid": bool(control_ok), "verdict": verdict}
        del m

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
