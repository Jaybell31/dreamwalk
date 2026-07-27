#!/usr/bin/env python3
"""T22 -- CAF, Julius's deciding statistic for LEAD vs LIMITATION (R11-C).

    CAF = (R_rescued - R_collided) / (R_oracle - R_collided)

"Collision-attributable fraction of paired error reduction." It asks what share
of the recoverable gap a perfect tokenizer fix actually buys.

R_oracle MUST be "a genuine counterfactual ceiling ON THE SAME ITEMS -- not
aggregate performance on an easier subset" (Julius, verbatim). We use the
ENGLISH query for the same entity against the same document set: identical
items, identical documents, identical scoring, the only change being the query
language. That is the ceiling a perfect multilingual encoder would reach.

PRECOMMITTED INTERPRETATION (Julius, before this ran):
    CAF < 0.50 and upper 95% bound < 0.50  -> THE RESIDUAL IS THE LEAD
    CAF >= 0.50 and lower 95% bound > 0.50 -> TOKENIZATION IS THE LEAD
    interval crosses 0.50                  -> UNRESOLVED, say so

BOOTSTRAP: "by entity and document, not by individual query" -- we resample
PAIRS (each pair is one entity with its own document), which resamples the
entity and its document together as one unit. 10,000 replicates, percentile CI.

Also reported, per Julius: ABSOLUTE ERROR MASS over the full benchmark. A
dramatic rescue on a small collision subset can still explain little total
failure, and CAF alone would hide that.
"""
import json, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import importlib.util

OUT = "/home/jason/dream_harness/t22_caf.json"
spec = importlib.util.spec_from_file_location("t19", "/home/jason/dream_harness/t19_rescue.py")
t19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t19)
PAIRS = t19.PAIRS

spec2 = importlib.util.spec_from_file_location("t17", "/home/jason/dream_harness/t17_within_model.py")
t17 = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(t17)
ITEMS = t17.ITEMS

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base"]
B = 10000
rng = np.random.default_rng(20260727)

def main():
    docs = [p[4] for p in PAIRS]; n = len(PAIRS); idx = np.arange(n)
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    full_docs = [i[2] for i in ITEMS]; full_zh = [i[0] for i in ITEMS]
    nf = len(ITEMS)

    res = {"n_pairs": n, "bootstrap": B, "precommit": Julius_rule(), "models": {}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        hits = {}
        for key, col in (("collided", 1), ("rescued", 2), ("oracle", 0)):
            Q = m.encode([p[col] for p in PAIRS], normalize_embeddings=True, show_progress_bar=False)
            hits[key] = ((Q @ D.T).argmax(1) == idx).astype(float)
        c, r, o = hits["collided"], hits["rescued"], hits["oracle"]

        def caf_of(sel):
            den = o[sel].mean() - c[sel].mean()
            if den <= 0: return np.nan
            return (r[sel].mean() - c[sel].mean()) / den
        point = caf_of(np.arange(n))

        boot = []
        for _ in range(B):
            s = rng.integers(0, n, n)          # resample ENTITY+DOCUMENT units
            v = caf_of(s)
            if not np.isnan(v): boot.append(v)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        if not np.isnan(point) and hi < 0.50:   verdict = "RESIDUAL IS THE LEAD"
        elif not np.isnan(point) and lo > 0.50: verdict = "TOKENIZATION IS THE LEAD"
        else:                                    verdict = "UNRESOLVED (CI crosses 0.50)"

        # ABSOLUTE ERROR MASS on the full 61-item benchmark
        Df = m.encode(full_docs, normalize_embeddings=True, show_progress_bar=False)
        Qf = m.encode(full_zh, normalize_embeddings=True, show_progress_bar=False)
        hit_f = ((Qf @ Df.T).argmax(1) == np.arange(nf))
        n_coll = sum(1 for k in range(nf) if 100 in tk(full_zh[k], add_special_tokens=False)["input_ids"])
        err_total = int((~hit_f).sum())
        err_coll = int(sum(1 for k in range(nf)
                           if (not hit_f[k]) and 100 in tk(full_zh[k], add_special_tokens=False)["input_ids"]))
        share = err_coll / err_total if err_total else float("nan")

        print(f"  collided {c.mean():.3f}  rescued {r.mean():.3f}  ORACLE(english) {o.mean():.3f}")
        print(f"  CAF = {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   (bootstrap by entity+document)")
        print(f"  VERDICT: {verdict}")
        print(f"  ABSOLUTE ERROR MASS (full 61-item set): {err_total} failures total, "
              f"{err_coll} on collided items = {share:.1%} of all error")
        print(f"    (collided items are {n_coll}/{nf} = {n_coll/nf:.1%} of the benchmark)")

        res["models"][mname] = {
            "collided": float(c.mean()), "rescued": float(r.mean()), "oracle_english": float(o.mean()),
            "caf": float(point), "ci95": [float(lo), float(hi)], "verdict": verdict,
            "err_total": err_total, "err_on_collided": err_coll,
            "collision_share_of_error": float(share),
            "collided_items_frac": float(n_coll / nf)}
        del m

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

def Julius_rule():
    return ("CAF<0.50 & upper<0.50 -> residual is lead; CAF>=0.50 & lower>0.50 -> "
            "tokenization is lead; CI crossing 0.50 -> unresolved")

if __name__ == "__main__":
    main()
