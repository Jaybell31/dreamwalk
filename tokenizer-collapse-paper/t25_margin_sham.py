#!/usr/bin/env python3
"""T25 -- GPT'S CEILING-FREE NEGATIVE CONTROL + THE SHAM DETERMINISM TEST.

PART 1 -- MARGIN CONTROL (repairs what Julius broke)
Julius killed bge-m3 as a negative control: it is CEILING-LIMITED, so
1.000 -> 1.000 "cannot distinguish no-mechanism from ceiling". GPT's fix keeps
bge-m3 but replaces the ceilinged R@1 endpoint with a CONTINUOUS MARGIN:

    m_i     = cos(q_i, d_i+) - max_{j != i} cos(q_i, d_ij)
    delta_m = m_i(rescued spelling) - m_i(collided spelling)

Index the documents ONCE in canonical form; change ONLY the query spelling.
A margin has no ceiling -- a model already at R@1 1.000 can still show its
margin move. Architecture, corpus, candidate difficulty and candidate identity
are all held fixed, which is exactly what baseline-matching would have broken.

GPT'S PRE-REGISTERED GATES (his numbers):
  PASS (bge-m3 is a clean negative control):
        |mean delta_m| < 0.01 AND paired 95% CI contains 0
  FAIL collision-specific interpretation:
        lower 95% CI for delta_m > 0.01
  GRAY: point estimate in 0.00-0.01 with CI excluding neither threshold
The English-vocabulary models are run too, as the POSITIVE contrast: they
should show a large positive delta_m. If they do not, the margin metric itself
is broken and the control result is meaningless.

PART 2 -- SHAM / DETERMINISM TEST, and it settles an earlier retraction.
GPT: "change Unicode while requiring input_ids and attention_mask to remain
exactly identical. Any rank change under identical token sequences, or any
embedding cosine below 0.9999999, means the pipeline is nondeterministic or
contaminated."
This is also the receipt for a claim I RETRACTED earlier tonight. I wrote that
collided documents produce "bit-identical vectors" on the basis of a printed
cos=1.000000, then retracted it because exact equality was never measured.
Here it is measured properly: two DIFFERENT Chinese words that both tokenize to
exactly [100] are encoded and compared with np.array_equal and max elementwise
difference. Either the vectors are exactly equal (claim restored, with a
receipt) or they are not (retraction stands and the pipeline is nondeterministic).
"""
import json
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import importlib.util

OUT = "/home/jason/dream_harness/t25_margin_sham.json"
spec = importlib.util.spec_from_file_location("t19", "/home/jason/dream_harness/t19_rescue.py")
t19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t19)
PAIRS = t19.PAIRS

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]
B = 10000
rng = np.random.default_rng(20260727)


def margins(Q, D, idx):
    """m_i = cos(q_i, d_i+) - max_{j!=i} cos(q_i, d_ij), no ceiling."""
    S = Q @ D.T
    gold = S[idx, idx].copy()
    S2 = S.copy()
    S2[idx, idx] = -np.inf
    return gold - S2.max(1)


def main():
    docs = [p[4] for p in PAIRS]
    coll = [p[1] for p in PAIRS]
    resc = [p[2] for p in PAIRS]
    n = len(PAIRS)
    idx = np.arange(n)
    res = {"n": n, "gates": {"pass_abs_mean_below": 0.01, "fail_lower_ci_above": 0.01},
           "models": {}, "sham": {}}

    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # sham pair: two DIFFERENT words whose input_ids are byte-for-byte identical
    sham_a, sham_b = "车", "马"
    ia = tk(sham_a, add_special_tokens=False)["input_ids"]
    ib = tk(sham_b, add_special_tokens=False)["input_ids"]
    print(f"SHAM PAIR {sham_a!r} ids={ia}  vs  {sham_b!r} ids={ib}  identical={ia == ib}")

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        QC = m.encode(coll, normalize_embeddings=True, show_progress_bar=False)
        QR = m.encode(resc, normalize_embeddings=True, show_progress_bar=False)

        mC = margins(QC, D, idx)
        mR = margins(QR, D, idx)
        d = mR - mC
        point = float(d.mean())
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
        lo, hi = np.percentile(boot, [2.5, 97.5])

        if abs(point) < 0.01 and lo < 0 < hi:
            verdict = "PASS -- clean negative control (no collision effect)"
        elif lo > 0.01:
            verdict = "COLLISION EFFECT PRESENT (lower CI > 0.01)"
        else:
            verdict = "GRAY"

        print(f"  mean margin collided={mC.mean():+.4f}  rescued={mR.mean():+.4f}")
        print(f"  delta_m = {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {verdict}")

        # ---- sham / determinism, English-vocab models only (bge-m3 will differ) ----
        if mname != "BAAI/bge-m3":
            E = m.encode([sham_a, sham_b], normalize_embeddings=True, show_progress_bar=False)
            exact = bool(np.array_equal(E[0], E[1]))
            maxdiff = float(np.abs(E[0] - E[1]).max())
            cos = float(E[0] @ E[1])
            ok = cos >= 0.9999999
            print(f"  SHAM {sham_a}/{sham_b}: exactly_equal={exact}  max|diff|={maxdiff:.3e}  cos={cos:.9f}"
                  f"  -> {'deterministic (identical ids -> identical vector)' if ok else 'PIPELINE NONDETERMINISTIC/CONTAMINATED'}")
            res["sham"][mname] = {"input_ids_identical": ia == ib, "vectors_exactly_equal": exact,
                                  "max_abs_diff": maxdiff, "cosine": cos, "deterministic": ok}

        res["models"][mname] = {"margin_collided": float(mC.mean()), "margin_rescued": float(mR.mean()),
                                "delta_m": point, "ci95": [float(lo), float(hi)], "verdict": verdict}
        del m

    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
