#!/usr/bin/env python3
"""T26 -- JULIUS'S TWO R12 CORRECTIONS. Both attack MY conclusions.

--------------------------------------------------------------------------
PART 1 -- TOST EQUIVALENCE. "PASS clean negative control" is probably FALSE.
Julius, R12-E: a CI containing zero is NOT evidence of practical absence.
"The current interval of [-0.0248, +0.0331] permits an effect larger than
gte-base's observed +0.0266. Thus the data simultaneously allow 'no bge-m3
effect' and 'an effect as large as one of the detected collapsed-model
effects.'" That is exactly right and it invalidates the word PASS.

Correct procedure is TWO ONE-SIDED TESTS against the practical-null bound we
already declared (+/-0.01): equivalence requires the ENTIRE 90% CI to lie
inside [-0.01, +0.01].
  ABORT "clean negative control" if either endpoint falls outside +/-0.01.
He predicts it fails tonight. We run it and report whichever way it lands.

--------------------------------------------------------------------------
PART 2 -- FORCED-COLLISION ABLATION. The strongest remaining hostile attack.
Julius, R12-C: "You did not manipulate collision status alone; you replaced an
information-poor tokenization with a pretrained, semantically meaningful token.
The experiment therefore proves that better lexical representations improve
retrieval, not specifically that token collision caused the original failure."
The derangement (T24) and codepoint (T23) results kill ARBITRARY IDENTITY
coding but do NOT separate collision-removal from better-embedding-quality.

HIS INTERVENTION: take the RESCUED spelling and overwrite ONLY its target token
embedding with the COLLIDED token's ([UNK]) embedding vector. Everything else
identical -- document index, surrounding tokens, attention mask, sequence
length, weights. Three paired margins:
    m_raw_collided,  m_rescued,  m_rescued_with_forced_collision
DECISIVE PREDICTION: m_rescued > m_forced ~= m_raw_collided
  ABORT explicit collision-causality language if forced collision reproduces
    < 50% of the rescued-minus-collided margin difference in >= 2 of 3 models,
    or if the rescued-vs-forced contrast has p >= 0.01.
  STRONGLY CORROBORATING if it erases >= 75% of the rescue gain in all three
    models with p < 0.01 in at least two.

REPORTING FIX (Julius, same round): never print permutation p = 0.000. With B
permutations report p = (k+1)/(B+1). Applied throughout.
"""
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import importlib.util

OUT = "/home/jason/dream_harness/t26_tost_forced.json"
spec = importlib.util.spec_from_file_location("t19", "/home/jason/dream_harness/t19_rescue.py")
t19 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t19)
PAIRS = t19.PAIRS

EN_MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
             "BAAI/bge-base-en-v1.5",
             "thenlper/gte-base"]
B = 10000
NPERM = 10000
rng = np.random.default_rng(20260727)
UNK = 100


def margins(Q, D, idx):
    S = Q @ D.T
    gold = S[idx, idx].copy()
    S2 = S.copy(); S2[idx, idx] = -np.inf
    return gold - S2.max(1)


def perm_p(a, b, nperm=NPERM):
    """Paired permutation on sign flips. Never returns 0 -- (k+1)/(B+1)."""
    d = a - b
    obs = d.mean()
    k = 0
    for _ in range(nperm):
        s = rng.choice([-1.0, 1.0], size=len(d))
        if (d * s).mean() >= obs:
            k += 1
    return (k + 1) / (nperm + 1)


def main():
    docs = [p[4] for p in PAIRS]
    coll = [p[1] for p in PAIRS]
    resc = [p[2] for p in PAIRS]
    n = len(PAIRS); idx = np.arange(n)
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    res = {"n": n, "models": {}, "tost": {}}

    # ---------- PART 1: TOST on bge-m3 ----------
    print("=== PART 1 -- TOST EQUIVALENCE, bge-m3, bound +/-0.01 ===")
    m = SentenceTransformer("BAAI/bge-m3", device="cuda")
    D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    QC = m.encode(coll, normalize_embeddings=True, show_progress_bar=False)
    QR = m.encode(resc, normalize_embeddings=True, show_progress_bar=False)
    d = margins(QR, D, idx) - margins(QC, D, idx)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    equiv = (lo90 > -0.01) and (hi90 < 0.01)
    print(f"  delta_m = {d.mean():+.4f}   90% CI [{lo90:+.4f}, {hi90:+.4f}]   95% CI [{lo95:+.4f}, {hi95:+.4f}]")
    print(f"  TOST equivalence within +/-0.01: {'PASS -- practical equivalence established' if equiv else 'FAIL -- CANNOT claim clean negative control'}")
    res["tost"] = {"delta_m": float(d.mean()), "ci90": [float(lo90), float(hi90)],
                   "ci95": [float(lo95), float(hi95)], "equivalence": bool(equiv),
                   "bound": 0.01}
    del m

    # ---------- PART 2: forced-collision ablation ----------
    print("\n=== PART 2 -- FORCED-COLLISION ABLATION ===")
    # one-token matched subset: rescued form must be a SINGLE non-UNK token
    subset = []
    for k, p in enumerate(PAIRS):
        ri = tk(p[2], add_special_tokens=False)["input_ids"]
        ci = tk(p[1], add_special_tokens=False)["input_ids"]
        if len(ri) == 1 and ri[0] != UNK and len(ci) >= 1:
            subset.append(k)
    print(f"  one-token matched subset: {len(subset)}/{n} pairs")

    for mname in EN_MODELS:
        print(f"\n  --- {mname} ---", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        dev = m.device
        emb_layer = m[0].auto_model.get_input_embeddings()
        unk_vec = emb_layer.weight.data[UNK].clone()

        sub_docs = [docs[k] for k in subset]
        si = np.arange(len(subset))
        D = m.encode(sub_docs, normalize_embeddings=True, show_progress_bar=False)
        QC = m.encode([coll[k] for k in subset], normalize_embeddings=True, show_progress_bar=False)
        QR = m.encode([resc[k] for k in subset], normalize_embeddings=True, show_progress_bar=False)

        # FORCED: overwrite the rescued token's embedding row with UNK's vector
        forced = []
        for k in subset:
            tid = tk(resc[k], add_special_tokens=False)["input_ids"][0]
            saved = emb_layer.weight.data[tid].clone()
            emb_layer.weight.data[tid] = unk_vec          # only this row changes
            v = m.encode([resc[k]], normalize_embeddings=True, show_progress_bar=False)[0]
            emb_layer.weight.data[tid] = saved            # restore immediately
            forced.append(v)
        QF = np.vstack(forced)

        mC, mR, mF = margins(QC, D, si), margins(QR, D, si), margins(QF, D, si)
        gain = mR.mean() - mC.mean()
        erased = (mR.mean() - mF.mean()) / gain if gain != 0 else float("nan")
        p_rf = perm_p(mR, mF)

        print(f"    margins: collided {mC.mean():+.4f}  rescued {mR.mean():+.4f}  FORCED {mF.mean():+.4f}")
        print(f"    rescue gain {gain:+.4f};  forced-collision erases {erased:.1%} of it;  p(rescued vs forced) = {p_rf:.4f}")
        res["models"][mname] = {"m_collided": float(mC.mean()), "m_rescued": float(mR.mean()),
                                "m_forced": float(mF.mean()), "rescue_gain": float(gain),
                                "fraction_erased": float(erased), "p_rescued_vs_forced": float(p_rf),
                                "n_subset": len(subset)}
        del m

    fr = [v["fraction_erased"] for v in res["models"].values()]
    ps = [v["p_rescued_vs_forced"] for v in res["models"].values()]
    strong = all(f >= 0.75 for f in fr) and sum(1 for p in ps if p < 0.01) >= 2
    abort = (sum(1 for f in fr if f < 0.50) >= 2) or all(p >= 0.01 for p in ps)
    res["verdict"] = ("STRONGLY CORROBORATING -- collision causality supported" if strong
                      else "ABORT explicit collision-causality language" if abort
                      else "PARTIAL -- neither gate cleanly met")
    print(f"\nVERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
