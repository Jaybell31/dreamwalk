#!/usr/bin/env python3
"""T20 -- TWO SEAT-SET FALSIFIERS, RUN TOGETHER.

R11 produced an unusual convergence: Grok and Gemini INDEPENDENTLY answered
question C the same way -- the 65.7% residual failure among tokenizer-DISTINCT
items is the LEAD, not a limitation paragraph -- and each named a different
mechanical test that decides it. Julius had already flagged the same anomaly in
R10. Three seats, three routes, one target. So we run both tests.

--------------------------------------------------------------------------
TEST A -- GROK'S FREQUENCY CONFOUND KILLER (his E, verbatim gate)
  Claim under attack: "In-vocab CJK tokens are merely UN-COLLIDED; their
  superior retrieval is not driven by differential training frequency."
  This is Julius's R10 confound restated as a falsifiable statistic.
  Method: rank DISTINCT-arm items by a training-signal proxy, Spearman
  against per-item retrieval success.
  GROK'S ABORT: if Spearman rho >= 0.40 with p < 0.05, "merely un-collided"
  is DEAD and collision must be demoted to co-cause with training frequency.

  PROXY CHOICE, STATED HONESTLY. We do not have BERT's training corpus, so
  true token frequency is unavailable. We use the INPUT EMBEDDING L2 NORM as
  the proxy. Justification is not hand-waving: arXiv 2405.05417 (Magikarp,
  VERIFIED FETCHED) detects under-trained tokens precisely from input-embedding
  geometry -- tokens absent from training retain near-initialisation vectors.
  Norm is therefore a defensible monotone proxy for training exposure.
  WEAKNESS WE WILL PRINT: it is a PROXY. A null result here does NOT prove
  frequency is irrelevant; it proves this proxy did not detect it. We report
  token-ID rank as a second, independent proxy (WordPiece IDs are assigned in
  descending corpus frequency during vocab construction).

--------------------------------------------------------------------------
TEST B -- GEMINI'S SEMANTIC-HOLLOWNESS DECIDER (his C, verbatim gate)
  Claim: if tokenization is DISTINCT but the Chinese form still fails, is the
  residual failure SEMANTIC rather than lexical?
  Method: for DISTINCT items only, cosine between the Chinese surface form and
  its exact English equivalent, same model.
  GEMINI'S GATE: if cross-lingual cosine stays BELOW 0.75 while tokenization
  is distinct, the failure is undeniably semantic -> the 65.7% becomes the
  paper's lead and the framing moves from "tokenizer bug report" to
  "cross-lingual alignment failure".

Both tests run on the frozen 61-item T17 set. No new items, no reselection.
"""
import json, numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr
import importlib.util

OUT = "/home/jason/dream_harness/t20_falsifiers.json"
spec = importlib.util.spec_from_file_location("t17", "/home/jason/dream_harness/t17_within_model.py")
t17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t17)
ITEMS = t17.ITEMS                      # (zh, en, doc)

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base"]

def main():
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    docs = [i[2] for i in ITEMS]
    zh   = [i[0] for i in ITEMS]
    en   = [i[1] for i in ITEMS]
    n = len(ITEMS); idx = np.arange(n)

    distinct = [k for k in range(n) if 100 not in tk(zh[k], add_special_tokens=False)["input_ids"]]
    print(f"frozen T17 set: {n} items, DISTINCT arm = {len(distinct)}")

    res = {"n_items": n, "n_distinct": len(distinct), "models": {},
           "gates": {"grok_freq_spearman_abort": 0.40, "gemini_semantic_cos": 0.75}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D  = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        QZ = m.encode(zh,   normalize_embeddings=True, show_progress_bar=False)
        QE = m.encode(en,   normalize_embeddings=True, show_progress_bar=False)
        hit_zh = ((QZ @ D.T).argmax(1) == idx).astype(float)

        # ---- TEST A: frequency proxies vs success, DISTINCT arm only ----
        emb = m[0].auto_model.get_input_embeddings().weight.detach().cpu().numpy()
        norms, ids_rank = [], []
        for k in distinct:
            tid = [t for t in tk(zh[k], add_special_tokens=False)["input_ids"] if t != 100]
            norms.append(float(np.mean([np.linalg.norm(emb[t]) for t in tid])) if tid else np.nan)
            ids_rank.append(float(np.mean(tid)) if tid else np.nan)
        y = hit_zh[distinct]
        ok = ~np.isnan(norms)
        r_norm, p_norm = spearmanr(np.array(norms)[ok], y[ok])
        r_id,   p_id   = spearmanr(np.array(ids_rank)[ok], y[ok])
        # WordPiece IDs ascend as frequency DEscends -> flip sign for "frequency"
        r_id = -r_id
        grok_fires = (abs(r_norm) >= 0.40 and p_norm < 0.05) or (abs(r_id) >= 0.40 and p_id < 0.05)

        # ---- TEST B: cross-lingual cosine, DISTINCT arm only ----
        xling = float(np.mean([float(QZ[k] @ QE[k]) for k in distinct]))
        xling_fail = float(np.mean([float(QZ[k] @ QE[k]) for k in distinct if hit_zh[k] == 0]))
        gemini_semantic = xling < 0.75

        print(f"  DISTINCT-arm R@1 = {y.mean():.3f}  (residual failure {1-y.mean():.1%})")
        print(f"  A) freq proxy embed-norm  rho={r_norm:+.3f} p={p_norm:.4f}")
        print(f"  A) freq proxy token-id    rho={r_id:+.3f} p={p_id:.4f}")
        print(f"     GROK ABORT (|rho|>=0.40 & p<.05): {'FIRES -- frequency confound REAL' if grok_fires else 'does NOT fire'}")
        print(f"  B) cross-lingual cos(zh,en) DISTINCT = {xling:.3f}   (failed items only: {xling_fail:.3f})")
        print(f"     GEMINI GATE (<0.75 => semantic): {'SEMANTIC FAILURE CONFIRMED' if gemini_semantic else 'not semantic by this gate'}")

        res["models"][mname] = {
            "distinct_r1": float(y.mean()), "residual_failure": float(1 - y.mean()),
            "spearman_embed_norm": float(r_norm), "p_embed_norm": float(p_norm),
            "spearman_token_id_freq": float(r_id), "p_token_id": float(p_id),
            "grok_abort_fires": bool(grok_fires),
            "xlingual_cos_distinct": xling, "xlingual_cos_failed_only": xling_fail,
            "gemini_semantic_confirmed": bool(gemini_semantic)}
        del m

    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
