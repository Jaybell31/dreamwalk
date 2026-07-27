#!/usr/bin/env python3
"""T33 -- THE REMEDY TEST. Does anything actually FIX the OOV retrieval gap?

Everything so far is diagnosis. T32 established the honest shape of the problem:
OOV entities lose 2.7-6.2x recall vs in-vocab names AT EVERY CHUNK SIZE, and
chunk tuning does not touch it. So the practitioner question is no longer "how
bad" but "what works". This is the last piece before write-up.

FOUR REMEDIES, all cheap enough to actually ship:
  dense_raw   baseline: dense embedding, entity as-is
  dense_sub   plain-word/latin substitution of the entity in BOTH query and doc
              (the T9/T11 fix ladder, generalised beyond emoji)
  bm25        pure lexical retrieval -- no embedding model at all
  hybrid      dense_raw + bm25, reciprocal rank fusion (the standard production
              answer, tested rather than assumed)

WHY BM25 IS THE INTERESTING ARM: the whole defect is that the tokenizer destroys
the entity string. BM25 never tokenizes into wordpieces -- it matches surface
terms. If the diagnosis is right, BM25 should be LARGELY IMMUNE for sku/pharma
(which are exact ASCII string matches) and should FAIL for cjk under a
whitespace tokenizer (no spaces between Japanese terms) unless char-ngrams are
used. That asymmetric prediction is what makes this a test and not a demo.

PRE-REGISTERED PREDICTIONS:
  P1 bm25 >> dense_raw for sku and pharma (exact ASCII surface forms).
  P2 bm25 fails for cjk with whitespace tokenization; char-ngram BM25 rescues
     it. If plain BM25 already nails cjk, my tokenization model is wrong.
  P3 hybrid >= max(dense, bm25) -- if fusion is WORSE than its best component,
     the standard production advice is actively harmful here and that is a
     finding worth reporting on its own.
  ABORT the remedy recommendation if no arm beats dense_raw by >= 0.10 absolute
  on the OOV categories -- then we have a problem with no cheap fix, and the
  honest output is "this needs a vocabulary change", not a tip.
"""
import json
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t33_remedies.json"
MODELS = ["BAAI/bge-base-en-v1.5", "thenlper/gte-base", "BAAI/bge-m3"]
N = 120
DOC_WORDS = 64

REAL = """Kensington Fairbanks Ashford Brentwood Carlton Dunmore Everly Fenwick
Glenmore Hartley Inglewood Jarvis Kirkland Langley Merton Norwood Oakhurst
Pembroke Quinton Ravenswood Stanhope Thornbury Upton Vernon Westbrook Yardley
Aldridge Bramwell Coventry Dorset Ellsworth Fairmont Granville Havelock
Ivybridge Jamestown Kingsley Lockwood Marlowe Newbury Ormsby Prescott Quimby
Redmond Sheffield Tarrant Ullswater Vandermeer Whitmore Yarrow Ainsley
Blackwood Cranfield Devonport Eastleigh Fitzroy Galloway Hollingsworth Ivorton
Jeffries Kimberley Lyndhurst Middleton Northfield Oldbury Pinehurst Queensbury
Rockwell Southgate Templeton Underwood Vanbrugh Wexford Yorkshire Abernathy
Balfour Chadwick Delacroix Edgerton Fallowfield Gresham Hawthorne Illingworth
Jessup Kensworth Lambourne Montgomery Netherfield Oakleigh Pendleton Quarrington
Rutherford Stapleton Thistlewood Ulverston Vaughan Wintersgill Ashcombe
Bexhill Chesterton Dunwoody Elmhurst Fothergill Greenhalgh Huddleston Inverness
Jorgenson Kettering Lindisfarne Marchbanks Norrington Ottershaw Pickering
Quenby Ravensworth Shackleton Tewkesbury Uppingham Verwood Wadsworth""".split()

PH_A = ["Zolpi", "Craxa", "Vemli", "Tarso", "Nuvig", "Palbo", "Ristu", "Ombra",
        "Xelda", "Frava", "Quinta", "Yavro", "Dexur", "Melvo", "Sorbi"]
PH_B = ["draxen", "tinib", "zumab", "prazil", "vastat", "olone", "cyclav",
        "fenac", "mustine", "tropium"]
CJK_A = ["中村", "山田", "小林", "佐藤", "田中", "渡辺", "伊藤", "高橋", "吉田", "松本",
         "井上", "木村", "森田", "斎藤", "清水"]
CJK_B = ["建設", "工業", "商事", "電機", "製薬", "運輸", "食品", "化学", "重工", "銀行"]
CJK_SUB_A = {"中村": "Nakamura", "山田": "Yamada", "小林": "Kobayashi", "佐藤": "Sato",
             "田中": "Tanaka", "渡辺": "Watanabe", "伊藤": "Ito", "高橋": "Takahashi",
             "吉田": "Yoshida", "松本": "Matsumoto", "井上": "Inoue", "木村": "Kimura",
             "森田": "Morita", "斎藤": "Saito", "清水": "Shimizu"}
CJK_SUB_B = {"建設": "Construction", "工業": "Industries", "商事": "Trading",
             "電機": "Electric", "製薬": "Pharmaceutical", "運輸": "Transport",
             "食品": "Foods", "化学": "Chemical", "重工": "Heavy Industries",
             "銀行": "Bank"}
CATS = ["real", "sku", "pharma", "cjk"]


def mint(cat, i):
    if cat == "real":
        return REAL[i]
    if cat == "sku":
        return f"{chr(65+i%26)}{chr(65+(i//26)%26)}-{1000+(i*37)%9000}{chr(65+(i*7)%26)}"
    if cat == "pharma":
        return PH_A[i % len(PH_A)] + PH_B[(i // len(PH_A)) % len(PH_B)]
    if cat == "cjk":
        return CJK_A[i % len(CJK_A)] + CJK_B[(i // len(CJK_A)) % len(CJK_B)]
    raise ValueError(cat)


def substitute(cat, i, name):
    """Plain-word/latin form of the entity -- the generalised T9/T11 fix."""
    if cat == "cjk":
        return CJK_SUB_A[CJK_A[i % len(CJK_A)]] + " " + CJK_SUB_B[CJK_B[(i // len(CJK_A)) % len(CJK_B)]]
    if cat == "sku":
        # spell out the code so wordpieces align: "AA-1000A" -> "model A A 1000 A"
        return "model " + " ".join(re.findall(r"[A-Z]|\d+", name))
    if cat == "pharma":
        a = PH_A[i % len(PH_A)]
        b = PH_B[(i // len(PH_A)) % len(PH_B)]
        return f"drug {a} {b}"
    return name


def tok_ws(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def tok_ngram(s, n=2):
    """char n-grams -- needed for languages without whitespace word boundaries."""
    t = tok_ws(s)
    cjk = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", s)
    grams = ["".join(cjk[i:i + n]) for i in range(max(0, len(cjk) - n + 1))]
    return t + grams + cjk


class BM25:
    def __init__(self, docs, tokfn, k1=1.5, b=0.75):
        self.tok = tokfn; self.k1 = k1; self.b = b
        self.docs = [tokfn(d) for d in docs]
        self.N = len(self.docs)
        self.len = np.array([len(d) for d in self.docs], dtype=float)
        self.avg = self.len.mean() if self.N else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: np.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q):
        qt = self.tok(q)
        s = np.zeros(self.N)
        for t in qt:
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for j in range(self.N):
                f = self.tf[j].get(t, 0)
                if f:
                    s[j] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[j] / self.avg))
        return s


def rrf(*rank_lists, k=60):
    out = np.zeros(len(rank_lists[0]))
    for r in rank_lists:
        order = np.argsort(-r)
        ranks = np.empty(len(r), dtype=int)
        ranks[order] = np.arange(len(r))
        out += 1.0 / (k + ranks + 1)
    return out


def main():
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="test")
    pool = [r["text"] for r in ds if len(r["text"]) > 150]
    docs_words, k = [], 0
    while len(docs_words) < N and k < len(pool):
        buf = []
        while len(buf) < DOC_WORDS and k < len(pool):
            buf += pool[k].split(); k += 1
        if len(buf) >= DOC_WORDS:
            docs_words.append(buf[:DOC_WORDS])
    if len(docs_words) < N:
        print(f"ABORT: only {len(docs_words)} docs"); return
    body = [" ".join(w) for w in docs_words]
    print(f"corpus: {N} docs x {DOC_WORDS} words (mid chunk size, where both arms are live)")

    idx = np.arange(N)
    res = {"n": N, "doc_words": DOC_WORDS, "cats": {}}

    # ---- lexical arms are model-independent: compute once ----
    lex = {}
    for c in CATS:
        names = [mint(c, i) for i in range(N)]
        subs = [substitute(c, i, names[i]) for i in range(N)]
        docs_raw = [f"{names[i]}. {body[i]}" for i in range(N)]
        docs_sub = [f"{subs[i]}. {body[i]}" for i in range(N)]
        b_ws = BM25(docs_raw, tok_ws)
        b_ng = BM25(docs_raw, tok_ngram)
        S_ws = np.vstack([b_ws.scores(q) for q in names])
        S_ng = np.vstack([b_ng.scores(q) for q in names])
        lex[c] = {"names": names, "subs": subs, "docs_raw": docs_raw,
                  "docs_sub": docs_sub, "S_ws": S_ws, "S_ng": S_ng}
        print(f"  {c:7} bm25_ws R@1={(S_ws.argmax(1)==idx).mean():.3f}   "
              f"bm25_ngram R@1={(S_ng.argmax(1)==idx).mean():.3f}")

    for mname in MODELS:
        short = mname.split("/")[-1]
        print(f"\n=== {short} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        for c in CATS:
            L = lex[c]
            Draw = m.encode(L["docs_raw"], normalize_embeddings=True, show_progress_bar=False)
            Qraw = m.encode(L["names"], normalize_embeddings=True, show_progress_bar=False)
            Dsub = m.encode(L["docs_sub"], normalize_embeddings=True, show_progress_bar=False)
            Qsub = m.encode(L["subs"], normalize_embeddings=True, show_progress_bar=False)
            S_dense = Qraw @ Draw.T
            S_sub = Qsub @ Dsub.T
            arms = {
                "dense_raw": S_dense,
                "dense_sub": S_sub,
                "bm25": L["S_ws"],
                "bm25_ngram": L["S_ng"],
                "hybrid": np.vstack([rrf(S_dense[i], L["S_ws"][i]) for i in range(N)]),
                "hybrid_ngram": np.vstack([rrf(S_dense[i], L["S_ng"][i]) for i in range(N)]),
            }
            r = {a: float((S.argmax(1) == idx).mean()) for a, S in arms.items()}
            res["cats"].setdefault(c, {})[short] = r
            print(f"  {c:7} " + "  ".join(f"{a}={v:.3f}" for a, v in r.items()))
        del m

    print("\n" + "=" * 78)
    print("BEST REMEDY PER CATEGORY (mean over models), vs dense_raw baseline")
    summary = {}
    for c in CATS:
        per = res["cats"][c]
        arms = list(next(iter(per.values())).keys())
        mean = {a: float(np.mean([per[s][a] for s in per])) for a in arms}
        base = mean["dense_raw"]
        best = max((a for a in arms if a != "dense_raw"), key=lambda a: mean[a])
        summary[c] = {"mean": mean, "baseline": base, "best_arm": best,
                      "best": mean[best], "gain": mean[best] - base}
        print(f"  {c:7} dense_raw={base:.3f}  BEST={best} {mean[best]:.3f}  "
              f"gain={mean[best]-base:+.3f}")
    res["summary"] = summary

    # P1: bm25 >> dense_raw for sku/pharma
    p1 = all(summary[c]["mean"]["bm25"] > summary[c]["mean"]["dense_raw"] + 0.10
             for c in ("sku", "pharma"))
    # P2: plain bm25 fails cjk, ngram rescues it
    p2 = (summary["cjk"]["mean"]["bm25"] < 0.30
          and summary["cjk"]["mean"]["bm25_ngram"] > summary["cjk"]["mean"]["bm25"] + 0.10)
    # P3: hybrid >= best component
    p3 = all(summary[c]["mean"]["hybrid"] >= max(summary[c]["mean"]["dense_raw"],
                                                 summary[c]["mean"]["bm25"]) - 1e-9
             for c in CATS)
    oov_gain = float(np.mean([summary[c]["gain"] for c in ("sku", "pharma", "cjk")]))
    res.update({"P1_bm25_beats_dense_ascii": bool(p1),
                "P2_ngram_rescues_cjk": bool(p2),
                "P3_hybrid_not_worse": bool(p3),
                "mean_oov_gain": oov_gain})
    print(f"\nP1 bm25 >> dense on sku/pharma      : {p1}")
    print(f"P2 plain bm25 fails cjk, ngram fixes: {p2}  "
          f"(bm25={summary['cjk']['mean']['bm25']:.3f} ngram={summary['cjk']['mean']['bm25_ngram']:.3f})")
    print(f"P3 hybrid never worse than best comp: {p3}")
    print(f"mean OOV gain from best remedy      : {oov_gain:+.3f}")
    res["verdict"] = ("CHEAP REMEDY EXISTS" if oov_gain >= 0.10
                      else "ABORT -- no cheap remedy; needs vocabulary change")
    print(f"VERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
