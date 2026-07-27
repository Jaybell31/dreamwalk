#!/usr/bin/env python3
"""T35 -- BUILD THE FIX. Confidence-weighted fusion vs RRF.

T33/T34 killed naive RRF hybrid in 8/8 tests -- it lands below its best
component because rank-only fusion cannot tell that an arm is returning noise.
Telling practitioners "don't use RRF" is a warning. Showing them what to use
instead is a fix. This test builds it.

DIAGNOSIS RESTATED: RRF sees only ORDER. A component returning pure noise still
produces a rank-1 document, and RRF gives it the same 1/(k+1) vote as a
confident correct hit. The information RRF discards is the SHAPE of the score
distribution -- a component that knows the answer has one score far above its
own runners-up; a component guessing has a flat top.

FUSION METHODS TESTED (all cheap, all deployable):
  rrf         reciprocal rank fusion, k=60           (the standard, our baseline)
  zscore      z-normalise each arm's scores, then sum
  minmax      min-max normalise to [0,1], then sum
  maxnorm     divide by each arm's own max
  gap_weight  weight each arm by its OWN top1-minus-top2 gap  <- THE CANDIDATE
  gap_gate    hard-select the arm with the larger normalised gap (winner-take-all)

gap_weight/gap_gate are the direct implementation of the diagnosis: the top1-top2
gap IS the arm's self-reported confidence, computed per query, requiring no
training, no labels, no tuning. If the diagnosis is right they must beat RRF.

PRE-REGISTERED:
  * PRIMARY: does any method beat max(dense, bm25) -- i.e. actually EARN the
    fusion -- averaged over all categories and perturbations? RRF scored 0/4.
  * The honest bar is not "beats RRF" (a low bar, RRF is broken here). It is
    "beats the better single arm", which is the only reason to fuse at all.
  * ABORT the fusion recommendation entirely if NO method clears that bar --
    then the correct public advice is "pick the right single arm per entity
    type", which is a legitimate and simpler finding.
  * Report the failure cases of the winner. A fix I cannot break is a fix I
    have not tested hard enough.
"""
import json
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t35_fusion.json"
MODELS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"]
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
CATS = ["real", "sku", "pharma", "cjk"]
PERTS = ["exact", "typo1", "drop1", "partial", "morph"]
FUSERS = ["rrf", "zscore", "minmax", "maxnorm", "gap_weight", "gap_gate"]


def mint(cat, i):
    if cat == "real":
        return REAL[i]
    if cat == "sku":
        return f"{chr(65+i%26)}{chr(65+(i//26)%26)}-{1000+(i*37)%9000}{chr(65+(i*7)%26)}"
    if cat == "pharma":
        return PH_A[i % len(PH_A)] + PH_B[(i // len(PH_A)) % len(PH_B)]
    return CJK_A[i % len(CJK_A)] + CJK_B[(i // len(CJK_A)) % len(CJK_B)]


def perturb(name, mode, i, cat):
    if mode == "exact":
        return name
    if mode == "morph":
        return f"information about {name} please"
    if cat == "cjk":
        ch = list(name)
        if mode == "typo1":
            j = i % len(ch); ch[j] = CJK_A[(i + 3) % len(CJK_A)][0]; return "".join(ch)
        if mode == "drop1":
            j = i % len(ch); return "".join(ch[:j] + ch[j + 1:])
        if mode == "partial":
            return "".join(ch[:max(1, len(ch) - 1)])
    s = list(name); alpha = "abcdefghijklmnopqrstuvwxyz"
    if mode == "typo1":
        j = 1 + (i % max(1, len(s) - 1))
        s[j] = alpha[(alpha.find(s[j].lower()) + 1) % 26] if s[j].lower() in alpha else "x"
        return "".join(s)
    if mode == "drop1":
        j = 1 + (i % max(1, len(s) - 1)); return "".join(s[:j] + s[j + 1:])
    if mode == "partial":
        return name[:max(3, int(len(name) * 0.6))]
    return name


def tok_ws(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def tok_ngram(s, n=2):
    t = tok_ws(s)
    cjk = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff]", s)
    return t + ["".join(cjk[i:i + n]) for i in range(max(0, len(cjk) - n + 1))] + cjk


class BM25:
    def __init__(self, docs, tokfn, k1=1.5, b=0.75):
        self.tok = tokfn; self.k1 = k1; self.b = b
        self.docs = [tokfn(d) for d in docs]
        self.N = len(self.docs)
        self.len = np.array([len(d) for d in self.docs], dtype=float)
        self.avg = self.len.mean()
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: np.log(1 + (self.N - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q):
        s = np.zeros(self.N)
        for t in self.tok(q):
            if t not in self.idf:
                continue
            idf = self.idf[t]
            for j in range(self.N):
                f = self.tf[j].get(t, 0)
                if f:
                    s[j] += idf * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.len[j] / self.avg))
        return s


def _ranks(r):
    o = np.argsort(-r); k = np.empty(len(r), int); k[o] = np.arange(len(r)); return k


def _gap(r):
    """Self-reported confidence: top1 - top2, scaled by spread. A noisy arm has
    a flat top and scores ~0; a confident arm stands well clear of its runner-up."""
    if r.max() <= r.min():
        return 0.0
    p = np.sort(r)[::-1]
    return float((p[0] - p[1]) / (r.max() - r.min() + 1e-12))


def _z(r):
    sd = r.std()
    return (r - r.mean()) / sd if sd > 1e-12 else np.zeros_like(r)


def _mm(r):
    rng_ = r.max() - r.min()
    return (r - r.min()) / rng_ if rng_ > 1e-12 else np.zeros_like(r)


def fuse(method, d, b):
    if method == "rrf":
        return 1.0 / (60 + _ranks(d) + 1) + 1.0 / (60 + _ranks(b) + 1)
    if method == "zscore":
        return _z(d) + _z(b)
    if method == "minmax":
        return _mm(d) + _mm(b)
    if method == "maxnorm":
        return (d / (abs(d).max() + 1e-12)) + (b / (abs(b).max() + 1e-12))
    gd, gb = _gap(d), _gap(b)
    if method == "gap_weight":
        return gd * _mm(d) + gb * _mm(b)
    if method == "gap_gate":
        return _mm(d) if gd >= gb else _mm(b)
    raise ValueError(method)


def main():
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="test")
    pool = [r["text"] for r in ds if len(r["text"]) > 150]
    dw, k = [], 0
    while len(dw) < N and k < len(pool):
        buf = []
        while len(buf) < DOC_WORDS and k < len(pool):
            buf += pool[k].split(); k += 1
        if len(buf) >= DOC_WORDS:
            dw.append(buf[:DOC_WORDS])
    body = [" ".join(w) for w in dw]
    print(f"corpus: {N} docs x {DOC_WORDS} words")
    idx = np.arange(N)
    rows = []

    for c in CATS:
        names = [mint(c, i) for i in range(N)]
        docs = [f"{names[i]}. {body[i]}" for i in range(N)]
        bm = BM25(docs, tok_ngram if c == "cjk" else tok_ws)
        qs = {p: [perturb(names[i], p, i, c) for i in range(N)] for p in PERTS}
        Sbm = {p: np.vstack([bm.scores(q) for q in qs[p]]) for p in PERTS}
        for mname in MODELS:
            short = mname.split("/")[-1]
            m = SentenceTransformer(mname, device="cuda")
            D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            for p in PERTS:
                Q = m.encode(qs[p], normalize_embeddings=True, show_progress_bar=False)
                Sd = Q @ D.T
                rec = {"cat": c, "model": short, "pert": p,
                       "dense": float((Sd.argmax(1) == idx).mean()),
                       "bm25": float((Sbm[p].argmax(1) == idx).mean())}
                for f in FUSERS:
                    F = np.vstack([fuse(f, Sd[i], Sbm[p][i]) for i in range(N)])
                    rec[f] = float((F.argmax(1) == idx).mean())
                rec["best_single"] = max(rec["dense"], rec["bm25"])
                rows.append(rec)
            del m
        print(f"  done {c}", flush=True)

    print("\n" + "=" * 86)
    print("MEAN OVER ALL 40 CELLS (4 cats x 2 models x 5 perturbations)")
    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ["dense", "bm25", "best_single"] + FUSERS}
    for k in ["dense", "bm25", "best_single"] + FUSERS:
        mark = ""
        if k in FUSERS:
            mark = "  <- BEATS best single arm" if mean[k] > mean["best_single"] else "  (below best single)"
        print(f"  {k:12} {mean[k]:.4f}{mark}")

    print("\nPER-CELL WIN RATE vs best single arm (higher is better, 40 cells)")
    winrate = {}
    for f in FUSERS:
        w = sum(1 for r in rows if r[f] > r["best_single"] + 1e-9)
        t = sum(1 for r in rows if abs(r[f] - r["best_single"]) <= 1e-9)
        winrate[f] = w / len(rows)
        print(f"  {f:12} wins {w:2}/40  ties {t:2}  losses {len(rows)-w-t:2}   ({100*w/len(rows):.0f}%)")

    best = max(FUSERS, key=lambda f: mean[f])
    print(f"\nBEST FUSER: {best}  mean={mean[best]:.4f}  vs best_single={mean['best_single']:.4f}"
          f"  vs rrf={mean['rrf']:.4f}")

    print(f"\nWHERE {best} LOSES (a fix I cannot break is untested):")
    losses = sorted((r for r in rows if r[best] < r["best_single"] - 1e-9),
                    key=lambda r: r[best] - r["best_single"])[:8]
    if not losses:
        print("  none -- it never underperforms the better arm")
    for r in losses:
        print(f"  {r['cat']:7} {r['model'][:12]:12} {r['pert']:8} "
              f"{best}={r[best]:.3f}  best_single={r['best_single']:.3f} "
              f"(dense={r['dense']:.3f} bm25={r['bm25']:.3f})")

    res = {"rows": rows, "mean": mean, "winrate": winrate, "best_fuser": best,
           "beats_best_single": bool(mean[best] > mean["best_single"]),
           "beats_rrf": bool(mean[best] > mean["rrf"])}
    res["verdict"] = ("CONFIDENCE-WEIGHTED FUSION EARNS ITS KEEP" if res["beats_best_single"]
                      else "ABORT fusion advice -- recommend per-entity single-arm selection")
    print(f"\nVERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
