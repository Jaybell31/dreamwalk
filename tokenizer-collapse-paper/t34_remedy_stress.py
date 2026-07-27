#!/usr/bin/env python3
"""T34 -- STRESS THE REMEDY. Where does BM25's 1.000 actually break?

T33 found BM25 scores exactly 1.000 on real/sku/pharma. I flagged that as
degenerate in the log before anyone could quote it: the entity appears verbatim
in exactly one document, so lexical matching is a lookup. Real users do not type
verbatim. This test pays off that debt.

PERTURBED QUERIES -- the entity in the DOCUMENT stays canonical; only the QUERY
is degraded, which is exactly the real-world asymmetry (clean index, messy user):
  exact      unmodified                                  (T33 replication)
  case       lowercased                                  ("xr-4520b")
  typo1      one character substituted
  drop1      one character deleted
  space      internal separator/space variation          ("XR 4520B")
  partial    a prefix of the entity only
  morph      surrounding query words added                ("info about X please")

WHY THIS MATTERS FOR THE HEADLINE: if BM25 collapses under mild perturbation
while dense degrades gracefully, then "just add BM25" is bad advice and the
hybrid story needs rewriting. If BM25 stays strong, the T33 recommendation
holds and gains a robustness envelope. Either way the vendor-facing sentence
changes, so this is the test that decides what we publish.

PRE-REGISTERED:
  * REPORT the full degradation curve per arm, no threshold gaming.
  * PRIMARY: does BM25 still beat dense_raw on OOV categories under the WORST
    perturbation? If yes, the remedy is robust and shippable.
  * If BM25 falls BELOW dense on any perturbation, say so plainly and revise
    the recommendation to hybrid-with-confidence-weighting.
  * Also re-test RRF hybrid under perturbation: P3 failed at exact-match, but
    fusion may EARN its keep once the lexical arm is no longer perfect. That
    is the strongest argument FOR hybrid and it deserves a fair hearing.
"""
import json
import re
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t34_remedy_stress.json"
MODELS = ["BAAI/bge-base-en-v1.5", "BAAI/bge-m3"]
N = 120
DOC_WORDS = 64
rng = np.random.default_rng(20260727)

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
PERTS = ["exact", "case", "typo1", "drop1", "space", "partial", "morph"]


def mint(cat, i):
    if cat == "real":
        return REAL[i]
    if cat == "sku":
        return f"{chr(65+i%26)}{chr(65+(i//26)%26)}-{1000+(i*37)%9000}{chr(65+(i*7)%26)}"
    if cat == "pharma":
        return PH_A[i % len(PH_A)] + PH_B[(i // len(PH_A)) % len(PH_B)]
    if cat == "cjk":
        return CJK_A[i % len(CJK_A)] + CJK_B[(i // len(CJK_A)) % len(CJK_B)]


def perturb(name, mode, i, cat):
    if mode == "exact":
        return name
    if mode == "case":
        return name.lower()
    if mode == "morph":
        return f"information about {name} please"
    if cat == "cjk":
        # character-level ops on CJK: drop/swap a character, partial = first char
        ch = list(name)
        if mode == "typo1":
            j = i % len(ch); ch[j] = CJK_A[(i + 3) % len(CJK_A)][0]; return "".join(ch)
        if mode == "drop1":
            j = i % len(ch); return "".join(ch[:j] + ch[j + 1:])
        if mode == "space":
            j = max(1, len(ch) // 2); return "".join(ch[:j]) + " " + "".join(ch[j:])
        if mode == "partial":
            return "".join(ch[:max(1, len(ch) - 1)])
    s = list(name)
    alpha = "abcdefghijklmnopqrstuvwxyz"
    if mode == "typo1":
        j = 1 + (i % max(1, len(s) - 1))
        s[j] = alpha[(alpha.find(s[j].lower()) + 1) % 26] if s[j].lower() in alpha else "x"
        return "".join(s)
    if mode == "drop1":
        j = 1 + (i % max(1, len(s) - 1)); return "".join(s[:j] + s[j + 1:])
    if mode == "space":
        return name.replace("-", " ") if "-" in name else \
            name[:len(name) // 2] + " " + name[len(name) // 2:]
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


def rrf(a, b, k=60):
    out = np.zeros(len(a))
    for r in (a, b):
        order = np.argsort(-r)
        ranks = np.empty(len(r), dtype=int); ranks[order] = np.arange(len(r))
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
    body = [" ".join(w) for w in docs_words]
    print(f"corpus: {N} docs x {DOC_WORDS} words. DOCUMENTS stay canonical; only QUERIES degrade.")
    idx = np.arange(N)
    res = {"n": N, "perts": PERTS, "cats": {}}

    for c in CATS:
        names = [mint(c, i) for i in range(N)]
        docs = [f"{names[i]}. {body[i]}" for i in range(N)]
        tokfn = tok_ngram if c == "cjk" else tok_ws
        bm = BM25(docs, tokfn)
        print(f"\n--- {c} (bm25 tokenizer: {'char-ngram' if c=='cjk' else 'whitespace'}) ---")
        print("  " + " ".join(f"{p:>8}" for p in PERTS) + "   <- perturbation")
        qsets = {p: [perturb(names[i], p, i, c) for i in range(N)] for p in PERTS}
        print(f"  e.g. {names[0]!r} -> " + ", ".join(f"{p}:{qsets[p][0]!r}" for p in PERTS[1:4]))
        Sbm = {p: np.vstack([bm.scores(q) for q in qsets[p]]) for p in PERTS}
        rbm = {p: float((Sbm[p].argmax(1) == idx).mean()) for p in PERTS}
        print("  bm25   " + " ".join(f"{rbm[p]:8.3f}" for p in PERTS))
        entry = {"bm25": rbm, "dense": {}, "hybrid": {}}
        for mname in MODELS:
            short = mname.split("/")[-1]
            m = SentenceTransformer(mname, device="cuda")
            D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            rd, rh = {}, {}
            for p in PERTS:
                Q = m.encode(qsets[p], normalize_embeddings=True, show_progress_bar=False)
                Sd = Q @ D.T
                rd[p] = float((Sd.argmax(1) == idx).mean())
                H = np.vstack([rrf(Sd[i], Sbm[p][i]) for i in range(N)])
                rh[p] = float((H.argmax(1) == idx).mean())
            entry["dense"][short] = rd
            entry["hybrid"][short] = rh
            print(f"  dense/{short[:12]:12} " + " ".join(f"{rd[p]:8.3f}" for p in PERTS))
            print(f"  hybr /{short[:12]:12} " + " ".join(f"{rh[p]:8.3f}" for p in PERTS))
            del m
        res["cats"][c] = entry

    print("\n" + "=" * 78)
    print("WORST-CASE PERTURBATION per category (excluding exact):")
    verdict_ok, hybrid_wins = 0, 0
    for c in CATS:
        e = res["cats"][c]
        worst_p = min(PERTS[1:], key=lambda p: e["bm25"][p])
        bm_w = e["bm25"][worst_p]
        dn_w = float(np.mean([e["dense"][s][worst_p] for s in e["dense"]]))
        hy_w = float(np.mean([e["hybrid"][s][worst_p] for s in e["hybrid"]]))
        beats = bm_w > dn_w
        verdict_ok += beats
        if hy_w > max(bm_w, dn_w):
            hybrid_wins += 1
        print(f"  {c:7} worst='{worst_p}'  bm25={bm_w:.3f}  dense={dn_w:.3f}  "
              f"hybrid={hy_w:.3f}   {'BM25 still wins' if beats else 'BM25 LOSES'}")
        e["worst"] = {"pert": worst_p, "bm25": bm_w, "dense": dn_w, "hybrid": hy_w}
    res["bm25_robust_cats"] = verdict_ok
    res["hybrid_best_cats"] = hybrid_wins
    print(f"\ncategories where BM25 survives its worst perturbation: {verdict_ok}/4")
    print(f"categories where HYBRID beats both components: {hybrid_wins}/4")
    res["verdict"] = ("REMEDY ROBUST -- lexical arm recommendation stands" if verdict_ok >= 3
                      else "REMEDY FRAGILE -- revise to confidence-weighted hybrid")
    print(f"VERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
