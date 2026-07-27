#!/usr/bin/env python3
"""T28 -- HARDEN T27 ON A REAL CORPUS. Removes the author-written-docs objection.

T27 (nonce ASCII English collapses to chance, bge-m3 too) is the biggest claim
of the night, but it had three weaknesses I logged at the time:
  1. n=20, author-written documents  -> reviewer says "you wrote the docs"
  2. bge-m3's real-arm R@1 was only 0.500, so its fractured 0.000 came from a
     weak baseline
  3. single document per entity

T28 fixes all three by using REAL Wikipedia-style text from a public dataset as
the document pool, and by injecting the entity name into REAL sentences rather
than writing bespoke descriptions.

DESIGN. Take N real paragraphs from a public corpus (ag_news -- news text, no
login, already local from earlier runs if cached). For each paragraph, mint an
entity name in three arms and PREPEND it as a subject line, so the document is
overwhelmingly real human text with one injected token:
    REAL      an in-vocabulary common English word (baseline/ceiling)
    NONCE     invented pronounceable word (Binglebop-style)
    FRACTURED rare-character run, maximally shattered
Query = the entity name alone. Correct document = the one carrying that name.
Everything else about the document is identical across arms, so the ONLY
difference is which name was injected. This is the same within-item design as
T19, applied to ASCII.

PRE-REGISTERED (before running):
  * If NONCE/FRACTURED collapse toward chance (1/N) while REAL stays high, the
    T27 result replicates on real text and the general-fallback claim holds.
  * ABORT the general-fallback claim if fractured R@1 >= 0.70 x real R@1 in
    >= 2 of 3 English-vocab models.
  * bge-m3 must ALSO be tested; if its real arm is now near ceiling AND its
    fractured arm still collapses, weakness (2) is repaired.
"""
import json
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t28_nonce_real.json"
MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]
N = 60
rng = np.random.default_rng(20260727)

REAL_NAMES = ["Hospital", "Airport", "Library", "Bakery", "Shipyard", "Vineyard",
              "Foundry", "Orchard", "Quarry", "Aviary", "Brewery", "Tannery",
              "Smithy", "Cannery", "Hatchery", "Granary", "Dairy", "Sawmill",
              "Apiary", "Pottery", "Harbour", "Cinema", "Museum", "Stadium",
              "Chapel", "Cottage", "Warehouse", "Laundry", "Nursery", "Observatory",
              "Barracks", "Lighthouse", "Windmill", "Ferry", "Clinic", "Studio",
              "Workshop", "Depot", "Garrison", "Monastery", "Playhouse", "Arcade",
              "Boathouse", "Greenhouse", "Toolshed", "Guildhall", "Almshouse",
              "Racecourse", "Bandstand", "Tollbooth", "Coalyard", "Brickworks",
              "Glassworks", "Ropewalk", "Saltern", "Limekiln", "Fishery",
              "Colliery", "Bellfoundry", "Papermill"]

SYL_A = ["bing", "snork", "frum", "wob", "grum", "plim", "krum", "ding", "snuff",
         "thrum", "clunk", "splodge", "grind", "bloop", "mung", "frob", "twizz",
         "blund", "snick", "chunk"]
SYL_B = ["le", "ler", "el", "bler", "ble", "ber", "wa", "gle", "er", "pi"]
SYL_C = ["bop", "wack", "dorf", "thunk", "snitch", "wash", "flax", "whort",
         "brick", "pick", "bosh", "wick", "puff", "snag", "whisk", "snee",
         "bonk", "fitch", "plop", "wisp"]
RARE = list("qxzvwkjbnrmghpfdt")


def mint_nonce(i):
    """Unique for i < len(A)*len(B)*len(C) = 4000. The earlier (i*3)/(i*7)
    stride formula silently COLLIDED at n=60 -- the uniqueness assert caught it,
    which is exactly why that assert exists. Use positional digits instead."""
    a = SYL_A[i % len(SYL_A)]
    b = SYL_B[(i // len(SYL_A)) % len(SYL_B)]
    c = SYL_C[(i // (len(SYL_A) * len(SYL_B))) % len(SYL_C)]
    return (a + b + c).capitalize()


def mint_fractured(i):
    """Rare-character run, PROVABLY unique: the last 3 characters encode i in
    base-17 (len(RARE)=17), so distinct i give distinct names for i < 4913.
    Two earlier stride formulas silently collided here; encode, don't stir."""
    base = [RARE[(i + k * 5) % len(RARE)] for k in range(8)]
    d = [RARE[(i // 17 ** p) % 17] for p in range(3)]
    return ("".join(base) + "".join(d)).capitalize()


def main():
    # ---- real document text ----
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="test")
        paras = [r["text"] for r in ds if len(r["text"]) > 220][:N]
        source = "ag_news/test"
    except Exception as e:
        print(f"dataset load failed ({e}); ABORT -- refusing to silently fall back "
              "to author-written text, that is the exact weakness this test removes")
        return
    if len(paras) < N:
        print(f"ABORT: only {len(paras)} paragraphs >220 chars, need {N}")
        return
    print(f"real corpus: {source}, {len(paras)} paragraphs, "
          f"mean {np.mean([len(p) for p in paras]):.0f} chars")

    names = {"real": REAL_NAMES[:N],
             "nonce": [mint_nonce(i) for i in range(N)],
             "fractured": [mint_fractured(i) for i in range(N)]}
    # uniqueness guard -- duplicate names would silently create ties
    for a, v in names.items():
        assert len(set(v)) == len(v), f"{a} names not unique"

    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print("\ntokens per name:")
    for a, v in names.items():
        tpw = np.mean([len(tk(w, add_special_tokens=False)["input_ids"]) for w in v])
        unk = sum(1 for w in v if 100 in tk(w, add_special_tokens=False)["input_ids"])
        print(f"  {a:10} {tpw:5.2f} tokens/name   [UNK]-bearing {unk}/{N}   e.g. {v[0]!r} -> "
              f"{tk.convert_ids_to_tokens(tk(v[0], add_special_tokens=False)['input_ids'])}")

    idx = np.arange(N)
    res = {"n": N, "source": source, "models": {}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        out = {}
        for a in ("real", "nonce", "fractured"):
            docs = [f"{names[a][i]}. {paras[i]}" for i in range(N)]
            D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            Q = m.encode(names[a], normalize_embeddings=True, show_progress_bar=False)
            S = Q @ D.T
            hit = (S.argmax(1) == idx)
            gold = S[idx, idx].copy(); S2 = S.copy(); S2[idx, idx] = -np.inf
            out[a] = {"r1": float(hit.mean()), "margin": float((gold - S2.max(1)).mean())}
            print(f"  {a:10} R@1={out[a]['r1']:.3f}  margin={out[a]['margin']:+.4f}  (chance {1/N:.3f})")
        ratio = out["fractured"]["r1"] / out["real"]["r1"] if out["real"]["r1"] > 0 else float("nan")
        out["fractured_over_real"] = float(ratio)
        print(f"  fractured/real = {ratio:.3f}")
        res["models"][mname] = out
        del m

    en = [v for k, v in res["models"].items() if "m3" not in k]
    n_ok = sum(1 for v in en if v["fractured_over_real"] >= 0.70)
    res["verdict"] = ("ABORT general-fallback claim" if n_ok >= 2
                      else "GENERAL FALLBACK DEFECT REPLICATES ON REAL TEXT")
    print(f"\n{n_ok}/3 EN models kept fractured >= 0.70 x real")
    print(f"VERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
