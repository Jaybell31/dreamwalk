#!/usr/bin/env python3
"""T29 -- REPAIR THE BASELINE, THEN RE-ASK. Last blocker before write-up.

T28 replicated the fracture defect on real news text, but every model's REAL arm
sat at 0.267-0.567 instead of near ceiling. bge-m3 in particular started at
0.483, so its collapse to 0.067 is measured from a weak baseline. Until the real
arm is near ceiling we cannot say "models that handle the name FINE collapse when
it is fractured" -- we can only say "models that handle it POORLY collapse".

DIAGNOSED CAUSE (from T28): the real names were common nouns -- "Hospital",
"Library", "Harbour". Those words also occur inside OTHER news paragraphs, so
the real arm loses to distractor collisions that invented names never face. The
weak baseline is an artifact of NAME CHOICE, not of model competence.

FIX: real names become distinctive in-vocabulary PROPER nouns (place/surname
style) that are single- or double-token and rare inside news text. Same
within-item design, same corpus, same everything else. If the diagnosis is
right, the real arm should jump toward ceiling while nonce/fractured stay low.

PRE-REGISTERED:
  * PRIMARY: does bge-m3's real arm clear 0.85? If yes, weakness (2) is repaired
    and the multilingual claim can be stated from a strong baseline.
  * The fracture claim then re-asks itself honestly: fractured/real must stay
    below 0.70 with the DENOMINATOR NOW LARGE. This is a harder test than T28,
    because a bigger real arm makes the ratio easier to satisfy... no: a bigger
    denominator SHRINKS the ratio only if fractured stays flat. If fractured
    also rises with the better names, the ratio could go either way. Genuinely
    open.
  * ABORT the "strong-baseline" claim if bge-m3 real < 0.85 even with proper
    nouns -- then the ceiling problem is the corpus, not the names, and I must
    say so rather than keep hunting for a design that flatters the result.
"""
import json
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t29_baseline_repair.json"
MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base",
          "BAAI/bge-m3"]
N = 60

# Distinctive in-vocab proper nouns. Rare inside news paragraphs, but real
# English-lexicon words the tokenizer knows -- the opposite of T28's common nouns.
REAL_NAMES = ["Kensington", "Fairbanks", "Ashford", "Brentwood", "Carlton",
              "Dunmore", "Everly", "Fenwick", "Glenmore", "Hartley",
              "Inglewood", "Jarvis", "Kirkland", "Langley", "Merton",
              "Norwood", "Oakhurst", "Pembroke", "Quinton", "Ravenswood",
              "Stanhope", "Thornbury", "Upton", "Vernon", "Westbrook",
              "Yardley", "Aldridge", "Bramwell", "Coventry", "Dorset",
              "Ellsworth", "Fairmont", "Granville", "Havelock", "Ivybridge",
              "Jamestown", "Kingsley", "Lockwood", "Marlowe", "Newbury",
              "Ormsby", "Prescott", "Quimby", "Redmond", "Sheffield",
              "Tarrant", "Ullswater", "Vandermeer", "Whitmore", "Yarrow",
              "Ainsley", "Blackwood", "Cranfield", "Devonport", "Eastleigh",
              "Fitzroy", "Galloway", "Hollingsworth", "Ivorton", "Jeffries"]

SYL_A = ["bing", "snork", "frum", "wob", "grum", "plim", "krum", "ding", "snuff",
         "thrum", "clunk", "splodge", "grind", "bloop", "mung", "frob", "twizz",
         "blund", "snick", "chunk"]
SYL_B = ["le", "ler", "el", "bler", "ble", "ber", "wa", "gle", "er", "pi"]
SYL_C = ["bop", "wack", "dorf", "thunk", "snitch", "wash", "flax", "whort",
         "brick", "pick", "bosh", "wick", "puff", "snag", "whisk", "snee",
         "bonk", "fitch", "plop", "wisp"]
RARE = list("qxzvwkjbnrmghpfdt")


def mint_nonce(i):
    a = SYL_A[i % len(SYL_A)]
    b = SYL_B[(i // len(SYL_A)) % len(SYL_B)]
    c = SYL_C[(i // (len(SYL_A) * len(SYL_B))) % len(SYL_C)]
    return (a + b + c).capitalize()


def mint_fractured(i):
    base = [RARE[(i + k * 5) % len(RARE)] for k in range(8)]
    d = [RARE[(i // 17 ** p) % 17] for p in range(3)]
    return ("".join(base) + "".join(d)).capitalize()


def main():
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="test")
    paras = [r["text"] for r in ds if len(r["text"]) > 220][:N]
    if len(paras) < N:
        print(f"ABORT: only {len(paras)} paragraphs")
        return
    print(f"real corpus: ag_news/test, {len(paras)} paragraphs")

    names = {"real": REAL_NAMES[:N],
             "nonce": [mint_nonce(i) for i in range(N)],
             "fractured": [mint_fractured(i) for i in range(N)]}
    for a, v in names.items():
        assert len(set(v)) == N, f"{a} names not unique"

    # Confound check: does the real name leak into OTHER paragraphs? That was
    # the diagnosed cause of T28's weak baseline, so measure it, don't assume.
    low = [p.lower() for p in paras]
    leaks = sum(1 for i, nm in enumerate(names["real"])
                if any(nm.lower() in low[j] for j in range(N) if j != i))
    print(f"real names appearing in a NON-matching paragraph: {leaks}/{N} "
          f"(T28's common nouns were the suspected cause of the weak baseline)")

    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print("\ntokens per name:")
    for a, v in names.items():
        tpw = np.mean([len(tk(w, add_special_tokens=False)["input_ids"]) for w in v])
        print(f"  {a:10} {tpw:5.2f} tokens/name   e.g. {v[0]!r}")

    idx = np.arange(N)
    res = {"n": N, "real_name_leaks": leaks, "models": {}}

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
            print(f"  {a:10} R@1={out[a]['r1']:.3f}  margin={out[a]['margin']:+.4f}")
        out["fractured_over_real"] = float(out["fractured"]["r1"] / out["real"]["r1"]) \
            if out["real"]["r1"] > 0 else float("nan")
        print(f"  fractured/real = {out['fractured_over_real']:.3f}")
        res["models"][mname] = out
        del m

    m3 = res["models"]["BAAI/bge-m3"]
    res["m3_real"] = m3["real"]["r1"]
    res["baseline_repaired"] = bool(m3["real"]["r1"] >= 0.85)
    en = [v for k, v in res["models"].items() if "m3" not in k]
    res["n_en_above_threshold"] = sum(1 for v in en if v["fractured_over_real"] >= 0.70)

    print("\n" + "=" * 68)
    print(f"bge-m3 real arm: {m3['real']['r1']:.3f}  (T28 was 0.483, target >= 0.85)")
    print(f"BASELINE REPAIRED: {res['baseline_repaired']}")
    print(f"bge-m3 fractured/real = {m3['fractured_over_real']:.3f}")
    print(f"EN models keeping fractured >= 0.70 x real: {res['n_en_above_threshold']}/3")
    if res["baseline_repaired"] and m3["fractured_over_real"] < 0.70:
        res["verdict"] = "STRONG-BASELINE MULTILINGUAL COLLAPSE CONFIRMED -- publication blocker cleared"
    elif not res["baseline_repaired"]:
        res["verdict"] = "ABORT strong-baseline claim -- ceiling problem is the corpus, not the names"
    else:
        res["verdict"] = "baseline repaired BUT fracture effect weakened -- restate claim"
    print(f"VERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
