#!/usr/bin/env python3
"""T23 -- DOES THE ONE-LINE FIX GENERALISE FROM EMOJI TO CJK?

T9 showed demojize() takes real-text emoji retrieval from 0.360 to 1.000 in
both collapsed models. That is a shippable one-line fix. The immediate question
any vendor or journalist asks next: DOES THE SAME TRICK FIX CHINESE?

You cannot "demojize" Chinese, but the underlying move -- replace the
out-of-vocabulary glyph with an IN-VOCABULARY ASCII NAME before encoding -- has
three natural CJK analogues, all one-line, all no-retraining:

  1. PINYIN      transliterate to romanised syllables (yi1yuan4)
  2. UNICODE     replace each collided char with its Unicode codepoint token
                 (<U+533B>) -- pure identity preservation, zero semantics
  3. TRANSLATE   substitute the English gloss (upper bound / cheat oracle:
                 this is what a perfect translator would give you)

PREREGISTERED PREDICTIONS, written before running:
  * UNICODE should beat RAW on any metric that needs only DISTINCTNESS,
    because it restores identity -- but it should NOT approach the English
    oracle, because <U+533B> carries no meaning. If UNICODE alone closes the
    gap, the failure was purely lexical and the misalignment story is wrong.
  * PINYIN sits between: distinct AND carries some phonetic signal, but the
    encoder never saw pinyin-to-meaning mappings in training.
  * TRANSLATE is the ceiling. If TRANSLATE does not reach the English oracle
    the harness is broken (sanity check).
GATE FOR "THE FIX GENERALISES": any no-translation method (unicode/pinyin)
must reach >= 0.70 x the English-oracle R@1 in >= 2 of 3 models. Otherwise the
honest headline is "the one-line fix works for emoji and NOT for CJK", which
is a sharper and more useful finding than pretending it generalises.

This directly tests the collision-vs-misalignment split that T22's CAF could
not resolve: UNICODE isolates the LEXICAL component with semantics held at
zero.
"""
import json, unicodedata
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import importlib.util

OUT = "/home/jason/dream_harness/t23_generalise.json"
spec = importlib.util.spec_from_file_location("t17", "/home/jason/dream_harness/t17_within_model.py")
t17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t17)
ITEMS = t17.ITEMS

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "thenlper/gte-base"]

try:
    from pypinyin import lazy_pinyin
    HAVE_PINYIN = True
except ImportError:
    HAVE_PINYIN = False


def to_unicode_names(s):
    """Replace every CJK char with its codepoint tag. Identity, no semantics."""
    out = []
    for ch in s:
        if ord(ch) > 0x2E80:
            out.append(f"<U+{ord(ch):04X}>")
        else:
            out.append(ch)
    return " ".join(out)


def to_pinyin(s):
    if not HAVE_PINYIN:
        return None
    return " ".join(lazy_pinyin(s))


def main():
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    docs = [i[2] for i in ITEMS]
    zh = [i[0] for i in ITEMS]
    en = [i[1] for i in ITEMS]
    n = len(ITEMS)
    idx = np.arange(n)
    collided = [k for k in range(n) if 100 in tk(zh[k], add_special_tokens=False)["input_ids"]]
    print(f"n={n}, collided arm={len(collided)}, pinyin available={HAVE_PINYIN}")

    arms = {"raw": zh,
            "unicode": [to_unicode_names(s) for s in zh],
            "translate_oracle": en}
    if HAVE_PINYIN:
        arms["pinyin"] = [to_pinyin(s) for s in zh]

    # show what the transforms actually produce -- no silent black boxes
    for k in collided[:3]:
        print(f"  e.g. {zh[k]!r} -> unicode {arms['unicode'][k]!r}"
              + (f" | pinyin {arms['pinyin'][k]!r}" if HAVE_PINYIN else ""))

    res = {"n": n, "n_collided": len(collided), "have_pinyin": HAVE_PINYIN, "models": {}}

    for mname in MODELS:
        print(f"\n=== {mname} ===", flush=True)
        m = SentenceTransformer(mname, device="cuda")
        D = m.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        out = {}
        for arm, texts in arms.items():
            Q = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            hit = ((Q @ D.T).argmax(1) == idx)
            out[arm] = {"all": float(hit.mean()),
                        "collided_only": float(hit[collided].mean())}
            print(f"  {arm:17} all={out[arm]['all']:.3f}   collided-arm={out[arm]['collided_only']:.3f}")

        oracle = out["translate_oracle"]["collided_only"]
        best_notrans = max(out[a]["collided_only"] for a in out if a not in ("translate_oracle", "raw"))
        ratio = best_notrans / oracle if oracle > 0 else float("nan")
        out["oracle_ratio_best_notranslation"] = float(ratio)
        out["generalises"] = bool(ratio >= 0.70)
        print(f"  best no-translation / oracle = {ratio:.3f}  "
              f"{'GENERALISES' if ratio >= 0.70 else 'DOES NOT GENERALISE (gate 0.70)'}")
        res["models"][mname] = out
        del m

    n_pass = sum(1 for v in res["models"].values() if v["generalises"])
    res["verdict"] = ("FIX GENERALISES TO CJK" if n_pass >= 2
                      else "FIX IS EMOJI-ONLY -- CJK NEEDS MORE THAN IDENTITY RESTORATION")
    print(f"\n{n_pass}/3 models pass. VERDICT: {res['verdict']}")
    json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
