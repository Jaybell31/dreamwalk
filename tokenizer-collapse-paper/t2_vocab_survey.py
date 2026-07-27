#!/usr/bin/env python3
"""T2 VOCAB SURVEY -- does the deployed embedding stack silently discard emoji?

MECHANICAL. No LLM judge. Two independent instruments per model:

  1. TOKENIZER: encode a probe, count how many ids are <unk>. This is the
     mechanism (Gemini's [UNK] id-100 claim) and it is decidable exactly.
  2. VECTORS: embed N sentences that differ ONLY in the emoji, then count
     distinct rows and the max off-diagonal cosine. This is the CONSEQUENCE
     and it is what actually hurts retrieval.

WHY BOTH: they can disagree, and the disagreement is the interesting part.
A byte-level BPE tokenizer never emits <unk>, so instrument 1 says "fine" --
but if every emoji byte-sequence lands in the same under-trained region the
vectors can still collapse. Conversely a model can emit <unk> for the emoji
and still separate the sentences using the carrier text. Reporting only one
instrument is how you publish a wrong headline.

POSITIVE CONTROL (standing rule 1): DISTINCT_WORDS. Same carrier, but the
varying token is a common English word instead of an emoji. Any model that
cannot separate those is broken or mis-loaded, and its emoji number means
NOTHING. A model failing the control is reported UNTESTABLE, never as a
finding.

NEGATIVE CONTROL: IDENTICAL. The same sentence repeated. distinct==1 by
construction. If this ever exceeds 1 the harness itself is nondeterministic
and every other number in the table is suspect.

CEILING CHECK (standing rule 3): we report distinct/N for the control too.
If the control does not reach 1.0 the metric has no headroom.
"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "thenlper/gte-base",
    "intfloat/e5-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-m3",
]

CARRIER = "the market signal is {}"
EMOJI_SIMPLE   = ["\U0001F680", "\U0001F480", "\U0001F402", "\U0001F43B",
                  "\u2764\uFE0F", "\U0001F602", "\U0001F62D", "\U0001F44D"]
EMOJI_COMPOUND = ["\U0001F1FA\U0001F1F8", "\U0001F1EF\U0001F1F5",
                  "\U0001F926\u200D\u2642\uFE0F", "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466",
                  "\U0001F469\U0001F3FD\u200D\U0001F4BB", "\U0001F3F4\u200D\u2620\uFE0F",
                  "1\uFE0F\u20E3", "\U0001F44D\U0001F3FF"]
WORDS          = ["rising", "falling", "bullish", "bearish",
                  "strong", "weak", "green", "red"]

SUITES = {
    "SIMPLE_EMOJI":   [CARRIER.format(e) for e in EMOJI_SIMPLE],
    "COMPOUND_EMOJI": [CARRIER.format(e) for e in EMOJI_COMPOUND],
    "DISTINCT_WORDS": [CARRIER.format(w) for w in WORDS],          # positive control
    "IDENTICAL":      [CARRIER.format("x")] * 8,                    # negative control
}


def analyse(model, tok, sents):
    v = model.encode(sents, normalize_embeddings=True, show_progress_bar=False)
    v = np.asarray(v, dtype=np.float64)
    # distinct rows, tolerant of float noise
    rounded = np.round(v, 6)
    distinct = len({r.tobytes() for r in rounded})
    sim = v @ v.T
    off = sim[~np.eye(len(sents), dtype=bool)]
    unk = tok.unk_token_id
    unk_frac = []
    for s in sents:
        ids = tok(s, add_special_tokens=False)["input_ids"]
        unk_frac.append(sum(1 for i in ids if i == unk) / max(len(ids), 1))
    return {
        "distinct": distinct,
        "n": len(sents),
        "max_offdiag_cos": float(off.max()),
        "mean_offdiag_cos": float(off.mean()),
        "unk_rate": float(np.mean(unk_frac)),
    }


def main():
    from sentence_transformers import SentenceTransformer
    rows = []
    for name in MODELS:
        try:
            m = SentenceTransformer(name, trust_remote_code=False, device="cpu")
            tok = m.tokenizer
            r = {"model": name, "vocab": int(tok.vocab_size),
                 "tokenizer": type(tok).__name__}
            for suite, sents in SUITES.items():
                r[suite] = analyse(m, tok, sents)
            # STANDING RULE 1: control gates the whole row.
            ctrl = r["DISTINCT_WORDS"]
            r["control_ok"] = (ctrl["distinct"] == ctrl["n"]
                               and ctrl["max_offdiag_cos"] < 0.999)
            r["sanity_ok"] = r["IDENTICAL"]["distinct"] == 1
            rows.append(r)
            v = r["vocab"]
            for suite in ("SIMPLE_EMOJI", "COMPOUND_EMOJI", "DISTINCT_WORDS"):
                s = r[suite]
                print(f"{name[:44]:46} v={v:<7} {suite:15} "
                      f"{s['distinct']}/{s['n']}  maxcos={s['max_offdiag_cos']:.4f} "
                      f"unk={s['unk_rate']*100:.0f}%", flush=True)
            print(f"{'':46} control_ok={r['control_ok']} sanity_ok={r['sanity_ok']}\n",
                  flush=True)
            del m
        except Exception as e:
            print(f"{name[:44]:46} LOAD FAILED {type(e).__name__}: {e}", flush=True)
            rows.append({"model": name, "error": f"{type(e).__name__}: {e}"})
    json.dump(rows, open("/home/jason/dream_harness/t2_vocab_survey.json", "w"), indent=1)
    print("\nwrote /home/jason/dream_harness/t2_vocab_survey.json")


if __name__ == "__main__":
    main()
