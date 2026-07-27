# Silent [UNK] Collapse: Tokenizer Vocabulary Gaps Break Text Embedding Retrieval

Reproduction repo. Every number in the write-up is produced by a script here.
Full paper: `PAPER_DRAFT.md`. This README is a shorter pointer, kept in sync
with it (last synced: round 7 reframe, Jul27).

## THE ONE-SENTENCE FINDING

A hospital and an airport become the identical vector for stock English-
vocabulary embedding models (all-MiniLM-L6-v2, ChromaDB/Qdrant/txtai's
shipped defaults) — 14 of 24 common Chinese entities collapse to the same
`[UNK][UNK]` token sequence, and Chinese-query retrieval scores 0.083-0.125
against a 0.917-1.000 English-query control on the identical corpus. Emoji
are the extreme point of the same mechanism (99.83% of the official Unicode
set collapses to one token). Crucially, fixing the collision (so every
Chinese character survives tokenization) only recovers PART of the loss:
these same models still fail 45.7-65.7% of cross-lingual retrievals they get
right in English — vocabulary-coverage collapse is the deterministic, silent
extreme of a larger, non-deterministic cross-lingual failure.

## WHY THIS REPO LOOKS PARANOID

Every correction made along the way is disclosed rather than cleaned up —
see `PAPER_DRAFT.md` for the full list (a moved gate, a fabricated citation
caught and removed, a "bit-identical" overclaim walked back to "cosine
1.000000 to six decimals", and the frame itself changed twice as new data
falsified the earlier "emoji-only" and "other scripts survive intact"
claims). The correction trail is the credibility argument, read it before
citing a number out of context.

## THE MECHANISM (short form)

1. **Identical-vector collapse** (deterministic, script-selective). Chinese
   and emoji hit `[UNK]` en masse on a 30522-token English WordPiece vocab;
   Korean/Arabic/Russian/Hindi mostly don't collide but still fail to
   retrieve. Both are reported, separately, because conflating them either
   overstates or understates the mechanism.
2. **General cross-lingual retrieval failure** (non-deterministic, general
   to every non-English script tested) — every language lands at or near
   chance against an English control that scores 0.917-1.000 on the same
   corpus, with or without collision.
3. `bge-m3` (250k multilingual vocab) is the causal control: no collision,
   0.958 on the same task — this is a vocabulary/training choice, not an
   architecture limit, and it is imperfect too (ties on 8/96 polarity pairs,
   loses 1/8 emoji). No model tested is fully immune.

## THE CURE, AND ITS LIMITS

`emoji.demojize()` (one line) takes collapsed models from 1/8 to 8/8 on
representational emoji retrieval and from a forced 0.500 coin-flip to
0.589-0.667 on a pre-registered semantic-polarity task (n=96) — but recovers
only 25-40% of the gap to an explicit-English-word control, and does nothing
for CJK. See `PAPER_DRAFT.md` §5 "WHAT TO ACTUALLY DO ABOUT IT" for the
full ranked list (BM25 lexical arm, char n-grams, max-norm fusion instead of
RRF, word-substitution, bigger-vocab encoder) — demojize alone is not a fix.

## FILES

```
reproduce.py            THE ARTIFACT — pip install chromadb && python3 reproduce.py
killshot_chromadb.py    30-line unmodified-ChromaDB reproduction (§2)
t2_vocab_survey.py      emoji census, 9-model survey (§1.0, §1.2)
t2_falsify_all_emoji.py falsification of the "all emoji" overclaim
t2_confirm.py           confirms the 99.83% figure
t5_blast_radius.py      script-by-script blast radius (§1.9)
t9_realcontext.py       real human-text emoji retrieval, n=150 (§1.3)
t9b_explain.py / t9c_argmax.py   degenerate-argmax artifact diagnosis
t10_tokenids.py         token-ID mechanism, cross-model [UNK]/id table (§1.1)
t11_semantic.py         semantic-polarity pilot, n=12 (§1.5, pre-registration)
t12_build.py / t12_eval.py / t12_generate.py   powered semantic-polarity run, n=96
t13_collisions.py       within-model DISTINCT vs COLLIDED contrast (§1.2)
t14_cjk.py              24-entity cross-lingual retrieval (§1.1, the promoted result)
t15_order.py            falsifies the insertion-order explanation
t16_attractor.py        confirms the query-side attractor mechanism
t20-t29                 falsifiers, hubness, CAF, generalisation, derangement,
                         margin/TOST shams, nonce controls, baseline repair
                         (earlier round's causal-attribution battery, folded
                         into §1.2's within-model contrast)
*.json                  raw results for the runs above
PAPER_DRAFT.md          the full paper (status: draft, ship-ready, see its
                         own STATUS line for what's blocking vs deferred)
```

## REPRODUCE

Minimal (no local deps beyond one pip install):
```bash
pip install chromadb && python3 reproduce.py
```

Full battery:
```bash
pip install sentence-transformers transformers scipy numpy emoji chromadb
python3 t2_vocab_survey.py && python3 t14_cjk.py && python3 t13_collisions.py
python3 t9_realcontext.py && python3 t12_eval.py
```

All models are public checkpoints. No API keys. Arm membership (collided vs
distinct) is derived from the tokenizer at runtime, never hand-assigned.

Anyone can check their own stack in one line:
```python
tok("your term", add_special_tokens=False)["input_ids"]   # id 100 (or your
                                                             # tokenizer's
                                                             # unk id) == gone
```
