# Silent [UNK] Collapse: Distinct Real-World Terms Receive Identical Vectors in Widely-Deployed Text Embedding Models

*(Working subtitle for press: "A hospital and an airport are the same vector.")*

REFRAME NOTE (round 7). The council split 2–1 for keeping an emoji-only lead.
Both dissenting votes rested explicitly on the premise that the CJK evidence
was "8 words" and experimentally weak — GPT: "the emoji result is already a
complete result"; Gemini: "you have 8 words for Chinese … Reviewer 2 will
annihilate you." That premise expired while they were writing: T14 landed a
24-entity cross-lingual retrieval test with a clean positive control
(0.917–1.000) and a collapsed-model score of 0.083. Grok's minority position —
"the core failure is not emoji-specific; it is systematic many-to-one collapse
… emoji is simply the purest instance" — is the only one that survives the new
data, and GPT's own framing agrees the mechanism is the contribution. We
therefore lead with the mechanism, keep emoji as the exhaustive extreme point
(it has the 99.83% census and the pre-registered n=96 test), and promote CJK
to the high-stakes result. Nothing is discarded; the emoji work is unchanged.

STATUS: DRAFT — numbers measured, framing settled (round 7, see REFRAME NOTE
above). Remaining before ship: none blocking for a Show HN / blog post; a
real-corpus CJK replication (T2Ranking) is the flagged v2 strengthening for
a peer-reviewed venue, not a launch blocker (see Limitations §4).
Repro: /home/jason/dream_harness/{t1..t40}*.py, killshot_chromadb.py, reproduce.py

## Abstract (150 words, publication-ready)

Text embedding models are the retrieval substrate of most production RAG
systems. We show that the English WordPiece vocabulary shared by widely
deployed encoders — the shipped defaults of ChromaDB, Qdrant's FastEmbed and
txtai — maps distinct real-world terms onto identical token sequences, and
therefore onto identical vectors. In Chinese, 14 of 24 common entities,
including hospital, airport, market, factory and prison, collapse to the
single sequence `[UNK][UNK]` and receive **one shared embedding**. On a
cross-lingual retrieval task these models score 0.083–0.125 against a
0.917–1.000 English-query control on the same corpus; a multilingual encoder
scores 0.958. Crucially, the effect is not mere cross-lingual weakness: within a
single model, Chinese words that survive tokenization retrieve at 0.34–0.54
while colliding words retrieve at 0.00–0.04 (Fisher p ≤ 0.0006), even though the
colliding concepts are *easier* to retrieve when queried in English. Emoji are
the extreme point of the same mechanism: 99.83% of the
official Unicode set collapses to one token, and in a pre-registered test
affected models cannot distinguish a production disaster from a success. The
failure is silent and requires no misconfiguration. It also nests inside a
larger one: on the same 24 documents these models retrieve correctly for
92–100% of English queries and 0–17% of Chinese, Korean, Arabic, Russian and
Hindi queries, while a multilingual encoder scores 0.958–1.000 on all six.
Identical-vector collapse is the deterministic, silent extreme of that failure.

## 1. The measured core

### 1.0 Tokenizer collapse: the emoji census
Of 5,042 official emoji sequences (Unicode emoji data via the `emoji` package),
5,033 (99.83%) tokenize to `[UNK]` under the 30522-token WordPiece vocabulary
shared by BERT-derived embedding models. The 9 survivors are legacy typographic
marks that predate emoji as a category: © ® ™ ↔ ▪ ♠ ♣ ♥ ♦. None represent
contemporary emoji usage.

NOTE: an earlier draft said "all emoji." That was falsified by our own test
(t2_falsify_all_emoji.py) and corrected. 99.83% is the defensible number.

### 1.1 The headline failure: every Chinese query becomes the same question

Run this against a stock ChromaDB install — no configuration, no custom
embedding function, the defaults exactly as shipped:

```
pip install chromadb && python3 reproduce.py
```

Eight English documents describe eight places. Queried in English, the
database returns the right document **8/8**. Queried in Chinese, it returns
the right document **1/8** — and the seven failures are not scattered. Asking
in Chinese for a farm, a factory, a prison or a pharmacy returns *the hospital
document*, every time, with a plausible distance score and no error.

The reason is not a tie-break or a ranking artifact; we tested and rejected
that explanation. Reversing the insertion order changes nothing, and the
returned distances are seven **distinct** values. The documents are fine. What
collapses is the *query*.

On the 30522-token English WordPiece vocabulary, 医院 (hospital), 机场
(airport), 市场 (market), 农场 (farm), 工厂 (factory), 监狱 (prison) and 药房
(pharmacy) all tokenize to exactly `[UNK][UNK]` — the unknown token, twice.

A NOTE ON THE ID, BECAUSE WE GOT THIS WRONG ONCE. Earlier drafts wrote the
sequence as the literal ids `[100, 100]`. That is correct for
all-MiniLM-L6-v2, bge-base-en-v1.5, e5-base-v2 and gte-base (vocab 30522,
`unk_token_id` = 100) but FALSE for all-mpnet-base-v2, whose vocab is 30527 and
whose `[UNK]` id is 104 — there the same 14 entities collapse to `[104, 104]`.
The mechanism is identical and the collapse is identical; only the integer
differs. We therefore state the claim in terms of the `[UNK]` TOKEN, never the
raw id. This correction came from an adversarial review round that asked us to
re-dump the token ids for all 14 surface forms across five models rather than
trust the number we had already written down.

All 21 pairs of these query vectors are bit-identical (max pairwise cosine
1.000000), while the English queries for the same concepts span cosines down
to 0.166. The document vectors remain fully distinct (0/21 identical pairs).

So every Chinese query is literally the **same point** in embedding space, and
that point has exactly one nearest neighbour. Every Chinese-speaking user, no
matter what they ask, is routed to a single fixed document — an *attractor*
determined by the corpus, not by the question. Swapping in an unrelated corpus
moves the attractor from the hospital write-up to a report about a football
match, and all seven queries follow it. The query carries zero information.

| model | Chinese query | English-query control | collision subset |
|---|---|---|---|
| all-MiniLM-L6-v2 | **0.083** (2/24) | 0.917 | **0.000** |
| bge-base-en-v1.5 | **0.125** (3/24) | 0.958 | 0.059 |
| gte-base | **0.125** (3/24) | 1.000 | 0.059 |
| bge-m3 (multilingual) | 0.958 | 1.000 | 0.941 |

(24-entity cross-lingual retrieval, chance = 0.042.) The English-query control
is the load-bearing part: same corpus, same documents, same model, 0.917–1.000.
The corpus is retrievable and the task is fair. Only the query language
changed. bge-m3, with a 250002-token multilingual vocabulary, scores 0.958 on
the identical task — the defect is a vocabulary-coverage choice, not a limit of
the architecture.

**This is not simply "English tokenizers are bad at Chinese" — but neither is
it confined to Chinese, and an earlier draft of this paper got that wrong.** We
previously wrote that Korean, Arabic, Devanagari and Cyrillic "survive intact"
on these models, on the strength of 8/8 distinct vectors. Adversarial review
asked the obvious follow-up we had not asked: *distinct is not the same as
working.* Measured on the identical 24-document task (§1.3), the answer is that
we were wrong, and the correction enlarges the result rather than shrinking it.
There are two separable failures:

| model | en | zh | ko | ar | ru | hi |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.917 | 0.083 | 0.042 | 0.042 | 0.000 | 0.000 |
| bge-base-en-v1.5 | 0.958 | 0.125 | 0.083 | 0.042 | 0.167 | 0.000 |
| gte-base | 1.000 | 0.125 | 0.083 | 0.042 | 0.167 | 0.000 |
| bge-m3 (multilingual) | 1.000 | 0.958 | 0.958 | 0.958 | 1.000 | 0.958 |

(top-1, n=24, chance = 0.042, same documents throughout)

1. **Identical-vector collapse** — deterministic and silent, and *this* is
   script-selective: 94 of 276 Chinese query pairs are bit-identical, versus
   0–3 of 276 for Korean, Arabic, Russian and Hindi. Every affected query is
   routed to one fixed attractor document. This is the novel mechanism.
2. **General cross-lingual retrieval failure** — non-deterministic, and *not*
   selective: every non-English language tested lands at or near chance
   (0.000–0.167) against a 0.917–1.000 English control on the same corpus.

Korean, Arabic, Russian and Hindi are therefore *injective but not functional*:
the encoder does not collide them, and it also cannot retrieve for them. That
those four scripts avoid collision while still failing is precisely why the two
mechanisms must be reported separately. bge-m3 scores 0.958–1.000 across all
six languages, so both failures are consequences of a vocabulary and training
choice, not of the architecture.

### 1.2 The within-model contrast: collision is the cause, not "bad at Chinese"

The table above compares collapsed models against bge-m3. That comparison
cannot, on its own, attribute the gap to collision — bge-m3 differs in
vocabulary *and* multilingual training *and* representation quality
simultaneously. A cross-model gap is separation, not causation. This objection
was raised against us in adversarial review and it was correct.

So we ran the contrast **inside a single model, in a single language.** The
30522-token vocabulary contains 488 CJK-bearing tokens. Consequently some
Chinese words tokenize to real, unique ids (山 → `[1831]`) while others collapse
to `[UNK]` (雨 → `[UNK]`). Arm membership is *derived from the tokenizer at
runtime*, never assigned by us. Same model, same corpus, same task, same
scoring, both arms Chinese — cross-lingual difficulty is held constant by
construction. n = 61 (35 distinct / 26 collided). Pre-registered gate: Δ ≥ 0.30
and Fisher exact p < 0.05.

| model | Chinese, DISTINCT | Chinese, COLLIDED | Δ | Fisher p | identical query pairs (collided / distinct) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.343 | **0.000** | **+0.343** | 0.00064 | 112/325 / 0/595 |
| bge-base-en-v1.5 | 0.514 | **0.038** | **+0.476** | 0.00005 | 112/325 / 0/595 |
| gte-base | 0.543 | **0.038** | **+0.504** | 0.00002 | 112/325 / 0/595 |
| bge-m3 (control) | 0.743 | 0.885 | −0.142 | 0.207 | **0/325** / 0/595 |

Two controls make this an attribution rather than a correlation.

**Concept difficulty.** The same entities queried in *English* against the same
corpus: distinct-arm 0.771 versus collided-arm 0.846 / 0.923 / 0.962. The
collided concepts are, if anything, *easier* in English. The documents and
concepts are equally retrievable; the only surviving difference is how the
Chinese query is encoded.

**Word length.** Collided words skew toward two-character compounds, so we
repeat the contrast on a length-matched two-character subset: 0.667 versus
0.000 / 0.053. The effect holds.

**bge-m3's null result is the control passing, not a failure.** It has zero
identical query pairs in either arm — there is no collision available to cause
anything, so the collision effect must vanish, and it does. The effect appears
in exactly the three models where collisions exist and disappears in the one
where they do not. It tracks the mechanism, not the model family.

The task above uses 24 entities; the mechanism is visible in the token IDs,
where **17 of the 24 fall into just two collision groups**:

```
[100, 100]      -> 14 entities SHARE ONE VECTOR:
                   hospital, airport, market, farm, factory, restaurant,
                   hotel, post office, pharmacy, supermarket, station,
                   theatre, prison, harbour
[100, 100, 100] ->  3 entities SHARE ONE VECTOR:
                   library, police station, gymnasium
```

No ranking function, reranker or similarity threshold downstream can separate
these, because there is nothing left to separate. Accuracy on the collision
subset is 0.000 for all-MiniLM-L6-v2 — not "poor", but *structurally
impossible*.

### 1.2 Emoji: the extreme point of the same mechanism

Nine models, 8 documents differing only by one emoji. `distinct` counts unique
embeddings; a collapsed model returns 1. All control and sanity checks passed
on every row (DISTINCT_WORDS = 8/8 everywhere, proving the harness measures
what it claims).

  model                        vocab    simple    compound   words   unk
  all-MiniLM-L6-v2             30522   1/8 1.0000  1/8 1.0000  8/8   20%
  all-mpnet-base-v2            30527   1/8 1.0000  1/8 1.0000  8/8   20%
  bge-base-en-v1.5             30522   1/8 1.0000  1/8 1.0000  8/8   20%
  bge-small-en-v1.5            30522   1/8 1.0000  1/8 1.0000  8/8   20%
  gte-base                     30522   1/8 1.0006  1/8 1.0006  8/8   20%
  e5-base-v2                   30522   1/8 1.0000  1/8 1.0000  8/8   20%
  paraphrase-MiniLM-L6-v2      30522   1/8 1.0000  1/8 1.0000  8/8   20%
  multi-qa-MiniLM-L6-cos-v1    30522   1/8 1.0000  1/8 1.0000  8/8   20%
  bge-m3                      250002   8/8 0.9682  8/8 0.9757  8/8    2%

8 of 9 collapse. The discriminator is the vocabulary: every 30522-token English
WordPiece model collapses; the 250002-token multilingual model does not.

### 1.2b The mechanism, at the token-ID level

We initially described this as the emoji being "deleted at tokenization." That
is incorrect and we correct it here. Reading the ids directly:

  all-MiniLM-L6-v2   "animal 🐶" -> [4111, 100]   for ALL EIGHT emoji
  bge-base-en-v1.5   "animal 🐶" -> [4111, 100]   for ALL EIGHT emoji
  bge-m3             "animal 🐶" -> [26249, 6, 246613]  (7 distinct of 8)

The emoji is not removed; it is REPLACED by the shared `[UNK]` token, id 100.
The correct description is MANY-TO-ONE `[UNK]` COLLAPSE. Eight semantically
different documents reduce to one identical token sequence, which is why the
embeddings are identical to machine precision rather than merely similar.

This also explains the control's imperfect score. bge-m3 yields 7 distinct
sequences for 8 emoji because one emoji (🐭) falls outside even its 250k
vocabulary and hits `<unk>` (id 3). Its single retrieval miss is therefore not
noise — it is the same defect at 1/8 the rate in a model with 8x the
vocabulary. Emoji coverage is a spectrum, not a binary property, and even our
immune control has a hole in it.

### 1.3 Retrieval on REAL human-written text (n=150)

The strongest objection to §1.2 is that the carrier sentences were written by
us. We therefore rebuilt the probe using only **real human messages** scraped
from our corpus: 389 messages containing exactly one emoji, carrier text
untouched. The distractor alphabet is the 8 most frequent real-usage emoji
(😂 ❤ 😭 😍 🤔 🙄 😡 🔥). Task: given the original message, retrieve it from 8
candidates differing only in that emoji. Chance = 1/8 = 0.125.

| model | raw | +demojize |
|---|---|---|
| all-MiniLM-L6-v2 | 0.360* | **1.000** |
| bge-base-en-v1.5 | 0.360* | **1.000** |
| bge-m3 (immune control) | **1.000** | 1.000 |

*\*The 0.360 is not partial retrieval — it is a degenerate-argmax artifact, and
it must be reported as such.* All eight candidates tokenize to the **same** id
sequence, so all eight cosine similarities are bit-identical: we verified
40/40 sampled trials have exactly one distinct similarity value. `argmax` then
deterministically returns index 0, which is 😂 — the most frequent emoji, and
therefore the gold answer in exactly 54/150 = **0.360** of trials. The observed
accuracy equals the index-0 base rate to three decimals. Information-
theoretically the raw retrieval carries **zero** bits; a random tie-break would
score 0.125. Any benchmark that reports top-1 accuracy without checking for
similarity ties will silently overstate collapsed-model performance.

The one-line `demojize` fix takes both collapsed models from zero information
to **perfect** retrieval on real human text.

### 1.4 Retrieval at chance (synthetic controls)
8-way retrieval, chance = 1/8. Collapsed models score 1/8. See T7 table below.

### 1.5 Does the collapse break SEMANTIC retrieval? (n=96, pre-registered)

Sections 1.2–1.4 prove *representational* collapse. A reviewer can still
object, and two of our reviewers did, independently: the task is tautological.
If candidates differ only by emoji and the query asks which emoji was present,
`demojize` simply writes the answer into the input. Nothing yet shows that
ordinary retrieval relevance suffers.

We therefore ran a test in which **the gold answer is a document, not an
emoji**, and **the corpus contains no emoji at all**.

*Design.* 96 items. Each query is a realistic status message whose polarity is
carried **only** by a terminal emoji — "the billing migration is finished 💀"
versus "…🚀". The corpus is the 192 emoji-free prose documents those messages
refer to: for each item, an incident write-up and a success write-up of the
same event. The system must retrieve the *right kind of document*. A validator
mechanically rejects any item whose documents contain the demojized words, so
`demojize` cannot leak the answer: `:skull:` appears nowhere in the corpus.
Primary metric is pair-restricted 2-way accuracy (chance = 0.500). The
hypothesis, metric, α and decision rule were **fixed before the run** — a
pilot at n=12 was underpowered (p=0.152), so we powered it and re-ran rather
than reinterpreting the pilot.

A **word control** replaces the emoji with explicit English ("a disaster" /
"a success") and upper-bounds what any tokenizer fix could achieve.

| model | raw | +demojize | McNemar *p* | word control | gap recovered |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.500 | 0.589 | 0.0095 | 0.745 | 36% |
| bge-base-en-v1.5 | 0.500 | 0.667 | <10⁻⁴ | 0.922 | 40% |
| bge-m3 (control) | 0.609 | 0.693 | 0.0259 | 0.943 | 25% |

**Three findings, and the third is the important one.**

1. **The harm is forced, not merely observed.** Raw accuracy is *exactly*
   0.500 (96/192) for both collapsed models, because 96/96 query pairs produce
   bit-identical vectors. The ranking cannot differ; the coin flip is a
   mathematical consequence of the tokenizer, not sampling noise. A retrieval
   system built on these models cannot distinguish a production disaster from
   a production success when the distinction is carried by an emoji — on a
   corpus with no emoji in it.

2. **The one-line fix works, and the effect is significant** on all three
   models under the pre-registered test.

3. **The fix is only partial, and this should not be glossed.** `demojize`
   recovers just 36% / 40% / 25% of the distance to the word control.
   `:skull:` is not as informative as "a disaster": the replacement token
   sequence is out-of-distribution relative to the text these encoders were
   trained on. Practitioners who apply `demojize` and declare the problem
   solved are still leaving most of the signal on the floor. Closing the
   remaining gap is an open problem, and we state it as one.

Note that bge-m3, our immune control, also ties on 8/96 pairs and also gains
significantly from `demojize`. No model in this study is fully safe.

### 1.6 The cure, with an immune control
`emoji.demojize()` converts 😂 -> `:face_with_tears_of_joy:`, which the same
vocabulary tokenizes into ordinary word pieces.

  model                     simple      compound
  all-MiniLM-L6-v2          1/8 -> 8/8   1/8 -> 8/8
  bge-base-en-v1.5          1/8 -> 8/8   1/8 -> 8/8
  e5-base-v2                1/8 -> 8/8   1/8 -> 8/8
  gte-base                  1/8 -> 8/8   1/8 -> 8/8
  bge-m3 (IMMUNE CONTROL)   8/8 -> 8/8   8/8 -> 8/8

bge-m3 is the positive control: it has genuine emoji coverage, scores perfectly
before the fix, and is unharmed by it. This rules out "the harness rewards
demojize" as an explanation and makes the fix safe to apply unconditionally.

### 1.8 The collapse is LENGTH-INDEPENDENT (T8)

A natural objection: surely one emoji cannot matter inside a long document.
We tested 8 documents that are word-for-word identical except for a single
emoji, at 5 / 25 / 100 / 250 / 500 words of realistic prose:

  all-MiniLM-L6-v2    every length   mean pairwise cos = 1.000000   top1 = 1/8
  bge-base-en-v1.5    every length   mean pairwise cos = 1.000000   top1 = 1/8
  bge-m3 (control)    every length   cos 0.98-0.99                  top1 = 7/8

There is no dilution curve, because there is no signal to dilute: every emoji
is replaced by the SAME `[UNK]` token (id 100), so the eight documents become
the *same token sequence* regardless of surrounding length. A 500-word support
ticket whose only distinguishing mark is 😡 versus 😍 is unretrievable.

This pre-registered gate was proposed by a reviewer specifically to catch us
overclaiming ("if a single emoji shifts a 500-word doc by < 0.01 cosine, narrow
the paper to short text"). It did not fire.

NOTE ON THE CONTROL: bge-m3 scores 7/8, not 8/8, on this task at every length.
We report 7/8.

### 1.9 A hypothesis we got wrong, and the correction

An earlier draft of this paper stated: *"The defect is specific to the emoji
block… say emoji, mean emoji."* **That claim was false, and our own data
falsified it.** We record it because the correction is the paper's main
result.

The original test used a coarse pass/fail over script blocks and we read it as
"non-emoji scripts survive." Re-running it per-item, on four models with the
30522 vocabulary (all-MiniLM-L6-v2, bge-base-en-v1.5, e5-base-v2, gte-base —
byte-identical results on all four), gives distinct embeddings out of 8 inputs:

| script / class | distinct | verdict |
|---|---|---|
| English words | 8/8 | intact |
| Korean | 8/8 | intact |
| Arabic | 8/8 | intact |
| Devanagari | 8/8 | intact |
| Cyrillic | 8/8 | intact |
| Mathematical operators | 7/8 | ∑ = ∫ |
| Currency symbols | 6/8 | ₪ = ₫ = ₽ |
| Japanese | 6/8 | 市場 = 農場 = 工場 |
| **Chinese** | **4/8** | 医院 = 机场 = 市场 = 农场 = 工厂 |
| **Emoji** | **1/8** | total collapse |

The correct statement is that this is a *vocabulary-coverage* defect, not an
emoji defect. Emoji are its extreme point (1/8) and Chinese its highest-stakes
point (4/8).

**And a second correction on top of the first.** The table above is a
measurement of *injectivity* — whether the encoder assigns distinct vectors —
and an earlier draft read the 8/8 rows as "Korean, Arabic, Devanagari and
Cyrillic are unaffected." That does not follow, and §1.1 shows it is false:
those four scripts are injective yet retrieve at 0.000–0.083 on a task where
English scores 0.917–1.000. They avoid the collision and still fail. So
selectivity applies to the *collapse mechanism*, not to retrieval success:
identical-vector collapse is script-selective (Chinese, Japanese, emoji), while
retrieval failure is general to every non-English script we tested. Reporting
only the first would have understated the problem; reporting only the second
would have missed the deterministic mechanism. Both are in §1.1.

We record this twice-corrected passage deliberately. Two independent reviewers
supplied the pre-registered gates that killed our own claims, and in one case a
reviewer's gate destroyed the rebuttal that same reviewer had proposed.

## 2. Why it is news: the hazard is the default

Verified by reading library source, not documentation:

  ChromaDB   chromadb/api/models/Collection.py:1691
             embedding_function=DefaultEmbeddingFunction() -> ONNXMiniLM_L6_V2
             -> all-MiniLM-L6-v2                                    COLLAPSED
  Qdrant     fastembed/text/text_embedding.py -> BAAI/bge-small-en-v1.5  COLLAPSED
  txtai      src/python/txtai/embeddings/base.py:844 -> all-MiniLM-L6-v2 COLLAPSED

A developer who installs ChromaDB and accepts the default cannot retrieve
emoji-bearing documents. Reproduced end-to-end in 30 lines against unmodified
ChromaDB (killshot_chromadb.py): three documents differing only by emoji, every
query returns the same wrong document (1/3), and demojize restores 3/3.

## 3. Prior art and what is actually new

- sentence-transformers issue #1177 (open since 2021). Maintainer Nils Reimers
  confirms emoji are absent from the vocabulary and writes that it is "unclear
  how much semantic meaning an emoji encodes." The question was posed and left
  open; we answer it with measurements.
- Łukawski (Qdrant, 2024), "word injection" post, states in prose that unknown
  tokens can cause identical embeddings. Single model, no quantification, no
  retrieval evaluation, no cure validation.

New here: SCOPE (8 models incl. three shipped defaults), MAGNITUDE (99.83%,
counted against the official emoji set), CONSEQUENCE (retrieval at chance,
measured end-to-end in a real vector DB), BOUNDARY (emoji-specific, other
scripts survive), and CURE (one line, validated, with an immune control).

## 4. Limitations

- **Corpora are author-constructed.** The CJK task (24 entities) and the
  polarity task (96 pairs) use documents we wrote, with mechanical validators
  for emoji-freeness and answer leakage. The real-text emoji probe (§1.3,
  n=150) uses genuine human messages. Replacing the CJK corpus with
  T2Ranking — real Chinese queries with expert relevance judgments, sampling
  frozen against the collision set — is the obvious strengthening step and we
  regard it as required for a v2.
- **Scope of scripts tested.** We cover English, Chinese, Japanese, Korean,
  Arabic, Devanagari, Cyrillic, mathematical operators, currency and emoji.
  Thai, Hebrew, Greek and Armenian are untested and may collide.
- **We measure retrieval, not end-to-end task accuracy.** We do not claim a
  quantified downstream QA degradation, only that the retrieval substrate
  returns a fixed wrong document.
- **The mitigation is partial and we say so.** `demojize` recovers 25–40% of
  the gap to an explicit-English control for emoji, and does *not* address
  CJK at all. The general remedy — an encoder whose vocabulary covers your
  users' scripts — is a procurement decision, not a one-line patch.
- **bge-m3 is a control, not an endorsement.** It also ties on 8/96 polarity
  pairs and loses one emoji in our probe set. Coverage is a spectrum.

## 5. Reproduction

`reproduce.py` is the artifact. It requires only `pip install chromadb`, uses
ChromaDB's shipped defaults with no configuration, and prints the failure, the
English-query control and the offending token IDs in one run:

```
pip install chromadb && python3 reproduce.py
  Chinese queries: 1/8 correct
  English queries: 8/8 correct   <-- same corpus, same model
  COLLISION: hospital, airport, market, farm, factory, prison, pharmacy
             all tokenize to [100, 100] -> ONE shared vector
```

Full experiment scripts are in `/home/jason/dream_harness`: `t2` (emoji
census and 9-model survey), `t5`/`t13` (script-by-script blast radius and
collision groups), `t9` (real human text), `t10` (token-ID mechanism),
`t11`/`t12` (pre-registered semantic polarity, n=12 pilot and n=96 powered),
`t14` (cross-lingual CJK retrieval), `t15`/`t16` (falsification of the
insertion-order explanation and confirmation of the query-side attractor).

Anyone can check their own stack in one line:

```python
tok("your term", add_special_tokens=False)["input_ids"]   # id 100 == gone
```
