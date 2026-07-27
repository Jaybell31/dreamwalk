#!/usr/bin/env python3
"""T11 -- THE DECISIVE TEST. Does the collapse break SEMANTIC retrieval?

THE KILL SHOT (Julius + Gemini, independently, round 6):

  JULIUS: "The retrieval task may still be tautological. If each candidate is
  produced by changing only the emoji and the query asks WHICH EMOJI was
  present, the experiment proves representational collapse -- not that
  ordinary RAG relevance suffers. Demojizing then writes the answer into the
  input as an English token, making 1.000 unsurprising."

  GEMINI: "A single [UNK] among 50 real words will be washed out by the
  surrounding text. The paper fails to show that [UNK] replacement destroys
  critical semantic payload. Therefore zero practical impact."

Both are right, and T9 does NOT answer them. This does.

DESIGN (why this is not tautological):
  * The QUERY is a realistic status message whose polarity is carried ONLY by
    a terminal emoji: "db migration finished 💀" vs "db migration finished 🚀".
  * The CORPUS is plain English prose with NO EMOJI ANYWHERE -- incident
    write-ups and success write-ups. Retrieval must map the query to the right
    KIND of document.
  * Nothing asks "which emoji was in the text". The gold answer is a document,
    not an emoji. demojize cannot "write the answer into the input": it writes
    :skull:, and the target doc never contains the words skull or rocket.
  * The two polarity queries are BYTE-IDENTICAL except the emoji, so under
    collapse they produce the SAME vector and therefore the SAME ranking. A
    system that cannot tell "the migration failed" from "the migration
    succeeded" is broken in a way any engineer understands.

CONTROLS:
  * bge-m3 = immune control, must SEPARATE the pair on raw text.
  * word-control: same pairs with the emoji replaced by an explicit English
    word ("a disaster"/"a success"). Every model must pass this. If a model
    fails the word-control, the corpus is too hard and the test is void.
  * tie-rate: fraction of pairs whose two raw queries are bit-identical.
"""
import json, numpy as np, torch
from sentence_transformers import SentenceTransformer

OUT = "/home/jason/dream_harness/t11_semantic.json"

# (carrier, bad_emoji, good_emoji, incident_doc, success_doc)
# Docs are ordinary ops/business prose. No emoji. No "skull"/"rocket"/"fire".
PAIRS = [
 ("the production database migration is finished", "\U0001F480", "\U0001F680",
  "Post-incident review: the schema migration left the primary replica in an inconsistent state and we served corrupted rows to customers for 40 minutes before rolling back.",
  "Release note: the schema migration completed cleanly on the primary and all replicas, with no downtime and no customer-visible impact."),
 ("we shipped the new payments flow", "\U0001F4A9", "\U0001F389",
  "Escalation summary: the new checkout path double-charged a subset of customers and finance is processing refunds while we hold the release.",
  "Launch summary: the new checkout path is live for all customers, conversion is up week over week and no refunds have been required."),
 ("the quarterly numbers are in", "\U0001F4C9", "\U0001F4C8",
  "Board memo: revenue came in materially below plan for the third consecutive quarter and we are opening a review of the cost base.",
  "Board memo: revenue beat plan for the third consecutive quarter and we are accelerating hiring against the new targets."),
 ("my code review came back", "\U0001F62D", "\U0001F60E",
  "Engineering note: the change was rejected in review, the author was asked to redesign the approach and the branch has been closed unmerged.",
  "Engineering note: the change was approved in review with no requested edits and has been merged to the mainline branch."),
 ("the client call just ended", "\U0001F621", "\U0001F91D",
  "Account risk: the client used the call to formally raise a complaint about missed deadlines and has asked about exiting the contract early.",
  "Account update: the client used the call to confirm they are renewing and expanding the contract for another year."),
 ("results from the load test", "\U0001F525", "\u2705",
  "Reliability report: the service fell over at 40 percent of target load, latency exceeded the error budget and we paged the on-call twice.",
  "Reliability report: the service held target load with latency well inside the error budget and no alerts were raised."),
 ("the security audit report landed", "\U0001F6A8", "\U0001F510",
  "Security finding: auditors identified a critical unauthenticated data exposure and we are treating it as a live incident with mandatory disclosure.",
  "Security finding: auditors closed the engagement with no critical or high findings and confirmed our controls meet the standard."),
 ("update on the hiring round", "\u274C", "\U0001F31F",
  "Recruiting update: the final candidate withdrew after the offer stage and the role is being reopened from scratch next quarter.",
  "Recruiting update: the final candidate signed the offer and starts next month, closing the role ahead of schedule."),
 ("status of the data pipeline backfill", "\U0001F4A5", "\U0001F44D",
  "Data incident: the backfill wrote duplicate records into the warehouse and every downstream dashboard has been showing inflated numbers since Tuesday.",
  "Data update: the backfill completed and reconciled exactly against source counts, so all downstream dashboards are now accurate."),
 ("feedback from the beta users", "\U0001F44E", "\U0001F44F",
  "Product signal: beta users reported the workflow is confusing and slower than the tool it replaces, and most stopped using it within a week.",
  "Product signal: beta users reported the workflow is clearer and faster than the tool it replaces, and weekly retention is strong."),
 ("the deployment to the eu region", "\u26A0", "\U0001F7E2",
  "Ops incident: the regional rollout tripped a cascading failure across availability zones and we invoked the disaster recovery runbook.",
  "Ops update: the regional rollout finished zone by zone with health checks green throughout and no rollback required."),
 ("news about the funding round", "\U0001F480", "\U0001F680",
  "Investor update: the lead withdrew from the round at the last stage and we are cutting burn to extend runway while we reassess.",
  "Investor update: the round closed oversubscribed with the lead confirmed and we are funded well beyond the original plan."),
]
WORD_BAD, WORD_GOOD = "a disaster", "a success"

MODELS = ["sentence-transformers/all-MiniLM-L6-v2",
          "BAAI/bge-base-en-v1.5",
          "BAAI/bge-m3"]

DOCS, GOLD_BAD, GOLD_GOOD = [], [], []
for _, _, _, bad_doc, good_doc in PAIRS:
    GOLD_BAD.append(len(DOCS));  DOCS.append(bad_doc)
    GOLD_GOOD.append(len(DOCS)); DOCS.append(good_doc)
N = len(PAIRS)
print(f"{N} polarity pairs, corpus = {len(DOCS)} emoji-free documents")
print(f"chance for a single query over the full corpus = 1/{len(DOCS)} = "
      f"{1/len(DOCS):.3f}")
assert not any(ord(c) > 0x2500 for d in DOCS for c in d), "corpus must be emoji-free"

try:
    import emoji as EMO
except ImportError:
    raise SystemExit("pip install emoji")

def build(mode):
    """Return (queries, golds). Two queries per pair: bad then good."""
    qs, gs = [], []
    for i, (carrier, be, ge, _, _) in enumerate(PAIRS):
        if mode == "word":
            qb, qg = f"{carrier} {WORD_BAD}", f"{carrier} {WORD_GOOD}"
        else:
            qb, qg = f"{carrier} {be}", f"{carrier} {ge}"
            if mode == "demojize":
                qb, qg = EMO.demojize(qb), EMO.demojize(qg)
        qs += [qb, qg]; gs += [GOLD_BAD[i], GOLD_GOOD[i]]
    return qs, gs

dev = "cuda" if torch.cuda.is_available() else "cpu"
results, ties = {}, {}
for name in MODELS:
    m = SentenceTransformer(name, device=dev)
    short = name.split("/")[-1]
    D = m.encode(DOCS, normalize_embeddings=True, show_progress_bar=False)
    for mode in ("raw", "demojize", "word"):
        qs, gs = build(mode)
        Q = m.encode(qs, normalize_embeddings=True, show_progress_bar=False)
        sims = Q @ D.T
        top1 = sims.argmax(1)
        acc = float(np.mean([int(t) == g for t, g in zip(top1, gs)]))
        # pair-restricted: choose between THIS pair's two docs only
        pair_hits = 0
        for i in range(N):
            for j, gold in ((2*i, GOLD_BAD[i]), (2*i+1, GOLD_GOOD[i])):
                two = [GOLD_BAD[i], GOLD_GOOD[i]]
                pick = two[int(np.argmax([sims[j][two[0]], sims[j][two[1]]]))]
                pair_hits += int(pick == gold)
        pacc = pair_hits / (2 * N)
        results[f"{short}|{mode}"] = {"corpus_top1": round(acc, 4),
                                      "pair_2way": round(pacc, 4)}
        if mode == "raw":
            same = sum(int(np.allclose(Q[2*i], Q[2*i+1], atol=1e-6))
                       for i in range(N))
            ties[short] = f"{same}/{N}"
        print(f"  {short:20} {mode:9} corpus_top1={acc:.3f}  pair2way={pacc:.3f}",
              flush=True)
        json.dump({"n_pairs": N, "corpus": len(DOCS), "acc": results,
                   "identical_raw_pairs": ties, "complete": False},
                  open(OUT, "w"), indent=1)
    print(f"  {short:20} identical raw query pairs: {ties[short]}", flush=True)
    del m
    torch.cuda.empty_cache()

json.dump({"n_pairs": N, "corpus": len(DOCS), "acc": results,
           "identical_raw_pairs": ties, "complete": True},
          open(OUT, "w"), indent=1)
print("\n" + "=" * 68)
print("SEMANTIC POLARITY TEST. Corpus has NO emoji. Query polarity is emoji-only.")
print(f"chance corpus_top1 = {1/len(DOCS):.3f} | chance pair_2way = 0.500")
for k, v in results.items():
    print(f"  {k:34} corpus_top1={v['corpus_top1']:.3f}  pair2way={v['pair_2way']:.3f}")
print("identical raw query pairs:", ties)
print(f"wrote {OUT}")
