"""The 0.360 is a DEGENERATE-ARGMAX artifact, not retrieval skill.

When every candidate collapses to the same token sequence, all 8 similarities
are EXACTLY equal. np.argmax then returns index 0 -- always. So a trial scores
a "hit" iff its gold emoji happens to sit at index 0 of the alphabet, which is
😂, the most frequent emoji in real usage.

Prediction: raw accuracy == share of trials whose true emoji is 😂,
and the number of DISTINCT similarity values per trial == 1.
"""
import collections, numpy as np, json
from sentence_transformers import SentenceTransformer
exec(open("/home/jason/dream_harness/t9_realcontext.py").read().split(
     "results = {}")[0].replace('print(', '_p('), g := {"_p": lambda *a, **k: None})

TRIALS, top8 = g["TRIALS"], g["top8"]
share = collections.Counter(e for _, e in TRIALS)
n = len(TRIALS)
print("alphabet order:", " ".join(top8))
print(f"index0 = {top8[0]}   share of trials = {share[top8[0]]}/{n} = "
      f"{share[top8[0]]/n:.3f}")
print("observed raw accuracy = 0.360")

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
ties = 0
for text, true_e in TRIALS[:40]:
    cands = [text.replace(true_e, alt) for alt in top8]
    E = m.encode(cands + [text], normalize_embeddings=True,
                 show_progress_bar=False)
    sims = E[:-1] @ E[-1]
    if len(np.unique(np.round(sims, 6))) == 1:
        ties += 1
print(f"trials where all 8 similarities are IDENTICAL: {ties}/40")
