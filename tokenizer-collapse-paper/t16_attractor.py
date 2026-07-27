#!/usr/bin/env python3
"""T16 -- confirm the CORRECTED mechanism: query-side collapse onto a fixed
attractor.

T15 falsified my "insertion order" story. New claim to verify:
  1. All Chinese query vectors are BIT-IDENTICAL to each other (one point).
  2. The English document vectors remain fully distinct (no doc-side problem).
  3. Therefore every Chinese query has the SAME nearest neighbour, whatever it
     happens to be -- a fixed attractor determined by the corpus, not the query.
  4. Changing the corpus moves the attractor, proving the query carries zero
     information: swap in a different doc set and every Chinese query follows
     the new winner.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

ZH = ["医院", "机场", "市场", "农场", "工厂", "监狱", "药房"]
EN = ["hospital", "airport", "market", "farm", "factory", "prison", "pharmacy"]
DOCS_A = [
 "Patients are admitted here for surgery and emergency treatment, and doctors and nurses provide medical care to the seriously ill.",
 "Passengers check in their luggage here before boarding international flights, and aircraft take off and land on the runways.",
 "Traders sell fresh produce and household goods from stalls here, and shoppers haggle over prices for vegetables and fish.",
 "Crops are planted and harvested in the surrounding fields here, and livestock such as cattle and chickens are raised for milk and meat.",
 "Production lines assemble manufactured goods here in shifts, and machinery stamps, welds and packages components for shipment.",
 "Inmates are held here in cells under supervision, guards patrol the corridors and wings, and visitors are searched before entering.",
 "Prescriptions are dispensed here by trained staff, customers buy painkillers and cold remedies, and advice is given on dosage.",
]
# A DIFFERENT corpus: if the attractor moves, the query truly carries no signal.
DOCS_B = [
 "The annual budget report was approved by the finance committee after a lengthy discussion of departmental spending limits.",
 "Volcanic activity on the island has increased, and geologists are monitoring seismic readings around the caldera each hour.",
 "The football match ended in a draw after both teams scored late in the second half in front of a capacity crowd.",
 "Researchers sequenced the genome of a deep sea organism and identified several previously unknown protein families.",
 "The orchestra performed a symphony by a nineteenth century composer, with a violin soloist joining for the final movement.",
 "Heavy rainfall caused localised flooding across the valley, and emergency services advised residents to avoid low roads.",
 "A new bridge across the estuary opened to traffic, cutting the journey time between the two towns by twenty minutes.",
]

m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
Q = m.encode(ZH, normalize_embeddings=True, show_progress_bar=False)
QE = m.encode(EN, normalize_embeddings=True, show_progress_bar=False)

pairs = [(i, j) for i in range(len(ZH)) for j in range(i+1, len(ZH))]
ident = sum(int(np.allclose(Q[i], Q[j], atol=1e-6)) for i, j in pairs)
print(f"1. Chinese query vectors identical: {ident}/{len(pairs)} pairs")
print(f"   max pairwise cosine among Chinese queries: "
      f"{max((Q[i]@Q[j]) for i, j in pairs):.6f}")
print(f"   min pairwise cosine among ENGLISH queries: "
      f"{min((QE[i]@QE[j]) for i, j in pairs):.6f}  (should be well below 1)")

for tag, DOCS in (("corpus A (places)", DOCS_A), ("corpus B (unrelated)", DOCS_B)):
    D = m.encode(DOCS, normalize_embeddings=True, show_progress_bar=False)
    dpairs = sum(int(np.allclose(D[i], D[j], atol=1e-6)) for i, j in pairs)
    nn = [int((Q @ D.T)[k].argmax()) for k in range(len(ZH))]
    print(f"\n2. {tag}: identical DOC pairs = {dpairs}/{len(pairs)} (expect 0)")
    print(f"3. nearest doc index for each Chinese query: {nn}")
    print(f"   all Chinese queries -> the SAME document? "
          f"{len(set(nn)) == 1}  (index {nn[0]})")
    print(f"   attractor text: {DOCS[nn[0]][:70]}...")

print("\n" + "=" * 66)
print("If the attractor index DIFFERS between corpus A and corpus B while all\n"
      "Chinese queries still land on ONE document in each, then the query\n"
      "carries zero information and the destination is a property of the\n"
      "CORPUS alone.")
