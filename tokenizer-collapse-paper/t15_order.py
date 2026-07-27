#!/usr/bin/env python3
"""T15 -- is the wrong answer DETERMINISTIC by insertion order?

reproduce.py showed farm/factory/prison/pharmacy in Chinese ALL return
"hospital", and hospital happens to be inserted first. If the returned
document is simply whichever tied document was added first, then:

  (a) the failure is fully deterministic, not noise -- a stronger claim;
  (b) reversing insertion order should move every wrong answer to the new
      first-inserted tied document;
  (c) the distance scores for tied documents must be EXACTLY equal.

If (b) fails, my claim in FOCUS.md is wrong and must be corrected before it
reaches the paper. Testing it rather than asserting it.
"""
import chromadb

CORPUS = {
    "hospital":   "Patients are admitted here for surgery and emergency treatment, and doctors and nurses provide medical care to the seriously ill.",
    "airport":    "Passengers check in their luggage here before boarding international flights, and aircraft take off and land on the runways.",
    "market":     "Traders sell fresh produce and household goods from stalls here, and shoppers haggle over prices for vegetables and fish.",
    "farm":       "Crops are planted and harvested in the surrounding fields here, and livestock such as cattle and chickens are raised for milk and meat.",
    "factory":    "Production lines assemble manufactured goods here in shifts, and machinery stamps, welds and packages components for shipment.",
    "prison":     "Inmates are held here in cells under supervision, guards patrol the corridors and wings, and visitors are searched before entering.",
    "pharmacy":   "Prescriptions are dispensed here by trained staff, customers buy painkillers and cold remedies, and advice is given on dosage.",
}
ZH = {"hospital": "医院", "airport": "机场", "market": "市场", "farm": "农场",
      "factory": "工厂", "prison": "监狱", "pharmacy": "药房"}

def run(order, tag):
    c = chromadb.Client()
    col = c.create_collection(tag)
    col.add(ids=order, documents=[CORPUS[i] for i in order])
    print(f"\n=== insertion order: {order[0]} first")
    out = {}
    for k in order:
        r = col.query(query_texts=[ZH[k]], n_results=len(order))
        out[k] = r["ids"][0][0]
    for k in sorted(out):
        print(f"   asked {k:9} -> {out[k]}")
    # distances for one query: are the tied docs exactly equal?
    r = col.query(query_texts=[ZH["hospital"]], n_results=len(order))
    ds = [round(d, 8) for d in r["distances"][0]]
    print(f"   distances for 医院 query: {ds}")
    print(f"   distinct distance values: {len(set(ds))} of {len(ds)}")
    c.delete_collection(tag)
    return out, set(ds)

fwd = list(CORPUS)
rev = fwd[::-1]
a, da = run(fwd, "fwd")
b, db = run(rev, "rev")

first_fwd, first_rev = fwd[0], rev[0]
all_to_first_fwd = all(v == first_fwd for v in a.values())
all_to_first_rev = all(v == first_rev for v in b.values())

print("\n" + "=" * 66)
print(f"forward: every query returned {first_fwd!r}? {all_to_first_fwd}")
print(f"reversed: every query returned {first_rev!r}? {all_to_first_rev}")
print(f"tied distances collapse to a single value? fwd={len(da)==1} rev={len(db)==1}")
if all_to_first_fwd and all_to_first_rev:
    print("\nCONFIRMED: the returned document is whichever tied document was\n"
          "INSERTED FIRST. The failure is deterministic, not random.")
else:
    print("\nNOT CONFIRMED -- claim must be corrected. Actual mapping above.")
