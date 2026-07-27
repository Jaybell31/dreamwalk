#!/usr/bin/env python3
"""
SILENT [UNK] COLLAPSE -- minimal reproducer against a STOCK ChromaDB install.

    pip install chromadb
    python3 reproduce.py

No configuration. No custom embedding function. This uses ChromaDB exactly as
it ships, which is all-MiniLM-L6-v2 via its default embedding function.

WHAT YOU WILL SEE
  A vector database is asked, in Chinese, for the document about a HOSPITAL.
  It returns the document about an AIRPORT -- or a farm, or a prison. Not
  because ranking is imperfect, but because the query for "hospital" and the
  query for "airport" produce the IDENTICAL vector. There is nothing left to
  rank.

WHY
  The default model carries a 30522-token English WordPiece vocabulary. Every
  Chinese character in these words is out of vocabulary, so each word is
  REPLACED (not deleted) by the unknown token [UNK], id 100. 医院 (hospital)
  and 机场 (airport) both tokenize to exactly [100, 100]. Same ids -> same
  vector -> same search results.

  This is silent. No exception, no warning, no log line. The database returns
  a confident answer with a plausible distance score.

CONTROL
  The same corpus, queried in English, retrieves correctly. The corpus is fine.
  The failure is entirely in how the query is encoded.
"""
import sys

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    sys.exit("pip install chromadb")

# Documents are ordinary English prose. No emoji, no Chinese, nothing exotic.
CORPUS = {
    "hospital":   "Patients are admitted here for surgery and emergency treatment, and doctors and nurses provide medical care to the seriously ill.",
    "airport":    "Passengers check in their luggage here before boarding international flights, and aircraft take off and land on the runways.",
    "market":     "Traders sell fresh produce and household goods from stalls here, and shoppers haggle over prices for vegetables and fish.",
    "farm":       "Crops are planted and harvested in the surrounding fields here, and livestock such as cattle and chickens are raised for milk and meat.",
    "factory":    "Production lines assemble manufactured goods here in shifts, and machinery stamps, welds and packages components for shipment.",
    "prison":     "Inmates are held here in cells under supervision, guards patrol the corridors and wings, and visitors are searched before entering.",
    "pharmacy":   "Prescriptions are dispensed here by trained staff, customers buy painkillers and cold remedies, and advice is given on dosage.",
    "library":    "Visitors borrow books here with a membership card, students study quietly at desks, and archives of newspapers are kept on shelves.",
}
ZH = {"hospital": "医院", "airport": "机场", "market": "市场", "farm": "农场",
      "factory": "工厂", "prison": "监狱", "pharmacy": "药房", "library": "图书馆"}

ids = list(CORPUS)
client = chromadb.Client()
col = client.create_collection("unk_collapse")          # stock defaults
col.add(ids=ids, documents=[CORPUS[i] for i in ids])

def ask(queries, label):
    print(f"\n--- {label} " + "-" * (58 - len(label)))
    hits = 0
    for key, q in queries.items():
        got = col.query(query_texts=[q], n_results=1)["ids"][0][0]
        ok = got == key
        hits += ok
        flag = "ok " if ok else "WRONG"
        print(f"  {flag}  asked for {key:9} (query {q!r:11}) -> returned {got}")
    print(f"  => {hits}/{len(queries)} correct")
    return hits

zh_hits = ask({k: ZH[k] for k in ids}, "QUERY IN CHINESE")
en_hits = ask({k: k for k in ids}, "QUERY IN ENGLISH (control)")

# Show the mechanism directly, using the tokenizer the database just used.
print("\n--- WHY: the token ids the default model actually produced " + "-" * 6)
try:
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    groups = {}
    for k in ids:
        t = tuple(tk(ZH[k], add_special_tokens=False)["input_ids"])
        groups.setdefault(t, []).append(k)
        print(f"  {ZH[k]:5} {k:9} -> {list(t)}")
    print()
    for t, members in groups.items():
        if len(t) and all(x == 100 for x in t) and len(members) > 1:
            print(f"  COLLISION: {', '.join(members)} all tokenize to {list(t)}")
            print(f"             -> ONE shared vector for {len(members)} different places.")
except ImportError:
    print("  (pip install transformers to see the token ids)")

print("\n" + "=" * 68)
print(f"Chinese queries: {zh_hits}/{len(ids)} correct")
print(f"English queries: {en_hits}/{len(ids)} correct   <-- same corpus, same model")
print("""
The corpus is retrievable. The model is fine at English. The database raised
no error. It simply cannot distinguish a hospital from an airport, because on
a 30522-token English vocabulary those two words are the same sequence of
unknown tokens.

CHECK YOUR OWN STACK:
    tok("your term", add_special_tokens=False)["input_ids"]
If you see id 100 (or your tokenizer's unk_token_id), that content is gone
before the model ever sees it. But note the defect is NOT limited to [UNK]:
invented ASCII English names with no unknown tokens at all fail the same way
purely from over-fragmentation, so a clean token dump does not clear you.

====================================================================
WHAT TO ACTUALLY DO ABOUT IT
Ordered by how much they buy you. Every number below comes from a
pre-registered run in this repo; see the named test for the full table.

1. ADD A LEXICAL (BM25) ARM. Biggest single win. The content is already in
   your index and reachable -- it is the DENSE arm that cannot see it. On
   exact queries a lexical arm scores 1.000 where dense scores 0.20-0.41.
   [T33]

2. INDEX CHARACTER N-GRAMS, NOT JUST WHITESPACE TOKENS. This is mandatory,
   not a CJK nicety. Whitespace BM25 scores 0.008 on Japanese company names;
   char-ngram BM25 scores 1.000 on the same corpus and model. It also buys
   TYPO RESILIENCE: a single-term name like "Zolpidraxen" drops 1.000 ->
   0.008 on one typo because there is no redundancy left when its only term
   is corrupted, while multi-term/n-grammed entities hold 0.775-0.900.
   N-grams manufacture that redundancy. [T33 P2, T34]

3. DO NOT FUSE WITH PLAIN RRF. Reciprocal rank fusion loses to its own
   better component in 37 of 40 tested cells, because it ranks by ORDER and
   discards score MAGNITUDE -- so an arm returning pure noise still casts a
   full-strength vote. Normalise each arm by its own maximum and sum:
       fused = dense/dense.max() + bm25/bm25.max()
   RRF 0.345 -> max-normalised 0.697 on identical arms, queries and corpus.
   That is 2.0x recall from one line of arithmetic. [T35]

4. SUBSTITUTING A PLAIN-WORD FORM of the entity helps, but only modestly
   (Japanese 0.197 -> 0.253) and it is not a substitute for 1-3. For emoji
   specifically, emoji.demojize() is significant but recovers only 25-40% of
   the gap; replacing the emoji with its plain English word is better. [T11,
   T12, T33]

5. A LARGER-VOCABULARY ENCODER (e.g. BAAI/bge-m3, 250002 tokens) helps and is
   worth doing, but it is NOT a fix. bge-m3 still collapses to 0.067 on
   fractured ASCII names and 0.017 at long chunk lengths. Nobody is immune;
   coverage is a spectrum, not a switch. [T27, T28, T30]

WHAT DOES NOT WORK: tuning your chunk size. The penalty is already maximal at
the shortest chunks (2.7-6.2x vs in-vocabulary names at 8 words) and shrinking
chunks does not close it. The gap lives in the tokenizer, upstream of anything
your retrieval config can reach. [T32]""")
