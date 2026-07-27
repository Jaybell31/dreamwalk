#!/usr/bin/env python3
"""THE KILL SHOT -- 60-second reproduction, default ChromaDB, no config.

Point: you do not have to misconfigure anything. Install a vector DB, accept
the default embedding model, store documents that contain emoji, and those
documents become unretrievable -- they all collapse to the SAME vector.

Run:  pip install chromadb emoji  &&  python3 killshot_chromadb.py
"""
import chromadb, emoji

# Three documents whose ONLY distinguishing content is the emoji. This is not a
# contrived edge case: it is a product catalog, a reaction log, a chat export.
DOCS = {
    "d1": "Customer reaction: 😂",   # face with tears of joy  -- amused
    "d2": "Customer reaction: 😡",   # pouting face            -- angry
    "d3": "Customer reaction: 😍",   # smiling face with hearts-- delighted
}

def run(label, transform):
    client = chromadb.Client()                      # default embedding function
    col = client.create_collection(label)           # -> all-MiniLM-L6-v2
    ids = list(DOCS)
    col.add(ids=ids, documents=[transform(DOCS[i]) for i in ids])

    print(f"\n=== {label} ===")
    ok = 0
    for target in ids:
        q = transform(DOCS[target])
        got = col.query(query_texts=[q], n_results=1)["ids"][0][0]
        hit = "OK " if got == target else "MISS"
        if got == target:
            ok += 1
        print(f"  query {DOCS[target]!r:34} -> returned {got}  {hit}")
    print(f"  correct: {ok}/{len(ids)}")
    return ok

raw   = run("raw_emoji", lambda s: s)
fixed = run("with_demojize", emoji.demojize)

print("\n" + "=" * 62)
print(f"  RAW default ChromaDB : {raw}/3 correct")
print(f"  + emoji.demojize()   : {fixed}/3 correct")
print("=" * 62)
print("The emoji are erased to [UNK] before embedding, so all three documents")
print("occupy the SAME point in vector space. Retrieval cannot distinguish them.")
