#!/usr/bin/env python3
"""T12 GENERATOR -- scale the semantic polarity test from n=12 to n>=72.

WHY: T11 (n=12, 24 queries) showed the RIGHT SHAPE but was UNDERPOWERED.
  raw pair2way = 0.500 exactly (forced: 12/12 query pairs bit-identical)
  demojize     = 0.667 (p=0.152)  <-- not significant. Cannot claim a fix.
  word ceiling = 0.792

Julius pre-registered this branch in round 6: "If demojizing significantly
improves relevance ... the practical RAG claim survives. If it does not,
narrow the paper to tokenization coverage and retrieval integrity."
So we power the test properly BEFORE deciding which paper we are writing.

DF (local qwen3.5:35b) is the MINER here, not the engineer: it drafts
candidate items, and a MECHANICAL validator decides what survives. The local
model never judges its own output.

HARD VALIDATION (every item must pass or it is discarded):
  1. incident_doc and success_doc contain NO emoji and NO codepoint > U+2500.
  2. Neither doc contains the emoji's demojized words (no answer leakage:
     e.g. a :skull: query may not retrieve a doc containing "skull").
  3. carrier contains no emoji.
  4. Both docs >= 12 words, and the two docs differ substantially.
  5. Deduplicated on carrier text.
Items are then used ONLY if the WORD-CONTROL passes at eval time; that filter
is applied per-item in the evaluator, and is pre-registered here.
"""
import json, re, subprocess, sys, unicodedata

MODEL = "qwen3.5:35b"
OUT = "/home/jason/dream_harness/t12_pairs.json"
TARGET = 90

SEED_EMOJI = [
    ("\U0001F480", "\U0001F680"), ("\U0001F4A9", "\U0001F389"),
    ("\U0001F4C9", "\U0001F4C8"), ("\U0001F62D", "\U0001F60E"),
    ("\U0001F621", "\U0001F91D"), ("\U0001F525", "\u2705"),
    ("\U0001F6A8", "\U0001F510"), ("\u274C", "\U0001F31F"),
    ("\U0001F4A5", "\U0001F44D"), ("\U0001F44E", "\U0001F44F"),
    ("\u26A0", "\U0001F7E2"), ("\U0001F614", "\U0001F600"),
]

DOMAINS = ["software deployment", "customer support", "financial reporting",
           "hiring and recruiting", "security and compliance",
           "data engineering", "product launch", "legal and contracts",
           "manufacturing operations", "clinical trial operations",
           "logistics and shipping", "marketing campaign",
           "academic research admin", "IT helpdesk", "sales pipeline"]

PROMPT = """You write evaluation data for an information-retrieval experiment.

Produce {n} items about: {domain}

Each item is a JSON object with exactly these keys:
  "carrier"  : a short neutral status message, 5-10 words, NO emoji, NO
               punctuation at the end. It must be genuinely AMBIGUOUS about
               whether the news is good or bad. Example:
               "the quarterly numbers are in"
  "bad_doc"  : 22-40 words of plain professional prose describing the BAD
               outcome of that situation. NO emoji.
  "good_doc" : 22-40 words of plain professional prose describing the GOOD
               outcome of the SAME situation. NO emoji.

CRITICAL RULES:
- The carrier alone must NOT reveal good or bad. Polarity lives only in docs.
- bad_doc and good_doc must be about the SAME event, opposite outcomes.
- Never use the words: skull, rocket, fire, thumbs, check mark, siren, poop,
  party, chart, warning, emoji, face, star, clap, lock, handshake, green.
- Write like an internal company document. No hype. No emoji anywhere.

Return ONLY a JSON array of {n} objects. No prose, no markdown fence.
"""

BANNED = {"skull","rocket","fire","thumbs","checkmark","check","siren","poop",
          "party","chart","warning","emoji","face","star","clap","lock",
          "handshake","green","tada","confetti"}

def has_emoji(s):
    return any(ord(c) > 0x2500 for c in s)

def ask(domain, n):
    p = PROMPT.format(n=n, domain=domain)
    r = subprocess.run(["ollama","run",MODEL,p], capture_output=True,
                       text=True, timeout=900)
    t = r.stdout.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(t[i:j+1])
    except Exception:
        return []

def valid(it):
    try:
        c, b, g = it["carrier"].strip(), it["bad_doc"].strip(), it["good_doc"].strip()
    except Exception:
        return None
    if has_emoji(c) or has_emoji(b) or has_emoji(g):
        return None
    if len(b.split()) < 12 or len(g.split()) < 12 or not (4 <= len(c.split()) <= 12):
        return None
    if b.lower() == g.lower():
        return None
    for d in (b, g, c):
        w = set(re.findall(r"[a-z]+", d.lower()))
        if w & BANNED:
            return None
    return {"carrier": c, "bad_doc": b, "good_doc": g}

items, seen = [], set()
for di, dom in enumerate(DOMAINS):
    if len(items) >= TARGET:
        break
    batch = ask(dom, 8)
    kept = 0
    for it in batch:
        v = valid(it) if isinstance(it, dict) else None
        if not v:
            continue
        k = v["carrier"].lower()
        if k in seen:
            continue
        seen.add(k)
        v["domain"] = dom
        v["bad_emoji"], v["good_emoji"] = SEED_EMOJI[len(items) % len(SEED_EMOJI)]
        items.append(v)
        kept += 1
    print(f"  {dom:28} raw={len(batch):2} kept={kept:2} total={len(items)}",
          flush=True)
    json.dump(items, open(OUT, "w"), indent=1)

json.dump(items, open(OUT, "w"), indent=1)
print(f"\nWROTE {len(items)} validated pairs -> {OUT}")
