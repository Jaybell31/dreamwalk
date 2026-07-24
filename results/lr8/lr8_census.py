#!/usr/bin/env python3
"""
Lonely Runner k=8 exact census / counterexample hunt.
8 moving runners, distinct positive integer speeds (gcd=1), observer at 0.
Gap ML(v) = max_{t in (0,1)} min_i ||v_i t||; conjecture ML >= 1/9.
EXACT: envelope maxima only at peaks t=j/(2 v_i) and crossings
t=m/(v_i -+ v_j). Pure integer arithmetic -> every line is a certificate.
Early exit once any t certifies min dist >= LOG threshold (not near-tight).
"""
import sys, json, time
from math import gcd
from itertools import combinations
from multiprocessing import Pool

K = 8
TARGET_NUM, TARGET_DEN = 1, 9      # conjectured bound
LOG_NUM, LOG_DEN = 113, 1000       # log census row if gap < 0.113

def candidates(v):
    seen = set()
    for i in range(K):
        q = 2 * v[i]
        for p in range(1, q):
            g = gcd(p, q); c = (p // g, q // g)
            if c not in seen:
                seen.add(c); yield c
    for i in range(K):
        for j in range(i + 1, K):
            for q in (v[j] - v[i], v[j] + v[i]):
                if q <= 1: continue
                for p in range(1, q):
                    g = gcd(p, q); c = (p // g, q // g)
                    if c not in seen:
                        seen.add(c); yield c

def gap_or_none(v):
    best_n, best_d = 0, 1
    for p, q in candidates(v):
        m = q  # min over runners of dist*q (common denom q at this t)
        for vi in v:
            r = (vi * p) % q
            d = r if r <= q - r else q - r
            if d < m: m = d
        if m * best_d > best_n * q:
            best_n, best_d = m, q
            if best_n * LOG_DEN >= LOG_NUM * best_d:
                return None  # certified not near-tight
    return (best_n, best_d)

def work(tup):
    g = 0
    for x in tup: g = gcd(g, x)
    if g != 1:
        return None  # scaled copy of smaller tuple
    r = gap_or_none(tup)
    if r is None:
        return None
    n, d = r
    if n * TARGET_DEN < TARGET_NUM * d: kind = "COUNTEREXAMPLE"
    elif n * TARGET_DEN == TARGET_NUM * d: kind = "TIGHT"
    else: kind = "NEAR"
    return {"speeds": list(tup), "gap": f"{n}/{d}", "gap_float": n / d, "kind": kind}

def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    out = f"/home/jason/dream_harness/lr8_probe/census_B{B}.jsonl"
    t0 = time.time(); n_done = n_hit = 0
    with Pool(14) as pool, open(out, "w") as f:
        for res in pool.imap_unordered(work, combinations(range(1, B + 1), K), chunksize=2000):
            n_done += 1
            if res:
                n_hit += 1
                f.write(json.dumps(res) + "\n"); f.flush()
                if res["kind"] == "COUNTEREXAMPLE":
                    print(f"!!! COUNTEREXAMPLE !!! {res}", flush=True)
            if n_done % 200000 == 0:
                print(f"[{time.time()-t0:.0f}s] {n_done} tuples, {n_hit} logged", flush=True)
    print(f"DONE B={B}: {n_done} tuples, {n_hit} near-tight, {time.time()-t0:.0f}s -> {out}", flush=True)

if __name__ == "__main__":
    main()
