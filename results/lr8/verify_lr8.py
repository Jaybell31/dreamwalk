#!/usr/bin/env python3
"""
INDEPENDENT VERIFIER — Lonely Runner Conjecture k=8 census claim.
Claim (Jul 23 2026): for ALL 8-subsets of {1..50} (gcd-reduced), the
loneliness gap ML(v) = max_t min_i ||v_i t|| satisfies ML(v) >= 1/9,
with equality ONLY at v = (1,2,3,4,5,6,7,8). No other tuple has
ML(v) < 0.113.

This verifier is INDEPENDENT of the census engine: it recomputes the
exact gap for any tuple you give it, using only the mathematical fact
that f(t) = min_i ||v_i t|| is piecewise linear, so its maximum over
(0,1) occurs at a peak t = j/(2 v_i) or a crossing t = m/(v_i +- v_j).
All arithmetic is exact integer — no floats touch the verdict.

Usage:
  python3 verify_lr8.py 1 2 3 4 5 6 7 8      -> gap = 1/9 (TIGHT)
  python3 verify_lr8.py 3 7 12 19 25 33 41 50 -> gap and verdict
Spot-check the census by sampling random tuples and confirming gap >= 1/9.
"""
import sys
from math import gcd
from fractions import Fraction

def exact_gap(v):
    cands = set()
    for vi in v:
        for p in range(1, 2 * vi):
            g = gcd(p, 2 * vi); cands.add((p // g, 2 * vi // g))
    n = len(v)
    for i in range(n):
        for j in range(i + 1, n):
            for q in (v[j] - v[i], v[j] + v[i]):
                if q > 1:
                    for p in range(1, q):
                        g = gcd(p, q); cands.add((p // g, q // g))
    best = Fraction(0)
    argbest = None
    for p, q in cands:
        m = min(min((vi * p) % q, q - (vi * p) % q) for vi in v)
        val = Fraction(m, q)
        if val > best:
            best, argbest = val, Fraction(p, q)
    return best, argbest

if __name__ == "__main__":
    v = sorted(int(x) for x in sys.argv[1:])
    assert len(v) == 8 and len(set(v)) == 8 and all(x > 0 for x in v)
    gap, t = exact_gap(v)
    bound = Fraction(1, 9)
    verdict = "TIGHT (=1/9)" if gap == bound else (
        "COUNTEREXAMPLE (<1/9)!!!" if gap < bound else "SAFE (>1/9)")
    print(f"speeds  : {v}")
    print(f"gap     : {gap} = {float(gap):.6f}  (witness time t = {t})")
    print(f"verdict : {verdict}")
