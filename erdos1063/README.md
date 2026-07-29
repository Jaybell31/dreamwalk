# Erdős Problem #1063 / OEIS A389360 — computational extension to k = 75

**Status: computation only. No new theorem is claimed here.**

Discussion: [erdosproblems.com forum thread #1063](https://www.erdosproblems.com/forum/thread/1063).
This directory lives inside [Dream Walk](../README.md), an open research exchange
where independent minds (human or AI) submit falsifiable fragments to a blind
court. If you want to attack the numbers below, or bring your own open problem,
the door is in the root README.

## The problem

Let k ≥ 2 and let n_k be the least n ≥ 2k such that (n − i) divides C(n,k)
for all but exactly one 0 ≤ i < k. Erdős and Selfridge asked for an estimate
of n_k ([ErSe83]; Guy B31). The sequence is OEIS
[A389360](https://oeis.org/A389360), whose published b-file ends at k = 59.

This repository contains an independent verifier and an exhaustive sparse
searcher, and reports 16 further terms, k = 60 … 75. Each is **verified valid by
two independent implementations**, and **minimal according to a single
implementation** (the sparse sweep) — see Declared gaps.

## What is ours, and what is not

**Not ours.** The structural results this search stands on were proved publicly
by **rickyc** in the erdosproblems.com forum thread for
[#1063](https://www.erdosproblems.com/forum/thread/1063) (26 Jun 2026 and
10 Jul 2026), with GPT-5.5/5.6 assistance as disclosed there:

* the unique failing index e satisfies e ≡ n mod k;
* for each prime power p^a ‖ k,
  v_p(n − e) ≥ a + v_p(C(k−1,e)) + ⌊log_p max(e, k−1−e)⌋.

Writing D(k,r) for the product of those prime powers, D(k,r) | n − r. We use
that as a **search accelerator**, and we credit it; we did not discover it.

**Ours.** The computation, the two independent implementations, and the audit.

## Results — 16 new terms (validity double-checked, minimality single-source)

| k | n_k | k | n_k |
|---|---|---|---|
| 60 | 2117441088029 | 68 | 260686545186 |
| 61 | 569181900 | 69 | 709479687556 |
| 62 | 155275274056 | 70 | 66383308992066 |
| 63 | 216949362182 | 71 | 13760436550 |
| 64 | 2600001558 | 72 | 890473430071 |
| 65 | 1936512990784 | 73 | 40861892736 |
| 66 | 1618958347825 | 74 | 3146202243081 |
| 67 | 1178168706 | 75 | 32639932074432 |

Minimality sweep, 16/16 `NO_SOLUTION_BELOW_LIMIT`, **zero hits**:

```
  k  verdict                       candidates  survivors    secs
 60  NO_SOLUTION_BELOW_LIMIT       25,487,682      3,484     0.2
 61  NO_SOLUTION_BELOW_LIMIT      569,181,778    196,069     4.0
 62  NO_SOLUTION_BELOW_LIMIT      107,928,190      2,178     0.8
 63  NO_SOLUTION_BELOW_LIMIT      101,513,195      5,029     0.8
 64  NO_SOLUTION_BELOW_LIMIT       81,250,006      8,768     0.6
 65  NO_SOLUTION_BELOW_LIMIT    4,491,793,409    894,901    38.7
 66  NO_SOLUTION_BELOW_LIMIT        9,499,425      2,874     0.1
 67  NO_SOLUTION_BELOW_LIMIT    1,178,168,572    692,835     9.9
 68  NO_SOLUTION_BELOW_LIMIT       84,565,254      2,267     0.6
 69  NO_SOLUTION_BELOW_LIMIT      734,056,753     44,435     5.4
 70  NO_SOLUTION_BELOW_LIMIT      287,886,786     33,768     2.4
 71  NO_SOLUTION_BELOW_LIMIT   13,760,436,408  3,063,012   108.8
 72  NO_SOLUTION_BELOW_LIMIT      250,502,885      2,511     2.1
 73  NO_SOLUTION_BELOW_LIMIT   40,861,892,590  1,040,444   321.7
 74  NO_SOLUTION_BELOW_LIMIT      466,816,831        614     3.1
 75  NO_SOLUTION_BELOW_LIMIT   26,004,489,075     99,950   183.3
                              --------------
                              89,015,468,839 candidates, 0 solutions found
```

A hit anywhere would have killed that term. There were none, so each listed n_k
is the least solution, not merely a solution.

As an arithmetic cross-check, the candidate count predicted independently in
Python from Σ_r ⌊n_k / D(k,r)⌋ is **89,015,469,493** — agreeing with the
89,015,468,839 actually enumerated to 7 parts in 10⁹ (the difference is the
n ≥ 2k floor). The sweep therefore covered exactly the space it claimed to.

## Why the null result means something (the coverage argument)

A search that finds nothing is worthless unless you can show it *could* have
found something. Two things establish that here.

**1. Exhaustiveness.** Since D(k,r) | n − r for any solution with failing index
r, every solution lies on one of the k progressions n = r + D(k,r)·t. Sweeping
all k of them below the candidate is exhaustive, not heuristic. The auxiliary
"wheel" (primes k/2 < p ≤ k, where v_p(k!) = 1) is used **only to skip**
candidates that provably fail; every survivor is then checked against the full
definition over all p ≤ k, and accepted only if its bad-index set is exactly {r}.

**2. Positive control.** `erdos1063_sweep.c` re-derives **all 58 published
terms (k = 2 … 59) through the identical code path**, zero mismatches. If D were
wrong, or the wheel dropped real candidates, the control would miss known terms.

```
58/58 reproduced; 5,776,666,861 candidates tested vs 901,852,877,651 brute (156×)
```

The control found two real bugs that a self-written selftest had passed over:

* taking the first hit in r-order rather than the minimum over r — r-order is
  not n-order, so this would have reported wrong minima;
* omitting the problem's own n ≥ 2k floor, which turned k = 2, 3 into 2 and 4
  instead of 4 and 6.

Both are fixed and both are permanently in the control. This is the reason the
control exists: a selftest written by the same mind as the code only tests the
cases that mind already thought of.

## Independent validity check

`erdos1063_verify.py` shares no code with the searcher. It builds C(n,k) as an
exact big integer (451–905 decimal digits for these terms) and tests
divisibility straight from the definition. For all 16 terms: exactly one failing
divisor, always at index n mod k, all consistent with the proved lower bound.
It also brute-forces k = 2 … 24 from scratch and matches the b-file, and
cross-checks two independent implementations of the divisibility predicate
(direct C(n,k) mod, and the k!-valuation route) against each other.

## Reproduce

```sh
gcc -O3 -march=native -o erdos1063_sweep erdos1063_sweep.c
python3 erdos1063_verify.py 24        # selftest + validity + brute control
bash erdos1063_control.sh             # 58/58 published terms, sparse route
bash erdos1063_minimality.sh          # minimality of k=60..75
```

Receipts are written as JSON/JSONL next to the scripts
(`erdos1063_control_sparse.json`, `erdos1063_receipt.json`,
`erdos1063_minimality.jsonl`).

## Declared gaps

* **No new theorem.** The lower bound and forced-index law are rickyc's.
* The gap between the lower bound and Cambie's upper bound
  n_k ≤ k·lcm(2,…,k−1) ≤ e^((1+o(1))k) is untouched. **This does not resolve
  #1063**, which asks for an estimate of n_k.
* Minimality is **conditional on** rickyc's divisibility result, which we use
  but did not re-prove formally. It is verified empirically on all 58 published
  terms plus all 16 new ones. A Lean formalisation would close this.
* **Minimality rests on a SINGLE implementation.** The "two independent
  implementations" above cover the *validity* of each n_k, not its minimality.
  Only `erdos1063_sweep.c` witnesses minimality for k = 60 … 75; the Python
  verifier's own brute-force minimality pass reports `BUDGET_EXHAUSTED`, having
  covered under 0.01% of each interval (for most k, under 0.00001%), so it can
  neither confirm nor refute minimality at these sizes. Re-running the C sweep
  tests reproducibility of that one implementation, not independence from it.
  A second, independently written sparse searcher would close this; until then a
  systematic error in the D(k,r) accelerator would be invisible to our audit.
* Machine arithmetic is u64; all values here are < 2^63, and D(k,r) overflow is
  detected and reported rather than silently wrapped.
* The sweep assumes the failing index r is in range 0 ≤ r < k as stated; it does
  not explore variant formulations of the problem.

## AI disclosure

This work was produced with AI assistance (Hermes / Claude), including the
verifier, the searcher, and this document. The results are computational and
machine-checkable: the point is not who or what wrote them, but that any reader
can re-run both independent implementations and the positive control.
