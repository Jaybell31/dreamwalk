# A389360 — extension of the b-file to k = 75

## What this is

`b389360_extended.txt` is a b-file for OEIS **A389360** covering **k = 2..75**.

    k = 2..59    as published (positive control: re-derived here, 58/58 match)
    k = 60..75   new, computed by exhaustive sweep

This is a **computational/data contribution**. It resolves no conjecture and
claims no new theorem. Erdős Problem #1063 (Erdős–Selfridge, [ErSe83]) asks one
to *estimate* n_k; this supplies 16 further exact terms and what they say about
the estimate, nothing more.

## The sequence

Let k >= 2. Define n_k >= 2k to be the least n such that n - i divides
C(n, k) for all but exactly one 0 <= i < k.

## New terms

    k    n_k                    k    n_k
    60   2117441088029          68   260686545186
    61   569181900              69   709479687556
    62   155275274056           70   66383308992066
    63   216949362182           71   13760436550
    64   2600001558             72   890473430071
    65   1936512990784          73   40861892736
    66   1618958347825          74   3146202243081
    67   1178168706             75   32639932074432

## Method

The search uses two structural results due to **rickyc** (erdosproblems.com
forum, thread 1063) as a *search accelerator*, and credits them as such:

  * the unique failing index satisfies `e = n mod k`;
  * for each `p^a || k`,
    `p^(a + v_p(C(k-1,e)) + floor(log_p max(e, k-1-e)))` divides `n - e`.

Every solution with failing index r therefore lies on the single progression
`n = r mod D(k,r)`, where `D(k,r)` is the product of those prime powers.
Sweeping all k such progressions below a candidate is **exhaustive**, not
heuristic — which is what makes minimality (not merely solution-hood) checkable.

These results are **used, not re-proved**. Minimality here is conditional on
them. A Lean formalisation of the divisibility result would close that gap.
They are verified empirically on all 58 published terms and all 16 new ones.

## Evidence

**Exhaustiveness.** 89,015,468,839 candidates tested below the reported terms,
zero hits. Independent arithmetic check: the count predicted from
`sum_r floor(n_k / D(k,r))` is 89,015,469,493 — agreement to 7 parts in 10^9
(the residual is the `n >= 2k` floor).

**Positive control.** The identical code path re-derives all 58 published terms
k = 2..59 with zero mismatches. It found two real bugs while doing so: taking
the first hit in r-order rather than the minimum over r, and omitting the
`n >= 2k` floor. A control that never fails has not been tested.

**Independent verifier.** `erdos1063_verify.py` shares no code with the
searcher. It builds C(n,k) as an exact big integer (451–905 decimal digits) and
tests divisibility straight from the definition. For all 16 terms it finds
exactly one failing divisor, always at index `n mod k`. Receipts:
`erdos1063_receipt.json`.

**Overflow.** Machine arithmetic is u64; all values are below 2^63 and D(k,r)
overflow is detected rather than silently wrapped.

## What the new terms say about the estimate

1. **No new extreme in 38 further terms.** `n_k^(1/k)` stays inside the band
   observed for k <= 37: over 2 <= k <= 75 the maximum is still k = 10 (2.1781)
   and the minimum still k = 31 (1.3368). The new terms span only 1.3658
   (k = 67) to 1.6048 (k = 60).

2. **The prime/composite separation sharpens rather than washing out.**
   Mean of `log(n_k)/k`:

        k range    primes   composites
         2-19      0.4954     0.5825
        20-39      0.3390     0.5122
        40-59      0.3521     0.4502
        60-75      0.3264     0.4105

   Regressing `log(n_k)/k` on k over k >= 20 gives slope -0.0026 for composite k
   but -0.0002 for prime k. The apparent decay lives almost entirely on the
   composite branch; the prime branch is near-flat around 0.33 across 55 terms.
   If the truth is `c^((1+o(1))k)`, these data are consistent with **two
   different constants for the two branches** rather than one.

3. **The lower bound remains very far from the truth here.** `n_k / LB(k)` runs
   from 1.3e6 (k = 64) to 6.4e8 (k = 75); `log(LB(k))/k` is roughly 0.07–0.23
   against `log(n_k)/k` of roughly 0.31–0.47.

## Reproduction

    ./erdos1063_control.sh        # positive control, k = 2..59
    ./erdos1063_minimality.sh     # exhaustive minimality sweep
    python3 erdos1063_verify.py   # independent big-int verifier

Runs in a few minutes on one machine.

## Provenance

`MANIFEST.sha256` pins every file in this directory. `b389360_extended.txt` is
a pure b-file (no comment header) so it can be submitted to OEIS as-is;
provenance lives here and in the manifest rather than inside the data.

## AI disclosure

The searcher, the independent verifier, and the prose here were produced with AI
assistance (Claude, in an agent harness). Every claim is computational and
machine-checkable: both implementations and the positive control are in this
repository and can be re-run in minutes. Judge the artifact, not the author.

## Status

Submitted as a comment to erdosproblems.com thread 1063 on **28 Jul 2026**
(account `jbell31`), where it remains pending moderator approval. This
repository is the citable record and does not depend on that approval.
