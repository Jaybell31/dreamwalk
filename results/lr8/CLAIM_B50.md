LONELY RUNNER CONJECTURE k=8 — EXACT CENSUS CERTIFICATE (Jul 23 2026)

CLAIM
For every set of 8 distinct integer speeds drawn from {1,...,50}
(reduced: gcd of the tuple = 1; tuples with gcd g>1 are time-rescaled
copies of smaller tuples), the loneliness gap

    ML(v) = max over t in (0,1) of  min_i || v_i * t ||
    (||x|| = distance from x to the nearest integer)

satisfies ML(v) >= 1/9, i.e. NO COUNTEREXAMPLE to the Lonely Runner
Conjecture k=8 exists with max speed <= 50. Moreover the bound is
attained ONLY by v = (1,2,3,4,5,6,7,8), and no other tuple even has
ML(v) < 0.113 — the near-tight desert is absolute.

SCOPE HONESTY
This is a finite census, not a proof of LRC k=8. A counterexample, if
one exists, has max speed > 50. It is consistent with (and evidence
for) the stronger folklore conjecture that the only tight tuple for
each k is (1,...,k).

NUMBERS
  tuples enumerated : 536,878,650  (= C(50,8))
  gcd-reduced tested: all with tuple-gcd 1
  counterexamples   : 0
  gap < 0.113       : 1 tuple — (1..8), gap exactly 1/9
  runtime           : 1502 s on 14 cores, pure integer arithmetic

METHOD (why the check is EXACT, no floats)
f(t) = min_i ||v_i t|| is piecewise linear in t. Its local maxima over
(0,1) can only occur where some ||v_i t|| peaks (t = j/(2 v_i)) or two
distance functions cross (||v_i t|| = ||v_j t|| iff v_i t = +- v_j t
mod 1, i.e. t = m/(v_j - v_i) or m/(v_j + v_i)). Evaluating f on this
finite rational set and taking the max gives ML(v) exactly. At
t = p/q every distance is (v_i p mod q)/q — integer arithmetic
throughout; comparisons by cross-multiplication.

VERIFY IT YOURSELF (independent, ~30 lines)
  python3 verify_lr8.py 1 2 3 4 5 6 7 8     -> gap 1/9 TIGHT
  python3 verify_lr8.py <any 8 speeds>       -> exact gap + verdict
Sample random 8-subsets of {1..50}, confirm gap >= 1/9 every time.
The census engine (lr8_census.py) and verifier were written
separately; agreement between them on samples is the cross-check.

FILES
  lr8_census.py    census engine (14-core, early-exit at 0.113)
  verify_lr8.py    single-tuple independent verifier (Fraction-exact)
  census_B50.jsonl the near-tight log (1 line: the tight tuple)
  sweep_B50.log    run log with progress stamps

NEXT CHUNKS
  B=60..80 census (each ~x5 cost; B=80 ~ 3e10 tuples, days not weeks)
  structure-restricted deep hunt: huge-speed tuples with special
  structure (near-arithmetic-progressions), where a counterexample
  would plausibly hide
  theorem conversion: view-obstruction covering argument for
  max-speed > B (open — this is the real prize)
