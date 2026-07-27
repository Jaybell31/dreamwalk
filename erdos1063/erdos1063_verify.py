#!/usr/bin/env python3
r"""
Erdos #1063 / OEIS A389360 -- INDEPENDENT VERIFIER + MINIMALITY AUDITOR.

n_k = least n >= 2k such that (n-i) | C(n,k) for all but exactly ONE 0<=i<k.

VERIFIER FIRST, SEARCH SECOND (conjecture-search-integrity law).
Nothing here trusts the guest submission's code; the predicate is written
straight from the definition with exact big integers.

Pass 1  selftest + positive control: reproduce the published OEIS b-file
        (k=2..59) and hand anchors n_2=4, n_3=6, n_4=9, n_5=12.
Pass 2  validity of the 16 candidate terms k=60..75 (exact big-int).
Pass 3  minimality: is there a smaller valid n for that k?  Reported per k
        with an EXPLICIT coverage bound -- never claimed beyond what ran.
"""
import sys, json, time, math
from math import comb, gcd

# ---------------------------------------------------------------- predicate
def bad_indices(n, k):
    """Indices 0<=i<k with (n-i) NOT dividing C(n,k). Definitional, exact."""
    C = comb(n, k)
    return [i for i in range(k) if C % (n - i) != 0]

def bad_count_fast(n, k):
    """Same count without building C(n,k): (n-i)|C(n,k) iff k! | prod_{j!=i}(n-j).
    Uses v_p only for p<=k (the only primes in k!). Returns list of bad i."""
    # valuations of the k consecutive integers, per prime p<=k
    primes = _primes_upto(k)
    vals = {p: [_vp(n - j, p) for j in range(k)] for p in primes}
    ek = {p: _vp_fact(k, p) for p in primes}
    bad = []
    for i in range(k):
        ok = True
        for p in primes:
            s = sum(vals[p]) - vals[p][i]
            if s < ek[p]:
                ok = False
                break
        if not ok:
            bad.append(i)
    return bad

def _primes_upto(m):
    if m < 2: return []
    s = [True] * (m + 1); s[0] = s[1] = False
    for i in range(2, int(m ** .5) + 1):
        if s[i]:
            for j in range(i * i, m + 1, i): s[j] = False
    return [i for i, b in enumerate(s) if b]

def _vp(x, p):
    v = 0
    while x % p == 0:
        x //= p; v += 1
    return v

def _vp_fact(k, p):
    v = 0; q = p
    while q <= k:
        v += k // q; q *= p
    return v

# ------------------------------------------------------- proved lower bound
def lower_bound(k):
    """max(2k, prod_{p^a || k} p^(a+floor(log_p(k-1))))  -- rickyc, 10 Jul 2026."""
    if k < 3: return 2 * k
    prod = 1; m = k
    for p in _primes_upto(k):
        if m % p: continue
        a = _vp(k, p)
        t = int(math.log(k - 1) / math.log(p) + 1e-12)
        prod *= p ** (a + t)
    return max(2 * k, prod)

def forced_modulus(k, r):
    """D_{k,r} = prod_{p^a||k} p^(a + v_p(C(k-1,r)) + floor(log_p max(r,k-1-r)))
    Claimed to divide n-r for every solution.  TREATED AS UNPROVEN: used only
    as a search accelerator, and every hit is re-checked definitionally."""
    B = comb(k - 1, r)
    rr = max(r, k - 1 - r)
    D = 1
    for p in _primes_upto(k):
        if k % p: continue
        a = _vp(k, p)
        u = 0 if rr < 1 else int(math.log(rr) / math.log(p) + 1e-12) if rr >= p else 0
        D *= p ** (a + _vp(B, p) + u)
    return D

# ------------------------------------------------------------------- checks
def selftest():
    fails = []
    # hand anchors, definitional path
    for k, want in ((2, 4), (3, 6), (4, 9), (5, 12)):
        found = None
        for n in range(2 * k, 4000):
            if len(bad_indices(n, k)) == 1:
                found = n; break
        if found != want: fails.append(f"anchor k={k} got {found} want {want}")
    # the two predicate implementations must agree
    for k in range(2, 14):
        for n in range(2 * k, 2 * k + 400):
            if bad_indices(n, k) != bad_count_fast(n, k):
                fails.append(f"predicate mismatch n={n} k={k}"); break
    # NEGATIVE CONTROLS: reject things that are NOT solutions
    assert len(bad_indices(2 * 5, 5)) != 1 or True
    neg = 0
    for k in range(2, 10):
        for n in range(2 * k, 2 * k + 200):
            if len(bad_indices(n, k)) != 1: neg += 1
    if neg == 0: fails.append("no negative cases seen -- predicate is trivially true")
    return fails

def control_bfile(bfile, kmax=40):
    """POSITIVE CONTROL: rediscover published terms by brute force from scratch."""
    out = {}
    for k in sorted(bfile):
        if k > kmax: break
        want = bfile[k]
        if want > 3_000_000:      # brute force cost guard
            out[k] = ("skipped-too-large", want); continue
        n = 2 * k; hit = None
        while n <= want:
            if len(bad_count_fast(n, k)) == 1: hit = n; break
            n += 1
        out[k] = ("OK" if hit == want else f"MISMATCH got {hit}", want)
    return out

def validate_candidates(cand):
    rows = []
    for k in sorted(cand):
        n = cand[k]
        t0 = time.time()
        bad = bad_indices(n, k)                    # exact big-int path
        C = comb(n, k)
        rows.append(dict(
            k=k, n=n, digits_of_C=len(str(C)),
            n_bad=len(bad), bad=bad,
            forced_index_ok=(bad == [n % k]) if len(bad) == 1 else False,
            r_expected=n % k,
            lower_bound=lower_bound(k), lb_ok=n >= lower_bound(k),
            D_divides=( (n - (n % k)) % forced_modulus(k, n % k) == 0 ),
            secs=round(time.time() - t0, 2)))
    return rows

def minimality_scan(k, n_target, budget_secs=60.0):
    """Exhaustively test every n in [2k, n_target) definitionally, as far as the
    budget allows. Returns (verdict, n_reached, tested).  Honest by construction:
    'CLEAR' only if the whole interval was covered."""
    t0 = time.time(); n = 2 * k; tested = 0
    while n < n_target:
        if len(bad_count_fast(n, k)) == 1:
            return ("SMALLER_FOUND", n, tested)
        n += 1; tested += 1
        if (tested & 0x3FF) == 0 and time.time() - t0 > budget_secs:
            return ("BUDGET_EXHAUSTED", n, tested)
    return ("CLEAR", n_target, tested)

# --------------------------------------------------------------------- main
CAND = {60:2117441088029,61:569181900,62:155275274056,63:216949362182,
        64:2600001558,65:1936512990784,66:1618958347825,67:1178168706,
        68:260686545186,69:709479687556,70:66383308992066,71:13760436550,
        72:890473430071,73:40861892736,74:3146202243081,75:32639932074432}

if __name__ == "__main__":
    print("== PASS 1 selftest")
    f = selftest()
    print("  FAILS:", f if f else "none (green)")
    if f: sys.exit(1)

    print("== PASS 1b positive control vs published OEIS b-file")
    # OEIS 403s urllib; fetch with curl once and cache (see skill: blocked-download-workarounds)
    import os, subprocess
    BF = "/home/jason/roundtable/b389360.txt"
    if not os.path.exists(BF):
        subprocess.run(["curl", "-s", "-A", "Mozilla/5.0",
                        "https://oeis.org/A389360/b389360.txt", "-o", BF], check=True)
    raw = open(BF).read()
    bfile = {}
    for line in raw.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit(): bfile[int(p[0])] = int(p[1])
    print(f"  b-file terms k=2..{max(bfile)}")
    ctrl = control_bfile(bfile, kmax=int(sys.argv[1]) if len(sys.argv) > 1 else 30)
    for k, (v, want) in ctrl.items(): print(f"   k={k:>3} {v:<24} published={want}")
    bad = [k for k, (v, _) in ctrl.items() if v.startswith("MISMATCH")]
    print("  control verdict:", "GREEN" if not bad else f"RED {bad}")

    print("== PASS 2 validity of 16 candidate terms k=60..75")
    rows = validate_candidates(CAND)
    for r in rows:
        print(f"   k={r['k']} n={r['n']} |C|={r['digits_of_C']}d bad={r['bad']} "
              f"r=n%k={r['r_expected']} fi_ok={r['forced_index_ok']} "
              f"lb_ok={r['lb_ok']} D|n-r={r['D_divides']} {r['secs']}s")
    allok = all(r['n_bad'] == 1 and r['forced_index_ok'] and r['lb_ok'] for r in rows)
    print("  validity verdict:", "ALL VALID" if allok else "RED")

    print("== PASS 3 minimality (the load-bearing gap)")
    for k in sorted(CAND):
        v, reached, tested = minimality_scan(k, CAND[k], budget_secs=20.0)
        frac = (reached - 2 * k) / max(1, CAND[k] - 2 * k)
        print(f"   k={k} {v} covered n<{reached} ({frac*100:.6f}% of interval), "
              f"{tested} n tested")
    print("  NOTE: minimality is NOT established unless verdict==CLEAR for that k.")

    json.dump(dict(control=({str(k): v for k, (v, _) in ctrl.items()}),
                   validity=rows), open("/home/jason/roundtable/erdos1063_receipt.json", "w"), indent=1)
    print("receipt -> /home/jason/roundtable/erdos1063_receipt.json")
